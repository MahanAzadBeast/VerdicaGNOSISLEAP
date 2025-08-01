"""
Fixed Expanded Multi-Source Data Extraction Pipeline
Addresses issues found in testing and uses existing successful oncoprotein data as foundation
"""

import modal
import pandas as pd
import numpy as np
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "requests",
    "pandas", 
    "numpy",
    "rdkit-pypi",
    "matplotlib",
    "seaborn"
])

app = modal.App("fixed-expanded-extractor")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

# EXPANDED TARGET LIST (23 targets)
EXPANDED_TARGETS = {
    # ONCOPROTEINS (Current - 10)
    "EGFR": {"chembl_id": "CHEMBL203", "category": "oncoprotein"},
    "HER2": {"chembl_id": "CHEMBL1824", "category": "oncoprotein"}, 
    "VEGFR2": {"chembl_id": "CHEMBL279", "category": "oncoprotein"},
    "BRAF": {"chembl_id": "CHEMBL5145", "category": "oncoprotein"},
    "MET": {"chembl_id": "CHEMBL3717", "category": "oncoprotein"},
    "CDK4": {"chembl_id": "CHEMBL331", "category": "oncoprotein"},
    "CDK6": {"chembl_id": "CHEMBL3974", "category": "oncoprotein"},
    "ALK": {"chembl_id": "CHEMBL4247", "category": "oncoprotein"},
    "MDM2": {"chembl_id": "CHEMBL5023", "category": "oncoprotein"},
    "PI3KCA": {"chembl_id": "CHEMBL4040", "category": "oncoprotein"},
    
    # TUMOR SUPPRESSORS (New - 7)
    "TP53": {"chembl_id": "CHEMBL4722", "category": "tumor_suppressor"},
    "RB1": {"chembl_id": "CHEMBL4462", "category": "tumor_suppressor"},
    "PTEN": {"chembl_id": "CHEMBL4792", "category": "tumor_suppressor"},
    "APC": {"chembl_id": "CHEMBL3778", "category": "tumor_suppressor"},
    "BRCA1": {"chembl_id": "CHEMBL5462", "category": "tumor_suppressor"},
    "BRCA2": {"chembl_id": "CHEMBL5856", "category": "tumor_suppressor"},
    "VHL": {"chembl_id": "CHEMBL5827", "category": "tumor_suppressor"},
    
    # METASTASIS SUPPRESSORS (New - 6)  
    "NDRG1": {"chembl_id": "CHEMBL1075104", "category": "metastasis_suppressor"},
    "KAI1": {"chembl_id": "CHEMBL1075318", "category": "metastasis_suppressor"},
    "KISS1": {"chembl_id": "CHEMBL1075167", "category": "metastasis_suppressor"},
    "NM23H1": {"chembl_id": "CHEMBL1075142", "category": "metastasis_suppressor"},
    "RKIP": {"chembl_id": "CHEMBL1075089", "category": "metastasis_suppressor"},
    "CASP8": {"chembl_id": "CHEMBL4681", "category": "metastasis_suppressor"}
}

def extract_chembl_target_data(target_name: str, chembl_id: str, limit: int = 2000) -> List[Dict]:
    """Extract bioactivity data for a single target from ChEMBL"""
    
    print(f"🎯 Extracting {target_name} ({chembl_id})...")
    
    base_url = "https://www.ebi.ac.uk/chembl/api/data"
    activities_url = f"{base_url}/activity"
    
    all_records = []
    offset = 0
    
    while len(all_records) < limit:
        params = {
            'target_chembl_id': chembl_id,
            'limit': min(1000, limit - len(all_records)),
            'offset': offset,
            'format': 'json'
        }
        
        try:
            time.sleep(0.2)  # Rate limiting
            response = requests.get(activities_url, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"   ⚠️ HTTP {response.status_code} for {target_name}")
                break
            
            data = response.json()
            activities = data.get('activities', [])
            
            if not activities:
                break
            
            # Process activities
            for activity in activities:
                record = process_chembl_activity(activity, target_name, EXPANDED_TARGETS[target_name])
                if record:
                    all_records.append(record)
            
            # Check pagination
            page_meta = data.get('page_meta', {})
            if not page_meta.get('next'):
                break
            
            offset += params['limit']
            print(f"   📊 {target_name}: {len(all_records)} records collected...")
            
        except Exception as e:
            print(f"   ❌ Error for {target_name}: {e}")
            break
    
    print(f"   ✅ {target_name}: {len(all_records)} total records")
    return all_records

def process_chembl_activity(activity: Dict, target_name: str, target_info: Dict) -> Optional[Dict]:
    """Process a single ChEMBL activity record"""
    
    try:
        # Extract key fields
        canonical_smiles = activity.get('canonical_smiles')
        standard_type = activity.get('standard_type')
        standard_value = activity.get('standard_value')
        standard_units = activity.get('standard_units')
        
        # Validate essential fields
        if not all([canonical_smiles, standard_type, standard_value, standard_units]):
            return None
        
        # Check if activity type is relevant
        if standard_type not in ['IC50', 'EC50', 'Ki']:
            return None
        
        # Validate SMILES (simple check)
        if len(canonical_smiles) < 5 or not canonical_smiles:
            return None
        
        # Standardize units to nM
        try:
            value_nm = standardize_to_nm(float(standard_value), standard_units)
            if value_nm is None:
                return None
        except:
            return None
        
        # Calculate pIC50/pEC50/pKi
        try:
            pic50 = -np.log10(value_nm / 1e9)  # Convert nM to M then -log10
            if not (0 < pic50 < 15):  # Reasonable range
                return None
        except:
            return None
        
        # Create record
        record = {
            'canonical_smiles': canonical_smiles,
            'target_name': target_name,
            'target_category': target_info['category'],
            'activity_type': standard_type,
            'standard_value': float(standard_value),
            'standard_units': standard_units,
            'standard_value_nm': value_nm,
            'pic50': pic50,
            'molecule_chembl_id': activity.get('molecule_chembl_id'),
            'assay_chembl_id': activity.get('assay_chembl_id'),
            'data_source': 'ChEMBL'
        }
        
        return record
        
    except Exception as e:
        return None

def standardize_to_nm(value: float, units: str) -> Optional[float]:
    """Standardize concentration units to nM"""
    
    if not value or value <= 0:
        return None
    
    # Conversion factors to nM
    conversions = {
        'nM': 1.0,
        'uM': 1000.0,
        'mM': 1000000.0,
        'M': 1000000000.0
    }
    
    if units in conversions:
        return value * conversions[units]
    
    return None

def deduplicate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply user-specified deduplication rules"""
    
    print("🔄 Applying deduplication rules...")
    
    # Group by compound-target pairs
    grouped = df.groupby(['canonical_smiles', 'target_name', 'activity_type'])
    
    deduplicated_records = []
    discarded_count = 0
    
    for (smiles, target, activity_type), group in grouped:
        if len(group) == 1:
            # Single measurement - keep as is
            deduplicated_records.append(group.iloc[0].to_dict())
            continue
        
        # Multiple measurements - check variance
        values = group['standard_value_nm'].values
        valid_values = values[~pd.isna(values)]
        
        if len(valid_values) < 2:
            # Use the single valid value
            best_record = group.dropna(subset=['standard_value_nm']).iloc[0]
            deduplicated_records.append(best_record.to_dict())
            continue
        
        # Check for >100-fold variance
        max_val = np.max(valid_values)
        min_val = np.min(valid_values)
        
        if max_val / min_val > 100:
            # Too much variance - discard
            discarded_count += len(group)
            print(f"   Discarded {target}/{activity_type}: {min_val:.1f}-{max_val:.1f} nM (>100x variance)")
            continue
        
        # Use median value
        median_value = np.median(valid_values)
        median_pic50 = -np.log10(median_value / 1e9) if activity_type in ['IC50', 'EC50', 'Ki'] else None
        
        # Create aggregated record
        aggregated_record = group.iloc[0].to_dict()
        aggregated_record.update({
            'standard_value_nm': median_value,
            'pic50': median_pic50,
            'source_count': len(group),
            'aggregation_method': 'median'
        })
        
        deduplicated_records.append(aggregated_record)
    
    result_df = pd.DataFrame(deduplicated_records)
    
    print(f"   ✅ Deduplication complete:")
    print(f"   📊 Original records: {len(df)}")
    print(f"   📊 Deduplicated records: {len(result_df)}")
    print(f"   🗑️ Discarded (>100x variance): {discarded_count}")
    
    return result_df

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=4.0,
    memory=16384,  # 16GB
    timeout=7200   # 2 hours timeout
)
def extract_fixed_expanded_dataset():
    """
    Fixed expanded dataset extraction with robust error handling
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🚀 FIXED EXPANDED MULTI-SOURCE EXTRACTION STARTED")
    print("=" * 80)
    print(f"📋 Total targets: {len(EXPANDED_TARGETS)}")
    print(f"🎯 Oncoproteins: {len([t for t in EXPANDED_TARGETS.values() if t['category'] == 'oncoprotein'])}")
    print(f"🔒 Tumor Suppressors: {len([t for t in EXPANDED_TARGETS.values() if t['category'] == 'tumor_suppressor'])}")
    print(f"🚫 Metastasis Suppressors: {len([t for t in EXPANDED_TARGETS.values() if t['category'] == 'metastasis_suppressor'])}")
    
    try:
        all_records = []
        target_stats = {}
        
        # Extract data for each target
        for target_idx, (target_name, target_info) in enumerate(EXPANDED_TARGETS.items(), 1):
            print(f"\n📍 [{target_idx}/{len(EXPANDED_TARGETS)}] Processing {target_name} ({target_info['category']})...")
            
            try:
                target_records = extract_chembl_target_data(
                    target_name, 
                    target_info['chembl_id'], 
                    limit=2000  # Reasonable limit per target
                )
                
                all_records.extend(target_records)
                
                target_stats[target_name] = {
                    'category': target_info['category'],
                    'total_records': len(target_records),
                    'chembl_id': target_info['chembl_id']
                }
                
                print(f"   ✅ {target_name}: {len(target_records)} records added")
                
            except Exception as e:
                print(f"   ❌ {target_name} failed: {e}")
                target_stats[target_name] = {
                    'category': target_info['category'],
                    'total_records': 0,
                    'error': str(e)
                }
        
        if not all_records:
            raise ValueError("❌ No bioactivity data retrieved from ChEMBL")
        
        print(f"\n📊 RAW DATA SUMMARY:")
        print(f"   📈 Total records: {len(all_records)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        
        print(f"   📊 DataFrame shape: {df.shape}")
        print(f"   📊 Unique targets: {df['target_name'].nunique()}")
        print(f"   📊 Unique compounds: {df['canonical_smiles'].nunique()}")
        
        # Apply data quality control
        print("\n🔍 APPLYING DATA QUALITY CONTROL...")
        
        # 1. Remove rows with missing essential data
        initial_count = len(df)
        df = df.dropna(subset=['canonical_smiles', 'target_name', 'standard_value_nm'])
        print(f"   📊 After removing missing data: {len(df)} records (removed {initial_count - len(df)})")
        
        # 2. Apply deduplication rules
        df = deduplicate_data(df)
        
        print(f"\n📊 FINAL CLEANED DATA:")
        print(f"   📈 Total records: {len(df)}")
        print(f"   📊 Unique targets: {df['target_name'].nunique()}")
        print(f"   📊 Unique compounds: {df['canonical_smiles'].nunique()}")
        
        # Create multi-task dataset
        print("\n🔄 Creating multi-task dataset matrix...")
        
        # Pivot for IC50 data (primary activity type)
        ic50_data = df[df['activity_type'] == 'IC50']
        
        if len(ic50_data) > 0:
            pivot_table = ic50_data.pivot_table(
                index='canonical_smiles',
                columns='target_name', 
                values='pic50',
                aggfunc='median'
            ).reset_index()
            
            print(f"   📊 IC50 matrix: {pivot_table.shape}")
        else:
            # Fallback - use all activity types
            pivot_table = df.pivot_table(
                index='canonical_smiles',
                columns='target_name',
                values='pic50',
                aggfunc='median'
            ).reset_index()
            
            print(f"   📊 Combined matrix: {pivot_table.shape}")
        
        # Save datasets
        print("\n💾 Saving expanded dataset...")
        
        datasets_dir = Path("/vol/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw data
        raw_data_path = datasets_dir / "expanded_fixed_raw_data.csv"
        df.to_csv(raw_data_path, index=False)
        
        # Save matrix
        matrix_path = datasets_dir / "expanded_fixed_ic50_matrix.csv"
        pivot_table.to_csv(matrix_path, index=False)
        
        # Save metadata
        metadata = {
            'extraction_method': 'Fixed_Expanded_ChEMBL',
            'targets': list(EXPANDED_TARGETS.keys()),
            'target_info': EXPANDED_TARGETS,
            'total_records': len(df),
            'total_targets': df['target_name'].nunique(),
            'total_compounds': df['canonical_smiles'].nunique(),
            'target_stats': target_stats,
            'matrix_shape': pivot_table.shape,
            'extraction_timestamp': datetime.now().isoformat(),
            'data_quality': {
                'deduplication': True,
                'variance_threshold': '100x',
                'missing_data_removed': True
            }
        }
        
        metadata_path = datasets_dir / "expanded_fixed_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate final report
        print("\n🎉 FIXED EXPANDED EXTRACTION COMPLETED!")
        print("=" * 80)
        print(f"📁 Dataset files:")
        print(f"  • Raw data: {raw_data_path}")
        print(f"  • IC50 matrix: {matrix_path}")
        print(f"  • Metadata: {metadata_path}")
        
        print(f"\n📊 Final dataset summary:")
        print(f"  • Total records: {len(df):,}")
        print(f"  • Unique targets: {df['target_name'].nunique()}")
        print(f"  • Unique compounds: {df['canonical_smiles'].nunique()}")
        print(f"  • Matrix shape: {pivot_table.shape}")
        
        # Category breakdown
        for category in ['oncoprotein', 'tumor_suppressor', 'metastasis_suppressor']:
            category_targets = [name for name, info in EXPANDED_TARGETS.items() if info['category'] == category]
            category_records = df[df['target_name'].isin(category_targets)]
            print(f"  • {category.replace('_', ' ').title()}: {len(category_records)} records across {len(category_targets)} targets")
        
        return {
            'status': 'success',
            'raw_data_path': str(raw_data_path),
            'matrix_path': str(matrix_path),
            'metadata_path': str(metadata_path),
            'total_records': len(df),
            'total_targets': df['target_name'].nunique(),
            'total_compounds': df['canonical_smiles'].nunique(),
            'matrix_shape': pivot_table.shape,
            'ready_for_training': True
        }
        
    except Exception as e:
        print(f"❌ EXTRACTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🚀 Fixed Expanded Multi-Source Dataset Extractor")