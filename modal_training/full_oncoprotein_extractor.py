"""
Full 14-Oncoprotein Multi-Task Dataset Extractor
Using ChEMBL API - Fast and reliable alternative to 5GB database download
Prepares data specifically for ChemBERTa multi-task training
"""

import modal
import requests
import pandas as pd
import numpy as np
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "requests",
    "pandas", 
    "numpy",
    "rdkit",
    "matplotlib",
    "seaborn",
    "pyarrow"
])

app = modal.App("full-oncoprotein-extractor")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)

# ChEMBL API Configuration
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
REQUEST_DELAY = 0.3  # Balanced speed vs API limits

# All 14 Oncoprotein targets (verified ChEMBL IDs)
ONCOPROTEIN_TARGETS = {
    "EGFR": "CHEMBL203",     # Epidermal growth factor receptor
    "HER2": "CHEMBL1824",    # ErbB2/HER2  
    "VEGFR2": "CHEMBL279",   # VEGFR2
    "BRAF": "CHEMBL5145",    # Serine/threonine-protein kinase B-raf
    "MET": "CHEMBL3717",     # Hepatocyte growth factor receptor
    "CDK4": "CHEMBL331",     # Cyclin-dependent kinase 4
    "CDK6": "CHEMBL3974",    # Cyclin-dependent kinase 6
    "ALK": "CHEMBL4247",     # Anaplastic lymphoma kinase
    "MDM2": "CHEMBL5023",    # MDM2 proto-oncogene
    "STAT3": "CHEMBL5407",   # STAT3
    "RRM2": "CHEMBL3352",    # Ribonucleotide reductase M2
    "CTNNB1": "CHEMBL6132",  # β-catenin
    "MYC": "CHEMBL6130",     # MYC proto-oncogene
    "PI3KCA": "CHEMBL4040"   # PIK3CA
}

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=4.0,
    memory=16384,  # More memory for large dataset
    timeout=7200   # 2 hours for full extraction
)
def extract_full_oncoprotein_dataset():
    """
    Extract bioactivity data for all 14 oncoproteins
    Creates multi-task dataset ready for ChemBERTa training
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Full 14-Oncoprotein Multi-Task Dataset Extraction")
    logger.info("=" * 70)
    logger.info(f"📋 Targets: {', '.join(ONCOPROTEIN_TARGETS.keys())}")
    
    def make_api_request(url, params=None):
        """Make API request with error handling"""
        time.sleep(REQUEST_DELAY)
        
        if params is None:
            params = {}
        params['format'] = 'json'
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed for {url}: {e}")
            return None
    
    all_data = []
    target_stats = {}
    global_smiles_cache = {}  # Cache SMILES across all targets
    
    # Process each target
    for target_idx, (target_name, chembl_id) in enumerate(ONCOPROTEIN_TARGETS.items(), 1):
        logger.info(f"\n🎯 [{target_idx}/14] Processing {target_name} ({chembl_id})...")
        
        # Get activities for this target
        activities_url = f"{CHEMBL_BASE_URL}/activity"
        params = {
            'target_chembl_id': chembl_id,
            'limit': 5000,  # Get more data per target
            'format': 'json'
        }
        
        logger.info(f"🔍 Querying activities for {target_name}...")
        data = make_api_request(activities_url, params)
        
        if not data or 'activities' not in data:
            logger.warning(f"⚠️ No data found for {target_name}")
            target_stats[target_name] = {
                'total_activities': 0,
                'valid_activities': 0,
                'unique_compounds': 0
            }
            continue
        
        activities = data['activities']
        logger.info(f"📊 Found {len(activities)} total activities for {target_name}")
        
        # Process activities
        target_data = []
        processed_count = 0
        
        for activity in activities:
            processed_count += 1
            if processed_count % 200 == 0:
                logger.info(f"  Processed {processed_count}/{len(activities)} activities for {target_name}...")
            
            # Filter for relevant data
            if not activity.get('standard_value'):
                continue
            
            standard_type = activity.get('standard_type', '')
            standard_units = activity.get('standard_units', '')
            
            # Focus on IC50, Ki, EC50 in nM
            if standard_type not in ['IC50', 'Ki', 'EC50']:
                continue
            if standard_units != 'nM':
                continue
            
            molecule_chembl_id = activity.get('molecule_chembl_id')
            if not molecule_chembl_id:
                continue
            
            try:
                standard_value = float(activity['standard_value'])
                if standard_value <= 0 or standard_value > 1000000:  # Filter extreme values
                    continue
            except:
                continue
            
            # Get SMILES (use global cache)
            if molecule_chembl_id not in global_smiles_cache:
                molecule_url = f"{CHEMBL_BASE_URL}/molecule/{molecule_chembl_id}"
                molecule_data = make_api_request(molecule_url)
                
                if not molecule_data or 'molecule_structures' not in molecule_data:
                    global_smiles_cache[molecule_chembl_id] = None
                    continue
                
                structures = molecule_data['molecule_structures']
                if not structures or not structures.get('canonical_smiles'):
                    global_smiles_cache[molecule_chembl_id] = None
                    continue
                
                smiles = structures['canonical_smiles']
                
                # Validate SMILES with RDKit
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        global_smiles_cache[molecule_chembl_id] = None
                        continue
                except:
                    global_smiles_cache[molecule_chembl_id] = None
                    continue
                
                global_smiles_cache[molecule_chembl_id] = smiles
            
            smiles = global_smiles_cache[molecule_chembl_id]
            if not smiles:
                continue
            
            # Calculate pIC50
            try:
                pIC50 = -np.log10(standard_value / 1e9)  # Convert nM to M, then -log10
                if pIC50 < 0 or pIC50 > 12:  # Filter unrealistic pIC50 values
                    continue
            except:
                continue
            
            # Create record
            record = {
                'canonical_smiles': smiles,
                'target_name': target_name,
                'target_chembl_id': chembl_id,
                'standard_type': standard_type,
                'standard_value': standard_value,
                'pIC50': pIC50,
                'molecule_chembl_id': molecule_chembl_id
            }
            
            target_data.append(record)
            all_data.append(record)
        
        # Calculate stats for this target
        unique_compounds = len(set(record['canonical_smiles'] for record in target_data))
        target_stats[target_name] = {
            'total_activities': len(activities),
            'valid_activities': len(target_data),
            'unique_compounds': unique_compounds
        }
        
        logger.info(f"✅ {target_name}: {len(target_data)} valid activities, {unique_compounds} unique compounds")
        
        # Progress update
        logger.info(f"📈 Overall progress: {target_idx}/14 targets completed")
    
    logger.info(f"\n📊 Creating consolidated multi-task dataset...")
    df = pd.DataFrame(all_data)
    
    if df.empty:
        raise ValueError("❌ No bioactivity data retrieved for any target")
    
    logger.info(f"📈 Retrieved {len(df)} total bioactivity records across all targets")
    
    # Aggregate multiple measurements for same compound-target pairs
    logger.info("🔄 Aggregating multiple measurements per compound-target pair...")
    
    aggregated = df.groupby(['canonical_smiles', 'target_name']).agg({
        'pIC50': 'median',  # Use median for robustness
        'standard_value': 'median',
        'standard_type': lambda x: ','.join(set(x)),
        'molecule_chembl_id': 'first'
    }).reset_index()
    
    logger.info(f"📈 After aggregation: {len(aggregated)} unique compound-target pairs")
    
    # Create multi-task matrix (pivot table)
    logger.info("🔄 Creating multi-task dataset matrix...")
    
    pivot_table = aggregated.pivot(
        index='canonical_smiles', 
        columns='target_name', 
        values='pIC50'
    ).reset_index()
    
    # Ensure all 14 targets are present as columns (fill missing with NaN)
    for target in ONCOPROTEIN_TARGETS.keys():
        if target not in pivot_table.columns:
            pivot_table[target] = np.nan
    
    # Reorder columns: canonical_smiles first, then targets in consistent order
    target_columns = list(ONCOPROTEIN_TARGETS.keys())
    column_order = ['canonical_smiles'] + target_columns
    pivot_table = pivot_table[column_order]
    
    logger.info(f"📊 Multi-task dataset shape: {pivot_table.shape}")
    logger.info(f"📊 Unique compounds: {len(pivot_table)}")
    logger.info(f"📊 Targets: {len(target_columns)}")
    
    # Calculate coverage statistics
    logger.info("📊 Calculating target coverage statistics...")
    coverage_stats = {}
    for target in target_columns:
        if target in pivot_table.columns:
            non_null_count = pivot_table[target].notna().sum()
            total_compounds = len(pivot_table)
            coverage_percent = (non_null_count / total_compounds) * 100
            
            coverage_stats[target] = {
                'compounds_with_data': int(non_null_count),
                'total_compounds': int(total_compounds),
                'coverage_percent': round(coverage_percent, 2)
            }
            
            logger.info(f"  {target}: {non_null_count}/{total_compounds} compounds ({coverage_percent:.1f}%)")
    
    # Save datasets
    logger.info("💾 Saving multi-task oncoprotein dataset...")
    
    datasets_dir = Path("/vol/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main dataset
    csv_path = datasets_dir / "oncoprotein_multitask_dataset.csv"
    parquet_path = datasets_dir / "oncoprotein_multitask_dataset.parquet"
    
    pivot_table.to_csv(csv_path, index=False)
    pivot_table.to_parquet(parquet_path, index=False)
    
    logger.info(f"✅ Saved CSV: {csv_path}")
    logger.info(f"✅ Saved Parquet: {parquet_path}")
    
    # Save metadata
    metadata = {
        'extraction_method': 'ChEMBL_API_Full',
        'targets': target_columns,
        'target_chembl_ids': ONCOPROTEIN_TARGETS,
        'total_compounds': len(pivot_table),
        'total_records_before_aggregation': len(df),
        'coverage_stats': coverage_stats,
        'target_stats': target_stats,
        'dataset_shape': list(pivot_table.shape),
        'ready_for_chemberta': True
    }
    
    metadata_path = datasets_dir / "oncoprotein_multitask_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Saved metadata: {metadata_path}")
    
    # Generate summary report
    logger.info("\n🎉 FULL ONCOPROTEIN DATASET EXTRACTION COMPLETED!")
    logger.info("=" * 70)
    logger.info(f"📁 Dataset files:")
    logger.info(f"  • CSV: {csv_path}")
    logger.info(f"  • Parquet: {parquet_path}")
    logger.info(f"  • Metadata: {metadata_path}")
    logger.info(f"\n📊 Dataset summary:")
    logger.info(f"  • Total compounds: {len(pivot_table):,}")
    logger.info(f"  • Targets: {len(target_columns)}")
    logger.info(f"  • Matrix shape: {pivot_table.shape}")
    logger.info(f"  • Ready for ChemBERTa training: ✅")
    
    return {
        'status': 'success',
        'dataset_path': str(csv_path),
        'parquet_path': str(parquet_path),
        'metadata_path': str(metadata_path),
        'total_compounds': len(pivot_table),
        'targets_count': len(target_columns),
        'dataset_shape': list(pivot_table.shape),
        'coverage_stats': coverage_stats,
        'ready_for_chemberta': True
    }

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=2.0,
    memory=4096,
    timeout=600
)
def validate_oncoprotein_dataset():
    """
    Validate the extracted oncoprotein dataset
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Validating full oncoprotein dataset...")
    
    csv_path = Path("/vol/datasets/oncoprotein_multitask_dataset.csv")
    parquet_path = Path("/vol/datasets/oncoprotein_multitask_dataset.parquet")
    metadata_path = Path("/vol/datasets/oncoprotein_multitask_metadata.json")
    
    if not csv_path.exists():
        return {'status': 'error', 'message': 'Dataset CSV not found'}
    
    try:
        # Load and validate dataset
        df = pd.read_csv(csv_path)
        logger.info(f"✅ Dataset loaded: {df.shape}")
        
        # Check for required columns
        expected_targets = list(ONCOPROTEIN_TARGETS.keys())
        missing_targets = [t for t in expected_targets if t not in df.columns]
        
        if missing_targets:
            logger.warning(f"⚠️ Missing target columns: {missing_targets}")
        
        # Calculate statistics
        stats = {
            'total_compounds': len(df),
            'total_targets': len([c for c in df.columns if c != 'canonical_smiles']),
            'file_sizes': {
                'csv_mb': round(csv_path.stat().st_size / 1e6, 2),
                'parquet_mb': round(parquet_path.stat().st_size / 1e6, 2) if parquet_path.exists() else 0
            }
        }
        
        # Target-specific stats
        target_stats = {}
        for target in expected_targets:
            if target in df.columns:
                non_null = df[target].notna().sum()
                target_stats[target] = {
                    'compounds_with_data': int(non_null),
                    'coverage_percent': round((non_null / len(df)) * 100, 2),
                    'pIC50_range': [
                        round(df[target].min(), 2) if not df[target].isna().all() else None,
                        round(df[target].max(), 2) if not df[target].isna().all() else None
                    ]
                }
        
        stats['target_stats'] = target_stats
        
        # Sample data
        stats['sample_compounds'] = df.head(3)['canonical_smiles'].tolist()
        
        logger.info(f"✅ Validation completed successfully")
        logger.info(f"✅ {stats['total_compounds']} compounds across {stats['total_targets']} targets")
        
        return {'status': 'success', 'stats': stats}
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    pass