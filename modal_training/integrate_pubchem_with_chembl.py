"""
Integrate PubChem BioAssay data with existing ChEMBL dataset
Combines both datasets using standardized format and deduplication
"""

import modal
import pandas as pd
import numpy as np
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Modal setup  
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy",
    "rdkit-pypi"
])

app = modal.App("integrate-pubchem-chembl")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=4.0,
    memory=16384,  # 16GB
    timeout=3600   # 1 hour
)
def integrate_pubchem_with_existing_data():
    """
    Integrate PubChem BioAssay data with existing ChEMBL expanded dataset
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🔗 INTEGRATING PUBCHEM WITH EXISTING CHEMBL DATASET")
    print("=" * 80)
    
    try:
        datasets_dir = Path("/vol/datasets")
        
        # Load existing ChEMBL data
        chembl_raw_path = datasets_dir / "expanded_fixed_raw_data.csv"
        if not chembl_raw_path.exists():
            raise FileNotFoundError("❌ ChEMBL dataset not found. Run fixed_expanded_extractor first.")
        
        print("📊 Loading existing ChEMBL dataset...")
        chembl_df = pd.read_csv(chembl_raw_path)
        print(f"   ChEMBL records: {len(chembl_df):,}")
        print(f"   ChEMBL targets: {chembl_df['target_name'].nunique()}")
        print(f"   ChEMBL compounds: {chembl_df['canonical_smiles'].nunique()}")
        
        # Load PubChem data
        pubchem_raw_path = datasets_dir / "pubchem_bioassay_raw_data.csv"
        if not pubchem_raw_path.exists():
            print("⚠️ PubChem dataset not found. Extracting now...")
            
            # Launch PubChem extraction
            from enhanced_pubchem_extractor import app as pubchem_app, extract_pubchem_bioassay_data
            
            with pubchem_app.run() as app_run:
                pubchem_result = extract_pubchem_bioassay_data.remote()
            
            if pubchem_result['status'] != 'success':
                raise Exception(f"PubChem extraction failed: {pubchem_result.get('error', 'Unknown error')}")
        
        print("📊 Loading PubChem BioAssay dataset...")
        pubchem_df = pd.read_csv(pubchem_raw_path)
        print(f"   PubChem records: {len(pubchem_df):,}")
        print(f"   PubChem targets: {pubchem_df['target_name'].nunique()}")
        print(f"   PubChem compounds: {pubchem_df['canonical_smiles'].nunique()}")
        
        # Standardize column formats
        print("\n🔧 Standardizing dataset formats...")
        
        # Ensure both datasets have same columns
        common_columns = [
            'canonical_smiles', 'target_name', 'target_category', 'activity_type',
            'standard_value', 'standard_units', 'standard_value_nm', 'pic50', 'data_source'
        ]
        
        # Add missing columns with defaults
        for col in common_columns:
            if col not in chembl_df.columns:
                chembl_df[col] = None
            if col not in pubchem_df.columns:
                pubchem_df[col] = None
        
        # Select common columns for integration
        chembl_standardized = chembl_df[common_columns].copy()
        pubchem_standardized = pubchem_df[common_columns].copy()
        
        print(f"   ChEMBL standardized: {len(chembl_standardized)} records")
        print(f"   PubChem standardized: {len(pubchem_standardized)} records")
        
        # Combine datasets
        print("\n🔗 Combining datasets...")
        combined_df = pd.concat([chembl_standardized, pubchem_standardized], ignore_index=True)
        
        print(f"   Combined raw records: {len(combined_df):,}")
        print(f"   Combined targets: {combined_df['target_name'].nunique()}")
        print(f"   Combined compounds: {combined_df['canonical_smiles'].nunique()}")
        
        # Apply advanced deduplication across sources
        print("\n🔄 Applying cross-source deduplication...")
        deduplicated_df = apply_cross_source_deduplication(combined_df)
        
        print(f"   Final deduplicated records: {len(deduplicated_df):,}")
        print(f"   Final targets: {deduplicated_df['target_name'].nunique()}")
        print(f"   Final compounds: {deduplicated_df['canonical_smiles'].nunique()}")
        
        # Data source breakdown
        source_breakdown = deduplicated_df['data_source'].value_counts()
        print(f"\n📊 Final data source breakdown:")
        for source, count in source_breakdown.items():
            print(f"   • {source}: {count:,} records ({count/len(deduplicated_df)*100:.1f}%)")
        
        # Create integrated multi-task matrix
        print("\n🔄 Creating integrated multi-task matrix...")
        
        # Pivot for IC50 data (primary activity type)
        ic50_data = deduplicated_df[deduplicated_df['activity_type'] == 'IC50']
        
        if len(ic50_data) > 0:
            pivot_table = ic50_data.pivot_table(
                index='canonical_smiles',
                columns='target_name', 
                values='pic50',
                aggfunc='median'
            ).reset_index()
            
            print(f"   IC50 matrix: {pivot_table.shape}")
        else:
            # Fallback - use all activity types
            pivot_table = deduplicated_df.pivot_table(
                index='canonical_smiles',
                columns='target_name',
                values='pic50',
                aggfunc='median'
            ).reset_index()
            
            print(f"   Combined matrix: {pivot_table.shape}")
        
        # Save integrated datasets
        print("\n💾 Saving integrated dataset...")
        
        # Save raw integrated data
        integrated_raw_path = datasets_dir / "integrated_chembl_pubchem_raw_data.csv"
        deduplicated_df.to_csv(integrated_raw_path, index=False)
        
        # Save matrix
        integrated_matrix_path = datasets_dir / "integrated_chembl_pubchem_ic50_matrix.csv"
        pivot_table.to_csv(integrated_matrix_path, index=False)
        
        # Create comprehensive metadata
        target_categories = deduplicated_df.groupby('target_category')['target_name'].nunique().to_dict()
        
        metadata = {
            'integration_method': 'ChEMBL_PubChem_Combined',
            'data_sources': ['ChEMBL', 'PubChem_BioAssay'],
            'original_datasets': {
                'chembl': {
                    'records': len(chembl_df),
                    'targets': chembl_df['target_name'].nunique(),
                    'compounds': chembl_df['canonical_smiles'].nunique()
                },
                'pubchem': {
                    'records': len(pubchem_df),
                    'targets': pubchem_df['target_name'].nunique(), 
                    'compounds': pubchem_df['canonical_smiles'].nunique()
                }
            },
            'integrated_dataset': {
                'total_records': len(deduplicated_df),
                'total_targets': deduplicated_df['target_name'].nunique(),
                'total_compounds': deduplicated_df['canonical_smiles'].nunique(),
                'target_categories': target_categories,
                'source_breakdown': source_breakdown.to_dict()
            },
            'matrix_shape': pivot_table.shape,
            'integration_timestamp': datetime.now().isoformat(),
            'data_quality': {
                'cross_source_deduplication': True,
                'variance_threshold': '100x',
                'standardized_units': 'nM',
                'pic50_calculation': True,
                'missing_data_removed': True
            }
        }
        
        metadata_path = datasets_dir / "integrated_chembl_pubchem_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate final report
        print("\n🎉 DATASET INTEGRATION COMPLETED!")
        print("=" * 80)
        print(f"📁 Integrated dataset files:")
        print(f"  • Raw data: {integrated_raw_path}")
        print(f"  • IC50 matrix: {integrated_matrix_path}")
        print(f"  • Metadata: {metadata_path}")
        
        print(f"\n📊 Integration summary:")
        print(f"  • Original ChEMBL: {len(chembl_df):,} records")
        print(f"  • Original PubChem: {len(pubchem_df):,} records")
        print(f"  • Integrated total: {len(deduplicated_df):,} records")
        print(f"  • Boost: {((len(deduplicated_df) - len(chembl_df)) / len(chembl_df) * 100):+.1f}%")
        print(f"  • Final targets: {deduplicated_df['target_name'].nunique()}")
        print(f"  • Final compounds: {deduplicated_df['canonical_smiles'].nunique():,}")
        print(f"  • Matrix shape: {pivot_table.shape}")
        
        # Category breakdown
        print(f"\n📊 Target category breakdown:")
        for category, count in target_categories.items():
            category_records = deduplicated_df[deduplicated_df['target_category'] == category]
            print(f"  • {category.replace('_', ' ').title()}: {len(category_records):,} records across {count} targets")
        
        return {
            'status': 'success',
            'integrated_raw_path': str(integrated_raw_path),
            'integrated_matrix_path': str(integrated_matrix_path),
            'metadata_path': str(metadata_path),
            'original_chembl_records': len(chembl_df),
            'original_pubchem_records': len(pubchem_df),
            'integrated_total_records': len(deduplicated_df),
            'boost_percentage': ((len(deduplicated_df) - len(chembl_df)) / len(chembl_df) * 100),
            'total_targets': deduplicated_df['target_name'].nunique(),
            'total_compounds': deduplicated_df['canonical_smiles'].nunique(),
            'matrix_shape': pivot_table.shape,
            'ready_for_training': True
        }
        
    except Exception as e:
        print(f"❌ INTEGRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def apply_cross_source_deduplication(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply advanced deduplication across multiple data sources
    Prioritizes ChEMBL data when available, supplements with PubChem
    """
    
    print("🔄 Applying cross-source deduplication...")
    
    # Group by compound-target-activity combinations
    grouped = df.groupby(['canonical_smiles', 'target_name', 'activity_type'])
    
    deduplicated_records = []
    discarded_count = 0
    source_priority = {'ChEMBL': 1, 'PubChem_BioAssay': 2}
    
    for (smiles, target, activity_type), group in grouped:
        if len(group) == 1:
            # Single measurement - keep as is
            deduplicated_records.append(group.iloc[0].to_dict())
            continue
        
        # Multiple measurements from different sources
        group = group.copy()
        group['source_priority'] = group['data_source'].map(source_priority).fillna(999)
        
        # Check for values from both sources
        sources = group['data_source'].unique()
        
        if len(sources) == 1:
            # All from same source - apply standard deduplication
            values = group['standard_value_nm'].values
            valid_values = values[~pd.isna(values)]
            
            if len(valid_values) < 2:
                best_record = group.dropna(subset=['standard_value_nm']).iloc[0]
                deduplicated_records.append(best_record.to_dict())
                continue
            
            # Check for >100-fold variance
            max_val = np.max(valid_values)
            min_val = np.min(valid_values)
            
            if max_val / min_val > 100:
                discarded_count += len(group)
                continue
            
            # Use median value
            median_value = np.median(valid_values)
            best_record = group.iloc[0].to_dict()
            best_record.update({
                'standard_value_nm': median_value,
                'pic50': -np.log10(median_value / 1e9) if activity_type in ['IC50', 'EC50', 'Ki'] else None,
                'source_count': len(group),
                'aggregation_method': 'median'
            })
            
            deduplicated_records.append(best_record)
            
        else:
            # Cross-source comparison
            chembl_records = group[group['data_source'] == 'ChEMBL']
            pubchem_records = group[group['data_source'] == 'PubChem_BioAssay']
            
            if len(chembl_records) > 0 and len(pubchem_records) > 0:
                # Both sources have data - compare values
                chembl_values = chembl_records['standard_value_nm'].dropna()
                pubchem_values = pubchem_records['standard_value_nm'].dropna()
                
                if len(chembl_values) > 0 and len(pubchem_values) > 0:
                    chembl_median = np.median(chembl_values)
                    pubchem_median = np.median(pubchem_values)
                    
                    # Check agreement between sources
                    ratio = max(chembl_median, pubchem_median) / min(chembl_median, pubchem_median)
                    
                    if ratio <= 10:  # Good agreement - use ChEMBL (higher priority)
                        best_record = chembl_records.iloc[0].to_dict()
                        best_record.update({
                            'standard_value_nm': chembl_median,
                            'pic50': -np.log10(chembl_median / 1e9) if activity_type in ['IC50', 'EC50', 'Ki'] else None,
                            'cross_source_agreement': True,
                            'pubchem_value_nm': pubchem_median,
                            'agreement_ratio': ratio
                        })
                        deduplicated_records.append(best_record)
                        
                    elif ratio <= 100:  # Moderate agreement - use average weighted by source reliability
                        # Weight ChEMBL more heavily (70% vs 30%)
                        weighted_value = (chembl_median * 0.7) + (pubchem_median * 0.3)
                        
                        best_record = chembl_records.iloc[0].to_dict()
                        best_record.update({
                            'standard_value_nm': weighted_value,
                            'pic50': -np.log10(weighted_value / 1e9) if activity_type in ['IC50', 'EC50', 'Ki'] else None,
                            'cross_source_agreement': 'moderate',
                            'aggregation_method': 'weighted_average',
                            'chembl_value_nm': chembl_median,
                            'pubchem_value_nm': pubchem_median,
                            'agreement_ratio': ratio
                        })
                        deduplicated_records.append(best_record)
                        
                    else:
                        # Poor agreement - discard both
                        discarded_count += len(group)
                        continue
                else:
                    # One source has valid values - use that
                    if len(chembl_values) > 0:
                        best_record = chembl_records.iloc[0].to_dict()
                        best_record.update({
                            'standard_value_nm': np.median(chembl_values),
                            'pic50': -np.log10(np.median(chembl_values) / 1e9) if activity_type in ['IC50', 'EC50', 'Ki'] else None
                        })
                    else:
                        best_record = pubchem_records.iloc[0].to_dict()
                        best_record.update({
                            'standard_value_nm': np.median(pubchem_values),
                            'pic50': -np.log10(np.median(pubchem_values) / 1e9) if activity_type in ['IC50', 'EC50', 'Ki'] else None
                        })
                    
                    deduplicated_records.append(best_record)
            else:
                # Only one source - use standard deduplication
                source_group = chembl_records if len(chembl_records) > 0 else pubchem_records
                values = source_group['standard_value_nm'].dropna()
                
                if len(values) > 0:
                    median_value = np.median(values)
                    best_record = source_group.iloc[0].to_dict()
                    best_record.update({
                        'standard_value_nm': median_value,
                        'pic50': -np.log10(median_value / 1e9) if activity_type in ['IC50', 'EC50', 'Ki'] else None
                    })
                    deduplicated_records.append(best_record)
    
    result_df = pd.DataFrame(deduplicated_records)
    
    print(f"   ✅ Cross-source deduplication complete:")
    print(f"   📊 Original records: {len(df)}")
    print(f"   📊 Deduplicated records: {len(result_df)}")
    print(f"   🗑️ Discarded (poor agreement/variance): {discarded_count}")
    
    return result_df

if __name__ == "__main__":
    print("🔗 PubChem-ChEMBL Dataset Integration")