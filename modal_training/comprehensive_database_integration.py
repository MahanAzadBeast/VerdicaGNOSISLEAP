"""
Comprehensive Multi-Database Integration
Integrates ChEMBL, PubChem, BindingDB, and DTC into unified training dataset
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

app = modal.App("comprehensive-database-integration")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=8.0,
    memory=32768,  # 32GB for large dataset processing
    timeout=7200   # 2 hours
)
def integrate_all_databases():
    """
    Integrate all four databases: ChEMBL, PubChem, BindingDB, DTC
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🔗 COMPREHENSIVE MULTI-DATABASE INTEGRATION")
    print("=" * 80)
    print("🎯 Integrating: ChEMBL + PubChem + BindingDB + DTC")
    
    try:
        datasets_dir = Path("/vol/datasets")
        
        # Step 1: Launch all database extractions if needed
        print("📊 STEP 1: Ensuring all databases are extracted...")
        print("-" * 60)
        
        database_results = {}
        
        # Check and extract PubChem (enhanced)
        pubchem_path = datasets_dir / "pubchem_enhanced_raw_data.csv"
        if not pubchem_path.exists():
            print("🧪 Extracting PubChem BioAssay data...")
            from enhanced_pubchem_fixed import app as pubchem_app, extract_enhanced_pubchem_data
            
            with pubchem_app.run() as app_run:
                pubchem_result = extract_enhanced_pubchem_data.remote()
            
            if pubchem_result['status'] != 'success':
                raise Exception(f"PubChem extraction failed: {pubchem_result.get('error')}")
            
            database_results['pubchem'] = pubchem_result
            print(f"   ✅ PubChem: {pubchem_result['total_records']} records")
        else:
            print("   ✅ PubChem: Already available")
        
        # Check and extract BindingDB
        bindingdb_path = datasets_dir / "bindingdb_raw_data.csv"
        if not bindingdb_path.exists():
            print("🔗 Extracting BindingDB data...")
            from bindingdb_extractor import app as bindingdb_app, extract_bindingdb_data
            
            with bindingdb_app.run() as app_run:
                bindingdb_result = extract_bindingdb_data.remote()
            
            if bindingdb_result['status'] != 'success':
                raise Exception(f"BindingDB extraction failed: {bindingdb_result.get('error')}")
            
            database_results['bindingdb'] = bindingdb_result
            print(f"   ✅ BindingDB: {bindingdb_result['total_records']} records")
        else:
            print("   ✅ BindingDB: Already available")
        
        # Check and extract DTC
        dtc_path = datasets_dir / "dtc_raw_data.csv"
        if not dtc_path.exists():
            print("🔬 Extracting DTC data...")
            from dtc_extractor import app as dtc_app, extract_dtc_data
            
            with dtc_app.run() as app_run:
                dtc_result = extract_dtc_data.remote()
            
            if dtc_result['status'] != 'success':
                raise Exception(f"DTC extraction failed: {dtc_result.get('error')}")
            
            database_results['dtc'] = dtc_result
            print(f"   ✅ DTC: {dtc_result['total_records']} records")
        else:
            print("   ✅ DTC: Already available")
        
        # Step 2: Load all databases
        print(f"\n📊 STEP 2: Loading all databases...")
        print("-" * 60)
        
        # Load ChEMBL (existing)
        chembl_path = datasets_dir / "expanded_fixed_raw_data.csv"
        if not chembl_path.exists():
            raise FileNotFoundError("ChEMBL dataset not found. Run extraction first.")
        
        print("Loading ChEMBL dataset...")
        chembl_df = pd.read_csv(chembl_path)
        print(f"   ChEMBL: {len(chembl_df):,} records, {chembl_df['target_name'].nunique()} targets")
        
        # Load PubChem
        print("Loading PubChem dataset...")
        pubchem_df = pd.read_csv(datasets_dir / "pubchem_enhanced_raw_data.csv")
        print(f"   PubChem: {len(pubchem_df):,} records, {pubchem_df['target_name'].nunique()} targets")
        
        # Load BindingDB
        print("Loading BindingDB dataset...")
        bindingdb_df = pd.read_csv(datasets_dir / "bindingdb_raw_data.csv")
        print(f"   BindingDB: {len(bindingdb_df):,} records, {bindingdb_df['target_name'].nunique()} targets")
        
        # Load DTC
        print("Loading DTC dataset...")
        dtc_df = pd.read_csv(datasets_dir / "dtc_raw_data.csv")
        print(f"   DTC: {len(dtc_df):,} records, {dtc_df['target_name'].nunique()} targets")
        
        # Step 3: Standardize all datasets
        print(f"\n🔧 STEP 3: Standardizing datasets...")
        print("-" * 60)
        
        # Ensure all datasets have same columns
        common_columns = [
            'canonical_smiles', 'target_name', 'target_category', 'activity_type',
            'standard_value', 'standard_units', 'standard_value_nm', 'pic50', 'data_source'
        ]
        
        datasets = {
            'ChEMBL': chembl_df,
            'PubChem': pubchem_df,
            'BindingDB': bindingdb_df,
            'DTC': dtc_df
        }
        
        standardized_datasets = {}
        
        for db_name, df in datasets.items():
            print(f"Standardizing {db_name}...")
            
            # Add missing columns with defaults
            for col in common_columns:
                if col not in df.columns:
                    if col == 'data_source':
                        df[col] = db_name
                    else:
                        df[col] = None
            
            # Select common columns
            standardized_df = df[common_columns].copy()
            
            # Ensure data_source is correct
            standardized_df['data_source'] = db_name
            
            standardized_datasets[db_name] = standardized_df
            print(f"   ✅ {db_name}: {len(standardized_df)} records standardized")
        
        # Step 4: Advanced multi-source integration
        print(f"\n🔗 STEP 4: Advanced multi-source integration...")
        print("-" * 60)
        
        # Combine all datasets
        combined_df = pd.concat(list(standardized_datasets.values()), ignore_index=True)
        print(f"   Combined raw records: {len(combined_df):,}")
        
        # Apply comprehensive deduplication
        print("Applying comprehensive cross-source deduplication...")
        deduplicated_df = apply_comprehensive_deduplication(combined_df)
        
        print(f"   Final integrated records: {len(deduplicated_df):,}")
        print(f"   Final targets: {deduplicated_df['target_name'].nunique()}")
        print(f"   Final compounds: {deduplicated_df['canonical_smiles'].nunique()}")
        
        # Data source breakdown
        source_breakdown = deduplicated_df['data_source'].value_counts()
        print(f"\n📊 Final data source breakdown:")
        for source, count in source_breakdown.items():
            print(f"   • {source}: {count:,} records ({count/len(deduplicated_df)*100:.1f}%)")
        
        # Step 5: Create comprehensive training matrices
        print(f"\n🔄 STEP 5: Creating comprehensive training matrices...")
        print("-" * 60)
        
        # Create activity-specific matrices
        activity_matrices = {}
        
        for activity_type in deduplicated_df['activity_type'].unique():
            if pd.isna(activity_type):
                continue
                
            activity_df = deduplicated_df[deduplicated_df['activity_type'] == activity_type]
            
            if len(activity_df) > 100:  # Only create matrix if sufficient data
                pivot_table = activity_df.pivot_table(
                    index='canonical_smiles',
                    columns='target_name',
                    values='pic50',
                    aggfunc='median'
                ).reset_index()
                
                activity_matrices[activity_type] = pivot_table
                print(f"   ✅ {activity_type} matrix: {pivot_table.shape}")
        
        # Create comprehensive matrix (all activity types combined)
        print("Creating comprehensive multi-activity matrix...")
        comprehensive_matrix = deduplicated_df.pivot_table(
            index='canonical_smiles',
            columns='target_name',
            values='pic50',
            aggfunc='median'
        ).reset_index()
        
        print(f"   ✅ Comprehensive matrix: {comprehensive_matrix.shape}")
        
        # Step 6: Save integrated datasets
        print(f"\n💾 STEP 6: Saving comprehensive integrated dataset...")
        print("-" * 60)
        
        # Save raw integrated data
        integrated_raw_path = datasets_dir / "comprehensive_integrated_raw_data.csv"
        deduplicated_df.to_csv(integrated_raw_path, index=False)
        print(f"   ✅ Raw data: {integrated_raw_path}")
        
        # Save comprehensive matrix
        integrated_matrix_path = datasets_dir / "comprehensive_integrated_matrix.csv"
        comprehensive_matrix.to_csv(integrated_matrix_path, index=False)
        print(f"   ✅ Matrix: {integrated_matrix_path}")
        
        # Save activity-specific matrices
        matrix_paths = {}
        for activity_type, matrix in activity_matrices.items():
            matrix_path = datasets_dir / f"comprehensive_{activity_type.lower()}_matrix.csv"
            matrix.to_csv(matrix_path, index=False)
            matrix_paths[activity_type] = str(matrix_path)
            print(f"   ✅ {activity_type} matrix: {matrix_path}")
        
        # Create comprehensive metadata
        target_categories = deduplicated_df.groupby('target_category')['target_name'].nunique().to_dict()
        activity_breakdown = deduplicated_df['activity_type'].value_counts().to_dict()
        
        metadata = {
            'integration_method': 'Comprehensive_Multi_Database',
            'databases_integrated': list(standardized_datasets.keys()),
            'total_databases': len(standardized_datasets),
            'original_datasets': {
                db_name: {
                    'records': len(df),
                    'targets': df['target_name'].nunique(),
                    'compounds': df['canonical_smiles'].nunique()
                } for db_name, df in standardized_datasets.items()
            },
            'integrated_dataset': {
                'total_records': len(deduplicated_df),
                'total_targets': deduplicated_df['target_name'].nunique(),
                'total_compounds': deduplicated_df['canonical_smiles'].nunique(),
                'target_categories': target_categories,
                'activity_types': activity_breakdown,
                'source_breakdown': source_breakdown.to_dict()
            },
            'matrices': {
                'comprehensive_matrix_shape': comprehensive_matrix.shape,
                'activity_specific_matrices': {k: v.shape for k, v in activity_matrices.items()},
                'matrix_paths': matrix_paths
            },
            'integration_timestamp': datetime.now().isoformat(),
            'data_quality': {
                'cross_source_deduplication': True,
                'variance_threshold': '100x',
                'standardized_units': 'nM',
                'pic50_calculated': True,
                'missing_data_removed': True,
                'comprehensive_integration': True
            }
        }
        
        metadata_path = datasets_dir / "comprehensive_integrated_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate final report
        print(f"\n🎉 COMPREHENSIVE DATABASE INTEGRATION COMPLETED!")
        print("=" * 80)
        print(f"📁 Integrated dataset files:")
        print(f"  • Raw data: {integrated_raw_path}")
        print(f"  • Comprehensive matrix: {integrated_matrix_path}")
        for activity_type, path in matrix_paths.items():
            print(f"  • {activity_type} matrix: {path}")
        print(f"  • Metadata: {metadata_path}")
        
        print(f"\n📊 Integration summary:")
        original_total = sum(len(df) for df in standardized_datasets.values())
        print(f"  • Original total records: {original_total:,}")
        print(f"  • Integrated records: {len(deduplicated_df):,}")
        print(f"  • Deduplication efficiency: {((original_total - len(deduplicated_df)) / original_total * 100):.1f}% overlap removed")
        print(f"  • Final targets: {deduplicated_df['target_name'].nunique()}")
        print(f"  • Final compounds: {deduplicated_df['canonical_smiles'].nunique():,}")
        print(f"  • Training matrix: {comprehensive_matrix.shape}")
        
        # Category and activity breakdown
        print(f"\n📊 Target category breakdown:")
        for category, count in target_categories.items():
            category_records = deduplicated_df[deduplicated_df['target_category'] == category]
            print(f"  • {category.replace('_', ' ').title()}: {len(category_records):,} records across {count} targets")
        
        print(f"\n📊 Activity type breakdown:")
        for activity_type, count in activity_breakdown.items():
            print(f"  • {activity_type}: {count:,} records")
        
        return {
            'status': 'success',
            'integrated_raw_path': str(integrated_raw_path),
            'integrated_matrix_path': str(integrated_matrix_path),
            'metadata_path': str(metadata_path),
            'original_total_records': original_total,
            'integrated_total_records': len(deduplicated_df),
            'deduplication_efficiency': ((original_total - len(deduplicated_df)) / original_total * 100),
            'total_targets': deduplicated_df['target_name'].nunique(),
            'total_compounds': deduplicated_df['canonical_smiles'].nunique(),
            'comprehensive_matrix_shape': comprehensive_matrix.shape,
            'activity_matrices': {k: v.shape for k, v in activity_matrices.items()},
            'databases_integrated': list(standardized_datasets.keys()),
            'ready_for_training': True
        }
        
    except Exception as e:
        print(f"❌ COMPREHENSIVE INTEGRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def apply_comprehensive_deduplication(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply comprehensive deduplication across all four data sources
    Priority: ChEMBL > PubChem > BindingDB > DTC
    """
    
    print("🔄 Applying comprehensive cross-source deduplication...")
    
    # Define source priorities
    source_priority = {'ChEMBL': 1, 'PubChem': 2, 'BindingDB': 3, 'DTC': 4}
    
    # Group by compound-target-activity combinations
    grouped = df.groupby(['canonical_smiles', 'target_name', 'activity_type'])
    
    deduplicated_records = []
    discarded_count = 0
    
    for (smiles, target, activity_type), group in grouped:
        if len(group) == 1:
            # Single measurement - keep as is
            deduplicated_records.append(group.iloc[0].to_dict())
            continue
        
        # Multiple measurements from different sources
        group = group.copy()
        group['source_priority'] = group['data_source'].map(source_priority).fillna(999)
        
        # Get unique sources for this compound-target pair
        sources = group['data_source'].unique()
        
        if len(sources) == 1:
            # All from same source - apply standard deduplication
            values = group['standard_value_nm'].dropna()
            
            if len(values) < 2:
                best_record = group.dropna(subset=['standard_value_nm']).iloc[0]
                deduplicated_records.append(best_record.to_dict())
                continue
            
            # Check for >100-fold variance
            max_val = np.max(values)
            min_val = np.min(values)
            
            if max_val / min_val > 100:
                discarded_count += len(group)
                continue
            
            # Use median value
            median_value = np.median(values)
            best_record = group.iloc[0].to_dict()
            best_record.update({
                'standard_value_nm': median_value,
                'pic50': -np.log10(median_value / 1e9) if activity_type in ['IC50', 'EC50', 'Ki'] else None,
                'source_count': len(group),
                'aggregation_method': 'median'
            })
            
            deduplicated_records.append(best_record)
            
        else:
            # Cross-source comparison with priority system
            # Sort by source priority
            group_sorted = group.sort_values('source_priority')
            
            # Get values from each source
            source_values = {}
            for source in sources:
                source_data = group[group['data_source'] == source]
                source_values[source] = source_data['standard_value_nm'].dropna()
            
            # Find the highest priority source with valid data
            best_source = None
            best_value = None
            
            for source in ['ChEMBL', 'PubChem', 'BindingDB', 'DTC']:
                if source in source_values and len(source_values[source]) > 0:
                    best_source = source
                    best_value = np.median(source_values[source])
                    break
            
            if best_source and best_value:
                # Check agreement between sources (if multiple sources available)
                if len(sources) > 1:
                    all_medians = [np.median(vals) for vals in source_values.values() if len(vals) > 0]
                    if len(all_medians) > 1:
                        max_median = np.max(all_medians)
                        min_median = np.min(all_medians)
                        
                        if max_median / min_median > 100:
                            # Too much disagreement - discard
                            discarded_count += len(group)
                            continue
                
                # Use the best source data
                best_record = group[group['data_source'] == best_source].iloc[0].to_dict()
                best_record.update({
                    'standard_value_nm': best_value,
                    'pic50': -np.log10(best_value / 1e9) if activity_type in ['IC50', 'EC50', 'Ki'] else None,
                    'cross_source_data': True,
                    'sources_available': list(sources),
                    'selected_source': best_source
                })
                
                deduplicated_records.append(best_record)
    
    result_df = pd.DataFrame(deduplicated_records)
    
    print(f"   ✅ Comprehensive deduplication complete:")
    print(f"   📊 Original records: {len(df)}")
    print(f"   📊 Deduplicated records: {len(result_df)}")
    print(f"   🗑️ Discarded (high variance/disagreement): {discarded_count}")
    print(f"   📊 Efficiency: {((len(df) - len(result_df)) / len(df) * 100):.1f}% overlap removed")
    
    return result_df

if __name__ == "__main__":
    print("🔗 Comprehensive Multi-Database Integration System")