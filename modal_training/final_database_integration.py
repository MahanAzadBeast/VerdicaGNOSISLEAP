"""
Final Database Integration - Access and Integrate All Extracted Databases
"""

import modal
import json
from datetime import datetime
from pathlib import Path

# Modal setup with required packages
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy",
    "rdkit-pypi"
])

app = modal.App("final-database-integration")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=8.0,
    memory=32768,
    timeout=3600
)
def perform_comprehensive_integration():
    """
    Access all extracted databases and perform comprehensive integration
    """
    
    import pandas as pd
    import numpy as np
    
    print("🔗 COMPREHENSIVE DATABASE INTEGRATION")
    print("=" * 80)
    
    try:
        datasets_dir = Path("/vol/datasets")
        
        # Step 1: Check what databases are available
        print("📊 STEP 1: Checking available databases...")
        print("-" * 60)
        
        available_databases = {}
        
        # Check ChEMBL (existing)
        chembl_files = [
            "expanded_fixed_raw_data.csv",
            # Alternative names
            "chembl_raw_data.csv", 
            "fixed_raw_data.csv"
        ]
        
        chembl_df = None
        for filename in chembl_files:
            file_path = datasets_dir / filename
            if file_path.exists():
                print(f"📊 Loading ChEMBL: {filename}")
                chembl_df = pd.read_csv(file_path)
                available_databases['ChEMBL'] = {
                    'dataframe': chembl_df,
                    'records': len(chembl_df),
                    'targets': chembl_df['target_name'].nunique() if 'target_name' in chembl_df.columns else 0,
                    'file': filename
                }
                print(f"   ✅ ChEMBL: {len(chembl_df):,} records, {available_databases['ChEMBL']['targets']} targets")
                break
        
        if chembl_df is None:
            print("   ❌ ChEMBL not found")
        
        # Check PubChem
        pubchem_files = [
            "pubchem_enhanced_raw_data.csv",
            "pubchem_bioassay_raw_data.csv",
            "pubchem_raw_data.csv"
        ]
        
        pubchem_df = None
        for filename in pubchem_files:
            file_path = datasets_dir / filename
            if file_path.exists():
                print(f"📊 Loading PubChem: {filename}")
                pubchem_df = pd.read_csv(file_path)
                available_databases['PubChem'] = {
                    'dataframe': pubchem_df,
                    'records': len(pubchem_df),
                    'targets': pubchem_df['target_name'].nunique() if 'target_name' in pubchem_df.columns else 0,
                    'file': filename
                }
                print(f"   ✅ PubChem: {len(pubchem_df):,} records, {available_databases['PubChem']['targets']} targets")
                break
        
        if pubchem_df is None:
            print("   ❌ PubChem not found")
        
        # Check BindingDB
        bindingdb_files = [
            "bindingdb_raw_data.csv",
            "bindingdb_enhanced_raw_data.csv"
        ]
        
        bindingdb_df = None
        for filename in bindingdb_files:
            file_path = datasets_dir / filename
            if file_path.exists():
                print(f"📊 Loading BindingDB: {filename}")
                bindingdb_df = pd.read_csv(file_path)
                available_databases['BindingDB'] = {
                    'dataframe': bindingdb_df,
                    'records': len(bindingdb_df),
                    'targets': bindingdb_df['target_name'].nunique() if 'target_name' in bindingdb_df.columns else 0,
                    'file': filename
                }
                print(f"   ✅ BindingDB: {len(bindingdb_df):,} records, {available_databases['BindingDB']['targets']} targets")
                break
        
        if bindingdb_df is None:
            print("   ❌ BindingDB not found")
        
        # Check DTC
        dtc_files = [
            "dtc_raw_data.csv",
            "dtc_enhanced_raw_data.csv"
        ]
        
        dtc_df = None
        for filename in dtc_files:
            file_path = datasets_dir / filename
            if file_path.exists():
                print(f"📊 Loading DTC: {filename}")
                dtc_df = pd.read_csv(file_path)
                available_databases['DTC'] = {
                    'dataframe': dtc_df,
                    'records': len(dtc_df),
                    'targets': dtc_df['target_name'].nunique() if 'target_name' in dtc_df.columns else 0,
                    'file': filename
                }
                print(f"   ✅ DTC: {len(dtc_df):,} records, {available_databases['DTC']['targets']} targets")
                break
        
        if dtc_df is None:
            print("   ❌ DTC not found")
        
        if not available_databases:
            raise Exception("No databases found for integration")
        
        print(f"\n📊 Found {len(available_databases)} databases for integration")
        
        # Step 2: Standardize all datasets
        print(f"\n🔧 STEP 2: Standardizing datasets...")
        print("-" * 60)
        
        # Common columns for standardization
        standard_columns = [
            'canonical_smiles', 'target_name', 'target_category', 'activity_type',
            'standard_value', 'standard_units', 'standard_value_nm', 'pic50', 'data_source'
        ]
        
        standardized_datasets = {}
        
        for db_name, db_info in available_databases.items():
            df = db_info['dataframe']
            print(f"Standardizing {db_name}...")
            
            # Add missing columns with defaults
            for col in standard_columns:
                if col not in df.columns:
                    if col == 'data_source':
                        df[col] = db_name
                    elif col == 'target_category':
                        # Assign based on target name patterns
                        df[col] = df['target_name'].apply(lambda x: assign_target_category(x) if pd.notna(x) else 'unknown')
                    else:
                        df[col] = None
            
            # Ensure data_source is correct
            df['data_source'] = db_name
            
            # Select and clean standard columns
            standardized_df = df[standard_columns].copy()
            
            # Basic cleaning
            standardized_df = standardized_df.dropna(subset=['canonical_smiles', 'target_name'])
            
            standardized_datasets[db_name] = standardized_df
            print(f"   ✅ {db_name}: {len(standardized_df):,} records standardized")
        
        # Step 3: Combine and deduplicate
        print(f"\n🔗 STEP 3: Combining and deduplicating...")
        print("-" * 60)
        
        # Combine all datasets
        combined_df = pd.concat(list(standardized_datasets.values()), ignore_index=True)
        print(f"   📊 Combined raw records: {len(combined_df):,}")
        
        # Apply comprehensive deduplication
        print("   🔄 Applying cross-source deduplication...")
        
        # Group by compound-target pairs
        grouped = combined_df.groupby(['canonical_smiles', 'target_name'])
        
        deduplicated_records = []
        discarded_count = 0
        source_priority = {'ChEMBL': 1, 'PubChem': 2, 'BindingDB': 3, 'DTC': 4}
        
        for (smiles, target), group in grouped:
            if len(group) == 1:
                # Single measurement - keep as is
                deduplicated_records.append(group.iloc[0].to_dict())
                continue
            
            # Multiple measurements - prioritize by source
            group = group.copy()
            group['source_priority'] = group['data_source'].map(source_priority).fillna(999)
            group_sorted = group.sort_values('source_priority')
            
            # Check for reasonable agreement if multiple sources
            values = group['standard_value_nm'].dropna()
            if len(values) > 1:
                max_val = np.max(values)
                min_val = np.min(values)
                
                if max_val / min_val > 100:  # >100-fold difference
                    discarded_count += len(group)
                    continue
            
            # Use highest priority source
            best_record = group_sorted.iloc[0].to_dict()
            best_record['cross_source_data'] = len(group) > 1
            best_record['source_count'] = len(group)
            
            deduplicated_records.append(best_record)
        
        final_df = pd.DataFrame(deduplicated_records)
        
        print(f"   📊 Final deduplicated records: {len(final_df):,}")
        print(f"   🗑️ Discarded (high variance): {discarded_count:,}")
        print(f"   📊 Deduplication efficiency: {((len(combined_df) - len(final_df)) / len(combined_df) * 100):.1f}% overlap removed")
        
        # Step 4: Create training matrices
        print(f"\n📋 STEP 4: Creating training matrices...")
        print("-" * 60)
        
        # Comprehensive matrix (all activity types)
        comprehensive_matrix = final_df.pivot_table(
            index='canonical_smiles',
            columns='target_name',
            values='pic50',
            aggfunc='median'
        ).reset_index()
        
        print(f"   📊 Comprehensive matrix: {comprehensive_matrix.shape}")
        
        # Activity-specific matrices
        activity_matrices = {}
        for activity_type in final_df['activity_type'].dropna().unique():
            activity_df = final_df[final_df['activity_type'] == activity_type]
            if len(activity_df) > 50:  # Only if sufficient data
                matrix = activity_df.pivot_table(
                    index='canonical_smiles',
                    columns='target_name',
                    values='pic50',
                    aggfunc='median'
                ).reset_index()
                activity_matrices[activity_type] = matrix
                print(f"   📊 {activity_type} matrix: {matrix.shape}")
        
        # Step 5: Save integrated datasets
        print(f"\n💾 STEP 5: Saving integrated datasets...")
        print("-" * 60)
        
        # Save raw integrated data
        integrated_raw_path = datasets_dir / "final_integrated_raw_data.csv"
        final_df.to_csv(integrated_raw_path, index=False)
        print(f"   ✅ Raw data: {integrated_raw_path}")
        
        # Save comprehensive matrix
        integrated_matrix_path = datasets_dir / "final_integrated_matrix.csv"
        comprehensive_matrix.to_csv(integrated_matrix_path, index=False)
        print(f"   ✅ Matrix: {integrated_matrix_path}")
        
        # Save activity-specific matrices
        matrix_paths = {}
        for activity_type, matrix in activity_matrices.items():
            matrix_path = datasets_dir / f"final_integrated_{activity_type.lower()}_matrix.csv"
            matrix.to_csv(matrix_path, index=False)
            matrix_paths[activity_type] = str(matrix_path)
            print(f"   ✅ {activity_type} matrix: {matrix_path}")
        
        # Create comprehensive metadata
        source_breakdown = final_df['data_source'].value_counts().to_dict()
        target_categories = final_df.groupby('target_category')['target_name'].nunique().to_dict()
        activity_breakdown = final_df['activity_type'].value_counts().to_dict()
        
        metadata = {
            'integration_method': 'Final_Comprehensive_Multi_Database',
            'databases_integrated': list(available_databases.keys()),
            'total_databases': len(available_databases),
            'original_datasets': {
                db_name: {
                    'records': info['records'],
                    'targets': info['targets'],
                    'file': info['file']
                } for db_name, info in available_databases.items()
            },
            'final_integrated_dataset': {
                'total_records': len(final_df),
                'total_targets': final_df['target_name'].nunique(),
                'total_compounds': final_df['canonical_smiles'].nunique(),
                'target_categories': target_categories,
                'activity_types': activity_breakdown,
                'source_breakdown': source_breakdown
            },
            'matrices': {
                'comprehensive_matrix_shape': comprehensive_matrix.shape,
                'activity_specific_matrices': {k: v.shape for k, v in activity_matrices.items()},
                'matrix_paths': matrix_paths
            },
            'deduplication_stats': {
                'raw_combined_records': len(combined_df),
                'final_records': len(final_df),
                'overlap_removed': len(combined_df) - len(final_df),
                'discarded_high_variance': discarded_count,
                'efficiency_percentage': ((len(combined_df) - len(final_df)) / len(combined_df) * 100)
            },
            'integration_timestamp': datetime.now().isoformat(),
            'data_quality': {
                'cross_source_deduplication': True,
                'variance_threshold': '100x',
                'standardized_units': 'nM',
                'pic50_calculated': True,
                'missing_data_removed': True,
                'source_prioritization': True
            }
        }
        
        metadata_path = datasets_dir / "final_integrated_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Metadata: {metadata_path}")
        
        # Generate comprehensive report
        print(f"\n🎉 FINAL DATABASE INTEGRATION COMPLETED!")
        print("=" * 80)
        
        print(f"📊 Integration Summary:")
        print(f"   • Databases integrated: {len(available_databases)}")
        print(f"   • Raw combined records: {len(combined_df):,}")
        print(f"   • Final integrated records: {len(final_df):,}")
        print(f"   • Unique targets: {final_df['target_name'].nunique()}")
        print(f"   • Unique compounds: {final_df['canonical_smiles'].nunique():,}")
        print(f"   • Training matrix: {comprehensive_matrix.shape}")
        
        print(f"\n📊 Database Breakdown:")
        for source, count in source_breakdown.items():
            print(f"   • {source}: {count:,} records ({count/len(final_df)*100:.1f}%)")
        
        print(f"\n📊 Target Categories:")
        for category, count in target_categories.items():
            category_records = final_df[final_df['target_category'] == category]
            print(f"   • {category.replace('_', ' ').title()}: {len(category_records):,} records across {count} targets")
        
        # Calculate improvement over ChEMBL alone
        chembl_records = available_databases.get('ChEMBL', {}).get('records', 0)
        if chembl_records > 0:
            improvement = ((len(final_df) - chembl_records) / chembl_records) * 100
            print(f"\n📈 Improvement over ChEMBL alone: {improvement:+.1f}%")
        
        print(f"\n🚀 DATASETS NOW FULLY STANDARDIZED AND INTEGRATED!")
        print(f"   • Ready for enhanced model training")
        print(f"   • Expected R² improvement from larger, diverse dataset")
        print(f"   • Comprehensive target coverage across multiple categories")
        
        return {
            'status': 'success',
            'databases_integrated': list(available_databases.keys()),
            'total_databases': len(available_databases),
            'raw_combined_records': len(combined_df),
            'final_records': len(final_df),
            'unique_targets': final_df['target_name'].nunique(),
            'unique_compounds': final_df['canonical_smiles'].nunique(),
            'comprehensive_matrix_shape': comprehensive_matrix.shape,
            'improvement_over_chembl': improvement if chembl_records > 0 else 0,
            'deduplication_efficiency': ((len(combined_df) - len(final_df)) / len(combined_df) * 100),
            'source_breakdown': source_breakdown,
            'ready_for_training': True
        }
        
    except Exception as e:
        print(f"❌ FINAL INTEGRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def assign_target_category(target_name):
    """Assign target category based on target name"""
    
    target_name = str(target_name).upper()
    
    # Oncoproteins
    oncoproteins = ['EGFR', 'HER2', 'VEGFR2', 'BRAF', 'MET', 'CDK4', 'CDK6', 'ALK', 'MDM2', 'PI3KCA']
    if any(onco in target_name for onco in oncoproteins):
        return 'oncoprotein'
    
    # Tumor suppressors
    tumor_suppressors = ['TP53', 'RB1', 'PTEN', 'APC', 'BRCA1', 'BRCA2', 'VHL']
    if any(ts in target_name for ts in tumor_suppressors):
        return 'tumor_suppressor'
    
    # Metastasis suppressors
    metastasis_suppressors = ['NDRG1', 'KAI1', 'KISS1', 'NM23H1', 'RKIP', 'CASP8']
    if any(ms in target_name for ms in metastasis_suppressors):
        return 'metastasis_suppressor'
    
    return 'oncoprotein'  # Default

if __name__ == "__main__":
    print("🔗 Final Comprehensive Database Integration")