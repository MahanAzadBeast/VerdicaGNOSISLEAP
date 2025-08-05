"""
Model 1 Combiner - Merge ChEMBL and BindingDB data
Create comprehensive Model 1 training dataset
"""

import modal
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy",
    "rdkit-pypi"
])

app = modal.App("model1-combiner")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

def standardize_smiles(smiles: str) -> Optional[str]:
    """Standardize SMILES string"""
    if pd.isna(smiles) or not isinstance(smiles, str):
        return None
    smiles = smiles.strip()
    return smiles if len(smiles) >= 5 else None

def extract_uniprot_id(target_info: str) -> Optional[str]:
    """Extract UniProt ID from target information"""
    if pd.isna(target_info) or not isinstance(target_info, str):
        return None
    return target_info.upper().replace(' ', '_').replace('-', '_')

def calculate_pic50(ic50_nm: float) -> Optional[float]:
    """Calculate pIC50 = -log10(IC50 [M])"""
    if pd.isna(ic50_nm) or ic50_nm <= 0:
        return None
    ic50_m = ic50_nm / 1e9  # Convert nM to M
    return -np.log10(ic50_m)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    timeout=1800
)
def combine_model1_datasets():
    """Combine ChEMBL and BindingDB data into Model 1 training dataset"""
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 MODEL 1 COMBINER: CHEMBL + BINDINGDB")
    print("=" * 80)
    print("🎯 Goal: Create comprehensive Model 1 training dataset")
    print("📊 Sources: ChEMBL oncology + BindingDB SDF")
    
    try:
        datasets_dir = Path("/vol/datasets")
        
        # Load ChEMBL data
        print("\n📥 STEP 1: Loading ChEMBL oncology data...")
        
        chembl_path = datasets_dir / "chembl_oncology_bioactivities.csv"
        
        if not chembl_path.exists():
            raise Exception(f"ChEMBL data not found: {chembl_path}")
        
        chembl_df = pd.read_csv(chembl_path)
        print(f"   ✅ ChEMBL loaded: {len(chembl_df):,} records")
        print(f"   📊 ChEMBL unique compounds: {chembl_df['SMILES'].nunique()}")
        print(f"   📊 ChEMBL unique targets: {chembl_df['UniProt_ID'].nunique()}")
        
        # Load BindingDB data
        print("\n📥 STEP 2: Loading BindingDB SDF data...")
        
        bindingdb_path = datasets_dir / "bindingdb_sdf_oncology.csv"
        
        if not bindingdb_path.exists():
            raise Exception(f"BindingDB data not found: {bindingdb_path}")
        
        bindingdb_df = pd.read_csv(bindingdb_path)
        print(f"   ✅ BindingDB loaded: {len(bindingdb_df):,} records")
        print(f"   📊 BindingDB unique compounds: {bindingdb_df['SMILES'].nunique()}")
        print(f"   📊 BindingDB unique targets: {bindingdb_df['UniProt_ID'].nunique()}")
        
        # Standardize ChEMBL data format
        print("\n🔧 STEP 3: Standardizing ChEMBL data...")
        
        chembl_processed = []
        
        for idx, row in chembl_df.iterrows():
            smiles = standardize_smiles(row.get('SMILES'))
            if not smiles:
                continue
            
            target_name = row.get('UniProt_ID', '')
            uniprot_id = extract_uniprot_id(str(target_name))
            
            affinity_type = row.get('Assay_Type', 'IC50')
            affinity_nm = row.get('Original_Value_nM')
            
            if pd.notna(affinity_nm) and affinity_nm > 0:
                if 0.1 <= affinity_nm <= 1e7:
                    record = {
                        'SMILES': smiles,
                        'target_name': target_name,
                        'uniprot_id': uniprot_id,
                        'assay_type': affinity_type.upper(),
                        'affinity_nm': affinity_nm,
                        'data_source': 'ChEMBL'
                    }
                    chembl_processed.append(record)
        
        chembl_clean_df = pd.DataFrame(chembl_processed)
        print(f"   ✅ ChEMBL processed: {len(chembl_clean_df):,} valid records")
        
        # Standardize BindingDB data format
        print("\n🔧 STEP 4: Standardizing BindingDB data...")
        
        bindingdb_processed = []
        
        for idx, row in bindingdb_df.iterrows():
            smiles = standardize_smiles(row.get('SMILES'))
            if not smiles:
                continue
            
            target_name = row.get('UniProt_ID', '')
            uniprot_id = extract_uniprot_id(str(target_name))
            
            affinity_type = row.get('Assay_Type', 'IC50')
            affinity_nm = row.get('Original_Value_nM')
            
            if pd.notna(affinity_nm) and affinity_nm > 0:
                if 0.1 <= affinity_nm <= 1e7:
                    record = {
                        'SMILES': smiles,
                        'target_name': target_name,
                        'uniprot_id': uniprot_id,
                        'assay_type': affinity_type.upper(),
                        'affinity_nm': affinity_nm,
                        'data_source': 'BindingDB'
                    }
                    bindingdb_processed.append(record)
        
        bindingdb_clean_df = pd.DataFrame(bindingdb_processed)
        print(f"   ✅ BindingDB processed: {len(bindingdb_clean_df):,} valid records")
        
        # Combine datasets
        print("\n🔗 STEP 5: Combining ChEMBL + BindingDB...")
        
        combined_df = pd.concat([chembl_clean_df, bindingdb_clean_df], ignore_index=True)
        print(f"   ✅ Combined dataset: {len(combined_df):,} total records")
        
        # Deduplicate by (SMILES + UniProt ID + Assay Type)
        print("\n🔧 STEP 6: Deduplicating by (SMILES + UniProt + Assay)...")
        
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(
            subset=['SMILES', 'uniprot_id', 'assay_type'], 
            keep='first'
        )
        print(f"   Removed {initial_count - len(combined_df):,} duplicates")
        
        # Apply log transformations
        print("\n📈 STEP 7: Applying pIC50, pKi, pEC50 transformations...")
        
        combined_df['pIC50'] = None
        combined_df['pKi'] = None
        combined_df['pEC50'] = None
        
        for idx, row in combined_df.iterrows():
            assay_type = row['assay_type']
            affinity_nm = row['affinity_nm']
            
            p_value = calculate_pic50(affinity_nm)
            
            if assay_type == 'IC50':
                combined_df.at[idx, 'pIC50'] = p_value
            elif assay_type == 'KI':
                combined_df.at[idx, 'pKi'] = p_value
            elif assay_type == 'EC50':
                combined_df.at[idx, 'pEC50'] = p_value
        
        # Final statistics
        print("\n📊 STEP 8: Final dataset analysis...")
        
        total_compounds = combined_df['SMILES'].nunique()
        total_targets = combined_df['uniprot_id'].nunique()
        
        print(f"   🎯 FINAL MODEL 1 DATASET:")
        print(f"     • Total records: {len(combined_df):,}")
        print(f"     • Unique compounds: {total_compounds:,}")
        print(f"     • Unique targets: {total_targets}")
        
        # Source breakdown
        print(f"\n   📊 Source breakdown:")
        for source, count in combined_df['data_source'].value_counts().items():
            source_compounds = combined_df[combined_df['data_source'] == source]['SMILES'].nunique()
            print(f"     • {source}: {count:,} records, {source_compounds} compounds")
        
        # Assay type breakdown
        print(f"\n   📊 Assay type breakdown:")
        for assay, count in combined_df['assay_type'].value_counts().items():
            print(f"     • {assay}: {count:,} records")
        
        # Save final dataset
        print(f"\n💾 STEP 9: Saving Model 1 training dataset...")
        
        final_path = datasets_dir / "gnosis_model1_binding_training.csv"
        combined_df.to_csv(final_path, index=False)
        
        print(f"   ✅ Saved: {final_path}")
        
        # Create metadata
        metadata = {
            'model': 'Model1_Ligand_Activity_Predictor',
            'creation_date': datetime.now().isoformat(),
            'data_sources': {
                'chembl': 'ChEMBL_oncology_bioactivities_full',
                'bindingdb': 'BindingDB_SDF_bulk_oncology'
            },
            'total_statistics': {
                'total_records': len(combined_df),
                'unique_compounds': total_compounds,
                'unique_targets': total_targets,
                'chembl_records': len(chembl_clean_df),
                'bindingdb_records': len(bindingdb_clean_df)
            },
            'assay_distribution': combined_df['assay_type'].value_counts().to_dict(),
            'source_distribution': combined_df['data_source'].value_counts().to_dict(),
            'ready_for_training': True,
            'training_approach': 'Multi_task_regression',
            'outputs': ['pIC50', 'pKi', 'pEC50'],
            'real_experimental_data': True,
            'no_synthetic_data': True
        }
        
        metadata_path = datasets_dir / "gnosis_model1_training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n🎉 MODEL 1 TRAINING DATASET COMPLETED!")
        print("=" * 80)
        print(f"🎯 **FINAL MODEL 1 STATISTICS:**")
        print(f"  • **Total Records**: {len(combined_df):,} (100% real experimental)")
        print(f"  • **Unique Compounds**: {total_compounds:,}")
        print(f"  • **Unique Targets**: {total_targets}")
        print(f"  • **ChEMBL**: {len(chembl_clean_df):,} records")
        print(f"  • **BindingDB**: {len(bindingdb_clean_df):,} records")
        print(f"  • **File**: gnosis_model1_binding_training.csv")
        print(f"✅ **READY FOR MODEL 1 TRAINING**")
        
        return {
            'status': 'success',
            'total_records': len(combined_df),
            'unique_compounds': total_compounds,
            'unique_targets': total_targets,
            'chembl_records': len(chembl_clean_df),
            'bindingdb_records': len(bindingdb_clean_df),
            'chembl_compounds': int(chembl_clean_df['SMILES'].nunique()),
            'bindingdb_compounds': int(bindingdb_clean_df['SMILES'].nunique()),
            'real_experimental_data': True,
            'ready_for_training': True
        }
        
    except Exception as e:
        print(f"❌ MODEL 1 COMBINATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {'status': 'failed', 'error': str(e)}

if __name__ == "__main__":
    print("🧬 Model 1 Combiner: ChEMBL + BindingDB")