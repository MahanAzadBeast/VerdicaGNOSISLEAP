"""
Model 2 Training Starter - GDSC + EPA InvitroDB Selectivity Index
Start training immediately with available data
"""

import modal
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy",
    "scikit-learn"
])

app = modal.App("model2-training-starter")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    timeout=1800
)
def create_model2_training_dataset():
    """Create Model 2 training dataset with GDSC + EPA InvitroDB for Selectivity Index"""
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 MODEL 2 TRAINING DATASET CREATION")
    print("=" * 80)
    print("🎯 GDSC Cancer Data + EPA InvitroDB Normal Cell Data")
    print("📊 Selectivity Index = Normal AC50 / Cancer IC50")
    print("🚀 STARTING TRAINING PREPARATION NOW")
    
    try:
        datasets_dir = Path("/vol/datasets")
        
        # Load GDSC cancer data
        print("\n📥 STEP 1: Loading GDSC cancer data...")
        
        gdsc_path = datasets_dir / "gdsc_comprehensive_training_data.csv"
        
        if not gdsc_path.exists():
            raise Exception(f"GDSC data not found: {gdsc_path}")
        
        gdsc_df = pd.read_csv(gdsc_path)
        print(f"   ✅ GDSC loaded: {len(gdsc_df):,} records")
        print(f"   📊 Columns: {list(gdsc_df.columns[:10])}...")
        
        # Load EPA InvitroDB normal cell data
        print("\n📥 STEP 2: Loading EPA InvitroDB normal cell data...")
        
        epa_path = datasets_dir / "gnosis_normalcell_ic50.csv"
        
        if not epa_path.exists():
            raise Exception(f"EPA data not found: {epa_path}")
        
        epa_df = pd.read_csv(epa_path)
        print(f"   ✅ EPA InvitroDB loaded: {len(epa_df):,} records")
        print(f"   📊 Unique compounds: {epa_df['smiles'].nunique()}")
        print(f"   📊 Cell lines: {epa_df['cell_line'].nunique()}")
        
        # Process GDSC data
        print("\n🔧 STEP 3: Processing GDSC cancer data...")
        
        gdsc_processed = []
        
        for idx, row in gdsc_df.iterrows():
            smiles = row.get('SMILES')
            if pd.isna(smiles) or len(str(smiles)) < 5:
                continue
                
            cell_line = row.get('CELL_LINE_ID', row.get('CELL_LINE_NAME', 'Unknown'))
            tissue_type = row.get('CANCER_TYPE', row.get('TISSUE_TYPE', 'Unknown'))
            
            # Get IC50 data
            ic50_um = row.get('IC50_uM', row.get('IC50_nM'))
            if pd.isna(ic50_um):
                continue
                
            # Convert nM to uM if needed
            if 'IC50_nM' in gdsc_df.columns and pd.notna(row.get('IC50_nM')):
                ic50_um = row.get('IC50_nM') / 1000
            
            if ic50_um <= 0 or ic50_um > 1000:  # Reasonable range
                continue
                
            # Calculate pIC50
            pic50_cancer = -np.log10(ic50_um / 1e6)  # uM to M
            
            record = {
                'SMILES': str(smiles),
                'cell_line_id': cell_line,
                'tissue_type': tissue_type,
                'ic50_um_cancer': ic50_um,
                'pic50_cancer': pic50_cancer,
                'data_source_cancer': 'GDSC'
            }
            
            # Add genomic features if available
            for col in gdsc_df.columns:
                if any(keyword in col.lower() for keyword in ['mutation', 'cnv', 'expression']):
                    record[f'genomic_{col}'] = row.get(col)
            
            gdsc_processed.append(record)
        
        gdsc_clean_df = pd.DataFrame(gdsc_processed)
        print(f"   ✅ GDSC processed: {len(gdsc_clean_df):,} records")
        print(f"   📊 Unique compounds: {gdsc_clean_df['SMILES'].nunique()}")
        
        # Process EPA normal data
        print("\n🔧 STEP 4: Processing EPA normal cell data...")
        
        # Aggregate EPA data by SMILES (median across assays)
        epa_aggregated = epa_df.groupby('smiles').agg({
            'ac50_nM': 'median',
            'pIC50': 'median',
            'cell_line': lambda x: '; '.join(x.unique()[:3]),
            'tissue_type': lambda x: '; '.join(x.unique()[:3]),
            'data_source': 'first'
        }).reset_index()
        
        epa_aggregated = epa_aggregated.rename(columns={'smiles': 'SMILES'})
        
        print(f"   ✅ EPA aggregated: {len(epa_aggregated)} unique compounds")
        
        # Align GDSC + EPA by SMILES
        print("\n🔗 STEP 5: Aligning cancer + normal data by SMILES...")
        
        # Start with GDSC as base
        aligned_df = gdsc_clean_df.copy()
        
        # Add EPA normal data where available
        aligned_df = aligned_df.merge(
            epa_aggregated[['SMILES', 'ac50_nM', 'pIC50', 'cell_line', 'tissue_type']],
            on='SMILES',
            how='left',
            suffixes=('', '_normal')
        )
        
        # Rename EPA columns for clarity
        aligned_df = aligned_df.rename(columns={
            'ac50_nM': 'ac50_nm_normal',
            'pIC50': 'pac50_normal',
            'cell_line': 'normal_cell_lines',
            'tissue_type_normal': 'normal_tissue_types'
        })
        
        # Calculate Selectivity Index where both values available
        print("\n📈 STEP 6: Calculating Selectivity Index...")
        
        aligned_df['has_selectivity_data'] = aligned_df['ac50_nm_normal'].notna()
        
        # Convert normal AC50 from nM to uM for consistent units
        aligned_df['ac50_um_normal'] = aligned_df['ac50_nm_normal'] / 1000
        
        # Calculate Selectivity Index (Normal AC50 / Cancer IC50)
        mask = aligned_df['has_selectivity_data']
        
        aligned_df['selectivity_index'] = None
        aligned_df.loc[mask, 'selectivity_index'] = (
            aligned_df.loc[mask, 'ac50_um_normal'] / aligned_df.loc[mask, 'ic50_um_cancer']
        )
        
        # Calculate log selectivity index
        si_mask = aligned_df['selectivity_index'].notna() & (aligned_df['selectivity_index'] > 0)
        aligned_df['log_selectivity_index'] = pd.Series(dtype=float)
        if si_mask.sum() > 0:
            si_values = aligned_df.loc[si_mask, 'selectivity_index']
            aligned_df.loc[si_mask, 'log_selectivity_index'] = [np.log10(x) for x in si_values]
        
        # Classification
        def classify_selectivity(si):
            if pd.isna(si):
                return "No_Normal_Data"
            elif si > 10:
                return "Highly_Selective"
            elif si > 3:
                return "Moderately_Selective"  
            elif si > 1:
                return "Low_Selectivity"
            else:
                return "Non_Selective"
        
        aligned_df['selectivity_class'] = aligned_df['selectivity_index'].apply(classify_selectivity)
        
        print(f"   ✅ Selectivity Index calculated: {aligned_df['selectivity_index'].notna().sum():,} compounds")
        
        # Final statistics
        print(f"\n📊 STEP 7: Final dataset analysis...")
        
        total_records = len(aligned_df)
        unique_compounds = aligned_df['SMILES'].nunique()
        with_selectivity = aligned_df['has_selectivity_data'].sum()
        
        print(f"   🎯 MODEL 2 TRAINING DATASET:")
        print(f"     • Total records: {total_records:,}")
        print(f"     • Unique compounds: {unique_compounds}")
        print(f"     • With selectivity data: {with_selectivity:,}")
        print(f"     • Cancer-only data: {total_records - with_selectivity:,}")
        
        # Selectivity distribution
        print(f"\n   📊 Selectivity distribution:")
        for sel_class, count in aligned_df['selectivity_class'].value_counts().items():
            print(f"     • {sel_class}: {count:,} records")
        
        # Save training dataset
        print(f"\n💾 STEP 8: Saving Model 2 training dataset...")
        
        model2_path = datasets_dir / "gnosis_model2_cytotox_training.csv"
        aligned_df.to_csv(model2_path, index=False)
        
        print(f"   ✅ Saved: {model2_path}")
        
        # Create metadata
        metadata = {
            'model': 'Model2_Cytotoxicity_Selectivity',
            'creation_date': datetime.now().isoformat(),
            'data_sources': {
                'cancer_data': 'GDSC_comprehensive',
                'normal_data': 'EPA_InvitroDB_v4.1'
            },
            'total_statistics': {
                'total_records': total_records,
                'unique_compounds': unique_compounds,
                'with_selectivity_index': int(with_selectivity),
                'cancer_only_records': int(total_records - with_selectivity)
            },
            'selectivity_distribution': {k: int(v) for k, v in aligned_df['selectivity_class'].value_counts().to_dict().items()},
            'ready_for_training': True,
            'training_approach': 'Multi_task_regression',
            'outputs': ['Cancer_IC50', 'Normal_AC50', 'Selectivity_Index']
        }
        
        metadata_path = datasets_dir / "gnosis_model2_training_metadata.json"
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        print(f"\n🎉 MODEL 2 TRAINING DATASET READY!")
        print("=" * 80)
        print(f"🚀 READY TO START TRAINING:")
        print(f"  • Training file: gnosis_model2_cytotox_training.csv")
        print(f"  • Records: {total_records:,} total")
        print(f"  • Selectivity Index: {with_selectivity:,} compounds")
        print(f"  • Multi-task approach: Cancer IC50 + Normal AC50 + SI")
        print(f"  • ChemBERTa embeddings: {unique_compounds} unique SMILES")
        print(f"  • Genomic features: Included for cancer context")
        
        return {
            'status': 'success',
            'total_records': total_records,
            'unique_compounds': unique_compounds,
            'with_selectivity': int(with_selectivity),
            'ready_for_training': True,
            'training_file': str(model2_path)
        }
        
    except Exception as e:
        print(f"❌ MODEL 2 TRAINING PREPARATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🚀 Model 2 Training Dataset Creator")