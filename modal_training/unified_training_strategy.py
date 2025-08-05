"""
Unified Training Strategy for GNOSIS AI Platform
Two-Module Architecture with Real Data Integration
"""

import modal
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy",
    "scikit-learn",
    "rdkit-pypi"
])

app = modal.App("unified-training-strategy")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

class UnifiedTrainingStrategy:
    """Unified training strategy for GNOSIS platform"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_available_datasets(self, datasets_dir: Path) -> Dict[str, Any]:
        """Analyze all available real datasets"""
        
        self.logger.info("📊 Analyzing available real datasets...")
        
        datasets_status = {}
        
        # Expected dataset files
        dataset_files = {
            # Module 1: Ligand Activity Predictor  
            'chembl_data': 'unified_protein_ligand_data.csv',
            'bindingdb_data': 'real_bindingdb_kaggle_data.csv',
            'bindingdb_training': 'bindingdb_training_data.csv',
            
            # Module 2: Cytotoxicity Prediction (Selectivity Index)
            'gdsc_data': 'gdsc_comprehensive_training_data.csv',
            'toxcast_data': 'real_toxcast_epa_data.csv',
            'normal_toxicity': 'toxcast_normal_cell_toxicity.csv'
        }
        
        for dataset_name, filename in dataset_files.items():
            file_path = datasets_dir / filename
            
            if file_path.exists():
                try:
                    # Quick analysis
                    df = pd.read_csv(file_path)
                    
                    datasets_status[dataset_name] = {
                        'available': True,
                        'path': str(file_path),
                        'total_records': len(df),
                        'unique_compounds': df['SMILES'].nunique() if 'SMILES' in df.columns else 0,
                        'columns': len(df.columns),
                        'sample_columns': list(df.columns[:8]),
                        'size_mb': file_path.stat().st_size / (1024 * 1024)
                    }
                    
                    self.logger.info(f"   ✅ {dataset_name}: {len(df):,} records")
                    
                except Exception as e:
                    datasets_status[dataset_name] = {
                        'available': True,
                        'error': str(e)
                    }
                    self.logger.warning(f"   ⚠️ {dataset_name}: Error - {e}")
            else:
                datasets_status[dataset_name] = {'available': False}
                self.logger.warning(f"   ❌ {dataset_name}: Not found")
        
        return datasets_status
    
    def create_module1_dataset(self, datasets_dir: Path, datasets_status: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Create unified dataset for Module 1: Ligand Activity Predictor"""
        
        self.logger.info("🔧 Creating Module 1 dataset (Ligand Activity Predictor)...")
        
        module1_datasets = []
        metadata = {
            'module': 'Ligand_Activity_Predictor',
            'purpose': 'IC50/EC50/Ki prediction for oncoproteins and tumor suppressors',
            'data_sources': []
        }
        
        # 1. ChEMBL data
        if datasets_status.get('chembl_data', {}).get('available'):
            chembl_path = Path(datasets_status['chembl_data']['path'])
            chembl_df = pd.read_csv(chembl_path)
            
            # Standardize ChEMBL format
            if 'standard_value_nm' in chembl_df.columns:
                chembl_df = chembl_df.rename(columns={'standard_value_nm': 'affinity_nm'})
            if 'standard_type' in chembl_df.columns:
                chembl_df = chembl_df.rename(columns={'standard_type': 'affinity_type'})
            
            chembl_df['data_source'] = 'ChEMBL'
            module1_datasets.append(chembl_df)
            metadata['data_sources'].append('ChEMBL')
            
            self.logger.info(f"   ✅ ChEMBL: {len(chembl_df)} records")
        
        # 2. BindingDB data
        bindingdb_loaded = False
        
        # Try training format first
        if datasets_status.get('bindingdb_training', {}).get('available'):
            bindingdb_path = Path(datasets_status['bindingdb_training']['path'])
            bindingdb_df = pd.read_csv(bindingdb_path)
            
            # Standardize BindingDB format
            if 'affinity_value_nm' in bindingdb_df.columns:
                bindingdb_df = bindingdb_df.rename(columns={'affinity_value_nm': 'affinity_nm'})
            
            bindingdb_df['data_source'] = 'BindingDB'
            module1_datasets.append(bindingdb_df)
            metadata['data_sources'].append('BindingDB')
            bindingdb_loaded = True
            
            self.logger.info(f"   ✅ BindingDB (training): {len(bindingdb_df)} records")
        
        # Try raw format if training not available
        elif datasets_status.get('bindingdb_data', {}).get('available'):
            bindingdb_path = Path(datasets_status['bindingdb_data']['path'])
            bindingdb_df = pd.read_csv(bindingdb_path)
            
            # Convert BindingDB format
            if 'IC50_nM' in bindingdb_df.columns:
                bindingdb_df['affinity_nm'] = bindingdb_df['IC50_nM']
                bindingdb_df['affinity_type'] = 'IC50'
            elif 'Ki_nM' in bindingdb_df.columns:
                bindingdb_df['affinity_nm'] = bindingdb_df['Ki_nM']  
                bindingdb_df['affinity_type'] = 'Ki'
            
            if 'target' in bindingdb_df.columns:
                bindingdb_df['target_name'] = bindingdb_df['target']
            
            bindingdb_df['data_source'] = 'BindingDB'
            module1_datasets.append(bindingdb_df)
            metadata['data_sources'].append('BindingDB')
            bindingdb_loaded = True
            
            self.logger.info(f"   ✅ BindingDB (raw): {len(bindingdb_df)} records")
        
        if not bindingdb_loaded:
            self.logger.warning("   ⚠️ No BindingDB data available")
        
        # Combine datasets
        if module1_datasets:
            combined_df = pd.concat(module1_datasets, ignore_index=True)
            
            # Standardize columns
            required_columns = ['SMILES', 'target_name', 'affinity_nm', 'affinity_type', 'data_source']
            
            for col in required_columns:
                if col not in combined_df.columns:
                    self.logger.warning(f"   ⚠️ Missing column: {col}")
            
            # Quality control
            initial_count = len(combined_df)
            
            # Remove invalid data
            combined_df = combined_df.dropna(subset=['SMILES', 'affinity_nm'])
            combined_df = combined_df[combined_df['SMILES'].str.len() > 5]
            combined_df = combined_df[combined_df['affinity_nm'] > 0]
            combined_df = combined_df[combined_df['affinity_nm'] < 10000000]  # < 10 mM
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['SMILES', 'target_name'], keep='first')
            
            # Add computed fields
            combined_df['pIC50'] = combined_df['affinity_nm'].apply(lambda x: -np.log10(x/1e9) if x > 0 else None)
            combined_df['log_affinity'] = np.log10(combined_df['affinity_nm'])
            
            # Classification
            def classify_affinity(affinity_nm):
                if affinity_nm < 10:
                    return "High_Affinity"
                elif affinity_nm < 100:
                    return "Moderate_Affinity"
                elif affinity_nm < 1000:
                    return "Low_Affinity"
                else:
                    return "Very_Low_Affinity"
            
            combined_df['affinity_class'] = combined_df['affinity_nm'].apply(classify_affinity)
            
            metadata.update({
                'total_records': len(combined_df),
                'unique_compounds': int(combined_df['SMILES'].nunique()),
                'unique_targets': int(combined_df['target_name'].nunique()),
                'quality_control': f"Removed {initial_count - len(combined_df)} invalid records",
                'affinity_distribution': combined_df['affinity_class'].value_counts().to_dict()
            })
            
            self.logger.info(f"   ✅ Module 1 combined: {len(combined_df)} records")
            self.logger.info(f"   📊 Unique compounds: {combined_df['SMILES'].nunique()}")
            self.logger.info(f"   📊 Unique targets: {combined_df['target_name'].nunique()}")
            
            return combined_df, metadata
        
        else:
            self.logger.error("   ❌ No Module 1 datasets available")
            return pd.DataFrame(), metadata
    
    def create_module2_dataset(self, datasets_dir: Path, datasets_status: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Create unified dataset for Module 2: Cytotoxicity Prediction (Selectivity Index)"""
        
        self.logger.info("🔧 Creating Module 2 dataset (Cytotoxicity Prediction)...")
        
        metadata = {
            'module': 'Cytotoxicity_Prediction_Selectivity',
            'purpose': 'Cancer vs Normal cell selectivity prediction',
            'data_sources': []
        }
        
        # 1. GDSC cancer data
        gdsc_df = None
        if datasets_status.get('gdsc_data', {}).get('available'):
            gdsc_path = Path(datasets_status['gdsc_data']['path'])
            gdsc_df = pd.read_csv(gdsc_path)
            
            self.logger.info(f"   ✅ GDSC cancer data: {len(gdsc_df)} records")
            metadata['data_sources'].append('GDSC')
        else:
            self.logger.error("   ❌ GDSC data not available")
            return pd.DataFrame(), metadata
        
        # 2. ToxCast normal cell data  
        normal_df = None
        if datasets_status.get('normal_toxicity', {}).get('available'):
            normal_path = Path(datasets_status['normal_toxicity']['path'])
            normal_df = pd.read_csv(normal_path)
            
            self.logger.info(f"   ✅ ToxCast normal data: {len(normal_df)} records")
            metadata['data_sources'].append('ToxCast')
        elif datasets_status.get('toxcast_data', {}).get('available'):
            normal_path = Path(datasets_status['toxcast_data']['path'])
            normal_df = pd.read_csv(normal_path)
            
            # Aggregate by SMILES
            normal_df = normal_df.groupby('SMILES').agg({
                'chemical_name': 'first',
                'ac50_um': 'median',
                'is_toxic_normal': lambda x: (x.sum() / len(x)) > 0.5
            }).reset_index()
            normal_df['normal_cell_ac50_um'] = normal_df['ac50_um']
            
            self.logger.info(f"   ✅ ToxCast aggregated: {len(normal_df)} records")
            metadata['data_sources'].append('ToxCast')
        else:
            self.logger.warning("   ⚠️ ToxCast data not available - creating selectivity from GDSC only")
        
        # Process GDSC data for training
        if gdsc_df is not None:
            # Clean and standardize GDSC columns
            gdsc_columns = gdsc_df.columns.tolist()
            smiles_col = None
            
            # Find SMILES column
            for col in ['SMILES', 'smiles', 'canonical_smiles', 'DRUG_SMILES']:
                if col in gdsc_columns:
                    smiles_col = col
                    break
            
            if smiles_col is None:
                self.logger.error("   ❌ No SMILES column found in GDSC data")
                return pd.DataFrame(), metadata
            
            # Find IC50/sensitivity column
            ic50_col = None
            for col in ['LN_IC50', 'IC50_nM', 'LOG_IC50', 'ln_ic50', 'AUC']:
                if col in gdsc_columns:
                    ic50_col = col
                    break
            
            if ic50_col is None:
                self.logger.error("   ❌ No IC50/sensitivity column found in GDSC data")
                return pd.DataFrame(), metadata
            
            # Create selectivity dataset
            selectivity_records = []
            
            for idx, row in gdsc_df.iterrows():
                smiles = row.get(smiles_col)
                cancer_ic50 = row.get(ic50_col)
                
                if pd.notna(smiles) and pd.notna(cancer_ic50) and len(str(smiles)) > 5:
                    
                    record = {
                        'SMILES': str(smiles),
                        'cancer_ic50_um': float(cancer_ic50) if 'nM' not in ic50_col else float(cancer_ic50)/1000,
                        'log_cancer_ic50': np.log10(float(cancer_ic50)) if cancer_ic50 > 0 else None,
                        'cell_line': row.get('CELL_LINE_NAME', 'Unknown'),
                        'drug_name': row.get('DRUG_NAME', 'Unknown'),
                        'data_source_cancer': 'GDSC'
                    }
                    
                    # Add normal cell toxicity if available
                    if normal_df is not None:
                        normal_match = normal_df[normal_df['SMILES'] == smiles]
                        
                        if len(normal_match) > 0:
                            normal_ac50 = normal_match.iloc[0]['normal_cell_ac50_um']
                            
                            record.update({
                                'normal_ac50_um': normal_ac50,
                                'log_normal_ac50': np.log10(normal_ac50) if normal_ac50 > 0 else None,
                                'selectivity_index': normal_ac50 / record['cancer_ic50_um'],
                                'log_selectivity_index': np.log10(normal_ac50 / record['cancer_ic50_um']) if record['cancer_ic50_um'] > 0 else None,
                                'data_source_normal': 'ToxCast',
                                'has_selectivity_data': True
                            })
                        else:
                            record.update({
                                'normal_ac50_um': None,
                                'selectivity_index': None,
                                'has_selectivity_data': False
                            })
                    else:
                        record['has_selectivity_data'] = False
                    
                    selectivity_records.append(record)
                
                # Progress tracking
                if idx > 0 and idx % 10000 == 0:
                    self.logger.info(f"     Processed {idx:,} GDSC records...")
            
            if selectivity_records:
                selectivity_df = pd.DataFrame(selectivity_records)
                
                # Classification
                def classify_selectivity(si):
                    if pd.isna(si):
                        return "No_Data"
                    elif si > 10:
                        return "Highly_Selective"
                    elif si > 3:
                        return "Moderately_Selective"
                    elif si > 1:
                        return "Low_Selectivity"
                    else:
                        return "Non_Selective"
                
                selectivity_df['selectivity_class'] = selectivity_df['selectivity_index'].apply(classify_selectivity)
                
                metadata.update({
                    'total_records': len(selectivity_df),
                    'unique_compounds': int(selectivity_df['SMILES'].nunique()),
                    'with_selectivity_data': int(selectivity_df['has_selectivity_data'].sum()),
                    'selectivity_distribution': selectivity_df['selectivity_class'].value_counts().to_dict()
                })
                
                self.logger.info(f"   ✅ Module 2 combined: {len(selectivity_df)} records")
                self.logger.info(f"   📊 Unique compounds: {selectivity_df['SMILES'].nunique()}")
                self.logger.info(f"   📊 With selectivity data: {selectivity_df['has_selectivity_data'].sum()}")
                
                return selectivity_df, metadata
            
            else:
                self.logger.error("   ❌ No valid records created for Module 2")
                return pd.DataFrame(), metadata
        
        else:
            return pd.DataFrame(), metadata

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=4.0,
    memory=16384,
    timeout=3600
)
def create_unified_training_datasets():
    """
    Create unified training datasets for both modules of GNOSIS platform
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 UNIFIED TRAINING DATASET CREATION - GNOSIS PLATFORM")
    print("=" * 80)
    print("🎯 Module 1: Ligand Activity Predictor (ChEMBL + BindingDB)")
    print("🎯 Module 2: Cytotoxicity Prediction (GDSC + ToxCast)")
    print("✅ Real experimental data only")
    
    try:
        strategy = UnifiedTrainingStrategy()
        
        datasets_dir = Path("/vol/datasets")
        
        # Step 1: Analyze available datasets
        print("\n📊 STEP 1: Analyzing available real datasets...")
        datasets_status = strategy.analyze_available_datasets(datasets_dir)
        
        # Step 2: Create Module 1 dataset
        print(f"\n🔧 STEP 2: Creating Module 1 dataset...")
        module1_df, module1_meta = strategy.create_module1_dataset(datasets_dir, datasets_status)
        
        # Step 3: Create Module 2 dataset  
        print(f"\n🔧 STEP 3: Creating Module 2 dataset...")
        module2_df, module2_meta = strategy.create_module2_dataset(datasets_dir, datasets_status)
        
        # Step 4: Save unified datasets
        print(f"\n💾 STEP 4: Saving unified training datasets...")
        
        # Save Module 1
        if len(module1_df) > 0:
            module1_path = datasets_dir / "gnosis_module1_ligand_activity_training.csv"
            module1_df.to_csv(module1_path, index=False)
            print(f"   ✅ Module 1 saved: {len(module1_df)} records")
        else:
            module1_path = None
            print(f"   ❌ Module 1: No data to save")
        
        # Save Module 2
        if len(module2_df) > 0:
            module2_path = datasets_dir / "gnosis_module2_cytotoxicity_training.csv"
            module2_df.to_csv(module2_path, index=False)
            print(f"   ✅ Module 2 saved: {len(module2_df)} records")
        else:
            module2_path = None
            print(f"   ❌ Module 2: No data to save")
        
        # Create comprehensive metadata
        unified_metadata = {
            'platform': 'GNOSIS_AI',
            'training_strategy': 'Two_Module_Architecture',
            'creation_date': datetime.now().isoformat(),
            'real_data_only': True,
            'modules': {
                'module1_ligand_activity': {
                    **module1_meta,
                    'training_file': str(module1_path) if module1_path else None,
                    'ready_for_training': len(module1_df) > 0
                },
                'module2_cytotoxicity_selectivity': {
                    **module2_meta,
                    'training_file': str(module2_path) if module2_path else None,
                    'ready_for_training': len(module2_df) > 0
                }
            },
            'datasets_analyzed': datasets_status,
            'training_recommendations': {}
        }
        
        # Add training recommendations
        if len(module1_df) > 0 and len(module2_df) > 0:
            unified_metadata['training_recommendations']['strategy'] = 'Train_Both_Modules_Separately'
            unified_metadata['training_recommendations']['rationale'] = 'Different objectives and data formats'
        elif len(module1_df) > 0:
            unified_metadata['training_recommendations']['strategy'] = 'Train_Module1_Only'
            unified_metadata['training_recommendations']['rationale'] = 'Module 2 data not available'
        elif len(module2_df) > 0:
            unified_metadata['training_recommendations']['strategy'] = 'Train_Module2_Only'
            unified_metadata['training_recommendations']['rationale'] = 'Module 1 data not available'
        else:
            unified_metadata['training_recommendations']['strategy'] = 'Data_Collection_Required'
            unified_metadata['training_recommendations']['rationale'] = 'No training data available'
        
        # Save metadata
        metadata_path = datasets_dir / "gnosis_unified_training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(unified_metadata, f, indent=2)
        
        # Generate comprehensive report
        print(f"\n🎉 UNIFIED TRAINING DATASETS CREATED!")
        print("=" * 80)
        print(f"📊 Training Dataset Summary:")
        
        if len(module1_df) > 0:
            print(f"  🎯 Module 1 (Ligand Activity Predictor):")
            print(f"    • Records: {len(module1_df):,}")
            print(f"    • Compounds: {module1_df['SMILES'].nunique()}")
            print(f"    • Targets: {module1_df['target_name'].nunique()}")
            print(f"    • Sources: {', '.join(module1_meta['data_sources'])}")
            print(f"    • ✅ Ready for training")
        else:
            print(f"  ❌ Module 1: No training data available")
        
        if len(module2_df) > 0:
            print(f"  🎯 Module 2 (Cytotoxicity Prediction):")
            print(f"    • Records: {len(module2_df):,}")
            print(f"    • Compounds: {module2_df['SMILES'].nunique()}")
            print(f"    • With selectivity: {module2_df['has_selectivity_data'].sum() if 'has_selectivity_data' in module2_df.columns else 0}")
            print(f"    • Sources: {', '.join(module2_meta['data_sources'])}")
            print(f"    • ✅ Ready for training")
        else:
            print(f"  ❌ Module 2: No training data available")
        
        print(f"\n🚀 TRAINING RECOMMENDATION:")
        print(f"  Strategy: {unified_metadata['training_recommendations']['strategy']}")
        print(f"  Rationale: {unified_metadata['training_recommendations']['rationale']}")
        
        return {
            'status': 'success',
            'module1_records': len(module1_df),
            'module2_records': len(module2_df),
            'training_strategy': unified_metadata['training_recommendations']['strategy'],
            'metadata_path': str(metadata_path),
            'ready_for_training': {
                'module1': len(module1_df) > 0,
                'module2': len(module2_df) > 0
            }
        }
        
    except Exception as e:
        print(f"❌ UNIFIED DATASET CREATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🧬 Unified Training Strategy for GNOSIS Platform")