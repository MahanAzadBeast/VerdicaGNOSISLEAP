"""
Diagnostic script to verify GDSC dataset contains >600 unique real compounds
This will run first to confirm we have the right dataset before training
"""

import modal
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("diagnostic-dataset-verification")

image = modal.Image.debian_slim().pip_install([
    "pandas==2.1.0", "numpy==1.24.3"
])

data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    timeout=1800,  # 30 minutes
    volumes={"/vol": data_volume}
)
def verify_gdsc_dataset():
    """Verify the GDSC dataset contains >600 unique real compounds"""
    
    logger.info("🔍 DIAGNOSTIC: VERIFYING GDSC DATASET")
    logger.info("Path: expanded-datasets > gdsc_dataset (/vol/gdsc_dataset/)")
    logger.info("Target: >600 unique real compounds")
    
    gdsc_dir = "/vol/gdsc_dataset"
    
    # 1. SCAN DIRECTORY STRUCTURE
    logger.info(f"📁 Scanning directory: {gdsc_dir}")
    
    if not os.path.exists(gdsc_dir):
        logger.error(f"❌ Directory does not exist: {gdsc_dir}")
        return {"error": "Directory not found", "exists": False}
    
    try:
        files = os.listdir(gdsc_dir)
        logger.info(f"Found {len(files)} files/directories:")
        
        for item in sorted(files):
            item_path = os.path.join(gdsc_dir, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                logger.info(f"  📄 {item} ({size:,} bytes)")
            else:
                logger.info(f"  📁 {item}/ (directory)")
                # Check subdirectories for CSV files
                try:
                    sub_files = os.listdir(item_path)
                    csv_sub_files = [f for f in sub_files if f.endswith('.csv')]
                    if csv_sub_files:
                        logger.info(f"    Contains {len(csv_sub_files)} CSV files")
                        for csv_file in csv_sub_files[:5]:  # Show first 5
                            csv_path = os.path.join(item_path, csv_file)
                            csv_size = os.path.getsize(csv_path)
                            logger.info(f"    📄 {csv_file} ({csv_size:,} bytes)")
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Error scanning directory: {e}")
        return {"error": f"Cannot scan directory: {e}"}
    
    # 2. FIND ALL CSV FILES (including subdirectories)
    csv_files = []
    
    def find_csv_files(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path) and item.endswith('.csv'):
                csv_files.append(item_path)
            elif os.path.isdir(item_path):
                try:
                    find_csv_files(item_path)
                except:
                    pass
    
    find_csv_files(gdsc_dir)
    
    logger.info(f"🎯 Found {len(csv_files)} total CSV files")
    
    # 3. ANALYZE EACH CSV FOR COMPOUNDS
    dataset_analysis = []
    
    for csv_path in csv_files:
        relative_path = csv_path.replace("/vol/", "")
        file_size = os.path.getsize(csv_path)
        
        logger.info(f"🔍 Analyzing: {relative_path} ({file_size:,} bytes)")
        
        try:
            # Read sample to check structure
            df_sample = pd.read_csv(csv_path, nrows=1000)
            total_rows = len(pd.read_csv(csv_path))
            
            logger.info(f"  📊 Structure: {total_rows:,} rows × {len(df_sample.columns)} columns")
            
            # Look for SMILES columns
            smiles_columns = []
            for col in df_sample.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['smiles', 'canonical', 'structure', 'compound_smiles']):
                    smiles_columns.append(col)
            
            logger.info(f"  🧬 SMILES columns found: {smiles_columns}")
            
            # Analyze each SMILES column
            compound_counts = {}
            valid_smiles_examples = {}
            
            if smiles_columns:
                df_full = pd.read_csv(csv_path)
                
                for smiles_col in smiles_columns:
                    # Count unique non-null SMILES
                    unique_smiles = df_full[smiles_col].dropna().unique()
                    compound_counts[smiles_col] = len(unique_smiles)
                    
                    # Get examples of SMILES to verify they're real
                    valid_examples = []
                    for smiles in unique_smiles[:5]:  # First 5 examples
                        if isinstance(smiles, str) and len(smiles) > 10:  # Basic validity check
                            valid_examples.append(smiles)
                    valid_smiles_examples[smiles_col] = valid_examples
                    
                    logger.info(f"    Column '{smiles_col}': {len(unique_smiles):,} unique compounds")
                    logger.info(f"    Examples: {valid_examples[:3]}")  # Show first 3
            
            # Look for activity columns
            activity_columns = []
            for col in df_sample.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['ic50', 'pic50', 'activity', 'response', 'auc', 'ln_ic50']):
                    activity_columns.append(col)
            
            logger.info(f"  📈 Activity columns: {activity_columns}")
            
            # Look for cell line columns
            cell_columns = []
            for col in df_sample.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['cell', 'line', 'cosmic', 'sample', 'tissue']):
                    cell_columns.append(col)
            
            logger.info(f"  🧪 Cell line columns: {cell_columns}")
            
            # Store analysis
            dataset_info = {
                'file_path': csv_path,
                'relative_path': relative_path,
                'file_size': file_size,
                'total_rows': total_rows,
                'total_columns': len(df_sample.columns),
                'smiles_columns': smiles_columns,
                'compound_counts': compound_counts,
                'max_compounds': max(compound_counts.values()) if compound_counts else 0,
                'valid_smiles_examples': valid_smiles_examples,
                'activity_columns': activity_columns,
                'cell_columns': cell_columns,
                'suitable_for_training': len(smiles_columns) > 0 and len(activity_columns) > 0
            }
            
            dataset_analysis.append(dataset_info)
            
            logger.info(f"  ✅ Max compounds in this file: {dataset_info['max_compounds']:,}")
            
        except Exception as e:
            logger.warning(f"  ❌ Error analyzing {relative_path}: {e}")
            continue
    
    # 4. FIND BEST DATASET
    if not dataset_analysis:
        logger.error("❌ No valid datasets found")
        return {"error": "No valid datasets", "datasets_found": 0}
    
    # Sort by compound count
    dataset_analysis.sort(key=lambda x: x['max_compounds'], reverse=True)
    
    logger.info("📊 DATASET SUMMARY (Top 5):")
    for i, dataset in enumerate(dataset_analysis[:5]):
        logger.info(f"  {i+1}. {dataset['relative_path']}")
        logger.info(f"     Compounds: {dataset['max_compounds']:,}")
        logger.info(f"     Rows: {dataset['total_rows']:,}")
        logger.info(f"     SMILES cols: {dataset['smiles_columns']}")
        logger.info(f"     Activity cols: {dataset['activity_columns']}")
        logger.info(f"     Cell cols: {dataset['cell_columns']}")
        logger.info(f"     Training ready: {'✅' if dataset['suitable_for_training'] else '❌'}")
        logger.info("")
    
    best_dataset = dataset_analysis[0]
    
    # 5. FINAL VERDICT
    logger.info("🏁 DIAGNOSTIC RESULTS:")
    logger.info(f"Best dataset: {best_dataset['relative_path']}")
    logger.info(f"Unique compounds: {best_dataset['max_compounds']:,}")
    logger.info(f"Target ≥600: {'✅ MET' if best_dataset['max_compounds'] >= 600 else '❌ NOT MET'}")
    logger.info(f"Has real SMILES: {'✅ YES' if best_dataset['valid_smiles_examples'] else '❌ NO'}")
    logger.info(f"Training ready: {'✅ YES' if best_dataset['suitable_for_training'] else '❌ NO'}")
    
    if best_dataset['max_compounds'] >= 600 and best_dataset['suitable_for_training']:
        logger.info("🎉 DATASET VERIFIED: Ready for ChemBERTa training!")
    else:
        logger.warning("⚠️ DATASET CONCERNS: May not meet requirements")
    
    return {
        'success': True,
        'best_dataset': best_dataset,
        'all_datasets': dataset_analysis,
        'meets_requirements': best_dataset['max_compounds'] >= 600 and best_dataset['suitable_for_training'],
        'total_datasets_found': len(dataset_analysis),
        'compounds_found': best_dataset['max_compounds']
    }

if __name__ == "__main__":
    with app.run():
        result = verify_gdsc_dataset.remote()
        
        print("\n" + "="*80)
        print("🔍 DATASET VERIFICATION COMPLETE")
        print("="*80)
        
        if result.get('success'):
            best = result['best_dataset']
            print(f"📁 Best Dataset: {best['relative_path']}")
            print(f"🧬 Unique Compounds: {best['max_compounds']:,}")
            print(f"📊 Total Rows: {best['total_rows']:,}")
            print(f"✅ Requirements Met: {'YES' if result['meets_requirements'] else 'NO'}")
            
            if result['meets_requirements']:
                print("\n🎉 SUCCESS: Dataset ready for training!")
                print("Next step: Run full ChemBERTa training")
            else:
                print("\n⚠️  ISSUES FOUND:")
                if best['max_compounds'] < 600:
                    print(f"   • Insufficient compounds: {best['max_compounds']} < 600")
                if not best['suitable_for_training']:
                    print("   • Missing required columns for training")
        else:
            print("❌ DIAGNOSTIC FAILED:")
            print(f"Error: {result.get('error', 'Unknown error')}")