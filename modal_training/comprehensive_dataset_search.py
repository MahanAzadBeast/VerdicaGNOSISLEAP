"""
Extended search for comprehensive GDSC dataset with >600 compounds
Check multiple locations and volumes
"""

import modal
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("comprehensive-dataset-search")

image = modal.Image.debian_slim().pip_install([
    "pandas==2.1.0", "numpy==1.24.3"
])

data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    timeout=1800,
    volumes={"/vol": data_volume}
)
def comprehensive_dataset_search():
    """Search comprehensively for GDSC dataset with >600 compounds"""
    
    logger.info("🔍 COMPREHENSIVE SEARCH FOR GDSC DATASET")
    logger.info("Searching all possible locations for real GDSC data with >600 compounds")
    
    # 1. CHECK ALL DIRECTORIES IN VOLUME
    logger.info("📁 Exploring volume structure...")
    
    vol_dirs = []
    try:
        for item in os.listdir("/vol"):
            item_path = os.path.join("/vol", item)
            if os.path.isdir(item_path):
                vol_dirs.append(item)
                logger.info(f"  📁 Found directory: {item}")
    except Exception as e:
        logger.error(f"Error listing /vol: {e}")
        return {"error": "Cannot access volume"}
    
    # 2. SEARCH FOR GDSC-RELATED FILES
    potential_datasets = []
    
    def search_gdsc_files(directory, max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
            
        try:
            items = os.listdir(directory)
            for item in items:
                item_path = os.path.join(directory, item)
                
                if os.path.isfile(item_path):
                    # Check for GDSC-related CSV files
                    if (item.endswith('.csv') and 
                        any(term in item.lower() for term in ['gdsc', 'drug', 'compound', 'cancer', 'screen', 'cytotox', 'ic50'])):
                        
                        try:
                            file_size = os.path.getsize(item_path)
                            # Only consider files >1MB (likely to have substantial data)
                            if file_size > 1_000_000:
                                potential_datasets.append({
                                    'path': item_path,
                                    'relative_path': item_path.replace('/vol/', ''),
                                    'filename': item,
                                    'size': file_size
                                })
                                logger.info(f"  🎯 Found candidate: {item_path.replace('/vol/', '')} ({file_size:,} bytes)")
                        except:
                            pass
                            
                elif os.path.isdir(item_path) and current_depth < max_depth - 1:
                    search_gdsc_files(item_path, max_depth, current_depth + 1)
                    
        except Exception as e:
            logger.warning(f"Cannot search {directory}: {e}")
    
    # Search all directories
    for vol_dir in vol_dirs:
        vol_path = os.path.join("/vol", vol_dir)
        logger.info(f"🔍 Searching: {vol_dir}/")
        search_gdsc_files(vol_path)
    
    logger.info(f"Found {len(potential_datasets)} potential datasets")
    
    # 3. ANALYZE EACH POTENTIAL DATASET
    dataset_results = []
    
    for dataset in potential_datasets:
        logger.info(f"📊 Analyzing: {dataset['relative_path']}")
        
        try:
            # Quick sample analysis
            df_sample = pd.read_csv(dataset['path'], nrows=1000)
            
            # Count SMILES
            smiles_cols = []
            smiles_counts = {}
            
            for col in df_sample.columns:
                if any(term in col.lower() for term in ['smiles', 'canonical', 'structure']):
                    smiles_cols.append(col)
                    
            if smiles_cols:
                # Load full file to count compounds
                logger.info("  Loading full file to count compounds...")
                df_full = pd.read_csv(dataset['path'])
                
                for smiles_col in smiles_cols:
                    unique_count = df_full[smiles_col].nunique()
                    smiles_counts[smiles_col] = unique_count
                    logger.info(f"    {smiles_col}: {unique_count:,} unique compounds")
                
                max_compounds = max(smiles_counts.values()) if smiles_counts else 0
                
                # Check for activity data
                activity_cols = []
                for col in df_full.columns:
                    if any(term in col.lower() for term in ['ic50', 'pic50', 'auc', 'response', 'activity']):
                        activity_cols.append(col)
                
                # Check for cell line data
                cell_cols = []
                for col in df_full.columns:
                    if any(term in col.lower() for term in ['cell', 'line', 'cosmic', 'sample']):
                        cell_cols.append(col)
                
                result = {
                    'dataset': dataset,
                    'total_rows': len(df_full),
                    'total_columns': len(df_full.columns),
                    'smiles_columns': smiles_cols,
                    'compound_counts': smiles_counts,
                    'max_compounds': max_compounds,
                    'activity_columns': activity_cols,
                    'cell_columns': cell_cols,
                    'has_training_data': len(activity_cols) > 0 and len(smiles_cols) > 0,
                    'meets_compound_target': max_compounds >= 600
                }
                
                dataset_results.append(result)
                
                logger.info(f"    📊 {len(df_full):,} rows, {max_compounds:,} compounds")
                logger.info(f"    🧬 SMILES: {smiles_cols}")
                logger.info(f"    📈 Activity: {activity_cols}")
                logger.info(f"    🧪 Cell lines: {cell_cols}")
                logger.info(f"    ✅ Target met: {'YES' if max_compounds >= 600 else 'NO'}")
                
            else:
                logger.info("    ❌ No SMILES columns found")
                
        except Exception as e:
            logger.warning(f"    ❌ Error analyzing {dataset['relative_path']}: {e}")
    
    # 4. FIND BEST DATASET
    if not dataset_results:
        logger.error("❌ No suitable datasets found")
        return {"success": False, "error": "No datasets with SMILES found"}
    
    # Sort by compound count
    dataset_results.sort(key=lambda x: x['max_compounds'], reverse=True)
    
    logger.info("\n📊 COMPREHENSIVE RESULTS:")
    logger.info("-" * 80)
    
    for i, result in enumerate(dataset_results):
        dataset = result['dataset']
        logger.info(f"{i+1}. {dataset['relative_path']}")
        logger.info(f"   Size: {dataset['size']:,} bytes")
        logger.info(f"   Rows: {result['total_rows']:,}")
        logger.info(f"   Compounds: {result['max_compounds']:,}")
        logger.info(f"   Target ≥600: {'✅' if result['meets_compound_target'] else '❌'}")
        logger.info(f"   Training ready: {'✅' if result['has_training_data'] else '❌'}")
        logger.info("")
    
    best_result = dataset_results[0]
    
    # 5. FINAL ASSESSMENT
    logger.info("🏁 SEARCH COMPLETE:")
    logger.info(f"Best dataset: {best_result['dataset']['relative_path']}")
    logger.info(f"Compounds: {best_result['max_compounds']:,}")
    logger.info(f"Meets target: {'✅ YES' if best_result['meets_compound_target'] else '❌ NO'}")
    
    if best_result['meets_compound_target'] and best_result['has_training_data']:
        logger.info("🎉 SUCCESS: Found suitable dataset for training!")
    else:
        logger.warning("⚠️ ISSUE: No dataset meets all requirements")
        
        # Look for the largest available dataset
        logger.info("\nRecommendation:")
        if best_result['max_compounds'] > 300:
            logger.info(f"Use best available: {best_result['max_compounds']} compounds")
            logger.info("This may still achieve good results")
        else:
            logger.info("Consider looking for additional data sources")
    
    return {
        'success': True,
        'best_dataset': best_result,
        'all_results': dataset_results,
        'meets_requirements': best_result['meets_compound_target'] and best_result['has_training_data'],
        'total_datasets_found': len(dataset_results)
    }

if __name__ == "__main__":
    with app.run():
        result = comprehensive_dataset_search.remote()
        
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE DATASET SEARCH COMPLETE")
        print("="*80)
        
        if result.get('success'):
            best = result['best_dataset']
            dataset_info = best['dataset']
            
            print(f"📁 Best Dataset: {dataset_info['relative_path']}")
            print(f"📊 File Size: {dataset_info['size']:,} bytes")
            print(f"🧬 Unique Compounds: {best['max_compounds']:,}")
            print(f"📊 Total Rows: {best['total_rows']:,}")
            print(f"✅ Target ≥600: {'YES' if best['meets_compound_target'] else 'NO'}")
            print(f"🎯 Training Ready: {'YES' if best['has_training_data'] else 'NO'}")
            
            if result['meets_requirements']:
                print("\n🎉 EXCELLENT: Found dataset meeting all requirements!")
                print("Ready to proceed with ChemBERTa training!")
            else:
                print(f"\n⚠️ PARTIAL: Best available has {best['max_compounds']} compounds")
                print("Recommend proceeding with available data")
                
            print(f"\nTotal datasets evaluated: {result['total_datasets_found']}")
        else:
            print("❌ SEARCH FAILED:")
            print(f"Error: {result.get('error', 'Unknown error')}")