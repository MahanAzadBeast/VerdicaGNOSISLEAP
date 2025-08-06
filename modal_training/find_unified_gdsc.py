"""
Targeted search for gdsc_unified_with_SMILES.csv file mentioned by previous agent
"""

import modal
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("find-unified-gdsc")

image = modal.Image.debian_slim().pip_install(["pandas==2.1.0"])

data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    timeout=900,
    volumes={"/vol": data_volume}
)
def find_unified_gdsc():
    """Search specifically for gdsc_unified_with_SMILES.csv"""
    
    logger.info("🎯 TARGETED SEARCH FOR: gdsc_unified_with_SMILES.csv")
    logger.info("Expected: 575,197 records, 621 unique drugs")
    
    target_file = "gdsc_unified_with_SMILES.csv"
    found_files = []
    
    # Search recursively through all directories
    def search_for_file(directory, target_filename, max_depth=4, current_depth=0):
        if current_depth >= max_depth:
            return
            
        try:
            items = os.listdir(directory)
            for item in items:
                item_path = os.path.join(directory, item)
                
                if os.path.isfile(item_path):
                    if item == target_filename:
                        found_files.append(item_path)
                        logger.info(f"🎉 FOUND: {item_path}")
                        return True
                    elif target_filename.lower() in item.lower():
                        found_files.append(item_path)
                        logger.info(f"🔍 Similar file: {item_path}")
                        
                elif os.path.isdir(item_path) and current_depth < max_depth - 1:
                    if search_for_file(item_path, target_filename, max_depth, current_depth + 1):
                        return True
                        
        except Exception as e:
            logger.warning(f"Cannot search {directory}: {e}")
        
        return False
    
    # Start search from /vol
    logger.info("🔍 Searching entire volume for target file...")
    search_for_file("/vol", target_file)
    
    if not found_files:
        logger.warning("❌ Target file not found. Checking all CSV files with 'unified' in name...")
        
        # Broader search for any file with 'unified' in name
        def find_unified_files(directory, max_depth=4, current_depth=0):
            if current_depth >= max_depth:
                return
                
            try:
                items = os.listdir(directory)
                for item in items:
                    item_path = os.path.join(directory, item)
                    
                    if os.path.isfile(item_path):
                        if ('unified' in item.lower() and item.endswith('.csv')):
                            found_files.append(item_path)
                            logger.info(f"📄 Found unified file: {item_path}")
                            
                    elif os.path.isdir(item_path) and current_depth < max_depth - 1:
                        find_unified_files(item_path, max_depth, current_depth + 1)
                        
            except Exception as e:
                logger.warning(f"Cannot search {directory}: {e}")
        
        find_unified_files("/vol")
    
    # Analyze found files
    results = []
    
    for file_path in found_files:
        logger.info(f"📊 Analyzing: {file_path}")
        
        try:
            file_size = os.path.getsize(file_path)
            logger.info(f"  Size: {file_size:,} bytes")
            
            # Quick sample
            df_sample = pd.read_csv(file_path, nrows=1000)
            
            # Count total rows
            df_full = pd.read_csv(file_path)
            total_rows = len(df_full)
            
            logger.info(f"  Rows: {total_rows:,}")
            logger.info(f"  Columns: {len(df_full.columns)}")
            
            # Look for SMILES and count unique compounds
            smiles_cols = []
            for col in df_full.columns:
                if any(term in col.lower() for term in ['smiles', 'canonical']):
                    smiles_cols.append(col)
            
            unique_compounds = 0
            if smiles_cols:
                for smiles_col in smiles_cols:
                    count = df_full[smiles_col].nunique()
                    unique_compounds = max(unique_compounds, count)
                    logger.info(f"  {smiles_col}: {count:,} unique compounds")
            
            results.append({
                'file_path': file_path,
                'file_size': file_size,
                'total_rows': total_rows,
                'unique_compounds': unique_compounds,
                'smiles_columns': smiles_cols,
                'is_target_file': file_path.endswith(target_file),
                'matches_expected': total_rows > 500000 and unique_compounds > 600
            })
            
            if file_path.endswith(target_file):
                logger.info(f"  🎯 THIS IS THE TARGET FILE!")
                logger.info(f"  Expected vs Found:")
                logger.info(f"    Rows: 575,197 vs {total_rows:,}")
                logger.info(f"    Compounds: 621 vs {unique_compounds:,}")
            
        except Exception as e:
            logger.error(f"  ❌ Error analyzing {file_path}: {e}")
    
    # Summary
    if results:
        logger.info("\n📊 SEARCH RESULTS SUMMARY:")
        best_match = max(results, key=lambda x: x['unique_compounds'])
        
        for result in sorted(results, key=lambda x: x['unique_compounds'], reverse=True):
            logger.info(f"  📄 {result['file_path'].replace('/vol/', '')}")
            logger.info(f"     Rows: {result['total_rows']:,}")
            logger.info(f"     Compounds: {result['unique_compounds']:,}")
            logger.info(f"     Target: {'✅' if result['is_target_file'] else '❌'}")
            logger.info(f"     Meets spec: {'✅' if result['matches_expected'] else '❌'}")
            logger.info("")
        
        return {
            'found': True,
            'results': results,
            'best_match': best_match,
            'target_found': any(r['is_target_file'] for r in results)
        }
    else:
        logger.error("❌ NO FILES FOUND")
        return {'found': False, 'error': 'No matching files found'}

if __name__ == "__main__":
    with app.run():
        result = find_unified_gdsc.remote()
        
        print("\n" + "="*80)
        print("🎯 TARGETED SEARCH RESULTS")
        print("="*80)
        
        if result.get('found'):
            print(f"Found {len(result['results'])} matching files")
            
            if result['target_found']:
                print("🎉 SUCCESS: Found gdsc_unified_with_SMILES.csv!")
            else:
                print("⚠️ Target file not found, but found similar files")
            
            best = result['best_match']
            print(f"\nBest match: {best['file_path'].replace('/vol/', '')}")
            print(f"Rows: {best['total_rows']:,}")
            print(f"Compounds: {best['unique_compounds']:,}")
            print(f"Meets >600 requirement: {'YES' if best['unique_compounds'] >= 600 else 'NO'}")
            
        else:
            print("❌ NO MATCHING FILES FOUND")
            print("Will proceed with 404-compound dataset as fallback")