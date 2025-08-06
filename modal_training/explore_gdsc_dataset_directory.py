"""
Explore the gdsc_dataset subdirectory in Modal storage
Based on URL: https://modal.com/storage/mahanazad19/main/expanded-datasets/gdsc_dataset
"""

import modal
import pandas as pd
import os

app = modal.App("explore-gdsc-dataset-directory")

image = modal.Image.debian_slim().pip_install(["pandas==2.1.0"])
data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    volumes={"/vol": data_volume},
    timeout=120
)
def explore_gdsc_dataset_directory():
    """Explore the gdsc_dataset subdirectory specifically"""
    
    print("🔍 EXPLORING gdsc_dataset SUBDIRECTORY")
    print("=" * 60)
    print("Path: /vol/gdsc_dataset/")
    print()
    
    # Check the subdirectory
    gdsc_dir = "/vol/gdsc_dataset"
    
    if os.path.exists(gdsc_dir):
        print("✅ Found gdsc_dataset directory!")
        
        # List all files in the subdirectory
        files = os.listdir(gdsc_dir)
        print(f"📁 Files in gdsc_dataset directory ({len(files)}):")
        
        csv_files = []
        other_files = []
        
        for file in sorted(files):
            file_path = os.path.join(gdsc_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                size_mb = size / (1024 * 1024)
                print(f"   📄 {file} ({size:,} bytes, {size_mb:.2f} MB)")
                
                if file.endswith('.csv'):
                    csv_files.append((file, file_path, size))
                else:
                    other_files.append((file, file_path))
        
        print()
        print(f"✅ Found {len(csv_files)} CSV files and {len(other_files)} other files")
        print()
        
        # Examine each CSV file for compound count
        best_dataset = None
        max_compounds = 0
        
        for filename, file_path, size in csv_files:
            print(f"📊 ANALYZING: {filename}")
            print("-" * 40)
            
            try:
                # Load the CSV file
                df = pd.read_csv(file_path)
                print(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")
                
                # Show all columns
                print(f"📋 Columns ({len(df.columns)}):")
                for i, col in enumerate(df.columns, 1):
                    print(f"   {i:2d}. {col}")
                print()
                
                # Look for SMILES columns
                smiles_columns = []
                for col in df.columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in ['smiles', 'canonical', 'structure', 'compound_structure']):
                        smiles_columns.append(col)
                
                print(f"🧬 SMILES COLUMNS FOUND: {len(smiles_columns)}")
                
                for smiles_col in smiles_columns:
                    unique_count = df[smiles_col].nunique()
                    non_null = df[smiles_col].count()
                    null_count = df[smiles_col].isnull().sum()
                    
                    print(f"   ⭐ {smiles_col}:")
                    print(f"      Unique SMILES: {unique_count:,}")
                    print(f"      Non-null: {non_null:,}")
                    print(f"      Null: {null_count:,}")
                    
                    if unique_count > 0:
                        # Show sample SMILES
                        sample_smiles = df[smiles_col].dropna().head(3).tolist()
                        print(f"      Sample SMILES:")
                        for i, smiles in enumerate(sample_smiles, 1):
                            smiles_str = str(smiles)[:60] + "..." if len(str(smiles)) > 60 else str(smiles)
                            print(f"        {i}. {smiles_str}")
                        
                        # Track best dataset
                        if unique_count > max_compounds:
                            max_compounds = unique_count
                            best_dataset = {
                                'filename': filename,
                                'filepath': file_path,
                                'smiles_column': smiles_col,
                                'unique_compounds': unique_count,
                                'total_rows': len(df),
                                'columns': list(df.columns)
                            }
                            print(f"      🏆 NEW BEST: {unique_count:,} unique compounds!")
                
                # Also look for activity/IC50 columns
                activity_cols = []
                for col in df.columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in ['ic50', 'pic50', 'ln_ic50', 'activity', 'response', 'auc']):
                        activity_cols.append(col)
                
                if activity_cols:
                    print(f"🎯 ACTIVITY COLUMNS: {activity_cols}")
                
                # Cell line columns
                cell_cols = []
                for col in df.columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in ['cell', 'line', 'cosmic', 'sample']):
                        cell_cols.append(col)
                
                if cell_cols:
                    print(f"🔬 CELL LINE COLUMNS: {cell_cols}")
                
                print()
                
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
                print()
        
        # Read any text/documentation files
        for filename, file_path in other_files:
            if filename.lower().endswith(('.txt', '.md', '.readme')):
                print(f"📄 READING: {filename}")
                print("-" * 30)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(content[:1000])  # First 1000 chars
                        if len(content) > 1000:
                            print("... (truncated)")
                except Exception as e:
                    print(f"❌ Error reading {filename}: {e}")
                print()
        
        # Summary
        print("🏆 SUMMARY")
        print("=" * 60)
        print(f"Maximum unique compounds found: {max_compounds:,}")
        
        if best_dataset:
            print(f"🎯 BEST DATASET FOR TRAINING:")
            print(f"   File: {best_dataset['filename']}")
            print(f"   SMILES column: {best_dataset['smiles_column']}")
            print(f"   Unique compounds: {best_dataset['unique_compounds']:,}")
            print(f"   Total rows: {best_dataset['total_rows']:,}")
            print(f"   All columns: {len(best_dataset['columns'])}")
            
            if best_dataset['unique_compounds'] >= 600:
                print("   ✅ MEETS >600 COMPOUNDS REQUIREMENT!")
            else:
                print(f"   ❌ Only {best_dataset['unique_compounds']} compounds (need 600+)")
        
        return {
            'directory_found': True,
            'csv_files_count': len(csv_files),
            'max_compounds': max_compounds,
            'best_dataset': best_dataset
        }
    
    else:
        print("❌ gdsc_dataset directory not found!")
        
        # Check what's in the root volume
        vol_root = "/vol"
        if os.path.exists(vol_root):
            root_files = os.listdir(vol_root)
            print(f"📁 Root volume contents: {root_files}")
            
            # Check if there are any subdirectories
            subdirs = [f for f in root_files if os.path.isdir(os.path.join(vol_root, f))]
            if subdirs:
                print(f"📁 Subdirectories found: {subdirs}")
        
        return {
            'directory_found': False,
            'error': 'gdsc_dataset directory not found'
        }

if __name__ == "__main__":
    with app.run():
        result = explore_gdsc_dataset_directory.remote()
        
        if result.get('directory_found'):
            max_compounds = result.get('max_compounds', 0)
            print(f"✅ Exploration complete! Found {max_compounds:,} unique compounds")
            
            if max_compounds >= 600:
                print("🎉 SUCCESS: Found dataset with >600 compounds!")
            else:
                print("⚠️ Need to find more compounds")
        else:
            print("❌ Could not access gdsc_dataset directory")