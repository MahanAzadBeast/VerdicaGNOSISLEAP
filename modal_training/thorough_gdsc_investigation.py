"""
Thorough Investigation of Real GDSC Data on Modal
Find the real dataset with >600 compounds that the last agent found
"""

import modal
import pandas as pd
import os

app = modal.App("thorough-gdsc-investigation")

image = modal.Image.debian_slim().pip_install([
    "pandas==2.1.0"
])

data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    volumes={"/vol": data_volume},
    timeout=300
)
def investigate_gdsc_thoroughly():
    """Thoroughly investigate all GDSC files to find >600 compounds"""
    
    print("🔍 THOROUGH GDSC INVESTIGATION ON MODAL")
    print("=" * 60)
    print("Looking for dataset with >600 compounds...")
    print()
    
    vol_path = "/vol"
    
    if not os.path.exists(vol_path):
        print("❌ Volume path does not exist!")
        return {"error": "Volume not accessible"}
    
    # 1. LIST ALL FILES WITH DETAILS
    all_files = os.listdir(vol_path)
    print(f"📁 TOTAL FILES FOUND: {len(all_files)}")
    print("-" * 40)
    
    csv_files = []
    other_files = []
    
    for filename in sorted(all_files):
        file_path = os.path.join(vol_path, filename)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            size_mb = size / (1024 * 1024)
            
            print(f"📄 {filename}")
            print(f"    Size: {size:,} bytes ({size_mb:.2f} MB)")
            
            if filename.endswith('.csv'):
                csv_files.append((filename, file_path, size))
                print(f"    Type: CSV Data File ⭐")
            elif filename.endswith(('.txt', '.md')):
                other_files.append((filename, file_path))
                print(f"    Type: Text/Markdown")
            print()
    
    print(f"✅ Found {len(csv_files)} CSV files and {len(other_files)} other files")
    print()
    
    # 2. EXAMINE EACH CSV FILE IN DETAIL
    print("📊 DETAILED CSV ANALYSIS")
    print("=" * 60)
    
    best_dataset_info = None
    max_unique_smiles = 0
    
    for filename, file_path, size in csv_files:
        print(f"🔍 ANALYZING: {filename}")
        print("-" * 30)
        
        try:
            # Load the file
            df = pd.read_csv(file_path)
            print(f"✅ Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Show all columns
            print(f"📋 ALL COLUMNS ({len(df.columns)}):")
            for i, col in enumerate(df.columns, 1):
                print(f"   {i:2d}. {col}")
            print()
            
            # Look for SMILES columns specifically
            smiles_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['smiles', 'canonical', 'structure', 'mol']):
                    smiles_columns.append(col)
            
            print(f"🧬 SMILES COLUMNS FOUND: {len(smiles_columns)}")
            for smiles_col in smiles_columns:
                print(f"   ⭐ {smiles_col}")
                
                # Count unique SMILES
                unique_smiles = df[smiles_col].dropna().nunique()
                print(f"      Unique SMILES: {unique_smiles:,}")
                print(f"      Non-null SMILES: {df[smiles_col].count():,}")
                
                if unique_smiles > 0:
                    # Show sample SMILES
                    sample_smiles = df[smiles_col].dropna().head(3).tolist()
                    print(f"      Sample SMILES:")
                    for i, smiles in enumerate(sample_smiles, 1):
                        print(f"        {i}. {smiles}")
                
                # Track best dataset
                if unique_smiles > max_unique_smiles:
                    max_unique_smiles = unique_smiles
                    best_dataset_info = {
                        'filename': filename,
                        'smiles_column': smiles_col,
                        'unique_smiles': unique_smiles,
                        'total_rows': len(df),
                        'all_columns': list(df.columns)
                    }
            print()
            
            # Look for activity/IC50 columns
            activity_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['ic50', 'pic50', 'ln_ic50', 'activity', 'response', 'auc']):
                    activity_columns.append(col)
            
            if activity_columns:
                print(f"🎯 ACTIVITY COLUMNS FOUND: {len(activity_columns)}")
                for act_col in activity_columns:
                    print(f"   ⭐ {act_col}")
                    
                    non_null = df[act_col].count()
                    if non_null > 0:
                        stats = df[act_col].describe()
                        print(f"      Non-null values: {non_null:,}")
                        print(f"      Range: {stats['min']:.3f} to {stats['max']:.3f}")
                        print(f"      Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
                print()
            
            # Look for cell line columns
            cell_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['cell', 'line', 'cosmic', 'sample', 'tissue']):
                    cell_columns.append(col)
            
            if cell_columns:
                print(f"🔬 CELL LINE COLUMNS FOUND: {len(cell_columns)}")
                for cell_col in cell_columns:
                    print(f"   ⭐ {cell_col}")
                    
                    unique_cells = df[cell_col].nunique()
                    non_null = df[cell_col].count()
                    print(f"      Unique cell lines: {unique_cells:,}")
                    print(f"      Non-null values: {non_null:,}")
                    
                    if unique_cells > 0 and unique_cells <= 20:
                        sample_cells = df[cell_col].unique()[:10]
                        print(f"      Sample cell lines: {list(sample_cells)}")
                print()
            
            # Show first few rows for context
            print(f"📋 FIRST 3 ROWS:")
            print(df.head(3).to_string(max_cols=8))
            print()
            
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
        
        print("=" * 60)
        print()
    
    # 3. READ TEXT FILES FOR CONTEXT
    print("📄 TEXT FILES EXAMINATION")
    print("-" * 30)
    
    for filename, file_path in other_files:
        print(f"📄 Reading: {filename}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content[:1000])  # First 1000 characters
                if len(content) > 1000:
                    print("\n... (truncated)")
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
        print("-" * 30)
    
    # 4. SUMMARY
    print("📊 INVESTIGATION SUMMARY")
    print("=" * 60)
    print(f"Total CSV files examined: {len(csv_files)}")
    print(f"Maximum unique SMILES found: {max_unique_smiles:,}")
    
    if best_dataset_info:
        print(f"\n🏆 BEST DATASET FOR TRAINING:")
        print(f"   File: {best_dataset_info['filename']}")
        print(f"   SMILES column: {best_dataset_info['smiles_column']}")
        print(f"   Unique SMILES: {best_dataset_info['unique_smiles']:,}")
        print(f"   Total rows: {best_dataset_info['total_rows']:,}")
        print(f"   Total columns: {len(best_dataset_info['all_columns'])}")
        
        if best_dataset_info['unique_smiles'] >= 600:
            print("   ✅ MEETS >600 COMPOUNDS REQUIREMENT!")
        else:
            print("   ❌ Below 600 compounds requirement")
    
    return {
        'csv_files_found': len(csv_files),
        'max_unique_smiles': max_unique_smiles,
        'best_dataset': best_dataset_info
    }

if __name__ == "__main__":
    with app.run():
        result = investigate_gdsc_thoroughly.remote()
        print(f"✅ Investigation complete! Found max {result.get('max_unique_smiles', 0)} unique SMILES")