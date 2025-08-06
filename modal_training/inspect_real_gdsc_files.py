"""
Inspect Real GDSC Files in Modal
Check exactly what's in gdsc_unique_drugs_with_SMILES.csv and gdsc_sample_10k.csv
"""

import modal
import pandas as pd
import os

app = modal.App("inspect-real-gdsc-files")

image = modal.Image.debian_slim().pip_install([
    "pandas==2.1.0", 
    "numpy==1.24.3"
])

data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    volumes={"/vol": data_volume},
    timeout=300
)
def inspect_real_gdsc_files():
    """Inspect the real GDSC files from your screenshot"""
    
    print("🔍 INSPECTING REAL GDSC FILES")
    print("=" * 60)
    
    # Files from your screenshot
    target_files = [
        "gdsc_unique_drugs_with_SMILES.csv",
        "gdsc_sample_10k.csv",
        "DATASET_MANIFEST.md",
        "INFO.txt"
    ]
    
    vol_path = "/vol"
    
    # List all files first
    if os.path.exists(vol_path):
        all_files = os.listdir(vol_path)
        print("📁 ALL FILES IN EXPANDED-DATASETS:")
        for f in sorted(all_files):
            file_path = os.path.join(vol_path, f)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"   📄 {f} ({size:,} bytes)")
        print()
    
    # Inspect each target file
    for filename in target_files:
        file_path = os.path.join(vol_path, filename)
        
        if os.path.exists(file_path):
            print(f"📊 INSPECTING: {filename}")
            print("-" * 40)
            
            if filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                    print(f"✅ Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
                    
                    print(f"📋 Columns:")
                    for i, col in enumerate(df.columns):
                        print(f"   {i+1:2d}. {col}")
                    
                    print(f"📋 First 3 rows:")
                    print(df.head(3))
                    
                    print(f"📋 Data types:")
                    print(df.dtypes)
                    
                    # Check for missing values
                    print(f"📋 Missing values:")
                    missing = df.isnull().sum()
                    for col, miss_count in missing.items():
                        if miss_count > 0:
                            print(f"   {col}: {miss_count:,} ({100*miss_count/len(df):.1f}%)")
                    
                    # Look for key columns
                    key_info = {}
                    
                    # SMILES columns
                    for col in df.columns:
                        if 'smiles' in col.lower():
                            key_info[col] = 'SMILES'
                            print(f"🧬 SMILES column found: {col}")
                            if len(df[col].dropna()) > 0:
                                sample_smiles = df[col].dropna().iloc[0]
                                print(f"   Sample SMILES: {sample_smiles}")
                    
                    # Activity columns  
                    for col in df.columns:
                        if any(x in col.lower() for x in ['ic50', 'pic50', 'ln_ic50', 'activity']):
                            key_info[col] = 'Activity'
                            print(f"🎯 Activity column found: {col}")
                            if len(df[col].dropna()) > 0:
                                stats = df[col].describe()
                                print(f"   Range: {stats['min']:.3f} - {stats['max']:.3f}")
                                print(f"   Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
                    
                    # Cell line columns
                    for col in df.columns:
                        if any(x in col.lower() for x in ['cell', 'line', 'cosmic']):
                            key_info[col] = 'Cell Line'
                            print(f"🔬 Cell line column found: {col}")
                            if len(df[col].dropna()) > 0:
                                unique_lines = df[col].nunique()
                                sample_lines = df[col].unique()[:5]
                                print(f"   Unique cell lines: {unique_lines}")
                                print(f"   Sample lines: {list(sample_lines)}")
                    
                except Exception as e:
                    print(f"❌ Error reading CSV {filename}: {e}")
            
            elif filename.endswith(('.txt', '.md')):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print("📄 File content:")
                        print(content[:500])  # First 500 characters
                        if len(content) > 500:
                            print("... (truncated)")
                except Exception as e:
                    print(f"❌ Error reading text file {filename}: {e}")
            
            print("=" * 60)
            print()
        
        else:
            print(f"❌ File not found: {filename}")
            print()
    
    return {"status": "inspection_complete"}

if __name__ == "__main__":
    with app.run():
        result = inspect_real_gdsc_files.remote()
        print("✅ Real GDSC file inspection complete!")