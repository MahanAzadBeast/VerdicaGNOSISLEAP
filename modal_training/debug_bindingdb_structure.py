"""
Debug BindingDB data structure to understand column names
"""

import modal
from pathlib import Path
import pandas as pd

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy"
])

app = modal.App("debug-bindingdb-structure")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    timeout=300
)
def debug_bindingdb_structure():
    """Debug BindingDB structure"""
    
    print("🔍 DEBUGGING BINDINGDB STRUCTURE")
    print("=" * 60)
    
    datasets_dir = Path("/vol/datasets")
    bindingdb_path = datasets_dir / "realistic_bindingdb_raw_data.csv"
    
    if bindingdb_path.exists():
        df = pd.read_csv(bindingdb_path)
        
        print(f"📄 BindingDB Dataset:")
        print(f"   • Shape: {df.shape}")
        print(f"   • Columns: {list(df.columns)}")
        
        print(f"\n📊 Sample data:")
        for i in range(min(3, len(df))):
            print(f"\n   Row {i+1}:")
            for col, val in df.iloc[i].items():
                print(f"     {col}: {val}")
        
        print(f"\n🔍 Looking for affinity columns:")
        affinity_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['ic50', 'ki', 'ec50', 'kd', 'affinity', 'nm', 'um']):
                affinity_cols.append(col)
        
        print(f"   Potential affinity columns: {affinity_cols}")
        
        if affinity_cols:
            for col in affinity_cols[:3]:
                sample_vals = df[col].dropna().head(5).tolist()
                print(f"   {col} samples: {sample_vals}")
        
        print(f"\n🎯 Looking for target columns:")
        target_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['target', 'protein', 'gene', 'uniprot']):
                target_cols.append(col)
        
        print(f"   Potential target columns: {target_cols}")
        
        if target_cols:
            for col in target_cols[:3]:
                sample_vals = df[col].dropna().head(5).tolist()
                print(f"   {col} samples: {sample_vals}")
    
    else:
        print(f"❌ BindingDB file not found: {bindingdb_path}")

if __name__ == "__main__":
    print("🔍 Debugging BindingDB structure...")