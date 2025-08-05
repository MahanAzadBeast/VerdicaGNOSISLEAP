"""
Check Modal datasets using modal run
"""

import modal
import pandas as pd
from pathlib import Path

app = modal.App("check-datasets")

# Setup volumes
expanded_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=False)
oncoprotein_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=False) 

@app.function(
    image=modal.Image.debian_slim().pip_install(["pandas"]),
    volumes={
        "/vol/expanded": expanded_volume,
        "/vol/oncoprotein": oncoprotein_volume
    }
)
def check_datasets():
    """Check what datasets exist in Modal volumes"""
    
    print("🔍 CHECKING MODAL DATASETS")
    print("=" * 60)
    
    for volume_name, path in [("expanded-datasets", "/vol/expanded"), ("oncoprotein-datasets", "/vol/oncoprotein")]:
        print(f"\n📁 {volume_name}:")
        
        vol_path = Path(path)
        if not vol_path.exists():
            print(f"   ❌ Volume not found")
            continue
            
        # Find CSV files
        csv_files = list(vol_path.glob("*.csv"))
        
        if not csv_files:
            print(f"   📂 No CSV files found")
            # List all files
            all_files = list(vol_path.glob("*"))
            if all_files:
                print(f"   📄 Files found: {[f.name for f in all_files[:10]]}")
            continue
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                size_mb = csv_file.stat().st_size / (1024 * 1024)
                print(f"   📊 {csv_file.name}: {len(df):,} rows, {size_mb:.1f}MB")
                print(f"      Columns: {list(df.columns)[:10]}")  # First 10 columns
                
                # Check for multi-assay data
                assay_cols = [col for col in df.columns if 'assay' in col.lower() or 'type' in col.lower()]
                if assay_cols:
                    assay_col = assay_cols[0]
                    assay_types = df[assay_col].value_counts().head(5)
                    print(f"      Assay types: {dict(assay_types)}")
                
                # Check for multi-target data
                target_cols = [col for col in df.columns if 'target' in col.lower()]
                if target_cols:
                    target_col = target_cols[0]
                    num_targets = df[target_col].nunique()
                    print(f"      Unique targets: {num_targets}")
                    if num_targets <= 20:
                        top_targets = df[target_col].value_counts().head(5)
                        print(f"      Top targets: {dict(top_targets)}")
                
                print()
                    
            except Exception as e:
                print(f"   ❌ Error reading {csv_file.name}: {e}")
    
    print("Done!")

@app.local_entrypoint()
def main():
    check_datasets.remote()