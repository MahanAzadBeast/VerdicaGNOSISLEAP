"""
List all datasets in Modal volumes to locate the comprehensive training data
"""

import modal
import pandas as pd
from pathlib import Path

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy"
])

app = modal.App("list-modal-datasets")

# All the volumes we've identified
expanded_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=False)
oncoprotein_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=False) 
trained_models_volume = modal.Volume.from_name("trained-models", create_if_missing=False)
molbert_volume = modal.Volume.from_name("molbert-models", create_if_missing=False)

@app.function(
    image=image,
    volumes={
        "/vol/expanded": expanded_volume,
        "/vol/oncoprotein": oncoprotein_volume, 
        "/vol/models": trained_models_volume,
        "/vol/molbert": molbert_volume
    },
    timeout=300
)
def list_all_datasets():
    """List all datasets in Modal volumes"""
    
    print("🔍 MODAL VOLUMES DATASET INVENTORY")
    print("=" * 80)
    
    volumes_info = {
        "expanded-datasets": "/vol/expanded",
        "oncoprotein-datasets": "/vol/oncoprotein", 
        "trained-models": "/vol/models",
        "molbert-models": "/vol/molbert"
    }
    
    for volume_name, volume_path in volumes_info.items():
        print(f"\n📁 VOLUME: {volume_name}")
        print("-" * 60)
        
        volume_dir = Path(volume_path)
        
        if volume_dir.exists():
            # List all files
            all_files = list(volume_dir.rglob("*"))
            
            if not all_files:
                print("   📂 Empty volume")
                continue
            
            # Categorize files
            csv_files = [f for f in all_files if f.suffix.lower() == '.csv']
            pkl_files = [f for f in all_files if f.suffix.lower() in ['.pkl', '.pickle']]
            model_files = [f for f in all_files if f.suffix.lower() in ['.pt', '.pth', '.h5', '.joblib']]
            other_files = [f for f in all_files if f not in csv_files + pkl_files + model_files and f.is_file()]
            
            # Show CSV files (most important for datasets)
            if csv_files:
                print(f"   📊 CSV FILES ({len(csv_files)}):")
                for csv_file in sorted(csv_files):
                    try:
                        # Get basic info about each CSV
                        df = pd.read_csv(csv_file)
                        size_mb = csv_file.stat().st_size / (1024 * 1024)
                        relative_path = csv_file.relative_to(volume_dir)
                        print(f"     📄 {relative_path}")
                        print(f"        📈 {len(df):,} rows × {len(df.columns)} cols, {size_mb:.1f}MB")
                        
                        # Show column names for key datasets
                        if 'training' in csv_file.name.lower() or 'combined' in csv_file.name.lower():
                            print(f"        🔍 Columns: {list(df.columns)}")
                            
                            # Show assay type distribution if present
                            if 'assay_type' in df.columns:
                                assay_dist = df['assay_type'].value_counts().head(10)
                                print(f"        🧪 Assay types: {dict(assay_dist)}")
                            elif 'Assay_Type' in df.columns:
                                assay_dist = df['Assay_Type'].value_counts().head(10)
                                print(f"        🧪 Assay types: {dict(assay_dist)}")
                            
                            # Show target distribution if present
                            target_cols = [col for col in df.columns if 'target' in col.lower()]
                            if target_cols:
                                target_col = target_cols[0]
                                target_count = df[target_col].nunique()
                                print(f"        🎯 Unique targets: {target_count}")
                                if target_count <= 20:
                                    targets = df[target_col].value_counts().head(10)
                                    print(f"        🎯 Top targets: {dict(targets)}")
                        
                        print()
                    except Exception as e:
                        print(f"     📄 {csv_file.name} (Error reading: {e})")
            
            # Show other file types
            if pkl_files:
                print(f"   🥒 PICKLE FILES ({len(pkl_files)}):")
                for pkl_file in sorted(pkl_files):
                    size_mb = pkl_file.stat().st_size / (1024 * 1024)
                    relative_path = pkl_file.relative_to(volume_dir)
                    print(f"     📦 {relative_path} ({size_mb:.1f}MB)")
            
            if model_files:
                print(f"   🤖 MODEL FILES ({len(model_files)}):")
                for model_file in sorted(model_files):
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    relative_path = model_file.relative_to(volume_dir)
                    print(f"     🧠 {relative_path} ({size_mb:.1f}MB)")
            
            if other_files:
                print(f"   📁 OTHER FILES ({len(other_files)}):")
                for other_file in sorted(other_files)[:10]:  # Show first 10
                    relative_path = other_file.relative_to(volume_dir)
                    print(f"     📄 {relative_path}")
                if len(other_files) > 10:
                    print(f"     ... and {len(other_files) - 10} more")
        
        else:
            print(f"   ❌ Volume directory not found: {volume_path}")
    
    print("\n" + "=" * 80)
    print("🎉 INVENTORY COMPLETE!")

if __name__ == "__main__":
    list_all_datasets.remote()