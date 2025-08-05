"""
Check what real datasets already exist on Modal volume
"""

import modal
from pathlib import Path
import pandas as pd

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy"
])

app = modal.App("check-existing-data")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    timeout=300
)
def check_existing_datasets():
    """Check what datasets already exist"""
    
    print("🔍 CHECKING EXISTING DATASETS ON MODAL VOLUME")
    print("=" * 60)
    
    datasets_dir = Path("/vol/datasets")
    
    if not datasets_dir.exists():
        print("❌ Datasets directory does not exist")
        return {}
    
    all_files = list(datasets_dir.rglob("*"))
    
    print(f"📁 Found {len(all_files)} files/directories:")
    
    datasets_found = {}
    
    for file_path in all_files:
        if file_path.is_file():
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  📄 {file_path.name}: {size_mb:.1f} MB")
                
                # Check if it's a CSV and get row count
                if file_path.suffix == '.csv':
                    try:
                        df = pd.read_csv(file_path, nrows=1)  # Just check structure
                        full_df = pd.read_csv(file_path)
                        print(f"      → {len(full_df):,} rows, {len(full_df.columns)} columns")
                        
                        # Categorize by filename patterns
                        filename_lower = file_path.name.lower()
                        if 'gdsc' in filename_lower:
                            datasets_found['GDSC'] = {
                                'path': str(file_path),
                                'size_mb': size_mb,
                                'rows': len(full_df),
                                'columns': len(full_df.columns)
                            }
                        elif 'tox21' in filename_lower or 'cytotox' in filename_lower:
                            datasets_found['Tox21'] = {
                                'path': str(file_path),
                                'size_mb': size_mb,
                                'rows': len(full_df),
                                'columns': len(full_df.columns)
                            }
                        elif 'binding' in filename_lower:
                            datasets_found['BindingDB'] = {
                                'path': str(file_path),
                                'size_mb': size_mb,
                                'rows': len(full_df),
                                'columns': len(full_df.columns)
                            }
                        elif 'chembl' in filename_lower:
                            datasets_found['ChEMBL'] = {
                                'path': str(file_path),
                                'size_mb': size_mb,
                                'rows': len(full_df),
                                'columns': len(full_df.columns)
                            }
                            
                    except Exception as e:
                        print(f"      → Error reading CSV: {e}")
                
            except Exception as e:
                print(f"  ❌ Error checking {file_path}: {e}")
    
    print(f"\n📊 DATASET SUMMARY:")
    for dataset_type, info in datasets_found.items():
        print(f"  ✅ {dataset_type}: {info['rows']:,} rows ({info['size_mb']:.1f} MB)")
    
    if not datasets_found:
        print("  ❌ No major datasets found")
    
    return datasets_found

if __name__ == "__main__":
    print("🔍 Checking existing datasets...")