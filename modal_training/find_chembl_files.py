"""
Find any ChEMBL-related files on Modal volume
"""

import modal
from pathlib import Path
import pandas as pd

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy"
])

app = modal.App("find-chembl-files")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    timeout=300
)
def find_chembl_files():
    """Find ChEMBL-related files"""
    
    print("🔍 SEARCHING FOR CHEMBL FILES")
    print("=" * 50)
    
    datasets_dir = Path("/vol/datasets")
    
    # Look for any files with chembl in name
    chembl_files = []
    
    for file_path in datasets_dir.rglob("*"):
        if file_path.is_file():
            filename = file_path.name.lower()
            if 'chembl' in filename or 'protein' in filename or 'ligand' in filename:
                chembl_files.append(file_path)
    
    print(f"📁 Found {len(chembl_files)} potential ChEMBL files:")
    
    for file_path in chembl_files:
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  📄 {file_path.name}: {size_mb:.1f} MB")
            
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, nrows=1)
                print(f"      → Columns: {len(df.columns)}")
                print(f"      → Sample columns: {list(df.columns[:5])}")
        except Exception as e:
            print(f"      → Error: {e}")
    
    # Also check for unified files
    print(f"\n🔍 Checking for unified extraction files...")
    
    unified_files = [
        "unified_protein_ligand_data.csv",
        "unified_extraction_metadata.json", 
        "chembl_protein_ligand_data.csv",
        "real_chembl_data.csv"
    ]
    
    for filename in unified_files:
        file_path = datasets_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename}: {size_mb:.1f} MB")
        else:
            print(f"  ❌ {filename}: Not found")

if __name__ == "__main__":
    print("🔍 Finding ChEMBL files...")