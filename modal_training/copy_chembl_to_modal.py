"""
Copy ChEMBL pickle file to Modal volume
"""

import modal
import pickle
import pandas as pd
from pathlib import Path

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy"
])

app = modal.App("copy-chembl-to-modal")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    mounts=[modal.Mount.from_local_dir("/app/backend/data", remote_path="/backend_data")],
    timeout=300
)
def copy_chembl_data():
    """Copy ChEMBL data from backend to Modal volume"""
    
    print("🔄 COPYING CHEMBL DATA TO MODAL VOLUME")
    print("=" * 50)
    
    try:
        # Check if file exists in mount
        backend_chembl_path = "/backend_data/chembl_ic50_data.pkl"
        
        if Path(backend_chembl_path).exists():
            # Load from backend
            with open(backend_chembl_path, 'rb') as f:
                chembl_df = pickle.load(f)
            
            print(f"✅ Loaded ChEMBL data: {len(chembl_df)} records")
            print(f"   Columns: {list(chembl_df.columns)}")
            print(f"   Sample:")
            print(chembl_df.head(2))
            
            # Save to Modal volume as CSV
            datasets_dir = Path("/vol/datasets")
            datasets_dir.mkdir(exist_ok=True)
            
            chembl_csv_path = datasets_dir / "real_chembl_data.csv"
            chembl_df.to_csv(chembl_csv_path, index=False)
            
            print(f"✅ Saved to Modal volume: {chembl_csv_path}")
            
            return {
                'status': 'success',
                'records': len(chembl_df),
                'unique_compounds': chembl_df['smiles'].nunique() if 'smiles' in chembl_df.columns else 0
            }
        
        else:
            print(f"❌ ChEMBL file not found: {backend_chembl_path}")
            return {'status': 'not_found'}
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    print("🔄 Copying ChEMBL data...")