"""
Inspect GDSC dataset structure to understand columns
"""

import modal
from pathlib import Path
import pandas as pd

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy"
])

app = modal.App("inspect-gdsc-structure")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    timeout=300
)
def inspect_gdsc_dataset():
    """Inspect GDSC dataset structure"""
    
    print("🔍 INSPECTING GDSC DATASET STRUCTURE")
    print("=" * 60)
    
    datasets_dir = Path("/vol/datasets")
    
    gdsc_files = [
        "real_gdsc_training_data.csv",
        "gdsc_comprehensive_training_data.csv"
    ]
    
    for filename in gdsc_files:
        filepath = datasets_dir / filename
        
        if filepath.exists():
            print(f"\n📄 {filename}:")
            
            try:
                # Read first few rows
                df = pd.read_csv(filepath, nrows=10)
                
                print(f"   • Total columns: {len(df.columns)}")
                print(f"   • Columns: {list(df.columns)}")
                
                # Look for SMILES-like columns
                potential_smiles = []
                for col in df.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in ['smiles', 'structure', 'canonical', 'drug']):
                        potential_smiles.append(col)
                
                if potential_smiles:
                    print(f"   • Potential SMILES columns: {potential_smiles}")
                    
                    # Sample values
                    for col in potential_smiles[:3]:
                        sample_values = df[col].dropna().head(3).tolist()
                        print(f"     - {col} samples: {sample_values}")
                
                # Look for IC50-like columns
                potential_ic50 = []
                for col in df.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in ['ic50', 'ln_ic50', 'log_ic50', 'auc', 'sensitivity']):
                        potential_ic50.append(col)
                
                if potential_ic50:
                    print(f"   • Potential IC50/sensitivity columns: {potential_ic50}")
                    
                    # Sample values
                    for col in potential_ic50[:3]:
                        sample_values = df[col].dropna().head(3).tolist()
                        print(f"     - {col} samples: {sample_values}")
                
                # Sample first row
                print(f"   • Sample row:")
                for i, (col, val) in enumerate(df.iloc[0].items()):
                    print(f"     {col}: {val}")
                    if i >= 10:  # Limit output
                        print(f"     ... and {len(df.columns)-11} more columns")
                        break
                        
            except Exception as e:
                print(f"   ❌ Error reading {filename}: {e}")

if __name__ == "__main__":
    print("🔍 Inspecting GDSC structure...")