"""
Check what data is actually available on Modal volumes
"""

import modal
from pathlib import Path
import pandas as pd

app = modal.App("data-checker")

image = modal.Image.debian_slim().pip_install([
    "pandas==2.1.0"
])

data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    volumes={"/vol": data_volume}
)
def check_available_data():
    """Check what data files are actually available"""
    
    print("🔍 CHECKING AVAILABLE DATA FILES")
    print("=" * 50)
    
    # Check root volume structure
    vol_path = Path("/vol")
    if vol_path.exists():
        print(f"📁 Volume root exists: {vol_path}")
        
        # List all subdirectories and files
        for item in vol_path.iterdir():
            if item.is_dir():
                print(f"📁 Directory: {item.name}")
                # List files in subdirectory
                try:
                    subfiles = list(item.glob("*.csv"))[:10]  # Limit to first 10 CSV files
                    if subfiles:
                        print(f"   CSV files: {[f.name for f in subfiles]}")
                except Exception as e:
                    print(f"   Error reading subdirectory: {e}")
            else:
                print(f"📄 File: {item.name}")
        
        # Check for specific GDSC files
        print("\n🔍 CHECKING FOR GDSC FILES")
        print("-" * 30)
        
        gdsc_patterns = [
            "gdsc*",
            "*gdsc*",
            "GDSC*",
            "*GDSC*",
            "*sensitivity*",
            "*drug*",
            "*compound*"
        ]
        
        found_files = []
        for pattern in gdsc_patterns:
            matches = list(vol_path.rglob(pattern))
            if matches:
                print(f"Pattern '{pattern}': {len(matches)} files")
                for match in matches[:5]:  # Show first 5 matches
                    print(f"   {match}")
                    found_files.extend(matches[:5])
        
        # Try to read sample data from any found CSV files
        print("\n📊 SAMPLE DATA FROM FOUND FILES")
        print("-" * 35)
        
        csv_files = list(vol_path.rglob("*.csv"))[:5]  # Check first 5 CSV files
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                print(f"\n📄 {csv_file.name}:")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)[:10]}")  # First 10 columns
                
                # Check for SMILES columns
                smiles_cols = [col for col in df.columns if 'smiles' in col.lower()]
                if smiles_cols:
                    print(f"   SMILES columns: {smiles_cols}")
                    sample_smiles = df[smiles_cols[0]].dropna().head(2).tolist()
                    print(f"   Sample SMILES: {sample_smiles}")
                    
            except Exception as e:
                print(f"   Error reading {csv_file.name}: {e}")
    
    else:
        print("❌ Volume root not found!")
        
    return "Data check complete"

if __name__ == "__main__":
    with app.run():
        result = check_available_data.remote()
        print(result)