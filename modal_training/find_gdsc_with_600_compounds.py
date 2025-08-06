"""
Find the GDSC dataset with >600 compounds
Check different volumes and locations
"""

import modal
import pandas as pd
import os

app = modal.App("find-gdsc-600-compounds")

image = modal.Image.debian_slim().pip_install(["pandas==2.1.0"])

# Try different volume names that might contain GDSC data
volumes_to_check = [
    ("expanded-datasets", modal.Volume.from_name("expanded-datasets")),
    ("trained-models", modal.Volume.from_name("trained-models")),
]

# Also try if there are datasets with different names
@app.function(
    image=image,
    volumes={
        "/vol1": modal.Volume.from_name("expanded-datasets"),
        "/vol2": modal.Volume.from_name("trained-models"),
    },
    timeout=120
)
def find_gdsc_600_compounds():
    """Find GDSC dataset with >600 compounds"""
    
    print("🔍 SEARCHING FOR GDSC DATASET WITH >600 COMPOUNDS")
    print("=" * 60)
    
    volumes_to_search = ["/vol1", "/vol2"]
    
    best_dataset = None
    max_compounds = 0
    
    for vol_path in volumes_to_search:
        print(f"📁 Searching volume: {vol_path}")
        
        if not os.path.exists(vol_path):
            print(f"   ❌ Volume not accessible")
            continue
        
        try:
            files = os.listdir(vol_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            print(f"   📄 CSV files found: {len(csv_files)}")
            
            for csv_file in csv_files:
                file_path = os.path.join(vol_path, csv_file)
                print(f"   🔍 Checking: {csv_file}")
                
                try:
                    # Quick check of file
                    df = pd.read_csv(file_path, nrows=1000)  # Sample first 1000 rows
                    
                    # Look for SMILES columns
                    smiles_cols = [col for col in df.columns if 'smiles' in col.lower()]
                    
                    if smiles_cols:
                        # Load full file to count unique SMILES
                        df_full = pd.read_csv(file_path)
                        
                        for smiles_col in smiles_cols:
                            unique_smiles = df_full[smiles_col].nunique()
                            print(f"     Column '{smiles_col}': {unique_smiles} unique SMILES")
                            
                            if unique_smiles > max_compounds:
                                max_compounds = unique_smiles
                                best_dataset = {
                                    'file': csv_file,
                                    'volume': vol_path,
                                    'smiles_column': smiles_col,
                                    'unique_compounds': unique_smiles,
                                    'total_rows': len(df_full)
                                }
                                print(f"     🏆 NEW BEST: {unique_smiles} compounds!")
                    
                    else:
                        print(f"     ❌ No SMILES columns found")
                
                except Exception as e:
                    print(f"     ❌ Error reading {csv_file}: {e}")
                    continue
        
        except Exception as e:
            print(f"   ❌ Error accessing {vol_path}: {e}")
        
        print()
    
    print("🏆 SEARCH RESULTS")
    print("-" * 30)
    print(f"Maximum compounds found: {max_compounds}")
    
    if best_dataset:
        print(f"Best dataset:")
        print(f"  File: {best_dataset['file']}")
        print(f"  Volume: {best_dataset['volume']}")
        print(f"  SMILES column: {best_dataset['smiles_column']}")
        print(f"  Unique compounds: {best_dataset['unique_compounds']:,}")
        print(f"  Total rows: {best_dataset['total_rows']:,}")
        
        if best_dataset['unique_compounds'] >= 600:
            print("  ✅ MEETS >600 REQUIREMENT!")
        else:
            print("  ❌ Below 600 requirement")
    else:
        print("❌ No suitable GDSC dataset found")
    
    # Also try alternative approach - look for files that might have been missed
    print("\n🔍 LOOKING FOR OTHER POTENTIAL FILES")
    print("-" * 40)
    
    alternative_patterns = ['gdsc', 'drug', 'compound', 'cancer', 'ic50', 'sensitivity']
    
    for vol_path in volumes_to_search:
        if os.path.exists(vol_path):
            all_files = os.listdir(vol_path)
            for pattern in alternative_patterns:
                matching_files = [f for f in all_files if pattern.lower() in f.lower()]
                if matching_files:
                    print(f"Files containing '{pattern}': {matching_files}")
    
    return {
        'max_compounds_found': max_compounds,
        'best_dataset': best_dataset
    }

if __name__ == "__main__":
    with app.run():
        result = find_gdsc_600_compounds.remote()
        max_found = result.get('max_compounds_found', 0)
        print(f"✅ Search complete! Maximum compounds found: {max_found}")