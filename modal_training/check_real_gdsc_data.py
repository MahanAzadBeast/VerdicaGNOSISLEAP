"""
Check what real GDSC data we actually have available
"""

import modal
from pathlib import Path
import pandas as pd

app = modal.App("gdsc-data-checker")

image = modal.Image.debian_slim().pip_install([
    "pandas==2.1.0",
    "numpy==1.24.3",
    "openpyxl==3.1.0"
])

data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    volumes={"/vol": data_volume}
)
def check_available_gdsc_data():
    """Check what GDSC data is available on Modal volume"""
    
    print("📊 CHECKING AVAILABLE REAL GDSC DATA")
    print("=" * 50)
    
    vol_path = Path("/vol")
    
    # Find all GDSC-related files
    gdsc_files = []
    for pattern in ['*gdsc*', '*GDSC*', '*Drug*', '*Cell*', '*fitted*']:
        matches = list(vol_path.rglob(pattern))
        gdsc_files.extend(matches)
    
    # Remove duplicates and sort
    gdsc_files = sorted(set(gdsc_files))
    
    print(f"🎯 Found {len(gdsc_files)} GDSC-related files:")
    
    usable_files = []
    
    for file_path in gdsc_files:
        try:
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"\n📄 {file_path.name}")
                print(f"   Path: {file_path}")
                print(f"   Size: {size_mb:.1f} MB")
                
                # Try to read if it's a data file
                if file_path.suffix.lower() in ['.csv', '.xlsx', '.txt']:
                    try:
                        if file_path.suffix.lower() == '.csv':
                            df = pd.read_csv(file_path, nrows=5)
                        elif file_path.suffix.lower() == '.xlsx':
                            df = pd.read_excel(file_path, nrows=5)
                        elif file_path.suffix.lower() == '.txt':
                            df = pd.read_csv(file_path, sep='\t', nrows=5)
                            
                        print(f"   Rows: {len(df)} (sample)")
                        print(f"   Columns: {list(df.columns)[:5]}...")
                        
                        # Check for key columns
                        key_columns = {
                            'smiles': any('smiles' in col.lower() for col in df.columns),
                            'ic50': any('ic50' in col.lower() or 'ln_ic50' in col.lower() for col in df.columns),
                            'cell_line': any('cell' in col.lower() and 'line' in col.lower() for col in df.columns),
                            'drug': any('drug' in col.lower() or 'compound' in col.lower() for col in df.columns)
                        }
                        
                        if any(key_columns.values()):
                            print(f"   🎯 KEY DATA FOUND: {key_columns}")
                            usable_files.append({
                                'path': str(file_path),
                                'size_mb': size_mb,
                                'columns': key_columns,
                                'type': 'drug_sensitivity' if key_columns['ic50'] else 'metadata'
                            })
                            
                    except Exception as e:
                        print(f"   ⚠️ Could not read: {e}")
        except Exception as e:
            print(f"   ❌ Error accessing {file_path}: {e}")
    
    print(f"\n✅ USABLE GDSC DATA FILES: {len(usable_files)}")
    
    # Find the best drug sensitivity file
    drug_sens_files = [f for f in usable_files if f['columns']['ic50'] and f['columns']['smiles']]
    
    if drug_sens_files:
        best_file = max(drug_sens_files, key=lambda x: x['size_mb'])
        print(f"\n🏆 BEST DRUG SENSITIVITY FILE:")
        print(f"   Path: {best_file['path']}")
        print(f"   Size: {best_file['size_mb']:.1f} MB")
        
        # Read full file to check size
        try:
            if best_file['path'].endswith('.csv'):
                full_df = pd.read_csv(best_file['path'])
            else:
                full_df = pd.read_excel(best_file['path'])
                
            print(f"   Total records: {len(full_df):,}")
            print(f"   Unique SMILES: {full_df.iloc[:,0].nunique() if len(full_df.columns) > 0 else 'N/A'}")
            
            return {
                'best_file': best_file['path'],
                'total_records': len(full_df),
                'columns': list(full_df.columns),
                'sample_data': full_df.head().to_dict()
            }
            
        except Exception as e:
            print(f"   ⚠️ Could not analyze full file: {e}")
            
    return {
        'usable_files': usable_files,
        'status': 'found_data' if usable_files else 'no_usable_data'
    }

if __name__ == "__main__":
    with app.run():
        result = check_available_gdsc_data.remote()
        
        print("\n📊 FINAL RESULT:")
        print(f"Status: {result.get('status', 'unknown')}")
        
        if 'best_file' in result:
            print(f"✅ Best GDSC file: {result['best_file']}")
            print(f"📊 Records: {result['total_records']:,}")
            print(f"📋 Columns: {result['columns']}")