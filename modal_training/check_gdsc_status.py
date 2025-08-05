"""
Check GDSC dataset status and size
"""

import modal
from pathlib import Path
import pandas as pd

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy"
])

app = modal.App("check-gdsc-status")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    timeout=300
)
def check_gdsc_datasets():
    """Check GDSC dataset status"""
    
    print("🔍 CHECKING GDSC DATASET STATUS")
    print("=" * 50)
    
    datasets_dir = Path("/vol/datasets")
    
    # Look for GDSC files
    gdsc_files = [
        "real_gdsc_training_data.csv",
        "gdsc_comprehensive_training_data.csv", 
        "real_gdsc_gdsc1_sensitivity.csv",
        "real_gdsc_gdsc2_sensitivity.csv",
    ]
    
    gdsc_status = {}
    
    for filename in gdsc_files:
        filepath = datasets_dir / filename
        
        if filepath.exists():
            try:
                size_mb = filepath.stat().st_size / (1024 * 1024)
                
                # Read just a sample to get structure
                df_sample = pd.read_csv(filepath, nrows=5)
                
                # Count total rows (more efficient)
                total_rows = sum(1 for line in open(filepath)) - 1  # Subtract header
                
                gdsc_status[filename] = {
                    'exists': True,
                    'size_mb': round(size_mb, 1),
                    'total_rows': total_rows,
                    'columns': len(df_sample.columns),
                    'sample_columns': list(df_sample.columns[:10])
                }
                
                print(f"✅ {filename}:")
                print(f"   • Size: {size_mb:.1f} MB")
                print(f"   • Rows: {total_rows:,}")
                print(f"   • Columns: {len(df_sample.columns)}")
                print(f"   • Sample cols: {', '.join(df_sample.columns[:5])}...")
                
            except Exception as e:
                gdsc_status[filename] = {
                    'exists': True,
                    'error': str(e)
                }
                print(f"❌ {filename}: Error - {e}")
        else:
            gdsc_status[filename] = {'exists': False}
            print(f"❌ {filename}: Not found")
        
        print()
    
    # Determine best GDSC dataset for training
    best_dataset = None
    max_rows = 0
    
    for filename, status in gdsc_status.items():
        if status.get('exists') and 'total_rows' in status:
            if status['total_rows'] > max_rows:
                max_rows = status['total_rows']
                best_dataset = filename
    
    print(f"🎯 BEST GDSC DATASET FOR TRAINING:")
    if best_dataset:
        print(f"   • File: {best_dataset}")
        print(f"   • Rows: {gdsc_status[best_dataset]['total_rows']:,}")
        print(f"   • Size: {gdsc_status[best_dataset]['size_mb']} MB")
        print(f"   • ✅ READY FOR TRAINING")
    else:
        print(f"   • ❌ NO SUITABLE GDSC DATASET FOUND")
    
    return {
        'gdsc_files_status': gdsc_status,
        'best_dataset': best_dataset,
        'ready_for_training': best_dataset is not None
    }

if __name__ == "__main__":
    print("🔍 Checking GDSC status...")