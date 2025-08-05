"""
Check Model 2 Dataset Structure
Quick script to see what columns are available in the Model 2 dataset
"""

import modal
import pandas as pd
from pathlib import Path

image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas")
app = modal.App("check-model2-data")

datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": datasets_volume}
)
def check_model2_dataset():
    """Check what columns are in the Model 2 dataset"""
    
    print("🔍 CHECKING MODEL 2 DATASET STRUCTURE")
    print("=" * 50)
    
    gdsc_path = Path("/data/gnosis_model2_cytotox_training.csv")
    
    if not gdsc_path.exists():
        print(f"❌ File not found: {gdsc_path}")
        return
    
    df = pd.read_csv(gdsc_path)
    
    print(f"✅ Dataset loaded: {len(df):,} records")
    print(f"📊 Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"📊 Shape: {df.shape}")
    
    # Show first few rows
    print("\n📋 First 3 rows:")
    print(df.head(3))
    
    # Show column data types
    print(f"\n📋 Column types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # Check for cancer-related columns
    cancer_cols = [col for col in df.columns if 'cancer' in col.lower() or 'cell' in col.lower() or 'type' in col.lower()]
    if cancer_cols:
        print(f"\n🧬 Cancer/Cell related columns: {cancer_cols}")
        for col in cancer_cols:
            print(f"  {col} unique values: {df[col].nunique()}")
            print(f"  {col} sample values: {df[col].unique()[:5]}")
    
    return {
        'total_records': len(df),
        'columns': list(df.columns),
        'shape': df.shape
    }

if __name__ == "__main__":
    with app.run():
        result = check_model2_dataset.remote()
        print("Dataset check completed:", result)