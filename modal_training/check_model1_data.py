"""
Check Model 1 Dataset Structure
Quick script to see what columns are available in the Model 1 dataset
"""

import modal
import pandas as pd
from pathlib import Path

image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas")
app = modal.App("check-model1-data")

datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": datasets_volume}
)
def check_model1_dataset():
    """Check what columns are in the Model 1 dataset"""
    
    print("🔍 CHECKING MODEL 1 DATASET STRUCTURE")
    print("=" * 50)
    
    model1_path = Path("/data/gnosis_model1_binding_training.csv")
    
    if not model1_path.exists():
        print(f"❌ File not found: {model1_path}")
        return
    
    df = pd.read_csv(model1_path)
    
    print(f"✅ Dataset loaded: {len(df):,} records")
    print(f"📊 Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"📊 Shape: {df.shape}")
    
    # Show first few rows
    print("\n📋 First 3 rows:")
    print(df.head(3))
    
    # Check for target-related columns
    target_cols = [col for col in df.columns if 'target' in col.lower() or 'protein' in col.lower()]
    if target_cols:
        print(f"\n🎯 Target/Protein related columns: {target_cols}")
        for col in target_cols:
            print(f"  {col} unique values: {df[col].nunique()}")
            print(f"  {col} sample values: {df[col].unique()[:5]}")
    
    # Check for activity-related columns
    activity_cols = [col for col in df.columns if 'activity' in col.lower() or 'ic50' in col.lower() or 'ki' in col.lower()]
    if activity_cols:
        print(f"\n🧪 Activity related columns: {activity_cols}")
        for col in activity_cols:
            if df[col].dtype in ['float64', 'int64']:
                print(f"  {col}: {df[col].min():.2f} - {df[col].max():.2f} (mean: {df[col].mean():.2f})")
            else:
                print(f"  {col} unique values: {df[col].nunique()}")
    
    return {
        'total_records': len(df),
        'columns': list(df.columns),
        'shape': df.shape
    }

if __name__ == "__main__":
    with app.run():
        result = check_model1_dataset.remote()
        print("Dataset check completed:", result)