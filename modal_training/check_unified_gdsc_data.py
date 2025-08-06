"""
Check what's actually in the unified GDSC database
"""

import modal
import pandas as pd
from pathlib import Path
import os

app = modal.App("check-unified-gdsc")

image = modal.Image.debian_slim().pip_install([
    "pandas==2.1.0", 
    "numpy==1.24.3"
])

data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    volumes={"/vol": data_volume},
    timeout=300
)
def check_unified_gdsc():
    """Check unified GDSC data structure"""
    
    print("🔍 CHECKING UNIFIED GDSC DATABASE")
    print("=" * 60)
    
    # Check what files exist
    vol_path = Path("/vol")
    if vol_path.exists():
        print(f"📁 Volume contents:")
        csv_files = list(vol_path.glob("*.csv"))
        for file in sorted(csv_files):
            file_size = file.stat().st_size / (1024 * 1024)  # MB
            print(f"   📄 {file.name} ({file_size:.1f}MB)")
        
        # Focus on the comprehensive GDSC file
        gdsc_file = vol_path / "gdsc_comprehensive_training_data.csv"
        if gdsc_file.exists():
            print(f"\n📊 ANALYZING: {gdsc_file.name}")
            
            # Load and examine
            df = pd.read_csv(gdsc_file)
            print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Show column names
            print(f"   Columns ({len(df.columns)}):")
            for i, col in enumerate(df.columns):
                print(f"     {i+1:2d}. {col}")
            
            # Check for key columns
            key_columns = ['SMILES', 'smiles', 'pIC50', 'pic50', 'IC50', 'ic50', 'CELL_LINE_NAME', 'cell_line']
            found_columns = []
            for col in key_columns:
                if col in df.columns:
                    found_columns.append(col)
                    print(f"   ✅ Found: {col}")
            
            print(f"   📋 Found key columns: {found_columns}")
            
            # Sample data
            if len(found_columns) >= 2:
                print(f"\n📋 SAMPLE DATA (first 5 rows):")
                sample_cols = found_columns[:5]  # Show first 5 found columns
                sample_data = df[sample_cols].head()
                for idx, row in sample_data.iterrows():
                    print(f"   Row {idx+1}: {dict(row)}")
            
            # Data quality checks
            print(f"\n🔍 DATA QUALITY CHECKS:")
            for col in found_columns[:3]:  # Check first 3 found columns
                null_count = df[col].isnull().sum()
                null_pct = 100 * null_count / len(df)
                unique_count = df[col].nunique()
                print(f"   {col}: {null_count:,} nulls ({null_pct:.1f}%), {unique_count:,} unique")
                
                # Show sample values
                sample_vals = df[col].dropna().head(3).tolist()
                print(f"     Sample values: {sample_vals}")
        
        else:
            print(f"❌ gdsc_comprehensive_training_data.csv not found!")
    
    else:
        print("❌ Volume path /vol not found!")
    
    return {"status": "complete"}

if __name__ == "__main__":
    with app.run():
        result = check_unified_gdsc.remote()
        print("✅ Check complete!")