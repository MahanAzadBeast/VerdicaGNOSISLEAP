"""
Diagnose Real GDSC Data Structure
Check what's actually in the Modal files
"""

import modal
import pandas as pd
import numpy as np
import os

app = modal.App("diagnose-real-gdsc-data")

image = modal.Image.debian_slim().pip_install([
    "pandas==2.1.0", 
    "numpy==1.24.3"
])

data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    volumes={"/vol": data_volume},
    timeout=600
)
def diagnose_real_gdsc_data():
    """Diagnose the real GDSC data structure"""
    
    print("🔍 DIAGNOSING REAL GDSC DATA")
    print("=" * 60)
    
    # 1. List all files in volume
    vol_path = "/vol"
    if os.path.exists(vol_path):
        files = os.listdir(vol_path)
        print(f"📁 Files in expanded-datasets volume:")
        for file in sorted(files):
            file_path = os.path.join(vol_path, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"   📄 {file} ({size:,} bytes, {size/1024:.1f} KB)")
        print()
    else:
        print("❌ Volume path /vol not found!")
        return {"error": "Volume not found"}
    
    # 2. Read and analyze each CSV file
    csv_files = [f for f in files if f.endswith('.csv')]
    
    for csv_file in csv_files:
        file_path = os.path.join(vol_path, csv_file)
        print(f"📊 ANALYZING: {csv_file}")
        print("-" * 40)
        
        try:
            # Load the file
            df = pd.read_csv(file_path)
            print(f"✅ Successfully loaded: {len(df):,} rows × {len(df.columns)} columns")
            
            # Show basic info
            print(f"📋 Columns ({len(df.columns)}):")
            for i, col in enumerate(df.columns):
                print(f"   {i+1:2d}. {col}")
            print()
            
            # Show sample data
            print("📋 First 5 rows:")
            print(df.head())
            print()
            
            # Data types
            print("📋 Data types:")
            print(df.dtypes)
            print()
            
            # Check for SMILES-like columns
            smiles_candidates = []
            for col in df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['smiles', 'canonical', 'structure']):
                    smiles_candidates.append(col)
                    print(f"🧬 SMILES candidate: {col}")
                    if len(df[col].dropna()) > 0:
                        sample_values = df[col].dropna().head(3).tolist()
                        print(f"   Sample values: {sample_values}")
            print()
            
            # Check for activity/IC50 columns
            activity_candidates = []
            for col in df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['ic50', 'pic50', 'ln_ic50', 'activity', 'response']):
                    activity_candidates.append(col)
                    print(f"🎯 Activity candidate: {col}")
                    if len(df[col].dropna()) > 0:
                        stats = df[col].describe()
                        print(f"   Stats: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
            print()
            
            # Check for cell line columns
            cell_line_candidates = []
            for col in df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['cell', 'line', 'cosmic', 'sample']):
                    cell_line_candidates.append(col)
                    print(f"🔬 Cell line candidate: {col}")
                    if len(df[col].dropna()) > 0:
                        unique_count = df[col].nunique()
                        sample_values = df[col].unique()[:5]
                        print(f"   Unique values: {unique_count}")
                        print(f"   Sample values: {list(sample_values)}")
            print()
            
            # Data quality summary
            print("📊 DATA QUALITY SUMMARY:")
            print(f"   Total records: {len(df):,}")
            print(f"   SMILES candidates: {len(smiles_candidates)} ({smiles_candidates})")
            print(f"   Activity candidates: {len(activity_candidates)} ({activity_candidates})")
            print(f"   Cell line candidates: {len(cell_line_candidates)} ({cell_line_candidates})")
            
            # Count non-null values for key candidates
            if smiles_candidates:
                for col in smiles_candidates:
                    non_null = df[col].count()
                    print(f"   {col} non-null: {non_null:,} ({100*non_null/len(df):.1f}%)")
            
            if activity_candidates:
                for col in activity_candidates:
                    non_null = df[col].count()
                    print(f"   {col} non-null: {non_null:,} ({100*non_null/len(df):.1f}%)")
            
            if cell_line_candidates:
                for col in cell_line_candidates:
                    non_null = df[col].count()
                    print(f"   {col} non-null: {non_null:,} ({100*non_null/len(df):.1f}%)")
            
            print("=" * 60)
            print()
            
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {e}")
            print("=" * 60)
            print()
    
    # 3. Try to read text files for context
    txt_files = [f for f in files if f.endswith('.txt') or f.endswith('.md')]
    
    for txt_file in txt_files:
        file_path = os.path.join(vol_path, txt_file)
        print(f"📄 READING: {txt_file}")
        print("-" * 40)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content[:1000])  # First 1000 characters
                if len(content) > 1000:
                    print("... (truncated)")
        except Exception as e:
            print(f"❌ Error reading {txt_file}: {e}")
        
        print("=" * 60)
        print()
    
    return {"status": "complete"}

if __name__ == "__main__":
    with app.run():
        result = diagnose_real_gdsc_data.remote()
        print("✅ Diagnosis complete!")