#!/usr/bin/env python3
"""
Investigate Dataset Differences and Preprocessing Issues
"""

import modal
import pandas as pd
import numpy as np
from pathlib import Path

app = modal.App("dataset-investigation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "pandas>=2.0.0",
        "numpy>=1.24.0"
    ])
)

datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    timeout=300
)
def investigate_dataset_preprocessing():
    """Investigate potential dataset preprocessing issues"""
    
    print("🔍 INVESTIGATING DATASET AND PREPROCESSING ISSUES")
    print("=" * 70)
    
    # Load the original dataset
    dataset_path = Path("/vol/datasets/oncoprotein_multitask_dataset.csv")
    if not dataset_path.exists():
        dataset_path = Path("/vol/datasets/oncoprotein_multitask_dataset.parquet")
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
    
    print(f"📊 Original dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Check the targets used by both models
    FOCUSED_TARGETS = [
        'EGFR', 'HER2', 'VEGFR2', 'BRAF', 'MET', 
        'CDK4', 'CDK6', 'ALK', 'MDM2', 'PI3KCA'
    ]
    
    print(f"\n🎯 Analyzing Focused Targets: {FOCUSED_TARGETS}")
    
    # Check for Imatinib specifically (the test molecule)
    imatinib_smiles = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
    imatinib_rows = df[df['canonical_smiles'] == imatinib_smiles]
    
    print(f"\n🧪 IMATINIB ANALYSIS:")
    print(f"   SMILES: {imatinib_smiles}")
    print(f"   Found in dataset: {len(imatinib_rows)} rows")
    
    if len(imatinib_rows) > 0:
        print(f"   📊 Imatinib target values in dataset:")
        for target in FOCUSED_TARGETS:
            if target in df.columns:
                value = imatinib_rows[target].iloc[0]
                if pd.notna(value):
                    print(f"      {target}: {value:.3f}")
                else:
                    print(f"      {target}: NaN")
    
    # Analyze value distributions for each target
    print(f"\n📊 VALUE DISTRIBUTION ANALYSIS:")
    print(f"{'Target':<10} {'Count':<8} {'Min':<10} {'Max':<10} {'Mean':<10} {'Median':<10}")
    print("-" * 70)
    
    for target in FOCUSED_TARGETS:
        if target in df.columns:
            values = df[target].dropna()
            if len(values) > 0:
                print(f"{target:<10} {len(values):<8} {values.min():<10.3f} {values.max():<10.3f} {values.mean():<10.3f} {values.median():<10.3f}")
            else:
                print(f"{target:<10} {'0':<8} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    # Check for unit issues - IC50 values should typically be in nM or μM range
    print(f"\n🔬 UNIT ANALYSIS (typical IC50 ranges):")
    print("   Expected ranges:")
    print("   - Very active compounds: 1-100 nM (0.001-0.1 μM)")  
    print("   - Active compounds: 100-1000 nM (0.1-1 μM)")
    print("   - Moderate compounds: 1-10 μM")
    print("   - Weak compounds: 10-100 μM")
    print("   - Inactive compounds: >100 μM")
    
    # Sample some data points to understand the distribution
    print(f"\n📋 RANDOM SAMPLE ANALYSIS:")
    sample_df = df.sample(n=min(10, len(df)))
    
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        print(f"\n   Sample {i+1} (Index {idx}):")
        print(f"      SMILES: {row['canonical_smiles'][:50]}...")
        for target in FOCUSED_TARGETS[:3]:  # Show first 3 targets
            if target in df.columns and pd.notna(row[target]):
                print(f"      {target}: {row[target]:.3f}")
    
    # Check for potential log-transformed data
    print(f"\n🧮 CHECKING FOR LOG-TRANSFORMATION ISSUES:")
    for target in FOCUSED_TARGETS[:3]:  # Check first 3 targets
        if target in df.columns:
            values = df[target].dropna()
            if len(values) > 0:
                # Check if values might be pIC50 (typically 4-10 range)
                if values.mean() > 3 and values.mean() < 12:
                    print(f"   {target}: Mean={values.mean():.2f} - MIGHT BE pIC50 (log scale)")
                    # Convert pIC50 to IC50 nM
                    ic50_nm = 10**(9 - values.mean())
                    print(f"      If pIC50, equivalent IC50 would be ~{ic50_nm:.1f} nM")
                else:
                    print(f"   {target}: Mean={values.mean():.2f} - Likely IC50 in nM or μM")
    
    return {
        "dataset_shape": df.shape,
        "imatinib_found": len(imatinib_rows) > 0,
        "target_stats": {target: {
            "count": len(df[target].dropna()) if target in df.columns else 0,
            "mean": float(df[target].dropna().mean()) if target in df.columns and len(df[target].dropna()) > 0 else None,
            "min": float(df[target].dropna().min()) if target in df.columns and len(df[target].dropna()) > 0 else None,
            "max": float(df[target].dropna().max()) if target in df.columns and len(df[target].dropna()) > 0 else None
        } for target in FOCUSED_TARGETS}
    }

if __name__ == "__main__":
    print("🔍 Starting dataset investigation...")
    
    with app.run():
        result = investigate_dataset_preprocessing.remote()
        
        print(f"\n📊 INVESTIGATION SUMMARY:")
        print(f"Dataset shape: {result['dataset_shape']}")
        print(f"Imatinib found: {result['imatinib_found']}")
        
        print(f"\n🎯 HYPOTHESIS:")
        print("The massive variance suggests one of these issues:")
        print("1. ChemBERTa trained on IC50 values (μM/nM)")
        print("2. Chemprop trained on pIC50 values (log scale)")
        print("3. Different unit conventions between training")
        print("4. Data preprocessing error in one model")
        
        print(f"\n🔧 RECOMMENDED SOLUTION:")
        print("Check training logs to see what scale/units each model used")
        print("Ensure both models use same data preprocessing and units")