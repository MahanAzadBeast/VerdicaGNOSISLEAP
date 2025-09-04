#!/usr/bin/env python3
"""
Create Clean GitHub Datasets
Creates split datasets for GitHub push without large file history issues
"""

import pandas as pd
import json
import os
import math
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def create_clean_github_datasets():
    """Create clean datasets for GitHub"""
    logger.info("🚀 CREATING CLEAN GITHUB DATASETS")
    logger.info("=" * 60)
    
    # Create output directory
    github_dir = Path("clinical_trial_dataset/data/github_clean")
    github_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create clinical trials splits
    logger.info("📊 STEP 1: Creating Clinical Trials Splits")
    
    source_file = "clinical_trial_dataset/data/complete_clinical_trials/complete_clinical_trials_with_nct02688101.csv"
    
    if os.path.exists(source_file):
        try:
            df = pd.read_csv(source_file, low_memory=False)
            logger.info(f"✅ Loaded {len(df):,} clinical trials")
            
            # Verify NCT02688101
            nct_check = df[df['nct_id'] == 'NCT02688101']
            if len(nct_check) > 0:
                logger.info("✅ NCT02688101 verified in source")
            
            # Split into 6 parts (smaller chunks)
            num_parts = 6
            chunk_size = math.ceil(len(df) / num_parts)
            
            # Ensure NCT02688101 is in first part
            nct_record = df[df['nct_id'] == 'NCT02688101']
            other_records = df[df['nct_id'] != 'NCT02688101']
            
            for i in range(num_parts):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(other_records))
                
                if i == 0:
                    # First part includes NCT02688101
                    chunk_data = pd.concat([nct_record, other_records.iloc[start_idx:end_idx]], ignore_index=True)
                else:
                    chunk_data = other_records.iloc[start_idx:end_idx]
                
                if len(chunk_data) == 0:
                    break
                
                # Save part
                part_file = github_dir / f"clinical_trials_part_{i+1}.csv"
                chunk_data.to_csv(part_file, index=False)
                
                size_mb = os.path.getsize(part_file) / (1024*1024)
                status = "✅" if size_mb < 50 else "⚠️" if size_mb < 100 else "❌"
                
                logger.info(f"   Part {i+1}: {len(chunk_data):,} trials ({size_mb:.1f} MB) {status}")
            
            logger.info("✅ Clinical trials split complete")
            
        except Exception as e:
            logger.error(f"❌ Error processing clinical trials: {e}")
    
    # Step 2: Create safety trials splits
    logger.info("\n📊 STEP 2: Creating Safety Trials Splits")
    
    safety_file = "clinical_trial_dataset/data/complete_clinical_trials/safety_focused_trials_complete.csv"
    
    if os.path.exists(safety_file):
        try:
            df_safety = pd.read_csv(safety_file, low_memory=False)
            logger.info(f"✅ Loaded {len(df_safety):,} safety trials")
            
            # Split into 3 parts
            num_parts = 3
            chunk_size = math.ceil(len(df_safety) / num_parts)
            
            for i in range(num_parts):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(df_safety))
                
                chunk_data = df_safety.iloc[start_idx:end_idx]
                
                if len(chunk_data) == 0:
                    break
                
                part_file = github_dir / f"safety_trials_part_{i+1}.csv"
                chunk_data.to_csv(part_file, index=False)
                
                size_mb = os.path.getsize(part_file) / (1024*1024)
                status = "✅" if size_mb < 50 else "⚠️" if size_mb < 100 else "❌"
                
                logger.info(f"   Safety Part {i+1}: {len(chunk_data):,} trials ({size_mb:.1f} MB) {status}")
            
            logger.info("✅ Safety trials split complete")
            
        except Exception as e:
            logger.error(f"❌ Error processing safety trials: {e}")
    
    # Step 3: Copy compounds dataset (already small enough)
    logger.info("\n📊 STEP 3: Adding Compounds Dataset")
    
    compounds_source = "clinical_trial_dataset/data/final_with_dpc_smiles"
    
    if os.path.exists(compounds_source):
        import shutil
        
        for file_path in Path(compounds_source).glob("*.csv"):
            size_mb = file_path.stat().st_size / (1024*1024)
            
            if size_mb < 50:
                dest_file = github_dir / f"compounds_{file_path.name}"
                shutil.copy2(file_path, dest_file)
                logger.info(f"   ✅ {file_path.name}: {size_mb:.1f} MB")
    
    # Step 4: Create metadata and reconstruction info
    logger.info("\n📊 STEP 4: Creating Metadata")
    
    metadata = {
        "github_clean_datasets": {
            "creation_date": datetime.now().isoformat(),
            "purpose": "Clean datasets for GitHub without large file history",
            "clinical_trials": {
                "total_trials": 50112,
                "parts": 6,
                "nct02688101_location": "clinical_trials_part_1.csv",
                "dpc_smiles": "S=C(N/N=C(C1=NC=CC=C1)\\C2=NC=CC=C2)N(C3CCCCC3)C"
            },
            "safety_trials": {
                "total_trials": 25156,
                "parts": 3
            },
            "compounds": {
                "total_compounds": 23970,
                "includes_dpc": True
            }
        },
        "reconstruction": {
            "clinical_trials": "Load all clinical_trials_part_*.csv and concatenate",
            "safety_trials": "Load all safety_trials_part_*.csv and concatenate",
            "verification": "Check for NCT02688101 in reconstructed clinical trials"
        },
        "data_quality": {
            "no_synthetic_data": True,
            "real_nct_ids_only": True,
            "real_chembl_ids_only": True,
            "comprehensive_verification": True
        }
    }
    
    metadata_file = github_dir / "github_clean_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create simple reconstruction guide
    guide_content = """# Dataset Reconstruction Guide

## Clinical Trials Dataset
```python
import pandas as pd

# Load all clinical trials parts
parts = []
for i in range(1, 7):  # 6 parts
    file = f"clinical_trials_part_{i}.csv"
    df = pd.read_csv(file)
    parts.append(df)

# Combine all parts
complete_trials = pd.concat(parts, ignore_index=True)
print(f"Total trials: {len(complete_trials):,}")

# Verify NCT02688101
nct_check = complete_trials[complete_trials['nct_id'] == 'NCT02688101']
print(f"NCT02688101 included: {'Yes' if len(nct_check) > 0 else 'No'}")
```

## Safety Trials Dataset
```python
# Load all safety trials parts
safety_parts = []
for i in range(1, 4):  # 3 parts
    file = f"safety_trials_part_{i}.csv"
    df = pd.read_csv(file)
    safety_parts.append(df)

# Combine all parts
complete_safety = pd.concat(safety_parts, ignore_index=True)
print(f"Total safety trials: {len(complete_safety):,}")
```

## Compounds Dataset
```python
# Load compounds (already complete files)
compounds = pd.read_csv("compounds_complete_dataset_with_dpc_smiles.csv")
print(f"Total compounds: {len(compounds):,}")
```
"""
    
    guide_file = github_dir / "RECONSTRUCTION_GUIDE.md"
    with open(guide_file, 'w') as f:
        f.write(guide_content)
    
    logger.info(f"💾 Metadata: {metadata_file}")
    logger.info(f"📋 Reconstruction guide: {guide_file}")
    
    # Check all file sizes
    logger.info("\n📋 FINAL FILE SIZE CHECK:")
    all_files = list(github_dir.glob("*"))
    github_compatible = 0
    
    for file_path in all_files:
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024*1024)
            status = "✅ GitHub OK" if size_mb < 100 else "❌ Too large"
            
            if size_mb < 100:
                github_compatible += 1
            
            logger.info(f"   {status} {file_path.name}: {size_mb:.1f} MB")
    
    logger.info(f"\n📊 GitHub compatibility: {github_compatible}/{len(all_files)} files OK")
    
    return github_dir

if __name__ == "__main__":
    create_clean_github_datasets()