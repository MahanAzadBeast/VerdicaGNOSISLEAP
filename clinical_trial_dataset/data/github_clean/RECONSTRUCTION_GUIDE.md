# Dataset Reconstruction Guide

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
