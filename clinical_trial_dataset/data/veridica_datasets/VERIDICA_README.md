# 🧩 Veridica Drug Data Pipeline (ChEMBL-centric)

## 🎯 Overview

Extensible data ingestion + enrichment pipeline that expands ChEMBL-based datasets with real APIs and open datasets while enforcing leakage-safe splits for approval prediction models.

## 🗂 Project Structure

```
veridica-data/
  README.md                    # This file
  pyproject.toml              # Dependencies and package config
  .env.example                # Environment variables template
  src/
    veridica_data/
      __init__.py
      utils/                  # Core utilities
        io.py                 # I/O operations
        chem.py               # Chemical structure operations
        mapping.py            # Name-to-structure mapping
        timeguard.py          # Temporal leakage protection
        qc.py                 # Quality control
        rate_limit.py         # API rate limiting
      schemas/                # Data schemas
        master_schema.py      # Master compound table schema
        tox_schema.py         # Toxicity data schema
        clinical_schema.py    # Clinical trial data schema
        ae_schema.py          # Adverse events schema
      connectors/             # Data source connectors
        chembl_fetch.py       # ChEMBL API connector
        pubchem_fetch.py      # PubChem API connector
        unichem_fetch.py      # UniChem cross-reference
        clintrials_fetch.py   # ClinicalTrials.gov API
        fda_openfda_fetch.py  # openFDA API connector
        tox21_fetch.py        # Tox21 data loader
        gdsc_depmap_fetch.py  # GDSC/DepMap connector
        sider_loader.py       # SIDER adverse events
      transforms/             # Data transformation
        standardize_smiles.py # SMILES standardization
        joiner.py            # Data joining operations
        labelers.py          # Label generation
        impute_descriptors.py # Molecular descriptor imputation
      pipelines/              # Data processing pipelines
        build_master.py       # Master compound table
        build_tox.py          # Toxicity data pipeline
        build_clinical.py     # Clinical data pipeline
        build_ae.py           # Adverse events pipeline
        merge_all.py          # Final merge pipeline
  tests/                      # Test suite
    test_mapping.py
    test_timeguard.py
    test_joiner.py
  Makefile                    # Build automation
```

## 🚀 Quick Start

1. **Setup environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   make setup
   ```

2. **Build master dataset:**
   ```bash
   make fetch-chembl
   ```

3. **Enrich with clinical data:**
   ```bash
   make fetch-clinical
   make fetch-tox
   make fetch-ae
   ```

4. **Create final merged dataset:**
   ```bash
   make merge
   ```

## 🔐 Environment Variables

Create `.env` file with:
- `OPENFDA_API_KEY`: openFDA API key
- `CLINICALTRIALS_BASE`: ClinicalTrials.gov API base URL
- `OPENFDA_BASE`: openFDA API base URL
- `HTTP_TIMEOUT`: Request timeout in seconds
- `MAX_QPS`: Maximum queries per second for rate limiting

## 🧠 Key Design Principles

### 1. Master Index (One Row Per Compound)
- **Keys**: `chembl_id`, `canonical_smiles`, `inchi`, `inchikey`
- **Descriptors**: Complete molecular property block
- **Temporal**: `first_seen_date`, `source_first_seen`, `created_at`

### 2. Sidecar Tables (Sparse Allowed)
- **clinical.csv**: Clinical trial data with temporal information
- **tox.csv**: Toxicity surrogates from multiple sources
- **ae.csv**: Adverse events (labels only, not features)

### 3. Leakage Guards
- Temporal filtering to prevent future information leakage
- Separate AE data from approval features
- Pre-approval knowledge only for training

## 📊 Data Sources

### Real Pharmaceutical Databases
- **ChEMBL**: Primary compound and bioactivity source
- **PubChem**: Chemical structure validation
- **UniChem**: Cross-reference mapping
- **ClinicalTrials.gov**: Real clinical trial data
- **openFDA**: FDA approval information
- **Tox21**: Toxicity screening data
- **SIDER**: Adverse event data

### No Synthetic Data
- ✅ All data from authentic pharmaceutical sources
- ❌ No AI-generated or synthetic compounds
- ❌ No fake clinical data or outcomes
- ✅ Real molecular structures and clinical outcomes only

## 🎯 ML Training Outputs

### Approval Prediction Model
- **Features**: Molecular descriptors, clinical trial data (pre-approval)
- **Targets**: Binary approval status, clinical phase progression
- **Temporal Guard**: No future information leakage

### Toxicity Prediction Models
- **Features**: Molecular structures and properties
- **Targets**: hERG, AMES, DILI, CYP interactions
- **Sources**: Real experimental data from ChEMBL and Tox21

### Adverse Events Prediction
- **Features**: Molecular structures
- **Targets**: MedDRA adverse event categories
- **Sources**: Real post-market surveillance data

## 🔧 Usage Examples

### Load Master Dataset
```python
import pandas as pd
master = pd.read_parquet("data/master.parquet")
print(f"Compounds: {len(master):,}")
```

### Prepare Approval Training Data
```python
from veridica_data.utils.timeguard import cutoff
from veridica_data.transforms.labelers import make_labels

# Load merged data
merged = pd.read_parquet("out/veridica_master_merged.parquet")

# Apply temporal guard (no future leakage)
train_data = cutoff(merged, "first_seen_date", "2017-12-31")

# Create approval labels
train_approval = make_labels(train_data, label="approval_binary")
```

### Train ChemBERTA Model
```python
# Extract SMILES and labels for neural network training
X_smiles = train_approval['canonical_smiles'].tolist()
y_approved = train_approval['approved'].tolist()

# Ready for ChemBERTA molecular encoding
```

## 🏆 Success Metrics

- **Compound Coverage**: Expand beyond 20,963 ChEMBL compounds
- **Clinical Integration**: Real trial outcomes and approval data
- **Toxicity Coverage**: Comprehensive safety profiling
- **Temporal Safety**: No information leakage in training
- **Data Quality**: 100% real pharmaceutical data, zero synthetic

## 📝 License

Research use only. Complies with ChEMBL, PubChem, and ClinicalTrials.gov terms of use.