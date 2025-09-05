# 📊 Veridica ChEMBL Dataset Documentation

## 🎯 Overview

The **Veridica ChEMBL Dataset** is a comprehensive pharmaceutical machine learning dataset built from 100% real ChEMBL compound data. It contains **21,138 authentic pharmaceutical compounds** with molecular structures, clinical trial information, and toxicity safety profiles.

**Key Features:**
- ✅ **100% Real Data** - Zero synthetic compounds
- 🧬 **Complete SMILES Coverage** - Every compound has molecular structure
- 🏥 **Clinical Trial Integration** - Real trial outcomes and phases
- ⚠️ **Toxicity Safety Profiles** - hERG, CYP, AMES, DILI risk assessments
- 🤖 **ML-Ready** - Optimized for ChemBERTA and other pharmaceutical ML models
- ⏰ **Temporal Leakage Protection** - Proper train/validation splits

---

## 📁 Dataset Files (CSV Format)

### 1. **Master Merged Dataset** (`veridica_master_merged.csv`)
**Primary comprehensive dataset combining all data sources**

- **Size**: 21,138 compounds × 58 features
- **File Size**: 8.5 MB
- **Description**: Complete integrated dataset with molecular, clinical, and toxicity data

**Key Columns:**
- `chembl_id`: ChEMBL compound identifier
- `primary_drug`: Primary drug name
- `canonical_smiles`: Canonical SMILES molecular structure
- `max_clinical_phase`: Highest clinical phase reached (0-4)
- `mol_molecular_weight`, `mol_logp`, `mol_tpsa`: Molecular descriptors
- `tox_herg_risk`, `tox_cyp_risk`, `tox_ames_risk`: Toxicity risk levels
- `trial_activity_count`: Number of clinical trials
- `approved`: Binary approval status

**Use Cases:**
- Comprehensive pharmaceutical ML research
- Multi-task learning (approval + toxicity + phase prediction)
- Feature engineering and molecular property analysis

---

### 2. **ChemBERTA Ready Dataset** (`veridica_chembert_ready.csv`)
**Optimized for molecular transformer training**

- **Size**: 21,138 compounds × 4 features
- **File Size**: 1.6 MB
- **Description**: Clean SMILES + labels format for ChemBERTA neural networks

**Columns:**
- `canonical_smiles`: Molecular structure for transformer input
- `primary_drug`: Drug name
- `approved`: Binary approval target (0/1)
- `clinical_phase`: Clinical phase target (0-4)

**Use Cases:**
- ChemBERTA molecular transformer training
- SMILES-based neural network models
- Molecular property prediction
- Drug approval classification

**Example Usage:**
```python
import pandas as pd

# Load ChemBERTA dataset
df = pd.read_csv('csv_exports/veridica_chembert_ready.csv')

# Extract SMILES and targets
X_smiles = df['canonical_smiles'].tolist()
y_approved = df['approved'].tolist()
y_phase = df['clinical_phase'].tolist()

# Ready for ChemBERTA training!
```

---

### 3. **Toxicity Training Dataset** (`veridica_train_tox.csv`)
**Safety prediction and toxicology modeling**

- **Size**: 21,138 compounds × 26 features
- **File Size**: 3.2 MB
- **Description**: Molecular descriptors + toxicity targets for safety prediction

**Key Features:**
- Molecular descriptors: MW, LogP, TPSA, HBD/HBA, rotatable bonds
- Toxicity targets: hERG cardiotoxicity, CYP interactions, AMES mutagenicity
- Risk levels: High/Medium/Low classifications

**Toxicity Endpoints:**
- `tox_herg_risk`: Cardiotoxicity risk (hERG channel inhibition)
- `tox_cyp_risk`: CYP enzyme interaction risk
- `tox_ames_risk`: Mutagenicity risk (AMES test)
- `tox_dili_risk`: Drug-induced liver injury risk

**Use Cases:**
- Toxicity prediction models
- Safety screening algorithms
- QSAR (Quantitative Structure-Activity Relationship) studies
- Early-stage drug safety assessment

---

### 4. **Master Compound Table** (`master.csv`)
**Core compound information and molecular properties**

- **Size**: 20,963 compounds × 26 features
- **File Size**: 5.5 MB
- **Description**: Foundation table with ChEMBL compounds and molecular descriptors

**Key Features:**
- Complete molecular descriptor block
- Clinical phase information
- Data provenance and quality flags
- Temporal metadata

---

### 5. **Clinical Trials Table** (`clinical.csv`)
**Aggregated clinical trial outcomes**

- **Size**: 4,367 drugs × 14 features
- **File Size**: 0.6 MB
- **Description**: Clinical trial data aggregated by drug

**Key Metrics:**
- `max_clinical_phase`: Highest phase reached
- `trial_activity_count`: Total number of trials
- `active_trials`, `completed_trials`, `terminated_trials`: Trial status counts
- `first_trial_date`, `last_trial_date`: Temporal information

---

### 6. **Toxicity Safety Table** (`tox.csv`)
**Detailed toxicity risk profiles**

- **Size**: 20,963 compounds × 21 features
- **File Size**: 3.2 MB
- **Description**: Comprehensive toxicity risk assessment for each compound

**Safety Profiles:**
- Cardiotoxicity (hERG): 10,031 high-risk, 5,472 medium-risk, 5,285 low-risk
- CYP interactions: Enzyme-specific inhibition risks
- Mutagenicity: AMES test risk levels
- Hepatotoxicity: DILI risk assessment

---

## 🧬 Molecular Structure Examples

**Sample SMILES from the dataset:**
- **Prazosin** (Hypertension): `COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC`
- **Nicotine** (Neurological): `CN1CCC[C@H]1c1cccnc1`
- **Ofloxacin** (Antibiotic): `CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23`

---

## 🎯 Machine Learning Applications

### 1. **Drug Approval Prediction**
- **Dataset**: `veridica_chembert_ready.csv`
- **Features**: SMILES molecular structures
- **Target**: Binary approval classification
- **Model**: ChemBERTA transformer

### 2. **Toxicity Prediction**
- **Dataset**: `veridica_train_tox.csv`
- **Features**: Molecular descriptors (MW, LogP, TPSA, etc.)
- **Targets**: hERG, CYP, AMES, DILI risk levels
- **Models**: Random Forest, SVM, Neural Networks

### 3. **Clinical Phase Progression**
- **Dataset**: `veridica_master_merged.csv`
- **Features**: Molecular + toxicity + clinical features
- **Target**: Clinical phase (0-4)
- **Models**: Multi-class classification

### 4. **Multi-Task Learning**
- **Dataset**: `veridica_master_merged.csv`
- **Tasks**: Approval + Toxicity + Phase prediction
- **Features**: Complete feature set
- **Models**: Multi-head neural networks

---

## 📊 Dataset Statistics

### Clinical Phase Distribution
- **Phase IV (Approved)**: 1,010 compounds (4.8%)
- **Phase III**: 224 compounds (1.1%)
- **Phase II**: 1,473 compounds (7.0%)
- **Phase I**: 169 compounds (0.8%)
- **Preclinical**: 43 compounds (0.2%)
- **No Clinical Data**: 18,219 compounds (86.1%)

### Toxicity Risk Distribution
- **High hERG Risk**: 10,031 compounds (47.5%)
- **Medium hERG Risk**: 5,472 compounds (25.9%)
- **Low hERG Risk**: 5,285 compounds (25.0%)

### Data Completeness
- **SMILES Coverage**: 100% (21,138/21,138)
- **Molecular Descriptors**: 100% (21,138/21,138)
- **Clinical Data**: 13.8% (2,919/21,138)
- **Toxicity Data**: 100% (21,138/21,138)

---

## 🚀 Getting Started

### Quick Load in Python
```python
import pandas as pd

# Load main dataset
df = pd.read_csv('csv_exports/veridica_master_merged.csv')
print(f"Dataset shape: {df.shape}")
print(f"Compounds with SMILES: {df['canonical_smiles'].notna().sum()}")

# Load ChemBERTA-ready data
chembert_df = pd.read_csv('csv_exports/veridica_chembert_ready.csv')
smiles_list = chembert_df['canonical_smiles'].tolist()
```

### Quick Analysis
```python
# Clinical phase distribution
phase_dist = df['max_clinical_phase'].value_counts().sort_index()
print("Clinical phases:", phase_dist)

# Toxicity risk summary
herg_risk = df['tox_herg_risk'].value_counts()
print("hERG cardiotoxicity risk:", herg_risk)

# Molecular weight distribution
import matplotlib.pyplot as plt
df['mol_molecular_weight'].hist(bins=50)
plt.title('Molecular Weight Distribution')
plt.xlabel('Molecular Weight (Da)')
plt.show()
```

---

## ✅ Data Quality Assurance

### Validation Checks Passed
- ✅ **No synthetic compounds** - 100% real ChEMBL data
- ✅ **SMILES validation** - All structures are chemically valid
- ✅ **Unique identifiers** - No duplicate ChEMBL IDs
- ✅ **Temporal consistency** - Proper date handling
- ✅ **Feature completeness** - Molecular descriptors computed for all compounds

### Data Sources
- **Primary**: ChEMBL Database (European Bioinformatics Institute)
- **Clinical**: Real clinical trials dataset (ClinicalTrials.gov derived)
- **Toxicity**: Molecular property-based risk assessment

---

## 📝 Citation

If you use this dataset in your research, please cite:

```
Veridica ChEMBL Dataset (2025)
Comprehensive pharmaceutical machine learning dataset
Built from ChEMBL Database and real clinical trial data
21,138 authentic pharmaceutical compounds with molecular, clinical, and toxicity data
```

---

## 🔧 Technical Notes

### File Formats
- **Primary**: Parquet (optimized for ML workflows)
- **Inspection**: CSV (human-readable)
- **Compression**: Snappy (fast loading)

### Memory Requirements
- **Full dataset**: ~50MB RAM
- **ChemBERTA subset**: ~10MB RAM
- **Recommended**: 8GB+ RAM for full analysis

### Dependencies
```python
pandas>=1.5.0
numpy>=1.21.0
rdkit-pypi>=2022.9.1  # For SMILES processing
scikit-learn>=1.1.0   # For ML models
```

---

## 🎉 Ready for Pharmaceutical ML!

The **Veridica ChEMBL Dataset** is now ready for your machine learning applications. All files are in the `csv_exports/` directory for easy inspection and analysis.

**Happy modeling!** 🚀🧬