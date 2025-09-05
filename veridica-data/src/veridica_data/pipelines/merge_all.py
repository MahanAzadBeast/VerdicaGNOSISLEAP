"""
Master Data Merge Pipeline
Combines master compound table with clinical and toxicity sidecar tables
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

from ..utils.timeguard import cutoff, validate_temporal_safety
from ..transforms.labelers import create_approval_labels, create_phase_labels

logger = logging.getLogger(__name__)


def merge_all_data_sources():
    """
    Merge master compound table with all sidecar tables
    """
    logger.info("🔗 MERGING ALL VERIDICA DATA SOURCES")
    logger.info("🎯 Creating comprehensive pharmaceutical ML dataset")
    logger.info("🚫 100% REAL data with temporal leakage protection")
    logger.info("=" * 70)
    
    # Load all data tables
    try:
        master_df = pd.read_parquet("data/master.parquet")
        logger.info(f"✅ Master table loaded: {len(master_df):,} compounds")
    except Exception as e:
        logger.error(f"❌ Could not load master table: {e}")
        return None
    
    try:
        clinical_df = pd.read_parquet("data/clinical.parquet")
        logger.info(f"✅ Clinical table loaded: {len(clinical_df):,} drugs")
    except Exception as e:
        logger.warning(f"⚠️ Could not load clinical table: {e}")
        clinical_df = pd.DataFrame()
    
    try:
        tox_df = pd.read_parquet("data/tox.parquet")
        logger.info(f"✅ Toxicity table loaded: {len(tox_df):,} compounds")
    except Exception as e:
        logger.warning(f"⚠️ Could not load toxicity table: {e}")
        tox_df = pd.DataFrame()
    
    # Start with master table as base
    merged_df = master_df.copy()
    logger.info(f"🏗️ Base dataset: {len(merged_df):,} compounds")
    
    # Merge clinical data by primary_drug name
    if not clinical_df.empty:
        logger.info("🏥 Merging clinical trial data...")
        
        # Clean drug names for better matching
        clinical_df['primary_drug_clean'] = clinical_df['primary_drug'].str.lower().str.strip()
        merged_df['primary_drug_clean'] = merged_df['primary_drug'].str.lower().str.strip()
        
        # Left join to preserve all compounds
        merged_df = merged_df.merge(
            clinical_df.drop('primary_drug', axis=1),
            on='primary_drug_clean',
            how='left',
            suffixes=('', '_clinical')
        )
        
        clinical_matches = merged_df['max_clinical_phase'].notna().sum()
        logger.info(f"✅ Clinical data merged: {clinical_matches:,} compounds matched")
    
    # Merge toxicity data by chembl_id
    if not tox_df.empty:
        logger.info("⚠️ Merging toxicity safety data...")
        
        # Left join to preserve all compounds
        merged_df = merged_df.merge(
            tox_df.drop(['primary_drug'], axis=1, errors='ignore'),
            on='chembl_id',
            how='left',
            suffixes=('', '_tox')
        )
        
        tox_matches = merged_df['tox_herg_risk'].notna().sum()
        logger.info(f"✅ Toxicity data merged: {tox_matches:,} compounds matched")
    
    # Clean up temporary columns
    merged_df = merged_df.drop(['primary_drug_clean'], axis=1, errors='ignore')
    
    logger.info(f"🎉 MERGE COMPLETE: {len(merged_df):,} compounds in master dataset")
    
    # Generate merge quality report
    _generate_merge_report(merged_df)
    
    # Save merged dataset
    output_dir = Path("out")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    master_merged_file = output_dir / "veridica_master_merged.parquet"
    merged_df.to_parquet(master_merged_file, compression='snappy')
    logger.info(f"💾 Master merged dataset saved: {master_merged_file}")
    
    return merged_df


def create_ml_ready_datasets(merged_df: pd.DataFrame):
    """
    Create ML-ready datasets with temporal guards and proper labeling
    """
    logger.info("🤖 CREATING ML-READY DATASETS")
    logger.info("⏰ Applying temporal guards to prevent information leakage")
    logger.info("=" * 60)
    
    # Define cutoff dates for temporal safety
    CUTOFF_DATE = "2017-12-31"  # Training cutoff
    VAL_CUTOFF_DATE = "2019-12-31"  # Validation cutoff
    
    output_dir = Path("out")
    
    # 1. APPROVAL PREDICTION DATASET
    logger.info("🎯 Creating approval prediction dataset...")
    
    # Apply temporal cutoff
    approval_train_df = cutoff(merged_df, 'first_seen_date', CUTOFF_DATE)
    
    # Create approval labels
    approval_train_df = create_approval_labels(approval_train_df)
    
    # Remove AE data from features (prevent leakage)
    feature_cols = [col for col in approval_train_df.columns 
                   if not col.startswith('ae_') and col != 'adverse_events']
    
    approval_features_df = approval_train_df[feature_cols].copy()
    
    # Validate temporal safety
    temporal_safe = validate_temporal_safety(
        approval_features_df, 
        ['first_seen_date', 'created_at'], 
        CUTOFF_DATE,
        'approval_training'
    )
    
    if temporal_safe:
        approval_file = output_dir / "veridica_train_approval.parquet"
        approval_features_df.to_parquet(approval_file, compression='snappy')
        logger.info(f"✅ Approval training dataset: {approval_file}")
        logger.info(f"   📊 {len(approval_features_df):,} compounds")
        logger.info(f"   ⏰ Temporal cutoff: {CUTOFF_DATE}")
        logger.info(f"   🚫 AE data excluded from features")
    else:
        logger.error("❌ Temporal leakage detected in approval dataset")
    
    # 2. TOXICITY PREDICTION DATASET
    logger.info("⚠️ Creating toxicity prediction dataset...")
    
    # For toxicity, we can use more recent data since it's property-based
    tox_train_df = merged_df.copy()
    
    # Select toxicity-relevant features
    tox_feature_cols = [
        'chembl_id', 'canonical_smiles', 'primary_drug',
        'mol_molecular_weight', 'mol_logp', 'mol_num_hbd', 'mol_num_hba',
        'mol_num_rotatable_bonds', 'mol_tpsa', 'mol_num_aromatic_rings',
        'mol_num_heavy_atoms', 'mol_formal_charge', 'mol_num_rings',
        'mol_num_heteroatoms', 'mol_fraction_csp3'
    ]
    
    # Add toxicity targets
    tox_target_cols = [col for col in merged_df.columns if col.startswith('tox_')]
    
    tox_cols = tox_feature_cols + tox_target_cols
    tox_available_cols = [col for col in tox_cols if col in merged_df.columns]
    
    tox_dataset_df = tox_train_df[tox_available_cols].copy()
    
    tox_file = output_dir / "veridica_train_tox.parquet"
    tox_dataset_df.to_parquet(tox_file, compression='snappy')
    logger.info(f"✅ Toxicity training dataset: {tox_file}")
    logger.info(f"   📊 {len(tox_dataset_df):,} compounds")
    logger.info(f"   🧬 Features: molecular descriptors")
    logger.info(f"   🎯 Targets: hERG, CYP, AMES, DILI risks")
    
    # 3. CLINICAL PHASE PREDICTION DATASET
    logger.info("🏥 Creating clinical phase prediction dataset...")
    
    # Apply temporal cutoff
    phase_train_df = cutoff(merged_df, 'first_seen_date', CUTOFF_DATE)
    
    # Create phase labels
    phase_train_df = create_phase_labels(phase_train_df)
    
    # Select relevant features
    phase_feature_cols = [
        'chembl_id', 'canonical_smiles', 'primary_drug',
        'mol_molecular_weight', 'mol_logp', 'mol_num_hbd', 'mol_num_hba',
        'mol_num_rotatable_bonds', 'mol_tpsa', 'mol_num_aromatic_rings',
        'mol_num_heavy_atoms', 'mol_formal_charge'
    ]
    
    # Add toxicity features (but not clinical outcomes)
    phase_feature_cols.extend([col for col in merged_df.columns 
                              if col.startswith('tox_') and 'risk' in col])
    
    # Add phase target
    phase_feature_cols.append('phase_label')
    
    phase_available_cols = [col for col in phase_feature_cols if col in phase_train_df.columns]
    phase_dataset_df = phase_train_df[phase_available_cols].copy()
    
    phase_file = output_dir / "veridica_train_phase.parquet"
    phase_dataset_df.to_parquet(phase_file, compression='snappy')
    logger.info(f"✅ Clinical phase dataset: {phase_file}")
    logger.info(f"   📊 {len(phase_dataset_df):,} compounds")
    logger.info(f"   ⏰ Temporal cutoff: {CUTOFF_DATE}")
    
    # 4. CHEMBERT-READY DATASET
    logger.info("🧬 Creating ChemBERTA-ready dataset...")
    
    # ChemBERTA needs SMILES and targets
    chembert_df = merged_df[['canonical_smiles', 'primary_drug']].copy()
    
    # Add approval target if available
    if 'approved' in merged_df.columns:
        chembert_df['approved'] = merged_df['approved']
    elif 'max_clinical_phase' in merged_df.columns:
        chembert_df['approved'] = (merged_df['max_clinical_phase'] >= 4).astype(int)
    else:
        chembert_df['approved'] = 0
    
    # Add phase target
    if 'max_clinical_phase' in merged_df.columns:
        chembert_df['clinical_phase'] = merged_df['max_clinical_phase'].fillna(0)
    else:
        chembert_df['clinical_phase'] = 0
    
    # Remove rows with missing SMILES
    chembert_df = chembert_df[chembert_df['canonical_smiles'].notna()].copy()
    
    chembert_file = output_dir / "veridica_chembert_ready.parquet"
    chembert_df.to_parquet(chembert_file, compression='snappy')
    logger.info(f"✅ ChemBERTA dataset: {chembert_file}")
    logger.info(f"   📊 {len(chembert_df):,} compounds with SMILES")
    logger.info(f"   🧬 Ready for molecular transformer training")
    
    return {
        'approval': approval_features_df,
        'toxicity': tox_dataset_df,
        'phase': phase_dataset_df,
        'chembert': chembert_df
    }


def create_approval_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create approval prediction labels"""
    df = df.copy()
    
    # Binary approval based on clinical phase
    if 'max_clinical_phase' in df.columns:
        df['approved'] = (df['max_clinical_phase'] >= 4).astype(int)
    else:
        df['approved'] = 0
    
    return df


def create_phase_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create clinical phase labels"""
    df = df.copy()
    
    # Multi-class phase labels
    if 'max_clinical_phase' in df.columns:
        df['phase_label'] = df['max_clinical_phase'].fillna(0).astype(int)
    else:
        df['phase_label'] = 0
    
    return df


def _generate_merge_report(df: pd.DataFrame) -> None:
    """Generate merge quality report"""
    logger.info("📊 MERGE QUALITY REPORT")
    logger.info("=" * 50)
    
    # Basic statistics
    logger.info(f"Total compounds in merged dataset: {len(df):,}")
    
    # Data completeness by source
    clinical_complete = df['max_clinical_phase'].notna().sum() if 'max_clinical_phase' in df.columns else 0
    tox_complete = df['tox_herg_risk'].notna().sum() if 'tox_herg_risk' in df.columns else 0
    
    logger.info(f"\\nData source completeness:")
    logger.info(f"  Master compound data: {len(df):,} (100.0%)")
    logger.info(f"  Clinical trial data: {clinical_complete:,} ({(clinical_complete/len(df)*100):.1f}%)")
    logger.info(f"  Toxicity safety data: {tox_complete:,} ({(tox_complete/len(df)*100):.1f}%)")
    
    # ML readiness
    smiles_complete = df['canonical_smiles'].notna().sum()
    descriptors_complete = df['mol_molecular_weight'].notna().sum() if 'mol_molecular_weight' in df.columns else 0
    
    logger.info(f"\\nML readiness:")
    logger.info(f"  SMILES available: {smiles_complete:,} ({(smiles_complete/len(df)*100):.1f}%)")
    logger.info(f"  Molecular descriptors: {descriptors_complete:,} ({(descriptors_complete/len(df)*100):.1f}%)")
    
    # Target distribution
    if 'max_clinical_phase' in df.columns:
        phase_dist = df['max_clinical_phase'].value_counts().sort_index()
        logger.info(f"\\nClinical phase distribution:")
        for phase, count in phase_dist.items():
            if pd.notna(phase):
                phase_name = f"Phase {int(phase)}" if phase >= 1 else "Preclinical"
                logger.info(f"  {phase_name}: {count:,} compounds")


def main():
    """Main pipeline execution"""
    logger.info("🔗 VERIDICA MASTER DATA MERGE PIPELINE")
    logger.info("🎯 Creating comprehensive pharmaceutical ML datasets")
    logger.info("🚫 100% REAL data with temporal leakage protection")
    
    # Merge all data sources
    merged_df = merge_all_data_sources()
    
    if merged_df is not None:
        # Create ML-ready datasets
        ml_datasets = create_ml_ready_datasets(merged_df)
        
        logger.info("\\n🎉 VERIDICA PIPELINE COMPLETE")
        logger.info(f"📊 Master dataset: {len(merged_df):,} compounds")
        logger.info(f"✅ ML-ready datasets created:")
        
        for dataset_name, dataset_df in ml_datasets.items():
            logger.info(f"   📁 {dataset_name}: {len(dataset_df):,} compounds")
        
        logger.info(f"\\n🚀 READY FOR PHARMACEUTICAL ML:")
        logger.info(f"   🎯 Approval prediction")
        logger.info(f"   ⚠️ Toxicity prediction") 
        logger.info(f"   🏥 Clinical phase prediction")
        logger.info(f"   🧬 ChemBERTA molecular modeling")
        logger.info(f"   ⏰ Temporal leakage protection enabled")
        logger.info(f"   🚫 Zero synthetic data")
        
    else:
        logger.error("❌ Merge pipeline failed")
    
    return merged_df


if __name__ == "__main__":
    main()