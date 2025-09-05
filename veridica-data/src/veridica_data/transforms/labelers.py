"""
Label Generation for ML Training
Creates target labels for different prediction tasks
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def create_approval_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary approval labels for drug approval prediction
    
    Args:
        df: DataFrame with clinical phase information
        
    Returns:
        DataFrame with approval labels added
    """
    df = df.copy()
    
    # Binary approval based on clinical phase >= 4 (Phase IV/Post-market)
    if 'max_clinical_phase' in df.columns:
        df['approved'] = (df['max_clinical_phase'] >= 4).astype(int)
        logger.info(f"Created approval labels: {df['approved'].sum():,} approved compounds")
    else:
        df['approved'] = 0
        logger.warning("No clinical phase data available - setting all as not approved")
    
    # Alternative approval indicators
    if 'approval_date_first' in df.columns:
        df['has_approval_date'] = df['approval_date_first'].notna().astype(int)
        # Override approval if we have explicit approval date
        df['approved'] = np.maximum(df['approved'], df['has_approval_date'])
    
    return df


def create_phase_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create multi-class clinical phase labels
    
    Args:
        df: DataFrame with clinical phase information
        
    Returns:
        DataFrame with phase labels added
    """
    df = df.copy()
    
    # Multi-class phase labels (0=Preclinical, 1=Phase I, 2=Phase II, 3=Phase III, 4=Phase IV)
    if 'max_clinical_phase' in df.columns:
        df['phase_label'] = df['max_clinical_phase'].fillna(0).astype(int)
        
        # Ensure valid range
        df['phase_label'] = df['phase_label'].clip(0, 4)
        
        phase_counts = df['phase_label'].value_counts().sort_index()
        logger.info("Phase label distribution:")
        for phase, count in phase_counts.items():
            phase_name = f"Phase {int(phase)}" if phase > 0 else "Preclinical"
            logger.info(f"  {phase_name}: {count:,} compounds")
            
    else:
        df['phase_label'] = 0
        logger.warning("No clinical phase data available - setting all as preclinical")
    
    return df


def create_success_probability_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create continuous success probability labels based on multiple factors
    
    Args:
        df: DataFrame with clinical and molecular data
        
    Returns:
        DataFrame with success probability labels added
    """
    df = df.copy()
    
    # Initialize base success probability
    df['success_probability'] = 0.1  # Base 10% success rate
    
    # Adjust based on clinical phase
    if 'max_clinical_phase' in df.columns:
        phase_multipliers = {0: 0.1, 1: 0.2, 2: 0.4, 3: 0.7, 4: 1.0}
        
        for phase, multiplier in phase_multipliers.items():
            mask = (df['max_clinical_phase'] == phase)
            df.loc[mask, 'success_probability'] = multiplier
    
    # Adjust based on toxicity risk
    if 'tox_herg_risk' in df.columns:
        # Reduce probability for high toxicity risk
        high_tox_mask = (df['tox_herg_risk'] == 'high')
        df.loc[high_tox_mask, 'success_probability'] *= 0.7
        
        # Increase probability for low toxicity risk
        low_tox_mask = (df['tox_herg_risk'] == 'low')
        df.loc[low_tox_mask, 'success_probability'] *= 1.2
    
    # Adjust based on molecular properties (drug-likeness)
    if all(col in df.columns for col in ['mol_molecular_weight', 'mol_logp']):
        # Lipinski's Rule of Five compliance
        lipinski_violations = 0
        
        # MW <= 500 Da
        mw_violation = (df['mol_molecular_weight'] > 500)
        lipinski_violations += mw_violation.astype(int)
        
        # LogP <= 5
        logp_violation = (df['mol_logp'] > 5)
        lipinski_violations += logp_violation.astype(int)
        
        # HBD <= 5
        if 'mol_num_hbd' in df.columns:
            hbd_violation = (df['mol_num_hbd'] > 5)
            lipinski_violations += hbd_violation.astype(int)
        
        # HBA <= 10
        if 'mol_num_hba' in df.columns:
            hba_violation = (df['mol_num_hba'] > 10)
            lipinski_violations += hba_violation.astype(int)
        
        # Adjust probability based on violations
        violation_penalty = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.6, 4: 0.4}
        
        for violations, penalty in violation_penalty.items():
            mask = (lipinski_violations == violations)
            df.loc[mask, 'success_probability'] *= penalty
    
    # Ensure probability stays in [0, 1] range
    df['success_probability'] = df['success_probability'].clip(0, 1)
    
    logger.info(f"Success probability statistics:")
    logger.info(f"  Mean: {df['success_probability'].mean():.3f}")
    logger.info(f"  Median: {df['success_probability'].median():.3f}")
    logger.info(f"  Range: {df['success_probability'].min():.3f} - {df['success_probability'].max():.3f}")
    
    return df


def create_toxicity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create toxicity prediction labels
    
    Args:
        df: DataFrame with toxicity data
        
    Returns:
        DataFrame with toxicity labels added
    """
    df = df.copy()
    
    # Binary toxicity labels for key endpoints
    tox_endpoints = ['tox_herg_risk', 'tox_cyp_risk', 'tox_ames_risk', 'tox_dili_risk']
    
    for endpoint in tox_endpoints:
        if endpoint in df.columns:
            # Convert risk levels to binary (high risk = 1, others = 0)
            binary_col = endpoint.replace('_risk', '_toxic')
            df[binary_col] = (df[endpoint] == 'high').astype(int)
            
            toxic_count = df[binary_col].sum()
            total_count = df[binary_col].notna().sum()
            
            if total_count > 0:
                logger.info(f"{endpoint}: {toxic_count:,}/{total_count:,} ({toxic_count/total_count*100:.1f}%) high risk")
    
    # Composite toxicity score
    tox_binary_cols = [col for col in df.columns if col.endswith('_toxic')]
    
    if tox_binary_cols:
        df['composite_toxicity_score'] = df[tox_binary_cols].sum(axis=1) / len(tox_binary_cols)
        
        logger.info(f"Composite toxicity score statistics:")
        logger.info(f"  Mean: {df['composite_toxicity_score'].mean():.3f}")
        logger.info(f"  High toxicity (>0.5): {(df['composite_toxicity_score'] > 0.5).sum():,} compounds")
    
    return df


def create_temporal_splits(df: pd.DataFrame, 
                          date_col: str = 'first_seen_date',
                          train_cutoff: str = '2017-12-31',
                          val_cutoff: str = '2019-12-31') -> Dict[str, pd.DataFrame]:
    """
    Create temporal train/validation/test splits
    
    Args:
        df: DataFrame to split
        date_col: Column name for temporal splitting
        train_cutoff: Cutoff date for training data
        val_cutoff: Cutoff date for validation data
        
    Returns:
        Dictionary with train/val/test DataFrames
    """
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found - using random splits")
        return create_random_splits(df)
    
    # Convert dates
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    train_cutoff_dt = pd.to_datetime(train_cutoff)
    val_cutoff_dt = pd.to_datetime(val_cutoff)
    
    # Create temporal masks
    train_mask = df[date_col] <= train_cutoff_dt
    val_mask = (df[date_col] > train_cutoff_dt) & (df[date_col] <= val_cutoff_dt)
    test_mask = df[date_col] > val_cutoff_dt
    
    # Create splits
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    logger.info(f"Temporal splits created:")
    logger.info(f"  Train (≤{train_cutoff}): {len(train_df):,} compounds")
    logger.info(f"  Val ({train_cutoff} < date ≤ {val_cutoff}): {len(val_df):,} compounds")
    logger.info(f"  Test (>{val_cutoff}): {len(test_df):,} compounds")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def create_random_splits(df: pd.DataFrame, 
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
    """
    Create random train/validation/test splits
    
    Args:
        df: DataFrame to split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        
    Returns:
        Dictionary with train/val/test DataFrames
    """
    df = df.copy()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df[:train_end].copy()
    val_df = df[train_end:val_end].copy()
    test_df = df[val_end:].copy()
    
    logger.info(f"Random splits created:")
    logger.info(f"  Train: {len(train_df):,} compounds ({len(train_df)/n*100:.1f}%)")
    logger.info(f"  Val: {len(val_df):,} compounds ({len(val_df)/n*100:.1f}%)")
    logger.info(f"  Test: {len(test_df):,} compounds ({len(test_df)/n*100:.1f}%)")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def create_balanced_labels(df: pd.DataFrame, 
                          target_col: str,
                          balance_ratio: float = 0.5) -> pd.DataFrame:
    """
    Create balanced dataset by sampling
    
    Args:
        df: DataFrame with target column
        target_col: Column name for balancing
        balance_ratio: Target ratio for minority class
        
    Returns:
        Balanced DataFrame
    """
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found")
        return df.copy()
    
    # Get class counts
    value_counts = df[target_col].value_counts()
    
    if len(value_counts) < 2:
        logger.warning("Target column has less than 2 classes - no balancing needed")
        return df.copy()
    
    minority_class = value_counts.idxmin()
    majority_class = value_counts.idxmax()
    minority_count = value_counts[minority_class]
    majority_count = value_counts[majority_class]
    
    logger.info(f"Original class distribution:")
    for class_val, count in value_counts.items():
        logger.info(f"  {class_val}: {count:,} samples")
    
    # Calculate target counts
    target_minority = int(len(df) * balance_ratio)
    target_majority = len(df) - target_minority
    
    # Sample from each class
    minority_df = df[df[target_col] == minority_class].sample(
        n=min(target_minority, minority_count), 
        random_state=42
    )
    majority_df = df[df[target_col] == majority_class].sample(
        n=min(target_majority, majority_count),
        random_state=42
    )
    
    # Combine
    balanced_df = pd.concat([minority_df, majority_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    balanced_counts = balanced_df[target_col].value_counts()
    logger.info(f"Balanced class distribution:")
    for class_val, count in balanced_counts.items():
        logger.info(f"  {class_val}: {count:,} samples ({count/len(balanced_df)*100:.1f}%)")
    
    return balanced_df