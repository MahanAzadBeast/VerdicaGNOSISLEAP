"""
Temporal Guard Utilities
Prevents information leakage by filtering data based on temporal constraints
"""

import pandas as pd
from datetime import datetime
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def cutoff(df: pd.DataFrame, date_col: str, cutoff_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows before or on cutoff_date
    
    Args:
        df: Input DataFrame
        date_col: Column name containing dates
        cutoff_date: Cutoff date (inclusive)
        
    Returns:
        Filtered DataFrame with no information after cutoff_date
    """
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found in DataFrame")
        return df.copy()
    
    # Convert cutoff_date to datetime if string
    if isinstance(cutoff_date, str):
        try:
            cutoff_date = pd.to_datetime(cutoff_date)
        except Exception as e:
            logger.error(f"Error parsing cutoff_date '{cutoff_date}': {e}")
            return df.copy()
    
    # Convert date column to datetime
    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        
        # Filter data
        mask = df_copy[date_col] <= cutoff_date
        filtered_df = df_copy[mask].copy()
        
        logger.info(f"Temporal filter: {len(filtered_df):,}/{len(df):,} rows kept (cutoff: {cutoff_date})")
        
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error applying temporal cutoff: {e}")
        return df.copy()


def validate_temporal_safety(df: pd.DataFrame, 
                            date_cols: list, 
                            cutoff_date: Union[str, datetime],
                            split_name: str = "training") -> bool:
    """
    Validate that no future information leaks into training data
    
    Args:
        df: DataFrame to validate
        date_cols: List of date columns to check
        cutoff_date: Cutoff date for validation
        split_name: Name of the split being validated
        
    Returns:
        True if no leakage detected, False otherwise
    """
    if isinstance(cutoff_date, str):
        cutoff_date = pd.to_datetime(cutoff_date)
    
    leakage_detected = False
    
    for date_col in date_cols:
        if date_col not in df.columns:
            continue
        
        try:
            df_dates = pd.to_datetime(df[date_col], errors='coerce')
            future_mask = df_dates > cutoff_date
            future_count = future_mask.sum()
            
            if future_count > 0:
                logger.error(f"LEAKAGE DETECTED in {split_name}: {future_count} rows in '{date_col}' after {cutoff_date}")
                leakage_detected = True
            else:
                logger.info(f"Temporal safety OK for {split_name}: '{date_col}' ≤ {cutoff_date}")
                
        except Exception as e:
            logger.warning(f"Could not validate temporal safety for '{date_col}': {e}")
    
    return not leakage_detected


def add_temporal_features(df: pd.DataFrame, 
                         date_col: str, 
                         reference_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
    """
    Add temporal features based on a date column
    
    Args:
        df: Input DataFrame
        date_col: Date column to use for feature generation
        reference_date: Reference date for relative calculations
        
    Returns:
        DataFrame with additional temporal features
    """
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found")
        return df.copy()
    
    df_copy = df.copy()
    
    try:
        # Convert to datetime
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        
        if reference_date is None:
            reference_date = datetime.now()
        elif isinstance(reference_date, str):
            reference_date = pd.to_datetime(reference_date)
        
        # Add temporal features
        df_copy[f'{date_col}_year'] = df_copy[date_col].dt.year
        df_copy[f'{date_col}_month'] = df_copy[date_col].dt.month
        df_copy[f'{date_col}_quarter'] = df_copy[date_col].dt.quarter
        df_copy[f'{date_col}_dayofyear'] = df_copy[date_col].dt.dayofyear
        
        # Days since reference
        df_copy[f'{date_col}_days_since_ref'] = (reference_date - df_copy[date_col]).dt.days
        
        logger.info(f"Added temporal features for '{date_col}'")
        
        return df_copy
        
    except Exception as e:
        logger.error(f"Error adding temporal features: {e}")
        return df.copy()


def create_temporal_splits(df: pd.DataFrame,
                          date_col: str,
                          train_cutoff: Union[str, datetime],
                          val_cutoff: Union[str, datetime]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create temporal train/val/test splits to prevent leakage
    
    Args:
        df: Input DataFrame
        date_col: Date column for temporal splitting
        train_cutoff: Cutoff date for training data
        val_cutoff: Cutoff date for validation data (test is after this)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if isinstance(train_cutoff, str):
        train_cutoff = pd.to_datetime(train_cutoff)
    if isinstance(val_cutoff, str):
        val_cutoff = pd.to_datetime(val_cutoff)
    
    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        
        # Create temporal splits
        train_mask = df_copy[date_col] <= train_cutoff
        val_mask = (df_copy[date_col] > train_cutoff) & (df_copy[date_col] <= val_cutoff)
        test_mask = df_copy[date_col] > val_cutoff
        
        train_df = df_copy[train_mask].copy()
        val_df = df_copy[val_mask].copy()
        test_df = df_copy[test_mask].copy()
        
        logger.info(f"Temporal splits created:")
        logger.info(f"  Train: {len(train_df):,} rows (≤ {train_cutoff})")
        logger.info(f"  Val: {len(val_df):,} rows ({train_cutoff} < date ≤ {val_cutoff})")
        logger.info(f"  Test: {len(test_df):,} rows (> {val_cutoff})")
        
        return train_df, val_df, test_df
        
    except Exception as e:
        logger.error(f"Error creating temporal splits: {e}")
        return df.copy(), pd.DataFrame(), pd.DataFrame()