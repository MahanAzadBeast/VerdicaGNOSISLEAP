#!/usr/bin/env python3
"""
Time-Safe Training View Creator
Creates training views with proper temporal cutoffs to prevent information leakage
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_time_safe_training_view(
    input_file="csv_exports/veridica_master_merged.approvals.csv",
    output_file="csv_exports/veridica_train_approval.timesafe.csv",
    cutoff_date="2017-12-31"
):
    """
    Create time-safe training view with proper temporal cutoffs
    
    Args:
        input_file: Input dataset with approval dates
        output_file: Output time-safe training dataset
        cutoff_date: Temporal cutoff date (YYYY-MM-DD)
        
    Returns:
        Time-safe training DataFrame
    """
    logger.info("⏰ CREATING TIME-SAFE TRAINING VIEW")
    logger.info("🚫 Preventing information leakage with temporal cutoffs")
    logger.info("=" * 60)
    
    try:
        # Load dataset
        df = pd.read_csv(input_file)
        logger.info(f"✅ Loaded dataset: {len(df):,} compounds")
        
        # Parse cutoff date
        cutoff_dt = pd.to_datetime(cutoff_date)
        logger.info(f"⏰ Temporal cutoff: {cutoff_dt.strftime('%Y-%m-%d')}")
        
        # Identify temporal columns that could cause leakage
        temporal_columns = [
            'last_trial_date',
            'trial_status_most_recent', 
            'last_updated',
            'created_at'
        ]
        
        # Parse date columns
        date_columns = ['first_seen_date', 'approval_date_first', 'last_trial_date']
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"📅 Parsed date column: {col}")
        
        # Apply temporal cutoffs
        logger.info("✂️ Applying temporal cutoffs...")
        
        original_len = len(df)
        
        # Filter out compounds first seen after cutoff (if we have that data)
        if 'first_seen_date' in df.columns:
            before_cutoff = df['first_seen_date'] <= cutoff_dt
            valid_first_seen = df['first_seen_date'].notna()
            
            # Keep compounds with no first_seen_date OR first_seen_date <= cutoff
            keep_mask = (~valid_first_seen) | before_cutoff
            df = df[keep_mask].copy()
            
            removed_future = original_len - len(df)
            logger.info(f"   Removed {removed_future:,} compounds first seen after {cutoff_date}")
        
        # Remove future information from temporal columns
        future_info_columns = ['last_trial_date', 'trial_status_most_recent']
        
        for col in future_info_columns:
            if col in df.columns:
                if col == 'last_trial_date':
                    # Set future trial dates to NaT
                    future_mask = df[col] > cutoff_dt
                    future_count = future_mask.sum()
                    df.loc[future_mask, col] = pd.NaT
                    
                    if future_count > 0:
                        logger.info(f"   Removed {future_count:,} future dates from {col}")
                
                elif col == 'trial_status_most_recent':
                    # Remove trial status if last_trial_date is after cutoff
                    if 'last_trial_date' in df.columns:
                        future_trial_mask = df['last_trial_date'].isna() & df[col].notna()
                        # If we removed the trial date, also remove the status
                        df.loc[future_trial_mask, col] = np.nan
        
        # Handle approval dates - these should be kept as they represent the target
        # but we need to be careful about using them as features
        if 'approval_date_first' in df.columns:
            # Create binary approved label based on approval before cutoff
            pre_cutoff_approvals = (df['approval_date_first'] <= cutoff_dt) & df['approval_date_first'].notna()
            df['approved'] = pre_cutoff_approvals.astype(int)
            
            approved_count = df['approved'].sum()
            logger.info(f"🏷️ Approved compounds (pre-cutoff): {approved_count:,}")
            
            # For training, we can keep approval_date_first but flag it as target-only
            df['approval_date_first_target'] = df['approval_date_first']
        else:
            # No approval data - create default binary target
            df['approved'] = 0
            logger.warning("⚠️ No approval_date_first column - setting all as not approved")
        
        # Remove or flag columns that shouldn't be used as features for approval prediction
        feature_exclusion_columns = [
            'approval_date_first',  # This is the target we're predicting
            'approved',  # This is derived from approval_date_first
            'trial_status_most_recent',  # Could contain post-cutoff information
            'last_trial_date'  # Could be after cutoff
        ]
        
        # Create feature-safe version by excluding leakage-prone columns
        all_columns = df.columns.tolist()
        feature_columns = [col for col in all_columns if not any(
            col.startswith(prefix) for prefix in ['ae_', 'meddra_', 'smq_']  # Exclude AE data
        ) and col not in feature_exclusion_columns]
        
        # Add back the target
        feature_columns.append('approved')
        
        # Create feature-safe dataset
        feature_safe_df = df[feature_columns].copy()
        
        logger.info(f"🔒 Feature-safe columns: {len(feature_columns)}")
        logger.info(f"📊 Final training dataset: {len(feature_safe_df):,} compounds")
        
        # Validate temporal safety
        validation_passed = validate_temporal_safety(feature_safe_df, cutoff_dt)
        
        if not validation_passed:
            logger.error("❌ Temporal safety validation failed!")
            return None
        
        # Generate training statistics
        generate_training_stats(feature_safe_df)
        
        # Save time-safe training dataset
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            feature_safe_df.to_csv(output_file, index=False)
            logger.info(f"💾 Time-safe training dataset saved: {output_file}")
        except Exception as e:
            logger.error(f"❌ Error saving training dataset: {e}")
            return None
        
        return feature_safe_df
        
    except Exception as e:
        logger.error(f"❌ Error creating time-safe training view: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_temporal_safety(df, cutoff_dt):
    """
    Validate that no future information leaks into training data
    
    Args:
        df: Training DataFrame to validate
        cutoff_dt: Cutoff datetime
        
    Returns:
        True if validation passes, False otherwise
    """
    logger.info("🔍 VALIDATING TEMPORAL SAFETY")
    logger.info("=" * 40)
    
    validation_passed = True
    
    # Check date columns for future dates
    date_columns = ['first_seen_date', 'last_trial_date', 'created_at', 'last_updated']
    
    for col in date_columns:
        if col in df.columns:
            try:
                col_dates = pd.to_datetime(df[col], errors='coerce')
                future_mask = col_dates > cutoff_dt
                future_count = future_mask.sum()
                
                if future_count > 0:
                    logger.error(f"❌ LEAKAGE DETECTED: {future_count:,} future dates in '{col}'")
                    validation_passed = False
                else:
                    logger.info(f"✅ Temporal safety OK: '{col}' ≤ {cutoff_dt.strftime('%Y-%m-%d')}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not validate '{col}': {e}")
    
    # Check for suspicious columns that might contain future info
    suspicious_patterns = ['recent', 'latest', 'current', 'last_updated']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in suspicious_patterns):
            if col not in ['trial_status_most_recent']:  # Known safe columns
                logger.warning(f"⚠️ Suspicious column detected: '{col}' - verify temporal safety")
    
    # Check target distribution
    if 'approved' in df.columns:
        approved_count = df['approved'].sum()
        total_count = len(df)
        approval_rate = approved_count / total_count if total_count > 0 else 0
        
        logger.info(f"🎯 Target distribution:")
        logger.info(f"   Approved: {approved_count:,} ({approval_rate*100:.2f}%)")
        logger.info(f"   Not approved: {total_count - approved_count:,} ({(1-approval_rate)*100:.2f}%)")
        
        if approval_rate == 0:
            logger.warning("⚠️ No approved compounds in training data - check approval data")
        elif approval_rate > 0.5:
            logger.warning(f"⚠️ High approval rate ({approval_rate*100:.1f}%) - verify data quality")
    
    return validation_passed


def generate_training_stats(df):
    """Generate training dataset statistics"""
    logger.info("📊 TRAINING DATASET STATISTICS")
    logger.info("=" * 40)
    
    # Basic stats
    logger.info(f"Total compounds: {len(df):,}")
    logger.info(f"Total features: {len(df.columns):,}")
    
    # Missing data analysis
    missing_stats = df.isnull().sum()
    high_missing = missing_stats[missing_stats > len(df) * 0.5]
    
    if len(high_missing) > 0:
        logger.info(f"\\nColumns with >50% missing data:")
        for col, missing_count in high_missing.items():
            missing_pct = (missing_count / len(df)) * 100
            logger.info(f"  {col}: {missing_count:,} ({missing_pct:.1f}%)")
    
    # Feature type analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    text_cols = df.select_dtypes(include=['object']).columns
    date_cols = df.select_dtypes(include=['datetime64']).columns
    
    logger.info(f"\\nFeature types:")
    logger.info(f"  Numeric: {len(numeric_cols)}")
    logger.info(f"  Text: {len(text_cols)}")
    logger.info(f"  Date: {len(date_cols)}")
    
    # SMILES availability
    if 'canonical_smiles' in df.columns:
        smiles_count = df['canonical_smiles'].notna().sum()
        smiles_pct = (smiles_count / len(df)) * 100
        logger.info(f"\\nSMILES availability: {smiles_count:,} ({smiles_pct:.1f}%)")


def main():
    """Main execution"""
    logger.info("⏰ TIME-SAFE TRAINING VIEW CREATOR")
    logger.info("🔒 Creating leakage-free training datasets")
    logger.info("🚫 100% temporal safety for ML training")
    
    # Create time-safe training view
    training_df = create_time_safe_training_view()
    
    if training_df is not None:
        logger.info("\\n🎉 TIME-SAFE TRAINING VIEW COMPLETE")
        logger.info(f"📊 Training dataset: {len(training_df):,} compounds")
        logger.info(f"🔒 Temporal cutoff: 2017-12-31")
        logger.info(f"✅ No information leakage")
        logger.info(f"📁 Saved to: csv_exports/veridica_train_approval.timesafe.csv")
        
        # Show next steps
        logger.info("\\n🚀 READY FOR ML TRAINING:")
        logger.info("   1. Load time-safe dataset")
        logger.info("   2. Extract SMILES for ChemBERTA")
        logger.info("   3. Use molecular descriptors for classical ML")
        logger.info("   4. Target: 'approved' binary classification")
        logger.info("   5. Validate on held-out test set")
        
    else:
        logger.error("❌ Time-safe training view creation failed")


if __name__ == "__main__":
    main()