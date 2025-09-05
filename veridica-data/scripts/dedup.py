#!/usr/bin/env python3
"""
Deduplicate by ChEMBL ID
Removes duplicate ChEMBL IDs keeping the richest row (most complete data)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def deduplicate_by_chembl_id(input_file="csv_exports/veridica_master_merged.csv", 
                             output_file="csv_exports/veridica_master_merged.dedup.csv"):
    """
    Deduplicate by chembl_id keeping the richest row
    
    Args:
        input_file: Input CSV file path
        output_file: Output deduplicated CSV file path
        
    Returns:
        Deduplicated DataFrame
    """
    logger.info("🔧 DEDUPLICATING BY CHEMBL_ID")
    logger.info("📊 Keeping richest rows based on data completeness")
    logger.info("=" * 60)
    
    try:
        # Load dataset
        df = pd.read_csv(input_file)
        logger.info(f"✅ Loaded dataset: {len(df):,} rows × {len(df.columns)} columns")
        
        # Check for duplicates
        duplicate_mask = df.duplicated(subset=['chembl_id'], keep=False)
        duplicate_count = duplicate_mask.sum()
        unique_chembl_count = df['chembl_id'].nunique()
        
        logger.info(f"🔍 Duplicate analysis:")
        logger.info(f"   Total rows: {len(df):,}")
        logger.info(f"   Unique ChEMBL IDs: {unique_chembl_count:,}")
        logger.info(f"   Duplicate rows: {duplicate_count:,}")
        logger.info(f"   Expected deduplication: {len(df) - unique_chembl_count:,} rows removed")
        
        if duplicate_count == 0:
            logger.info("✅ No duplicates found - dataset already clean")
            return df
        
        # Score rows by completeness (non-null count)
        logger.info("📊 Scoring rows by data completeness...")
        non_null_counts = df.notna().sum(axis=1)
        df['_completeness_score'] = non_null_counts
        
        # Also consider data quality factors
        quality_bonus = 0
        
        # Bonus for having SMILES
        if 'canonical_smiles' in df.columns:
            quality_bonus += df['canonical_smiles'].notna().astype(int) * 10
        
        # Bonus for having clinical phase data
        if 'max_clinical_phase' in df.columns:
            quality_bonus += df['max_clinical_phase'].notna().astype(int) * 5
        
        # Bonus for having approval data
        if 'approval_date_first' in df.columns:
            quality_bonus += df['approval_date_first'].notna().astype(int) * 15
        
        # Bonus for having toxicity data
        tox_cols = [col for col in df.columns if col.startswith('tox_') and col.endswith('_risk')]
        if tox_cols:
            quality_bonus += df[tox_cols].notna().sum(axis=1) * 2
        
        df['_completeness_score'] += quality_bonus
        
        # Sort by ChEMBL ID and completeness score (descending)
        logger.info("🔄 Sorting by ChEMBL ID and completeness score...")
        df_sorted = df.sort_values(['chembl_id', '_completeness_score'], 
                                  ascending=[True, False])
        
        # Keep first (richest) row for each ChEMBL ID
        logger.info("✂️ Removing duplicates, keeping richest rows...")
        dedup_df = df_sorted.drop_duplicates(subset=['chembl_id'], keep='first')
        
        # Remove temporary scoring column
        dedup_df = dedup_df.drop(columns=['_completeness_score'])
        
        # Report results
        removed_count = len(df) - len(dedup_df)
        logger.info(f"✅ Deduplication complete:")
        logger.info(f"   Before: {len(df):,} rows")
        logger.info(f"   After: {len(dedup_df):,} rows")
        logger.info(f"   Removed: {removed_count:,} duplicate rows")
        logger.info(f"   Unique ChEMBL IDs: {dedup_df['chembl_id'].nunique():,}")
        
        # Validate no duplicates remain
        remaining_dupes = dedup_df['chembl_id'].duplicated().sum()
        if remaining_dupes > 0:
            logger.error(f"❌ Still have {remaining_dupes} duplicates after deduplication!")
            return None
        else:
            logger.info("✅ No duplicates remaining - validation passed")
        
        # Save deduplicated dataset
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            dedup_df.to_csv(output_file, index=False)
            logger.info(f"💾 Deduplicated dataset saved: {output_file}")
        except Exception as e:
            logger.error(f"❌ Error saving deduplicated dataset: {e}")
            return None
        
        return dedup_df
        
    except Exception as e:
        logger.error(f"❌ Error in deduplication: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_duplicates(df, chembl_col='chembl_id'):
    """
    Analyze duplicate patterns in the dataset
    
    Args:
        df: DataFrame to analyze
        chembl_col: Column name for ChEMBL ID
    """
    logger.info("🔍 ANALYZING DUPLICATE PATTERNS")
    logger.info("=" * 40)
    
    # Find duplicates
    duplicate_mask = df.duplicated(subset=[chembl_col], keep=False)
    duplicate_rows = df[duplicate_mask]
    
    if duplicate_rows.empty:
        logger.info("✅ No duplicates found")
        return
    
    # Group by ChEMBL ID to see duplicate sets
    duplicate_groups = duplicate_rows.groupby(chembl_col)
    
    logger.info(f"📊 Duplicate analysis:")
    logger.info(f"   Duplicate ChEMBL IDs: {duplicate_groups.ngroups:,}")
    logger.info(f"   Total duplicate rows: {len(duplicate_rows):,}")
    
    # Show examples of duplicate sets
    logger.info(f"\\n🔍 Sample duplicate sets:")
    
    sample_count = 0
    for chembl_id, group in duplicate_groups:
        if sample_count >= 5:  # Show first 5 examples
            break
        
        logger.info(f"\\n   ChEMBL ID: {chembl_id}")
        logger.info(f"   Duplicate count: {len(group)}")
        
        # Show key differences between duplicates
        key_cols = ['primary_drug', 'canonical_smiles', 'max_clinical_phase', 
                   'data_source', 'created_at']
        
        available_cols = [col for col in key_cols if col in group.columns]
        
        if available_cols:
            for idx, (_, row) in enumerate(group.iterrows()):
                logger.info(f"     Row {idx+1}:")
                for col in available_cols:
                    value = row[col]
                    if pd.isna(value):
                        value = "NULL"
                    elif isinstance(value, str) and len(str(value)) > 50:
                        value = str(value)[:47] + "..."
                    logger.info(f"       {col}: {value}")
        
        sample_count += 1
    
    if duplicate_groups.ngroups > 5:
        logger.info(f"   ... and {duplicate_groups.ngroups - 5} more duplicate sets")


def main():
    """Main execution"""
    logger.info("🔧 CHEMBL ID DEDUPLICATION")
    logger.info("📊 Removing duplicate ChEMBL IDs, keeping richest rows")
    
    # Load and analyze current dataset
    try:
        df = pd.read_csv("csv_exports/veridica_master_merged.csv")
        analyze_duplicates(df)
    except Exception as e:
        logger.error(f"Could not analyze duplicates: {e}")
    
    # Perform deduplication
    dedup_df = deduplicate_by_chembl_id()
    
    if dedup_df is not None:
        logger.info("\\n🎉 DEDUPLICATION COMPLETE")
        logger.info(f"📊 Clean dataset: {len(dedup_df):,} unique compounds")
        logger.info(f"📁 Saved to: csv_exports/veridica_master_merged.dedup.csv")
    else:
        logger.error("❌ Deduplication failed")


if __name__ == "__main__":
    main()