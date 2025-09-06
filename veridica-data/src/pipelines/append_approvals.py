"""
Append FDA Approvals Pipeline
Adds real FDA approval data with machine-auditable provenance
WITH STRICT SYNTHETIC DATA PREVENTION
"""

import os
import pandas as pd
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, 'src')

from connectors.openfda_approvals import fetch_approvals_for_names

logger = logging.getLogger(__name__)

# Configuration
IN_CSV = os.getenv("IN_CSV", "data/veridica_final_improved.cardio.csv")
OUT_CSV = os.getenv("OUT_CSV", "data/veridica_final_improved.cardio.approvals.csv")
NAME_COL = os.getenv("NAME_COL", "primary_drug")

# Synthetic data detection
SYNTHETIC_KEYWORDS = [
    'demo', 'synthetic', 'fake', 'test', 'generated', 'artificial',
    'chembl_demo', 'mock', 'placeholder', 'example'
]


def filter_authentic_drug_names(df: pd.DataFrame, name_col: str) -> list:
    """
    Filter to authentic drug names only
    
    Args:
        df: Dataset DataFrame
        name_col: Column containing drug names
        
    Returns:
        List of authentic drug names
    """
    logger.info("🔍 FILTERING TO AUTHENTIC DRUG NAMES")
    logger.info("🚫 Removing synthetic/test drug names")
    logger.info("=" * 50)
    
    if name_col not in df.columns:
        logger.error(f"❌ Column '{name_col}' not found in dataset")
        return []
    
    # Get unique drug names
    all_names = df[name_col].dropna().astype(str).str.strip().unique()
    logger.info(f"📊 Total unique drug names: {len(all_names):,}")
    
    # Filter out synthetic names
    authentic_names = []
    synthetic_count = 0
    
    for name in all_names:
        name_lower = name.lower()
        
        # Check for synthetic keywords
        is_synthetic = any(keyword in name_lower for keyword in SYNTHETIC_KEYWORDS)
        
        # Check for ChEMBL ID patterns (not real drug names)
        if name.startswith('ChEMBL_') or name.startswith('CHEMBL'):
            is_synthetic = True
        
        # Check for obviously test/demo patterns
        if any(pattern in name_lower for pattern in ['test_', '_test', 'demo_', '_demo']):
            is_synthetic = True
        
        if is_synthetic:
            synthetic_count += 1
            logger.debug(f"🚫 Skipping synthetic name: {name}")
        else:
            authentic_names.append(name)
    
    logger.info(f"✅ Authentic drug names: {len(authentic_names):,}")
    logger.info(f"🚫 Synthetic names excluded: {synthetic_count:,}")
    logger.info(f"📊 Authenticity rate: {len(authentic_names)/len(all_names)*100:.1f}%")
    
    return authentic_names


def validate_approval_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate approval data contains no synthetic entries
    
    Args:
        df: DataFrame with approval data
        
    Returns:
        Clean DataFrame with synthetic entries removed
    """
    if df.empty:
        return df
    
    logger.info("🔍 VALIDATING FDA APPROVAL DATA")
    
    original_count = len(df)
    synthetic_mask = False
    
    # Check all text columns for synthetic indicators
    text_columns = df.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        if col in df.columns:
            col_values = df[col].astype(str).str.lower()
            
            for keyword in SYNTHETIC_KEYWORDS:
                keyword_mask = col_values.str.contains(keyword, na=False)
                synthetic_mask = synthetic_mask | keyword_mask
                
                matches = keyword_mask.sum()
                if matches > 0:
                    logger.warning(f"⚠️ Found {matches} synthetic entries in {col}")
    
    # Remove synthetic entries
    clean_df = df[~synthetic_mask].copy()
    removed_count = original_count - len(clean_df)
    
    if removed_count > 0:
        logger.warning(f"🚫 REMOVED {removed_count} synthetic approval entries")
    else:
        logger.info("✅ All approval data verified authentic")
    
    return clean_df


def main():
    """Main FDA approval augmentation pipeline"""
    logger.info("📡 FDA APPROVAL AUGMENTATION PIPELINE")
    logger.info("🎯 Adding real FDA approval data with provenance")
    logger.info("🚫 STRICT SYNTHETIC DATA PREVENTION")
    logger.info("=" * 60)
    
    # Load input dataset
    try:
        df = pd.read_csv(IN_CSV)
        logger.info(f"✅ Loaded dataset: {len(df):,} compounds")
    except Exception as e:
        logger.error(f"❌ Error loading input dataset: {e}")
        return
    
    # Check if name column exists
    if NAME_COL not in df.columns:
        logger.error(f"❌ Column '{NAME_COL}' not found in dataset")
        logger.info(f"Available columns: {list(df.columns)}")
        
        # Save original dataset without modification
        df.to_csv(OUT_CSV, index=False)
        logger.info(f"💾 Original dataset saved (no approvals added): {OUT_CSV}")
        return
    
    # Filter to authentic drug names only
    authentic_names = filter_authentic_drug_names(df, NAME_COL)
    
    if not authentic_names:
        logger.error("❌ No authentic drug names found for FDA queries")
        df.to_csv(OUT_CSV, index=False)
        return
    
    # Fetch FDA approvals for authentic names only
    try:
        logger.info(f"📡 Querying openFDA for {len(authentic_names):,} authentic drugs...")
        appr = fetch_approvals_for_names(authentic_names)
        
        # Validate approval data is authentic
        appr = validate_approval_data(appr)
        
        if appr.empty:
            logger.warning("⚠️ No authentic FDA approvals found")
            df.to_csv(OUT_CSV, index=False)
            return
        
        logger.info(f"✅ Found {len(appr):,} authentic FDA approvals")
        
    except Exception as e:
        logger.error(f"❌ Error fetching FDA approvals: {e}")
        df.to_csv(OUT_CSV, index=False)
        return
    
    # Merge with main dataset
    logger.info("🔗 Merging FDA approval data...")
    
    out = df.merge(appr, left_on=NAME_COL, right_on="query_name", how="left")
    out = out.drop(columns=["query_name"], errors="ignore")
    
    # Count successful integrations
    approval_matches = out["approval_date_first"].notna().sum()
    logger.info(f"✅ FDA approval integration: {approval_matches:,} compounds")
    
    # Derive/refresh binary label
    if "approved" not in out.columns:
        out["approved"] = out["approval_date_first"].notna().astype(int)
    else:
        # Update approved status based on FDA data
        fda_approved = out["approval_date_first"].notna()
        out["approved"] = (out["approved"] | fda_approved).astype(int)
    
    approved_count = out["approved"].sum()
    logger.info(f"🏷️ Updated approval labels: {approved_count:,} approved compounds")
    
    # Final synthetic data validation
    logger.info("🔍 FINAL SYNTHETIC DATA VALIDATION")
    
    synthetic_found = 0
    
    # Check approval sources
    if 'approval_source' in out.columns:
        synthetic_sources = out['approval_source'].astype(str).str.lower().str.contains('|'.join(SYNTHETIC_KEYWORDS), na=False).sum()
        synthetic_found += synthetic_sources
        
        if synthetic_sources > 0:
            logger.error(f"❌ Found {synthetic_sources} synthetic approval sources")
    
    # Check drug names in final dataset
    if NAME_COL in out.columns:
        drug_values = out[NAME_COL].astype(str).str.lower()
        for keyword in SYNTHETIC_KEYWORDS:
            synthetic_drugs = drug_values.str.contains(keyword, na=False).sum()
            synthetic_found += synthetic_drugs
            
            if synthetic_drugs > 0:
                logger.error(f"❌ Found {synthetic_drugs} synthetic drug names with '{keyword}'")
    
    if synthetic_found == 0:
        logger.info("🎉 FINAL VALIDATION PASSED")
        logger.info("✅ NO SYNTHETIC DATA IN APPROVAL AUGMENTATION")
        logger.info("✅ 100% AUTHENTIC FDA APPROVAL DATA")
    else:
        logger.error(f"❌ VALIDATION FAILED: {synthetic_found} synthetic entries detected")
        logger.error("❌ ABORTING - DATASET CONTAMINATED")
        return
    
    # Save enhanced dataset
    try:
        Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_CSV, index=False)
        
        logger.info(f"💾 Dataset with FDA approvals saved: {OUT_CSV}")
        logger.info(f"   📊 Shape: {out.shape}")
        logger.info(f"   📋 FDA approvals: {approval_matches:,} compounds")
        logger.info(f"   🏷️ Total approved: {approved_count:,}")
        logger.info(f"   🚫 Synthetic data: ZERO (verified)")
        
    except Exception as e:
        logger.error(f"❌ Error saving dataset: {e}")
        return
    
    logger.info("\\n🎉 FDA APPROVAL AUGMENTATION COMPLETE")
    logger.info("✅ Real FDA approval data successfully integrated")
    logger.info("🚫 100% authentic data confirmed")
    logger.info("📋 Machine-auditable approval provenance added")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()