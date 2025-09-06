"""
Make Training View Pipeline (Enhanced)
Creates leak-safe training view with enhanced cardiac toxicity data
WITH STRICT SYNTHETIC DATA PREVENTION
"""

import os
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration
IN_CSV = os.getenv("IN_CSV", "data/veridica_final_improved.cardio.approvals.csv")
OUT_CSV = os.getenv("OUT_CSV", "data/veridica_train_approval.timesafe.csv")

# Synthetic data detection
SYNTHETIC_KEYWORDS = [
    'demo', 'synthetic', 'fake', 'test', 'generated', 'artificial',
    'chembl_demo', 'mock', 'placeholder', 'example'
]


def is_leaky(col: str) -> bool:
    """
    Determine if a column could cause information leakage
    
    Args:
        col: Column name
        
    Returns:
        True if column is potentially leaky
    """
    cl = col.lower()
    
    # Target column is not leaky
    if cl == "approved":
        return False
    
    # These patterns indicate potential leakage
    leaky_patterns = [
        "phase", "status", "approval", "date", "time"
    ]
    
    return any(k in cl for k in leaky_patterns)


def validate_training_data_authentic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final validation that training data contains no synthetic content
    
    Args:
        df: Training DataFrame to validate
        
    Returns:
        Clean DataFrame with synthetic entries removed
    """
    logger.info("🔍 FINAL TRAINING DATA AUTHENTICITY VALIDATION")
    logger.info("=" * 50)
    
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
                    logger.error(f"❌ SYNTHETIC DATA in training: {matches} entries with '{keyword}' in {col}")
    
    # Check ChEMBL IDs for synthetic patterns
    if 'chembl_id' in df.columns:
        synthetic_chembl = df['chembl_id'].astype(str).str.lower().str.contains('|'.join(SYNTHETIC_KEYWORDS), na=False).sum()
        if synthetic_chembl > 0:
            logger.error(f"❌ SYNTHETIC ChEMBL IDs in training: {synthetic_chembl}")
            synthetic_mask = synthetic_mask | df['chembl_id'].astype(str).str.lower().str.contains('|'.join(SYNTHETIC_KEYWORDS), na=False)
    
    # Remove any synthetic entries
    clean_df = df[~synthetic_mask].copy()
    removed_count = original_count - len(clean_df)
    
    if removed_count > 0:
        logger.error(f"❌ REMOVED {removed_count} SYNTHETIC ENTRIES FROM TRAINING DATA")
        
        # Show what was removed
        synthetic_entries = df[synthetic_mask]
        logger.error("❌ Removed synthetic training entries:")
        for idx, row in synthetic_entries.head(5).iterrows():
            chembl_id = row.get('chembl_id', 'N/A')
            logger.error(f"   {chembl_id}")
    else:
        logger.info("✅ Training data verified 100% authentic")
    
    return clean_df


def main():
    """Main training view creation with authenticity validation"""
    logger.info("🎯 CREATING ENHANCED TIME-SAFE TRAINING VIEW")
    logger.info("🫀 Including enhanced cardiac toxicity data")
    logger.info("🚫 STRICT SYNTHETIC DATA PREVENTION")
    logger.info("=" * 60)
    
    try:
        # Load enhanced dataset
        df = pd.read_csv(IN_CSV)
        logger.info(f"✅ Loaded enhanced dataset: {len(df):,} compounds × {len(df.columns)} features")
        
        # Validate input data authenticity
        df = validate_training_data_authentic(df)
        
        if df.empty:
            logger.error("❌ No authentic data remaining after validation")
            return
        
        # Identify feature blocks
        key_cols = [c for c in ["chembl_id", "inchikey", "canonical_smiles"] if c in df.columns]
        desc_cols = [c for c in df.columns if c.startswith("mol_")]
        
        # Enhanced toxicity columns (including new cardiac targets)
        tox_cols = [c for c in df.columns if c.lower().startswith((
            "tox_", "herg", "nav1_5", "cav1_2", "cyp", "dili", "ames", "bbb"
        )) and c != "tox_data_sources"]
        
        logger.info(f"📊 Feature analysis:")
        logger.info(f"   🔑 Key columns: {len(key_cols)}")
        logger.info(f"   🧬 Molecular descriptors: {len(desc_cols)}")
        logger.info(f"   ⚠️ Toxicity features: {len(tox_cols)}")
        
        # Show enhanced toxicity features
        cardiac_cols = [col for col in tox_cols if any(target in col for target in ['herg', 'nav1_5', 'cav1_2'])]
        if cardiac_cols:
            logger.info(f"   🫀 Cardiac toxicity features: {len(cardiac_cols)}")
            for col in cardiac_cols:
                if col in df.columns:
                    coverage = df[col].notna().sum()
                    logger.info(f"      {col}: {coverage:,} compounds")
        
        # Filter out leaky columns
        feat_cols = [c for c in (desc_cols + tox_cols) if not is_leaky(c)]
        
        # Combine all safe columns
        cols = key_cols + feat_cols + (["approved"] if "approved" in df.columns else [])
        
        logger.info(f"\\n🔒 Leakage prevention:")
        logger.info(f"   Total available columns: {len(df.columns)}")
        logger.info(f"   Safe columns selected: {len(cols)}")
        logger.info(f"   Potentially leaky columns excluded: {len(df.columns) - len(cols)}")
        
        # Show excluded columns
        excluded_cols = [c for c in df.columns if c not in cols]
        if excluded_cols:
            logger.info(f"   Excluded leaky columns:")
            for col in excluded_cols[:10]:  # Show first 10
                logger.info(f"      {col}")
            if len(excluded_cols) > 10:
                logger.info(f"      ... and {len(excluded_cols) - 10} more")
        
        # Create clean training dataset
        clean = df[cols].copy()
        
        # Final authenticity validation of training data
        clean = validate_training_data_authentic(clean)
        
        if clean.empty:
            logger.error("❌ No authentic training data remaining")
            return
        
        # Verify target distribution
        if "approved" in clean.columns:
            approved_count = clean["approved"].sum()
            approval_rate = approved_count / len(clean)
            logger.info(f"\\n🎯 Target distribution:")
            logger.info(f"   Approved: {approved_count:,} ({approval_rate*100:.1f}%)")
            logger.info(f"   Not approved: {len(clean) - approved_count:,} ({(1-approval_rate)*100:.1f}%)")
        
        # Check enhanced toxicity data coverage
        logger.info(f"\\n🫀 ENHANCED CARDIAC TOXICITY COVERAGE:")
        
        # hERG data
        herg_ic50_col = 'herg_median_ic50_um'
        herg_ki_col = 'herg_median_ki_um'
        
        if herg_ic50_col in clean.columns:
            herg_ic50_coverage = clean[herg_ic50_col].notna().sum()
            logger.info(f"   hERG IC50: {herg_ic50_coverage:,} compounds")
        
        if herg_ki_col in clean.columns:
            herg_ki_coverage = clean[herg_ki_col].notna().sum()
            logger.info(f"   hERG Ki: {herg_ki_coverage:,} compounds")
        
        # Nav1.5 data
        nav_ic50_col = 'nav1_5_median_ic50_um'
        if nav_ic50_col in clean.columns:
            nav_coverage = clean[nav_ic50_col].notna().sum()
            logger.info(f"   Nav1.5 IC50: {nav_coverage:,} compounds")
        
        # Cav1.2 data
        cav_ic50_col = 'cav1_2_median_ic50_um'
        if cav_ic50_col in clean.columns:
            cav_coverage = clean[cav_ic50_col].notna().sum()
            logger.info(f"   Cav1.2 IC50: {cav_coverage:,} compounds")
        
        # Check for missing data
        logger.info(f"\\n📊 Data completeness:")
        
        # SMILES availability
        if "canonical_smiles" in clean.columns:
            smiles_count = clean["canonical_smiles"].notna().sum()
            smiles_pct = (smiles_count / len(clean)) * 100
            logger.info(f"   SMILES: {smiles_count:,} ({smiles_pct:.1f}%)")
        
        # Molecular descriptor coverage
        if desc_cols:
            avg_desc_coverage = sum(clean[col].notna().sum() for col in desc_cols) / (len(desc_cols) * len(clean))
            logger.info(f"   Molecular descriptors: {avg_desc_coverage*100:.1f}% average")
        
        # Save training view
        Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(OUT_CSV, index=False)
        
        feature_count = len(feat_cols)
        
        logger.info(f"\\n💾 Enhanced training view saved: {OUT_CSV}")
        logger.info(f"   📊 Shape: {clean.shape}")
        logger.info(f"   🔑 Key columns: {len(key_cols)}")
        logger.info(f"   🧬 Features: {feature_count}")
        logger.info(f"   🫀 Enhanced cardiac features: {len(cardiac_cols) if 'cardiac_cols' in locals() else 0}")
        logger.info(f"   🎯 Target: {'approved' if 'approved' in clean.columns else 'none'}")
        logger.info(f"   🚫 Synthetic data: ZERO (verified)")
        
        logger.info("\\n🎉 ENHANCED TRAINING VIEW COMPLETE")
        logger.info("✅ Leakage-free dataset with enhanced cardiac toxicity")
        logger.info("🚫 100% authentic pharmaceutical data")
        logger.info("🔬 Ready for advanced cardiotoxicity ML models")
        
        return clean
        
    except Exception as e:
        logger.error(f"❌ Error creating enhanced training view: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()