"""
Make Training View Pipeline
Creates clean training view with leakage prevention
"""

import os
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration
IN_CSV = os.environ.get("IN_CSV", "data/veridica_final_improved.herg.prov.csv")
OUT_CSV = os.environ.get("OUT_CSV", "data/veridica_train_approval.timesafe.csv")


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
        "phase", "status", "approval", "date", "time",
        "trial_", "clinical_", "_recent", "last_", "current_"
    ]
    
    return any(pattern in cl for pattern in leaky_patterns)


def main():
    """Main training view creation"""
    logger.info("🎯 CREATING TIME-SAFE TRAINING VIEW")
    logger.info("🔒 Excluding leaky columns for approval prediction")
    logger.info("=" * 60)
    
    try:
        # Load dataset with hERG and provenance
        df = pd.read_csv(IN_CSV)
        logger.info(f"✅ Loaded dataset: {len(df):,} compounds × {len(df.columns)} features")
        
        # Identify feature blocks
        desc_cols = [c for c in df.columns if c.startswith("mol_")]
        tox_cols = [c for c in df.columns if c.lower().startswith(("tox_", "herg", "cyp", "dili", "ames", "bbb")) and c != "tox_data_sources"]
        key_cols = [c for c in ["chembl_id", "inchikey", "canonical_smiles"] if c in df.columns]
        
        logger.info(f"📊 Feature analysis:")
        logger.info(f"   🔑 Key columns: {len(key_cols)}")
        logger.info(f"   🧬 Molecular descriptors: {len(desc_cols)}")
        logger.info(f"   ⚠️ Toxicity features: {len(tox_cols)}")
        
        # Show toxicity features
        logger.info(f"   Toxicity features included:")
        for col in tox_cols:
            if col in df.columns:
                coverage = df[col].notna().sum()
                logger.info(f"      {col}: {coverage:,} ({coverage/len(df)*100:.1f}%)")
        
        # Filter out leaky columns
        safe_cols = key_cols + [c for c in (desc_cols + tox_cols) if not is_leaky(c)]
        
        # Add target if available
        if "approved" in df.columns:
            safe_cols.append("approved")
        
        logger.info(f"\\n🔒 Leakage prevention:")
        logger.info(f"   Total available columns: {len(df.columns)}")
        logger.info(f"   Safe columns selected: {len(safe_cols)}")
        logger.info(f"   Potentially leaky columns excluded: {len(df.columns) - len(safe_cols)}")
        
        # Show excluded columns
        excluded_cols = [c for c in df.columns if c not in safe_cols]
        if excluded_cols:
            logger.info(f"   Excluded columns:")
            for col in excluded_cols[:10]:  # Show first 10
                logger.info(f"      {col}")
            if len(excluded_cols) > 10:
                logger.info(f"      ... and {len(excluded_cols) - 10} more")
        
        # Create clean training dataset
        clean = df[safe_cols].copy()
        
        # Verify target distribution
        if "approved" in clean.columns:
            approved_count = clean["approved"].sum()
            approval_rate = approved_count / len(clean)
            logger.info(f"\\n🎯 Target distribution:")
            logger.info(f"   Approved: {approved_count:,} ({approval_rate*100:.1f}%)")
            logger.info(f"   Not approved: {len(clean) - approved_count:,} ({(1-approval_rate)*100:.1f}%)")
        
        # Check for missing data
        logger.info(f"\\n📊 Data completeness:")
        
        # SMILES coverage
        if "canonical_smiles" in clean.columns:
            smiles_coverage = clean["canonical_smiles"].notna().sum()
            logger.info(f"   SMILES: {smiles_coverage:,}/{len(clean):,} ({smiles_coverage/len(clean)*100:.1f}%)")
        
        # Molecular descriptor coverage
        if desc_cols:
            avg_desc_coverage = sum(clean[col].notna().sum() for col in desc_cols) / (len(desc_cols) * len(clean))
            logger.info(f"   Molecular descriptors: {avg_desc_coverage*100:.1f}% average")
        
        # hERG IC50 coverage
        if "tox_herg_ic50_uM" in clean.columns:
            herg_coverage = clean["tox_herg_ic50_uM"].notna().sum()
            logger.info(f"   hERG IC50 (numeric): {herg_coverage:,} ({herg_coverage/len(clean)*100:.1f}%)")
        
        # Save training view
        Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(OUT_CSV, index=False)
        
        feature_count = len(safe_cols) - len(key_cols) - (1 if "approved" in clean.columns else 0)
        
        logger.info(f"\\n💾 Training view saved: {OUT_CSV}")
        logger.info(f"   📊 Shape: {clean.shape}")
        logger.info(f"   🔑 Key columns: {len(key_cols)}")
        logger.info(f"   🧬 Features: {feature_count}")
        logger.info(f"   🎯 Target: {'approved' if 'approved' in clean.columns else 'none'}")
        
        logger.info("\\n🎉 TIME-SAFE TRAINING VIEW COMPLETE")
        logger.info("✅ Leakage-free dataset for approval prediction")
        logger.info("🔬 Ready for machine learning training")
        
        return clean
        
    except Exception as e:
        logger.error(f"❌ Error creating training view: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()