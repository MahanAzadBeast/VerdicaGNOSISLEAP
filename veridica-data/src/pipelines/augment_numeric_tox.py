"""
Augment Numeric Toxicity Pipeline
Adds numeric hERG IC50 data from ChEMBL to the dataset
"""

import os
import json
import pandas as pd
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, 'src')

from connectors.chembl_herg_ic50 import fetch_herg_ic50

logger = logging.getLogger(__name__)

# Configuration
IN_CSV = os.environ.get("IN_CSV", "data/veridica_final_improved.csv")
OUT_CSV = os.environ.get("OUT_CSV", "data/veridica_final_improved.herg.csv")
CACHE = os.environ.get("CACHE", "data/cache/chembl_herg.parquet")


def append_source_list(existing, new_item):
    """
    Append new item to existing source list
    
    Args:
        existing: Existing source list (JSON string, list, or NaN)
        new_item: New source to add
        
    Returns:
        Updated JSON string with new item
    """
    try:
        # existing may be JSON list or stringified list or NaN
        if pd.isna(existing):
            return json.dumps([new_item])
        
        if isinstance(existing, list):
            L = list(existing)
        else:
            s = str(existing).strip()
            if s.startswith("["):
                L = json.loads(s)
            else:
                L = []
        
        if new_item not in L:
            L.append(new_item)
            
        return json.dumps(L)
        
    except Exception as e:
        logger.debug(f"Error parsing source list: {e}")
        return json.dumps([new_item])


def main():
    """Main pipeline execution"""
    logger.info("🧬 AUGMENTING DATASET WITH NUMERIC hERG IC50 DATA")
    logger.info("🎯 Adding machine-auditable toxicity surrogates")
    logger.info("=" * 60)
    
    # Load input dataset
    try:
        df = pd.read_csv(IN_CSV)
        logger.info(f"✅ Loaded input dataset: {len(df):,} compounds")
    except Exception as e:
        logger.error(f"❌ Error loading input dataset: {e}")
        return
    
    # Fetch hERG IC50 data from ChEMBL
    try:
        herg = fetch_herg_ic50(cache_path=CACHE)
        logger.info(f"✅ Fetched hERG data: {len(herg):,} molecules")
    except Exception as e:
        logger.error(f"❌ Error fetching hERG data: {e}")
        return
    
    if herg.empty:
        logger.error("❌ No hERG data available - cannot proceed")
        return
    
    # Merge hERG data with main dataset
    logger.info("🔗 Merging hERG IC50 data with main dataset...")
    
    out = df.merge(herg, on="chembl_id", how="left")
    
    # Count matches
    herg_matches = out["tox_herg_ic50_uM"].notna().sum()
    logger.info(f"✅ hERG IC50 matches: {herg_matches:,}/{len(out):,} ({herg_matches/len(out)*100:.1f}%)")
    
    # Update tox_data_sources: append chembl source for rows that got hERG
    logger.info("📋 Updating toxicity data sources...")
    
    if "tox_data_sources" not in out.columns:
        out["tox_data_sources"] = pd.NA
    
    mask = out["tox_herg_ic50_uM"].notna()
    
    out.loc[mask, "tox_data_sources"] = out.loc[mask, "tox_data_sources"].apply(
        lambda x: append_source_list(x, "chembl:KCNH2_IC50")
    )
    
    updated_sources = out.loc[mask, "tox_data_sources"].notna().sum()
    logger.info(f"✅ Updated data sources: {updated_sources:,} compounds")
    
    # Generate hERG IC50 statistics
    if herg_matches > 0:
        herg_values = out["tox_herg_ic50_uM"].dropna()
        logger.info(f"📊 hERG IC50 statistics (µM):")
        logger.info(f"   Count: {len(herg_values)}")
        logger.info(f"   Mean: {herg_values.mean():.2f}")
        logger.info(f"   Median: {herg_values.median():.2f}")
        logger.info(f"   Range: {herg_values.min():.2f} - {herg_values.max():.2f}")
        
        # hERG risk classification based on IC50 values
        # Standard thresholds: <1µM = high risk, 1-10µM = medium risk, >10µM = low risk
        high_risk = (out["tox_herg_ic50_uM"] < 1).sum()
        medium_risk = ((out["tox_herg_ic50_uM"] >= 1) & (out["tox_herg_ic50_uM"] <= 10)).sum()
        low_risk = (out["tox_herg_ic50_uM"] > 10).sum()
        
        logger.info(f"📊 hERG risk distribution (based on IC50):")
        logger.info(f"   High risk (<1µM): {high_risk:,}")
        logger.info(f"   Medium risk (1-10µM): {medium_risk:,}")
        logger.info(f"   Low risk (>10µM): {low_risk:,}")
        
        # Update categorical hERG risk based on numeric data where available
        logger.info("🔄 Updating categorical hERG risk based on numeric IC50...")
        
        numeric_mask = out["tox_herg_ic50_uM"].notna()
        
        # Update risk categories based on IC50 values
        out.loc[numeric_mask & (out["tox_herg_ic50_uM"] < 1), "tox_herg_risk"] = "high"
        out.loc[numeric_mask & (out["tox_herg_ic50_uM"] >= 1) & (out["tox_herg_ic50_uM"] <= 10), "tox_herg_risk"] = "medium"
        out.loc[numeric_mask & (out["tox_herg_ic50_uM"] > 10), "tox_herg_risk"] = "low"
        
        updated_risk = numeric_mask.sum()
        logger.info(f"✅ Updated categorical risk for {updated_risk:,} compounds based on IC50 data")
    
    # Save augmented dataset
    try:
        Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_CSV, index=False)
        logger.info(f"💾 Augmented dataset saved: {OUT_CSV}")
        logger.info(f"   📊 {len(out):,} compounds")
        logger.info(f"   🫀 {herg_matches:,} with numeric hERG IC50")
        logger.info(f"   📋 Updated toxicity provenance")
        
    except Exception as e:
        logger.error(f"❌ Error saving augmented dataset: {e}")
        return
    
    logger.info("\\n🎉 NUMERIC TOXICITY AUGMENTATION COMPLETE")
    logger.info("✅ hERG IC50 data successfully integrated")
    logger.info("📋 Toxicity provenance updated")
    logger.info("🔬 Ready for machine-auditable toxicity modeling")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()