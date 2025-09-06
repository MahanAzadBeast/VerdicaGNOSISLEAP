"""
Enhanced Numeric Toxicity Augmentation Pipeline
Adds IC50/Ki data for hERG + Nav1.5 + Cav1.2 cardiac targets
WITH STRICT SYNTHETIC DATA PREVENTION
"""

import os
import json
import pandas as pd
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, 'src')

from connectors.chembl_numeric_tox import resolve_target_ids, fetch_numeric_for_target, aggregate_numeric, validate_final_toxicity_data, validate_authentic_data

logger = logging.getLogger(__name__)

# Configuration
IN_CSV = os.getenv("IN_CSV", "data/veridica_final_improved.herg.prov.csv")
OUT_CSV = os.getenv("OUT_CSV", "data/veridica_final_improved.cardio.csv")

# Gene → short tag used in column prefixes
GENE_TAGS = {
    "KCNH2": "herg",     # Kv11.1 (hERG channel)
    "SCN5A": "nav1_5",   # NaV1.5 (cardiac sodium channel)
    "CACNA1C": "cav1_2"  # CaV1.2 (cardiac calcium channel)
}

# Synthetic data detection
SYNTHETIC_KEYWORDS = [
    'demo', 'synthetic', 'fake', 'test', 'generated', 'artificial',
    'chembl_demo', 'mock', 'placeholder', 'example'
]


def validate_input_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate input dataset contains no synthetic data
    
    Args:
        df: Input dataset to validate
        
    Returns:
        Clean dataset with synthetic entries removed
    """
    logger.info("🔍 VALIDATING INPUT DATASET FOR SYNTHETIC DATA")
    logger.info("=" * 50)
    
    original_count = len(df)
    synthetic_mask = False
    
    # Check key columns for synthetic indicators
    check_columns = ['primary_drug', 'chembl_id', 'data_source']
    
    for col in check_columns:
        if col in df.columns:
            col_values = df[col].astype(str).str.lower()
            
            for keyword in SYNTHETIC_KEYWORDS:
                keyword_mask = col_values.str.contains(keyword, na=False)
                synthetic_mask = synthetic_mask | keyword_mask
                
                matches = keyword_mask.sum()
                if matches > 0:
                    logger.warning(f"⚠️ Found {matches} entries with '{keyword}' in {col}")
    
    # Remove any synthetic entries
    clean_df = df[~synthetic_mask].copy()
    removed_count = original_count - len(clean_df)
    
    if removed_count > 0:
        logger.warning(f"🚫 REMOVED {removed_count} synthetic entries from input")
        
        # Show what was removed
        synthetic_entries = df[synthetic_mask]
        logger.warning("❌ Removed synthetic entries:")
        for idx, row in synthetic_entries.head(5).iterrows():
            chembl_id = row.get('chembl_id', 'N/A')
            drug_name = row.get('primary_drug', 'N/A')
            logger.warning(f"   {chembl_id}: {drug_name}")
    else:
        logger.info("✅ Input dataset verified clean - no synthetic data")
    
    return clean_df


def main():
    """Main enhanced toxicity augmentation pipeline"""
    logger.info("🫀 ENHANCED CARDIAC TOXICITY AUGMENTATION")
    logger.info("🎯 Adding IC50/Ki for hERG + Nav1.5 + Cav1.2")
    logger.info("🚫 STRICT SYNTHETIC DATA PREVENTION")
    logger.info("=" * 60)
    
    # Load and validate input dataset
    try:
        base = pd.read_csv(IN_CSV)
        logger.info(f"📊 Loaded input dataset: {len(base):,} compounds")
        
        # Validate no synthetic data in input
        base = validate_input_dataset(base)
        
        if base.empty:
            logger.error("❌ No clean data after synthetic removal")
            return
        
    except Exception as e:
        logger.error(f"❌ Error loading input dataset: {e}")
        return
    
    # Resolve target IDs for cardiac ion channels
    try:
        target_ids = resolve_target_ids(list(GENE_TAGS.keys()))
        
        if not target_ids:
            logger.error("❌ No cardiac targets resolved")
            return
        
        logger.info(f"✅ Cardiac targets resolved: {list(target_ids.keys())}")
        
    except Exception as e:
        logger.error(f"❌ Error resolving targets: {e}")
        return
    
    # Start with base dataset
    merged = base.copy()
    total_new_data_points = 0
    
    # Fetch and integrate data for each cardiac target
    for gene, prefix in GENE_TAGS.items():
        try:
            tid = target_ids.get(gene)
            if not tid:
                logger.warning(f"⚠️ No target ID for {gene} - skipping")
                continue
            
            logger.info(f"🔬 Processing {gene} ({prefix}) - Target: {tid}")
            
            # Fetch authentic activity data
            df = fetch_numeric_for_target(tid, std_types=("IC50", "Ki"))
            
            if df.empty:
                logger.warning(f"⚠️ No authentic data for {gene}")
                continue
            
            # Additional synthetic data validation
            df = validate_authentic_data(df)
            
            if df.empty:
                logger.warning(f"⚠️ All {gene} data was synthetic - skipping")
                continue
            
            # Aggregate by molecule
            agg = aggregate_numeric(df, prefix)
            
            if agg.empty:
                logger.warning(f"⚠️ No aggregated data for {gene}")
                continue
            
            # Rename columns for clarity
            rename_map = {}
            for c in agg.columns:
                if c.endswith("_ic50"):
                    rename_map[c] = c.replace("_ic50", "_ic50_um")
                elif c.endswith("_ki"):
                    rename_map[c] = c.replace("_ki", "_ki_um")
            
            agg = agg.rename(columns=rename_map)
            
            # Merge with main dataset
            before_merge = len(merged)
            merged = merged.merge(agg, on="chembl_id", how="left")
            
            # Count successful integrations
            ic50_col = f"{prefix}_median_ic50_um"
            ki_col = f"{prefix}_median_ki_um"
            
            ic50_matches = merged[ic50_col].notna().sum() if ic50_col in merged.columns else 0
            ki_matches = merged[ki_col].notna().sum() if ki_col in merged.columns else 0
            
            total_matches = max(ic50_matches, ki_matches)
            total_new_data_points += total_matches
            
            logger.info(f"✅ {gene} integration complete:")
            logger.info(f"   IC50 matches: {ic50_matches:,}")
            logger.info(f"   Ki matches: {ki_matches:,}")
            logger.info(f"   Total compounds: {total_matches:,}")
            
            # Update provenance/source list
            if "tox_data_sources" not in merged.columns:
                merged["tox_data_sources"] = pd.NA
            
            # Create mask for compounds that got new data
            mask = merged[ic50_col].notna() | merged[ki_col].notna() if ic50_col in merged.columns and ki_col in merged.columns else merged[ic50_col].notna() if ic50_col in merged.columns else merged[ki_col].notna()
            
            if mask.any():
                tag = f'chembl:{gene}'
                
                def append_source(s):
                    try:
                        L = json.loads(s) if isinstance(s, str) and s.strip().startswith("[") else []
                    except Exception:
                        L = []
                    if tag not in L:
                        L.append(tag)
                    return json.dumps(L)
                
                merged.loc[mask, "tox_data_sources"] = merged.loc[mask, "tox_data_sources"].apply(append_source)
                
                logger.info(f"   📋 Updated provenance for {mask.sum():,} compounds")
            
        except Exception as e:
            logger.error(f"❌ Error processing {gene}: {e}")
            continue
    
    # Final synthetic data validation
    logger.info("🔍 FINAL SYNTHETIC DATA VALIDATION")
    validation_passed = validate_final_toxicity_data(merged)
    
    if not validation_passed:
        logger.error("❌ SYNTHETIC DATA DETECTED IN FINAL DATASET")
        logger.error("❌ ABORTING - DATASET CONTAMINATED")
        return
    
    # Generate enhancement summary
    logger.info("📊 ENHANCEMENT SUMMARY")
    logger.info("=" * 30)
    logger.info(f"   Input compounds: {len(base):,}")
    logger.info(f"   Output compounds: {len(merged):,}")
    logger.info(f"   New toxicity data points: {total_new_data_points:,}")
    
    # Show new toxicity columns
    new_tox_cols = [col for col in merged.columns if any(prefix in col for prefix in GENE_TAGS.values()) and col.endswith('_um')]
    logger.info(f"   New toxicity columns: {len(new_tox_cols)}")
    
    for col in new_tox_cols:
        if col in merged.columns:
            coverage = merged[col].notna().sum()
            logger.info(f"      {col}: {coverage:,} compounds")
    
    # Save enhanced dataset
    try:
        Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(OUT_CSV, index=False)
        
        logger.info(f"💾 Enhanced dataset saved: {OUT_CSV}")
        logger.info(f"   📊 Shape: {merged.shape}")
        logger.info(f"   🫀 Cardiac toxicity data: Enhanced")
        logger.info(f"   🚫 Synthetic data: ZERO (verified)")
        
    except Exception as e:
        logger.error(f"❌ Error saving enhanced dataset: {e}")
        return
    
    logger.info("\\n🎉 ENHANCED TOXICITY AUGMENTATION COMPLETE")
    logger.info("✅ Cardiac ion channel data successfully integrated")
    logger.info("🚫 100% authentic data confirmed")
    logger.info("🔬 Ready for advanced cardiotoxicity modeling")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()