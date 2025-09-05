"""
ChEMBL hERG IC50 Connector
Fetches numeric hERG IC50 activities from ChEMBL KCNH2 target
"""

from chembl_webresource_client.new_client import new_client
import pandas as pd
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# KCNH2 (hERG) target in ChEMBL is CHEMBL240
HERG_TARGET = "CHEMBL240"


def fetch_herg_ic50(cache_path: str | None = None) -> pd.DataFrame:
    """
    Fetch hERG IC50 activities from ChEMBL and return median per molecule in µM.
    
    Args:
        cache_path: Optional path to cache results
        
    Returns:
        DataFrame with chembl_id, tox_herg_ic50_uM, tox_herg_n_points
    """
    logger.info("🫀 FETCHING hERG IC50 DATA FROM CHEMBL")
    logger.info(f"🎯 Target: KCNH2 (ChEMBL {HERG_TARGET})")
    logger.info("=" * 50)
    
    try:
        activities = new_client.activity
        
        # Pull IC50-like measurements; you can add Ki if you want a broader proxy
        logger.info("📡 Querying ChEMBL for KCNH2 IC50 activities...")
        
        records = activities.filter(
            target_chembl_id=HERG_TARGET,
            standard_type="IC50"
        ).only([
            "molecule_chembl_id", "standard_value", "standard_units"
        ])
        
        df = pd.DataFrame(records)
        logger.info(f"✅ Retrieved {len(df):,} hERG IC50 records")
        
        if df.empty:
            logger.warning("⚠️ No hERG IC50 data found")
            return pd.DataFrame(columns=["chembl_id", "tox_herg_ic50_uM", "tox_herg_n_points"])
        
        # Clean types and units
        logger.info("🔧 Processing and normalizing units...")
        df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
        df = df.dropna(subset=["standard_value"])
        
        logger.info(f"📊 Valid numeric values: {len(df):,}")
        
        # Normalize to µM
        def to_uM(row):
            """Convert various units to micromolar (µM)"""
            u = str(row.get("standard_units", "")).lower()
            v = row["standard_value"]
            
            if u in ["nm", "nanomolar", "nanomole/l", "nm/l"]:
                return v / 1000.0  # nM → µM
            elif u in ["um", "µm", "micromolar", "umol/l", "µmol/l"]:
                return v  # Already µM
            elif u in ["mm", "millimolar", "mmol/l"]:
                return v * 1000.0  # mM → µM
            elif u in ["m", "molar", "mol/l"]:
                return v * 1000000.0  # M → µM
            else:
                logger.debug(f"Unknown unit: {u}")
                return pd.NA  # unknown units → drop later
        
        df["value_uM"] = df.apply(to_uM, axis=1)
        df = df.dropna(subset=["value_uM"])
        
        logger.info(f"📊 After unit normalization: {len(df):,} records")
        
        # Show unit distribution
        unit_dist = df["standard_units"].value_counts()
        logger.info("📋 Unit distribution:")
        for unit, count in unit_dist.head(10).items():
            logger.info(f"   {unit}: {count:,}")
        
        df = df.rename(columns={"molecule_chembl_id": "chembl_id"})
        
        # Aggregate: median IC50 per molecule (µM) + count
        logger.info("📊 Aggregating by molecule (median IC50)...")
        agg = df.groupby("chembl_id").agg(
            tox_herg_ic50_uM=("value_uM", "median"),
            tox_herg_n_points=("value_uM", "count")
        ).reset_index()
        
        logger.info(f"✅ hERG IC50 aggregation complete:")
        logger.info(f"   Unique molecules: {len(agg):,}")
        logger.info(f"   Median IC50 range: {agg['tox_herg_ic50_uM'].min():.2f} - {agg['tox_herg_ic50_uM'].max():.2f} µM")
        logger.info(f"   Average data points per molecule: {agg['tox_herg_n_points'].mean():.1f}")
        
        # Cache results if path provided
        if cache_path:
            from pathlib import Path
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            agg.to_parquet(cache_path, index=False)
            logger.info(f"💾 Cached results: {cache_path}")
        
        return agg
        
    except Exception as e:
        logger.error(f"❌ Error fetching hERG IC50 data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=["chembl_id", "tox_herg_ic50_uM", "tox_herg_n_points"])


def test_herg_fetcher():
    """Test the hERG IC50 fetcher"""
    logger.info("🧪 TESTING hERG IC50 FETCHER")
    
    # Test with a small sample
    herg_data = fetch_herg_ic50()
    
    if not herg_data.empty:
        logger.info(f"✅ Test successful: {len(herg_data):,} molecules with hERG data")
        
        # Show sample results
        sample = herg_data.head()
        logger.info("📊 Sample hERG IC50 data:")
        for _, row in sample.iterrows():
            chembl_id = row['chembl_id']
            ic50 = row['tox_herg_ic50_uM']
            n_points = row['tox_herg_n_points']
            logger.info(f"   {chembl_id}: {ic50:.2f} µM ({n_points} data points)")
    else:
        logger.error("❌ Test failed - no hERG data retrieved")
    
    return herg_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    test_herg_fetcher()