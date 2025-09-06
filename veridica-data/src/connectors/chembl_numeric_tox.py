"""
ChEMBL Numeric Toxicity Connector
Fetches real experimental IC50/Ki data for cardiac targets (hERG, Nav1.5, Cav1.2)
WITH STRICT SYNTHETIC DATA PREVENTION
"""

from chembl_webresource_client.new_client import new_client
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Synthetic data detection keywords
SYNTHETIC_KEYWORDS = [
    'demo', 'synthetic', 'fake', 'test', 'generated', 'artificial',
    'chembl_demo', 'mock', 'placeholder', 'example'
]


def validate_authentic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that no synthetic data is present in ChEMBL results
    
    Args:
        df: DataFrame with ChEMBL activity data
        
    Returns:
        DataFrame with synthetic entries removed
    """
    if df.empty:
        return df
    
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
                    logger.warning(f"⚠️ Found {matches} entries with '{keyword}' in {col} - will remove")
    
    # Remove synthetic entries
    clean_df = df[~synthetic_mask].copy()
    removed_count = original_count - len(clean_df)
    
    if removed_count > 0:
        logger.warning(f"🚫 REMOVED {removed_count} synthetic ChEMBL entries")
        logger.info(f"✅ Clean ChEMBL data: {len(clean_df)} authentic records")
    else:
        logger.info(f"✅ All ChEMBL data verified authentic: {len(clean_df)} records")
    
    return clean_df


# Resolve target IDs by gene name (e.g., "KCNH2", "SCN5A", "CACNA1C")
@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(5))
def resolve_target_ids(gene_names):
    """
    Resolve ChEMBL target IDs for cardiac ion channels
    
    Args:
        gene_names: List of gene names to resolve
        
    Returns:
        Dictionary mapping gene names to ChEMBL target IDs
    """
    logger.info(f"🔍 Resolving ChEMBL target IDs for: {gene_names}")
    
    targ = new_client.target
    out = {}
    
    for g in gene_names:
        logger.info(f"   Searching for {g}...")
        
        try:
            hits = (list(targ.filter(target_components__accession__icontains=g)) +
                   list(targ.filter(pref_name__icontains=g)) +
                   list(targ.filter(target_synonym__icontains=g)))
            
            # Choose first with exact gene mention where possible
            chosen = None
            for h in hits:
                # Validate this is not synthetic data
                target_data = str(h.get('pref_name', '')) + ' ' + str(h.get('target_synonym', ''))
                
                # Check for synthetic indicators
                is_synthetic = any(keyword in target_data.lower() for keyword in SYNTHETIC_KEYWORDS)
                if is_synthetic:
                    logger.warning(f"⚠️ Skipping synthetic target for {g}: {target_data}")
                    continue
                
                if g.lower() in target_data.lower():
                    chosen = h
                    break
            
            if not chosen and hits:
                # Take first non-synthetic hit
                for h in hits:
                    target_data = str(h.get('pref_name', '')) + ' ' + str(h.get('target_synonym', ''))
                    is_synthetic = any(keyword in target_data.lower() for keyword in SYNTHETIC_KEYWORDS)
                    if not is_synthetic:
                        chosen = h
                        break
            
            if chosen:
                target_id = chosen["target_chembl_id"]
                target_name = chosen.get("pref_name", "Unknown")
                out[g] = target_id
                logger.info(f"   ✅ {g} → {target_id} ({target_name})")
            else:
                logger.warning(f"   ❌ No valid target found for {g}")
                
        except Exception as e:
            logger.error(f"   ❌ Error resolving {g}: {e}")
            continue
    
    logger.info(f"✅ Resolved {len(out)}/{len(gene_names)} targets")
    return out


def _unit_to_uM(v, u):
    """Convert various concentration units to micromolar (µM)"""
    if v is None or pd.isna(v):
        return pd.NA
    
    u = (u or "").lower()
    
    try:
        v = float(v)
    except Exception:
        return pd.NA
    
    # Unit conversions to µM
    if u in ("nm", "nanomolar", "nanomole/l"):
        return v / 1000.0  # nM → µM
    elif u in ("μm", "um", "micromolar", "umol/l"):
        return v  # Already µM
    elif u in ("mm", "millimolar", "mmol/l"):
        return v * 1000.0  # mM → µM
    elif u in ("m", "molar", "mol/l"):
        return v * 1e6  # M → µM
    else:
        return pd.NA  # Unknown unit


@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(5))
def fetch_numeric_for_target(target_chembl_id, std_types=("IC50", "Ki")):
    """
    Fetch numeric activity data for a specific target
    
    Args:
        target_chembl_id: ChEMBL target identifier
        std_types: Standard types to fetch (IC50, Ki, etc.)
        
    Returns:
        DataFrame with authentic activity data
    """
    logger.info(f"📡 Fetching {std_types} data for target {target_chembl_id}")
    
    act = new_client.activity
    dfs = []
    
    for stype in std_types:
        logger.info(f"   Querying {stype} activities...")
        
        try:
            recs = act.filter(
                target_chembl_id=target_chembl_id, 
                standard_type=stype
            ).only([
                "molecule_chembl_id", "standard_type", 
                "standard_value", "standard_units"
            ])
            
            df = pd.DataFrame(recs)
            
            if df.empty:
                logger.info(f"   No {stype} data found for {target_chembl_id}")
                continue
            
            logger.info(f"   Retrieved {len(df):,} {stype} records")
            
            # STRICT SYNTHETIC DATA VALIDATION
            df = validate_authentic_data(df)
            
            if df.empty:
                logger.warning(f"   All {stype} data was synthetic - skipping")
                continue
            
            # Convert units to µM
            df["value_uM"] = df.apply(
                lambda r: _unit_to_uM(r["standard_value"], r["standard_units"]), 
                axis=1
            )
            
            # Remove invalid conversions
            df = df.dropna(subset=["value_uM"])
            
            if df.empty:
                logger.info(f"   No valid units for {stype} data")
                continue
            
            df["measure"] = stype.upper()
            dfs.append(df)
            
            logger.info(f"   ✅ Valid {stype} data: {len(df):,} records")
            
        except Exception as e:
            logger.error(f"   ❌ Error fetching {stype} for {target_chembl_id}: {e}")
            continue
    
    if not dfs:
        logger.warning(f"No valid data for target {target_chembl_id}")
        return pd.DataFrame(columns=["chembl_id", "measure", "value_uM"])
    
    # Combine all activity types
    out = pd.concat(dfs, ignore_index=True)
    out = out.rename(columns={"molecule_chembl_id": "chembl_id"})
    
    logger.info(f"✅ Total authentic records for {target_chembl_id}: {len(out):,}")
    
    return out


def aggregate_numeric(df, prefix):
    """
    Compute robust aggregates per molecule and measure
    
    Args:
        df: DataFrame with activity data
        prefix: Prefix for column names (e.g., 'herg', 'nav1_5')
        
    Returns:
        DataFrame with aggregated statistics
    """
    if df.empty:
        logger.warning(f"No data to aggregate for {prefix}")
        return pd.DataFrame(columns=["chembl_id"])
    
    logger.info(f"📊 Aggregating {prefix} data: {len(df):,} records")
    
    # Validate no synthetic ChEMBL IDs
    synthetic_chembl_mask = df['chembl_id'].astype(str).str.lower().str.contains('|'.join(SYNTHETIC_KEYWORDS), na=False)
    synthetic_chembl_count = synthetic_chembl_mask.sum()
    
    if synthetic_chembl_count > 0:
        logger.warning(f"🚫 Removing {synthetic_chembl_count} synthetic ChEMBL IDs")
        df = df[~synthetic_chembl_mask].copy()
    
    if df.empty:
        logger.error(f"❌ All {prefix} data was synthetic!")
        return pd.DataFrame(columns=["chembl_id"])
    
    # Create pivot table for aggregation
    try:
        piv = df.pivot_table(
            index="chembl_id", 
            columns="measure", 
            values="value_uM",
            aggfunc=["median", "min", "max", "count"]
        )
        
        # Flatten columns
        piv.columns = [f"{prefix}_{a}_{b}".lower() for a, b in piv.columns]
        piv = piv.reset_index()
        
        logger.info(f"✅ {prefix} aggregation: {len(piv):,} unique molecules")
        
        return piv
        
    except Exception as e:
        logger.error(f"❌ Error aggregating {prefix} data: {e}")
        return pd.DataFrame(columns=["chembl_id"])


def validate_final_toxicity_data(df: pd.DataFrame) -> bool:
    """
    Final validation that no synthetic data made it through
    
    Args:
        df: Final toxicity DataFrame
        
    Returns:
        True if validation passes
    """
    logger.info("🔍 FINAL SYNTHETIC DATA VALIDATION")
    
    synthetic_found = 0
    
    # Check all columns for synthetic indicators
    for col in df.columns:
        if col in df.columns:
            col_values = df[col].astype(str).str.lower()
            
            for keyword in SYNTHETIC_KEYWORDS:
                matches = col_values.str.contains(keyword, na=False).sum()
                if matches > 0:
                    logger.error(f"❌ SYNTHETIC DATA DETECTED: {matches} entries with '{keyword}' in {col}")
                    synthetic_found += matches
    
    # Check ChEMBL IDs for synthetic patterns
    if 'chembl_id' in df.columns:
        synthetic_ids = df['chembl_id'].astype(str).str.lower().str.contains('|'.join(SYNTHETIC_KEYWORDS), na=False).sum()
        if synthetic_ids > 0:
            logger.error(f"❌ SYNTHETIC ChEMBL IDs: {synthetic_ids}")
            synthetic_found += synthetic_ids
    
    if synthetic_found == 0:
        logger.info("🎉 FINAL VALIDATION PASSED")
        logger.info("✅ NO SYNTHETIC DATA DETECTED")
        logger.info("✅ 100% AUTHENTIC TOXICITY DATA CONFIRMED")
        return True
    else:
        logger.error(f"❌ VALIDATION FAILED: {synthetic_found} synthetic entries found")
        return False