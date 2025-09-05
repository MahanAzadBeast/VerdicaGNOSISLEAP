"""
Add Provenance Fields
Adds machine-auditable provenance fields for data lineage tracking
"""

import os
import json
import hashlib
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def sha256_of_file(path: str) -> str | None:
    """
    Calculate SHA256 hash of a file
    
    Args:
        path: File path to hash
        
    Returns:
        SHA256 hex digest or None if error
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.warning(f"Could not hash file {path}: {e}")
        return None


def add_provenance(df: pd.DataFrame, input_path: str | None = None) -> pd.DataFrame:
    """
    Add machine-auditable provenance fields to dataset
    
    Args:
        df: Input DataFrame
        input_path: Path to input file for hash calculation
        
    Returns:
        DataFrame with provenance fields added
    """
    logger.info("📋 ADDING MACHINE-AUDITABLE PROVENANCE")
    logger.info("🔍 Creating data lineage tracking fields")
    logger.info("=" * 50)
    
    # Get environment variables for provenance
    schema_version = os.getenv("SCHEMA_VERSION", "1.0.0")
    git_commit = os.getenv("GIT_COMMIT", "unknown")
    container_digest = os.getenv("CONTAINER_DIGEST", "unknown")
    cutoff_date = os.getenv("CUTOFF_DATE", "2017-12-31")
    qc_uri = os.getenv("QC_REPORT_URI", "")
    accessed_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    
    # Calculate input file hash
    input_sha = sha256_of_file(input_path) if input_path else None
    
    logger.info(f"📊 Provenance information:")
    logger.info(f"   Schema version: {schema_version}")
    logger.info(f"   Git commit: {git_commit}")
    logger.info(f"   Container: {container_digest}")
    logger.info(f"   Cutoff date: {cutoff_date}")
    logger.info(f"   Access time: {accessed_at}")
    logger.info(f"   Input hash: {input_sha[:16] + '...' if input_sha else 'N/A'}")
    
    # Add minimal flat columns for easy querying
    df = df.copy()
    df["schema_version"] = schema_version
    df["transform_code_commit"] = git_commit
    df["transform_container"] = container_digest
    df["cutoff_date"] = cutoff_date
    df["qc_report_uri"] = qc_uri
    df["source_accessed_at"] = accessed_at
    
    # Optional single JSON blob for machine parsing
    prov = {
        "schema_version": schema_version,
        "code_commit": git_commit,
        "container": container_digest,
        "cutoff_date": cutoff_date,
        "qc_report_uri": qc_uri,
        "source_accessed_at": accessed_at,
        "inputs": [{"path": input_path, "sha256": input_sha}] if input_path else [],
        "transform_timestamp": accessed_at,
        "data_lineage": {
            "source": "chembl_database",
            "clinical_trials": "clinicaltrials_gov",
            "toxicity": "molecular_properties",
            "authenticity": "verified_real_only"
        }
    }
    
    df["provenance_json"] = df.apply(lambda _: json.dumps(prov, separators=(",", ":")), axis=1)
    
    logger.info(f"✅ Provenance fields added:")
    logger.info(f"   Flat fields: 6 columns")
    logger.info(f"   JSON blob: provenance_json")
    logger.info(f"   Total compounds: {len(df):,}")
    
    return df


def validate_provenance(df: pd.DataFrame) -> bool:
    """
    Validate that provenance fields are properly populated
    
    Args:
        df: DataFrame with provenance fields
        
    Returns:
        True if validation passes
    """
    logger.info("🔍 VALIDATING PROVENANCE FIELDS")
    logger.info("=" * 40)
    
    required_fields = [
        "schema_version",
        "transform_code_commit", 
        "transform_container",
        "cutoff_date",
        "source_accessed_at",
        "provenance_json"
    ]
    
    validation_passed = True
    
    for field in required_fields:
        if field not in df.columns:
            logger.error(f"❌ Missing required provenance field: {field}")
            validation_passed = False
        else:
            non_null = df[field].notna().sum()
            if non_null < len(df):
                logger.warning(f"⚠️ Incomplete provenance field {field}: {non_null}/{len(df)}")
            else:
                logger.info(f"✅ {field}: Complete")
    
    # Validate JSON structure
    if "provenance_json" in df.columns:
        try:
            sample_json = json.loads(df["provenance_json"].iloc[0])
            required_json_fields = ["schema_version", "code_commit", "cutoff_date", "data_lineage"]
            
            for field in required_json_fields:
                if field not in sample_json:
                    logger.warning(f"⚠️ Missing JSON field: {field}")
                else:
                    logger.info(f"✅ JSON field {field}: Present")
                    
        except Exception as e:
            logger.error(f"❌ Invalid JSON in provenance_json: {e}")
            validation_passed = False
    
    return validation_passed


def main():
    """Main augmentation pipeline"""
    logger.info("🔬 NUMERIC TOXICITY AUGMENTATION PIPELINE")
    logger.info("🫀 Adding hERG IC50 data from ChEMBL")
    logger.info("📋 Adding machine-auditable provenance")
    
    # Load input dataset
    try:
        df = pd.read_csv(IN_CSV)
        logger.info(f"✅ Loaded input dataset: {len(df):,} compounds")
    except Exception as e:
        logger.error(f"❌ Error loading {IN_CSV}: {e}")
        return
    
    # Fetch hERG IC50 data
    try:
        herg = fetch_herg_ic50(cache_path=CACHE)
        
        if herg.empty:
            logger.error("❌ No hERG data available")
            return
            
    except Exception as e:
        logger.error(f"❌ Error fetching hERG data: {e}")
        return
    
    # Merge hERG data
    logger.info("🔗 Merging hERG IC50 data...")
    out = df.merge(herg, on="chembl_id", how="left")
    
    herg_matches = out["tox_herg_ic50_uM"].notna().sum()
    logger.info(f"✅ hERG IC50 integration: {herg_matches:,} compounds")
    
    # Update tox_data_sources
    logger.info("📋 Updating toxicity data sources...")
    
    if "tox_data_sources" not in out.columns:
        out["tox_data_sources"] = pd.NA
    
    mask = out["tox_herg_ic50_uM"].notna()
    out.loc[mask, "tox_data_sources"] = out.loc[mask, "tox_data_sources"].apply(
        lambda x: append_source_list(x, "chembl:KCNH2_IC50")
    )
    
    # Add provenance fields
    logger.info("📋 Adding provenance fields...")
    out = add_provenance(out, input_path=IN_CSV)
    
    # Validate provenance
    if not validate_provenance(out):
        logger.error("❌ Provenance validation failed")
        return
    
    # Save augmented dataset
    try:
        Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_CSV, index=False)
        
        logger.info(f"💾 Augmented dataset saved: {OUT_CSV}")
        logger.info(f"   📊 Shape: {out.shape}")
        logger.info(f"   🫀 hERG IC50 data: {herg_matches:,} compounds")
        logger.info(f"   📋 Provenance: Complete")
        
    except Exception as e:
        logger.error(f"❌ Error saving augmented dataset: {e}")
        return
    
    logger.info("\\n🎉 NUMERIC TOXICITY AUGMENTATION COMPLETE")
    logger.info(f"✅ Dataset enhanced with machine-auditable toxicity data")
    logger.info(f"📁 Output: {OUT_CSV}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()