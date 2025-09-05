"""
Append Provenance Pipeline
Adds machine-auditable provenance fields to the dataset
"""

import os
import pandas as pd
import logging
import sys

# Add src to path for imports
sys.path.insert(0, 'src')

from transforms.add_provenance import add_provenance

logger = logging.getLogger(__name__)

# Configuration
IN_CSV = os.environ.get("IN_CSV", "data/veridica_final_improved.herg.csv")
OUT_CSV = os.environ.get("OUT_CSV", "data/veridica_final_improved.herg.prov.csv")


def main():
    """Main provenance pipeline"""
    logger.info("📋 PROVENANCE PIPELINE")
    logger.info("🔍 Adding machine-auditable data lineage")
    logger.info("=" * 50)
    
    try:
        # Load dataset with hERG data
        df = pd.read_csv(IN_CSV)
        logger.info(f"✅ Loaded dataset: {len(df):,} compounds")
        
        # Add provenance fields
        df = add_provenance(df, input_path=IN_CSV)
        
        # Save with provenance
        df.to_csv(OUT_CSV, index=False)
        logger.info(f"💾 Dataset with provenance saved: {OUT_CSV}")
        logger.info(f"   📊 {len(df):,} compounds")
        logger.info(f"   📋 Provenance fields: Complete")
        
        logger.info("\\n🎉 PROVENANCE PIPELINE COMPLETE")
        logger.info("✅ Machine-auditable data lineage added")
        
    except Exception as e:
        logger.error(f"❌ Error in provenance pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()