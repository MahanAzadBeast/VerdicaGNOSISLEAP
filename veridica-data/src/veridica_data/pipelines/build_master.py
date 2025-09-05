"""
Build Master Compound Table Pipeline
Creates the master compound index from existing ChEMBL data
"""

import pandas as pd
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from ..connectors.chembl_fetch import ChEMBLFetcher
from ..utils.chem import canonicalize, compute_molecular_descriptors
from ..schemas.master_schema import create_master_record, validate_structure_consistency

logger = logging.getLogger(__name__)


def build_master_from_existing_chembl(
    chembl_file: str = "/workspace/clinical_trial_dataset/data/github_final/chembl_complete_dataset.csv",
    output_dir: str = "data"
) -> pd.DataFrame:
    """
    Build master compound table from existing ChEMBL dataset
    
    Args:
        chembl_file: Path to existing ChEMBL CSV file
        output_dir: Output directory for master table
        
    Returns:
        Master compound DataFrame
    """
    logger.info("🏗️ BUILDING MASTER COMPOUND TABLE FROM EXISTING CHEMBL")
    logger.info("=" * 70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load existing ChEMBL data
    try:
        df = pd.read_csv(chembl_file)
        logger.info(f"✅ Loaded existing ChEMBL dataset: {len(df):,} compounds")
    except Exception as e:
        logger.error(f"❌ Error loading ChEMBL file: {e}")
        return pd.DataFrame()
    
    # Process each compound to create master records
    master_records = []
    
    for idx, row in df.iterrows():
        try:
            # Extract ChEMBL data
            chembl_data = {
                'chembl_id': row['chembl_id'],
                'primary_drug': row['primary_drug'],
                'all_drug_names': row.get('all_drug_names', []),
                'max_clinical_phase': row.get('max_clinical_phase'),
                'clinical_status': row.get('clinical_status')
            }
            
            # Canonicalize SMILES and compute descriptors
            smiles = row['smiles']
            canonical_smiles, inchi, inchikey, computed_descriptors = canonicalize(smiles)
            
            if not canonical_smiles:
                logger.warning(f"Could not canonicalize SMILES for {row['chembl_id']}")
                continue
            
            # Use existing descriptors or computed ones
            descriptors = {}
            
            # Map existing molecular properties
            mol_prop_mapping = {
                'mol_molecular_weight': 'mol_molecular_weight',
                'mol_logp': 'mol_logp', 
                'mol_num_hbd': 'mol_num_hbd',
                'mol_num_hba': 'mol_num_hba',
                'mol_num_rotatable_bonds': 'mol_num_rotatable_bonds',
                'mol_tpsa': 'mol_tpsa',
                'mol_num_aromatic_rings': 'mol_num_aromatic_rings',
                'mol_num_heavy_atoms': 'mol_num_heavy_atoms',
                'mol_formal_charge': 'mol_formal_charge',
                'mol_num_rings': 'mol_num_rings',
                'mol_num_heteroatoms': 'mol_num_heteroatoms',
                'mol_fraction_csp3': 'mol_fraction_csp3'
            }
            
            for new_col, existing_col in mol_prop_mapping.items():
                if existing_col in row and pd.notna(row[existing_col]):
                    descriptors[new_col] = row[existing_col]
                elif new_col in computed_descriptors:
                    descriptors[new_col] = computed_descriptors[new_col]
                else:
                    descriptors[new_col] = None
            
            # Create master record
            master_record = create_master_record(
                chembl_data=chembl_data,
                canonical_smiles=canonical_smiles,
                inchi=inchi,
                inchikey=inchikey,
                descriptors=descriptors
            )
            
            # Add original ChEMBL metadata
            master_record.update({
                'max_clinical_phase': row.get('max_clinical_phase'),
                'clinical_status': row.get('clinical_status'),
                'compound_type': row.get('compound_type', 'Small molecule')
            })
            
            master_records.append(master_record)
            
            # Progress logging
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1:,}/{len(df):,} compounds")
                
        except Exception as e:
            logger.error(f"Error processing compound {row.get('chembl_id', 'unknown')}: {e}")
            continue
    
    # Create master DataFrame
    master_df = pd.DataFrame(master_records)
    
    if master_df.empty:
        logger.error("❌ No master records created")
        return pd.DataFrame()
    
    logger.info(f"✅ Created {len(master_df):,} master compound records")
    
    # Validate structure consistency
    inconsistencies = validate_structure_consistency(master_records)
    
    if inconsistencies:
        logger.warning(f"⚠️ Found {len(inconsistencies)} structure inconsistencies")
        for inc in inconsistencies[:5]:  # Show first 5
            logger.warning(f"   {inc}")
    else:
        logger.info("✅ Structure consistency validation passed")
    
    # Quality control report
    _generate_qc_report(master_df)
    
    # Save master table
    master_file = output_path / "master.parquet"
    
    try:
        master_df.to_parquet(master_file, compression='snappy')
        logger.info(f"💾 Master table saved: {master_file}")
    except Exception as e:
        logger.error(f"Error saving master table: {e}")
    
    return master_df


def _generate_qc_report(df: pd.DataFrame) -> None:
    """Generate quality control report for master table"""
    logger.info("📊 MASTER TABLE QC REPORT")
    logger.info("=" * 50)
    
    # Basic statistics
    logger.info(f"Total compounds: {len(df):,}")
    logger.info(f"Unique ChEMBL IDs: {df['chembl_id'].nunique():,}")
    logger.info(f"Unique InChIKeys: {df['inchikey'].nunique():,}")
    
    # SMILES validation
    valid_smiles = df['canonical_smiles'].notna().sum()
    logger.info(f"Valid SMILES: {valid_smiles:,}/{len(df):,} ({(valid_smiles/len(df)*100):.1f}%)")
    
    # Descriptor completeness
    descriptor_cols = [col for col in df.columns if col.startswith('mol_')]
    
    logger.info(f"\\nDescriptor completeness:")
    for col in descriptor_cols:
        non_null = df[col].notna().sum()
        completeness = (non_null / len(df) * 100)
        logger.info(f"  {col}: {non_null:,}/{len(df):,} ({completeness:.1f}%)")
    
    # Clinical phase distribution
    if 'max_clinical_phase' in df.columns:
        phase_counts = df['max_clinical_phase'].value_counts().sort_index()
        logger.info(f"\\nClinical phase distribution:")
        for phase, count in phase_counts.items():
            if pd.notna(phase):
                phase_name = f"Phase {int(phase)}" if phase >= 1 else "Preclinical"
                logger.info(f"  {phase_name}: {count:,} compounds")
    
    # Data source distribution
    if 'data_source' in df.columns:
        source_counts = df['data_source'].value_counts()
        logger.info(f"\\nData source distribution:")
        for source, count in source_counts.items():
            logger.info(f"  {source}: {count:,} compounds")


def main():
    """Main pipeline execution"""
    logger.info("🌟 MASTER COMPOUND TABLE BUILDER")
    logger.info("🔬 Using existing ChEMBL dataset as foundation")
    logger.info("🚫 NO synthetic data - 100% real pharmaceutical compounds")
    
    # Build master table
    master_df = build_master_from_existing_chembl()
    
    if not master_df.empty:
        logger.info("\\n🎉 MASTER TABLE BUILD COMPLETE")
        logger.info(f"📊 Total compounds: {len(master_df):,}")
        logger.info(f"✅ Ready for clinical and toxicity enrichment")
        logger.info(f"📁 Saved to: data/master.parquet")
    else:
        logger.error("❌ Master table build failed")
    
    return master_df


if __name__ == "__main__":
    main()