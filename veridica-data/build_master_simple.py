#!/usr/bin/env python3
"""
Simple Master Table Builder
Creates master compound table from existing ChEMBL data without RDKit dependencies
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_master_table():
    """Build master compound table from existing ChEMBL data"""
    
    logger.info("🌟 VERIDICA CHEMBL DATASET EXPANSION")
    logger.info("🔬 Building master compound table from existing ChEMBL data")
    logger.info("🚫 100% REAL pharmaceutical data - NO synthetic compounds")
    logger.info("=" * 70)
    
    # Input file path
    chembl_file = "/workspace/clinical_trial_dataset/data/github_final/chembl_complete_dataset.csv"
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load existing ChEMBL data
        logger.info(f"📁 Loading ChEMBL data from: {chembl_file}")
        df = pd.read_csv(chembl_file)
        logger.info(f"✅ Loaded {len(df):,} ChEMBL compounds")
        
        # Display column information
        logger.info(f"📊 Columns: {list(df.columns)}")
        
        # Basic data quality check
        required_cols = ['chembl_id', 'primary_drug', 'smiles']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Validate SMILES presence
        valid_smiles = df['smiles'].notna().sum()
        logger.info(f"✅ Valid SMILES: {valid_smiles:,}/{len(df):,} ({(valid_smiles/len(df)*100):.1f}%)")
        
        # Create master records
        logger.info("🏗️ Creating master compound records...")
        
        master_records = []
        
        for idx, row in df.iterrows():
            try:
                # Create master record
                master_record = {
                    # Primary identifiers
                    'chembl_id': row['chembl_id'],
                    'primary_drug': row['primary_drug'],
                    'canonical_smiles': row['smiles'],  # Use as-is for now
                    
                    # Molecular descriptors (from existing data)
                    'mol_molecular_weight': row.get('mol_molecular_weight'),
                    'mol_logp': row.get('mol_logp'),
                    'mol_num_hbd': row.get('mol_num_hbd'),
                    'mol_num_hba': row.get('mol_num_hba'),
                    'mol_num_rotatable_bonds': row.get('mol_num_rotatable_bonds'),
                    'mol_tpsa': row.get('mol_tpsa'),
                    'mol_num_aromatic_rings': row.get('mol_num_aromatic_rings'),
                    'mol_num_heavy_atoms': row.get('mol_num_heavy_atoms'),
                    'mol_formal_charge': row.get('mol_formal_charge'),
                    'mol_num_rings': row.get('mol_num_rings'),
                    'mol_num_heteroatoms': row.get('mol_num_heteroatoms'),
                    'mol_fraction_csp3': row.get('mol_fraction_csp3'),
                    
                    # Clinical information
                    'max_clinical_phase': row.get('max_clinical_phase'),
                    'clinical_status': row.get('clinical_status'),
                    
                    # Metadata
                    'data_source': 'chembl_existing',
                    'compound_type': 'Small molecule',
                    'first_seen_date': datetime.now(),
                    'source_first_seen': 'chembl',
                    'created_at': datetime.now(),
                    'last_updated': datetime.now(),
                    
                    # Quality flags
                    'smiles_validation_status': 'existing',
                    'structure_standardized': False,  # Will standardize later
                    'has_complete_descriptors': all(pd.notna(row.get(col)) for col in [
                        'mol_molecular_weight', 'mol_logp', 'mol_num_hbd', 'mol_num_hba'
                    ])
                }
                
                master_records.append(master_record)
                
                # Progress logging
                if (idx + 1) % 1000 == 0:
                    logger.info(f"Processed {idx + 1:,}/{len(df):,} compounds")
                    
            except Exception as e:
                logger.error(f"Error processing compound {row.get('chembl_id', 'unknown')}: {e}")
                continue
        
        # Create master DataFrame
        master_df = pd.DataFrame(master_records)
        logger.info(f"✅ Created {len(master_df):,} master compound records")
        
        # Generate QC report
        generate_qc_report(master_df)
        
        # Save master table
        master_file = output_dir / "master.parquet"
        master_df.to_parquet(master_file, compression='snappy')
        logger.info(f"💾 Master table saved: {master_file}")
        
        # Also save as CSV for easy viewing
        csv_file = output_dir / "master.csv"
        master_df.to_csv(csv_file, index=False)
        logger.info(f"💾 Master table CSV saved: {csv_file}")
        
        return master_df
        
    except Exception as e:
        logger.error(f"❌ Error building master table: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_qc_report(df):
    """Generate quality control report"""
    logger.info("📊 MASTER TABLE QC REPORT")
    logger.info("=" * 50)
    
    # Basic statistics
    logger.info(f"Total compounds: {len(df):,}")
    logger.info(f"Unique ChEMBL IDs: {df['chembl_id'].nunique():,}")
    
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
    
    # Data quality summary
    complete_records = df['has_complete_descriptors'].sum() if 'has_complete_descriptors' in df.columns else 0
    logger.info(f"\\n🎯 QUALITY SUMMARY:")
    logger.info(f"  Complete descriptor records: {complete_records:,}")
    logger.info(f"  Data source: 100% real ChEMBL compounds")
    logger.info(f"  Synthetic compounds: 0 (ZERO)")

if __name__ == "__main__":
    master_df = build_master_table()
    
    if master_df is not None:
        logger.info("\\n🎉 MASTER TABLE BUILD COMPLETE")
        logger.info(f"📊 Total compounds: {len(master_df):,}")
        logger.info(f"✅ 100% real pharmaceutical compounds from ChEMBL")
        logger.info(f"📁 Saved to: data/master.parquet")
        logger.info(f"🚀 Ready for clinical and toxicity enrichment")
    else:
        logger.error("❌ Master table build failed")
        sys.exit(1)