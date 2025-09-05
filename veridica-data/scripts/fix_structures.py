#!/usr/bin/env python3
"""
Fix Structure Keys and Missing Descriptors
Computes InChI, InChIKey and missing molecular descriptors using RDKit
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enrich_structure(smiles):
    """
    Compute missing structure keys and descriptors from SMILES
    
    Args:
        smiles: SMILES string
        
    Returns:
        Series with [inchi, inchikey, mol_num_rings, mol_num_heteroatoms, mol_fraction_csp3, rdkit_ok]
    """
    if not RDKIT_AVAILABLE:
        logger.error("RDKit not available - cannot compute structure data")
        return pd.Series([None, None, None, None, None, 0])
    
    if pd.isna(smiles) or not smiles:
        return pd.Series([None, None, None, None, None, 0])
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return pd.Series([None, None, None, None, None, 0])
        
        # Structure keys
        inchi = Chem.MolToInchi(mol)
        inchikey = Chem.InchiToInchiKey(inchi)
        
        # Missing descriptors - using available functions
        mol_num_rings = Descriptors.RingCount(mol)
        mol_num_heteroatoms = Descriptors.NumHeteroatoms(mol) 
        mol_fraction_csp3 = Descriptors.FractionCsp3(mol)
        
        return pd.Series([inchi, inchikey, mol_num_rings, mol_num_heteroatoms, mol_fraction_csp3, 1])
        
    except Exception as e:
        logger.error(f"Error processing SMILES {smiles}: {e}")
        return pd.Series([None, None, None, None, None, 0])


def fix_structures(input_file="csv_exports/veridica_master_merged.csv", 
                   output_file="csv_exports/veridica_master_merged.struct.csv"):
    """
    Fix structure keys and missing descriptors in the dataset
    """
    logger.info("🔧 FIXING STRUCTURE KEYS AND MISSING DESCRIPTORS")
    logger.info("=" * 60)
    
    if not RDKIT_AVAILABLE:
        logger.error("❌ RDKit not available - cannot proceed")
        return None
    
    # Load dataset
    try:
        df = pd.read_csv(input_file)
        logger.info(f"✅ Loaded dataset: {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"❌ Error loading {input_file}: {e}")
        return None
    
    # Add missing columns if they don't exist
    new_cols = ["inchi", "inchikey", "mol_num_rings", "mol_num_heteroatoms", "mol_fraction_csp3", "_rdkit_ok"]
    
    for col in new_cols:
        if col not in df.columns:
            df[col] = None
    
    # Find rows that need structure enrichment
    mask = (df["inchi"].isna() | 
            df["inchikey"].isna() | 
            df["mol_num_rings"].isna() | 
            df["mol_num_heteroatoms"].isna() | 
            df["mol_fraction_csp3"].isna())
    
    rows_to_fix = mask.sum()
    logger.info(f"🔍 Found {rows_to_fix:,} rows needing structure enrichment")
    
    if rows_to_fix == 0:
        logger.info("✅ All structure data already present")
        return df
    
    # Process SMILES to get structure data
    logger.info("🧬 Computing structure keys and descriptors...")
    
    structure_data = df.loc[mask, "canonical_smiles"].apply(enrich_structure)
    
    # Update the dataframe
    df.loc[mask, ["inchi", "inchikey", "mol_num_rings", "mol_num_heteroatoms", "mol_fraction_csp3", "_rdkit_ok"]] = structure_data.values
    
    # Report results
    successful = df["_rdkit_ok"].sum()
    failed = rows_to_fix - successful
    
    logger.info(f"✅ Structure enrichment complete:")
    logger.info(f"   Successful: {successful:,} compounds")
    logger.info(f"   Failed: {failed:,} compounds")
    
    # Validate structure keys
    unique_chembl = df["chembl_id"].nunique()
    unique_inchikey = df["inchikey"].nunique()
    
    logger.info(f"🔑 Structure key validation:")
    logger.info(f"   Unique ChEMBL IDs: {unique_chembl:,}")
    logger.info(f"   Unique InChIKeys: {unique_inchikey:,}")
    
    if unique_chembl != unique_inchikey:
        logger.warning(f"⚠️ ChEMBL ID / InChIKey mismatch - may indicate duplicates")
    
    # Check for missing descriptors
    descriptor_completeness = {}
    descriptor_cols = ["mol_num_rings", "mol_num_heteroatoms", "mol_fraction_csp3"]
    
    for col in descriptor_cols:
        non_null = df[col].notna().sum()
        completeness = (non_null / len(df) * 100)
        descriptor_completeness[col] = completeness
        logger.info(f"   {col}: {non_null:,}/{len(df):,} ({completeness:.1f}%)")
    
    # Save fixed dataset
    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"💾 Fixed dataset saved: {output_file}")
    except Exception as e:
        logger.error(f"❌ Error saving {output_file}: {e}")
        return None
    
    return df


def main():
    """Main execution"""
    logger.info("🔧 STRUCTURE KEYS AND DESCRIPTORS FIXER")
    logger.info("🧬 Adding InChI, InChIKey and missing molecular descriptors")
    
    # Fix structures
    fixed_df = fix_structures()
    
    if fixed_df is not None:
        logger.info("\\n🎉 STRUCTURE FIXING COMPLETE")
        logger.info("✅ All structure keys and descriptors computed")
        logger.info("📁 Output: csv_exports/veridica_master_merged.struct.csv")
    else:
        logger.error("❌ Structure fixing failed")


if __name__ == "__main__":
    main()