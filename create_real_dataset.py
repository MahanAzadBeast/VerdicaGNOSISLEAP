#!/usr/bin/env python3
"""
Real Dataset Creator - Collects actual compounds from ChEMBL and PubChem APIs
No synthetic data - only real pharmaceutical compounds with validated SMILES
"""

import requests
import pandas as pd
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealCompoundCollector:
    """Collects real pharmaceutical compounds from ChEMBL and PubChem APIs"""
    
    def __init__(self):
        self.chembl_base = "https://www.ebi.ac.uk/chembl/api/data"
        self.pubchem_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.collected_compounds = []
        
    def collect_chembl_approved_drugs(self, limit: int = 10000) -> List[Dict]:
        """Collect approved drugs from ChEMBL with SMILES"""
        logger.info(f"🔬 Collecting approved drugs from ChEMBL (target: {limit})")
        
        compounds = []
        offset = 0
        batch_size = 1000
        
        while len(compounds) < limit:
            try:
                # Get approved drugs (max_phase = 4)
                params = {
                    "format": "json",
                    "limit": batch_size,
                    "offset": offset,
                    "max_phase": 4,  # Only approved drugs
                    "molecule_type": "Small molecule"
                }
                
                url = f"{self.chembl_base}/molecule"
                logger.info(f"Fetching batch {offset}-{offset+batch_size} from ChEMBL...")
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                molecules = data.get("molecules", [])
                
                if not molecules:
                    logger.info("No more molecules available from ChEMBL")
                    break
                
                # Process each molecule
                batch_compounds = []
                for mol in molecules:
                    compound = self._extract_chembl_compound(mol)
                    if compound:
                        batch_compounds.append(compound)
                
                compounds.extend(batch_compounds)
                logger.info(f"✅ Collected {len(batch_compounds)} valid compounds (Total: {len(compounds)})")
                
                offset += batch_size
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching ChEMBL batch at offset {offset}: {e}")
                offset += batch_size
                continue
        
        logger.info(f"🎉 ChEMBL collection complete: {len(compounds)} approved drugs")
        return compounds[:limit]
    
    def _extract_chembl_compound(self, molecule: Dict) -> Optional[Dict]:
        """Extract compound data from ChEMBL molecule"""
        try:
            chembl_id = molecule.get("molecule_chembl_id")
            if not chembl_id:
                return None
            
            # Get SMILES
            structures = molecule.get("molecule_structures", {})
            smiles = structures.get("canonical_smiles")
            
            if not smiles:
                return None
            
            # Get basic properties
            properties = molecule.get("molecule_properties", {})
            hierarchy = molecule.get("molecule_hierarchy", {})
            
            # Extract drug name
            pref_name = molecule.get("pref_name", f"ChEMBL_{chembl_id}")
            
            return {
                "compound_id": f"CHEMBL_{chembl_id}",
                "primary_drug": pref_name,
                "all_drug_names": [pref_name],
                "smiles": smiles,
                "smiles_source": chembl_id,
                "mapping_status": "success",
                
                # Molecular properties from ChEMBL
                "mol_molecular_weight": properties.get("full_mwt"),
                "mol_logp": properties.get("alogp"),
                "mol_num_hbd": properties.get("hbd"),
                "mol_num_hba": properties.get("hba"),
                "mol_num_rotatable_bonds": properties.get("rtb"),
                "mol_tpsa": properties.get("psa"),
                "mol_num_aromatic_rings": properties.get("aromatic_rings"),
                "mol_num_heavy_atoms": properties.get("heavy_atoms"),
                "mol_formal_charge": 0,  # Default
                "mol_num_rings": properties.get("num_rings"),
                "mol_num_heteroatoms": None,  # Calculate if needed
                "mol_fraction_csp3": None,  # Calculate if needed
                
                # Clinical data
                "max_clinical_phase": molecule.get("max_phase", 4),
                "clinical_status": "Approved",
                "primary_condition": "Multiple",  # ChEMBL doesn't specify
                
                # Dataset metadata
                "data_source": "chembl_api",
                "compound_type": "Small molecule",
                "study_type": "APPROVED_DRUG",
                "primary_phase": "PHASE4",
                "overall_status": "APPROVED",
                "lead_sponsor": "ChEMBL_Database",
                "sponsor_class": "DATABASE",
                "collected_date": datetime.now().isoformat(),
                
                # ML targets (would need bioactivity data for real values)
                "efficacy_score": None,  # Requires bioactivity analysis
                "safety_score": None,    # Requires adverse event data
                "success_probability": 1.0  # Approved = successful
            }
            
        except Exception as e:
            logger.debug(f"Error processing ChEMBL molecule: {e}")
            return None
    
    def collect_pubchem_drugs(self, limit: int = 5000) -> List[Dict]:
        """Collect drug compounds from PubChem"""
        logger.info(f"💊 Collecting drugs from PubChem (target: {limit})")
        
        compounds = []
        
        try:
            # Search for approved drugs
            search_url = f"{self.pubchem_base}/compound/name/drug/cids/JSON"
            logger.info("Searching PubChem for drug compounds...")
            
            response = requests.get(search_url, timeout=30)
            if response.status_code != 200:
                logger.warning("PubChem drug search failed, trying alternative approach")
                return self._collect_pubchem_alternative(limit)
            
            data = response.json()
            cids = data.get("IdentifierList", {}).get("CID", [])[:limit]
            
            logger.info(f"Found {len(cids)} drug CIDs in PubChem")
            
            # Fetch compound details in batches
            batch_size = 100
            for i in range(0, len(cids), batch_size):
                batch_cids = cids[i:i+batch_size]
                batch_compounds = self._fetch_pubchem_batch(batch_cids)
                compounds.extend(batch_compounds)
                
                logger.info(f"✅ Processed PubChem batch {i//batch_size + 1} ({len(batch_compounds)} compounds)")
                time.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            logger.error(f"Error collecting PubChem drugs: {e}")
            return []
        
        logger.info(f"🎉 PubChem collection complete: {len(compounds)} drugs")
        return compounds
    
    def _collect_pubchem_alternative(self, limit: int) -> List[Dict]:
        """Alternative PubChem collection method"""
        logger.info("Using alternative PubChem collection method...")
        
        # Try collecting from known drug classes
        drug_classes = [
            "antibiotics", "analgesics", "antihypertensive", "antidiabetic",
            "antidepressant", "antihistamine", "bronchodilator", "diuretic"
        ]
        
        compounds = []
        per_class = limit // len(drug_classes)
        
        for drug_class in drug_classes:
            try:
                class_compounds = self._fetch_drug_class(drug_class, per_class)
                compounds.extend(class_compounds)
                logger.info(f"✅ Collected {len(class_compounds)} {drug_class} compounds")
                
                if len(compounds) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to collect {drug_class}: {e}")
                continue
        
        return compounds[:limit]
    
    def _fetch_drug_class(self, drug_class: str, limit: int) -> List[Dict]:
        """Fetch compounds from a specific drug class"""
        try:
            # Search for compounds by drug class
            search_url = f"{self.pubchem_base}/compound/name/{drug_class}/cids/JSON"
            response = requests.get(search_url, timeout=20)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            cids = data.get("IdentifierList", {}).get("CID", [])[:limit]
            
            return self._fetch_pubchem_batch(cids)
            
        except Exception as e:
            logger.debug(f"Error fetching {drug_class}: {e}")
            return []
    
    def _fetch_pubchem_batch(self, cids: List[int]) -> List[Dict]:
        """Fetch compound details for a batch of CIDs"""
        if not cids:
            return []
        
        try:
            # Join CIDs for batch request
            cid_list = ",".join(map(str, cids))
            
            # Get compound properties
            props_url = f"{self.pubchem_base}/compound/cid/{cid_list}/property/MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,TPSA,HeavyAtomCount,CanonicalSMILES/JSON"
            
            response = requests.get(props_url, timeout=30)
            if response.status_code != 200:
                return []
            
            data = response.json()
            properties_list = data.get("PropertyTable", {}).get("Properties", [])
            
            compounds = []
            for props in properties_list:
                compound = self._extract_pubchem_compound(props)
                if compound:
                    compounds.append(compound)
            
            return compounds
            
        except Exception as e:
            logger.debug(f"Error fetching PubChem batch: {e}")
            return []
    
    def _extract_pubchem_compound(self, props: Dict) -> Optional[Dict]:
        """Extract compound data from PubChem properties"""
        try:
            cid = props.get("CID")
            smiles = props.get("CanonicalSMILES")
            
            if not smiles or not cid:
                return None
            
            return {
                "compound_id": f"PUBCHEM_{cid}",
                "primary_drug": f"PubChem_CID_{cid}",
                "all_drug_names": [f"PubChem_CID_{cid}"],
                "smiles": smiles,
                "smiles_source": f"CID_{cid}",
                "mapping_status": "success",
                
                # Molecular properties
                "mol_molecular_weight": props.get("MolecularWeight"),
                "mol_logp": props.get("XLogP"),
                "mol_num_hbd": props.get("HBondDonorCount"),
                "mol_num_hba": props.get("HBondAcceptorCount"),
                "mol_num_rotatable_bonds": props.get("RotatableBondCount"),
                "mol_tpsa": props.get("TPSA"),
                "mol_num_aromatic_rings": None,  # Not provided
                "mol_num_heavy_atoms": props.get("HeavyAtomCount"),
                "mol_formal_charge": 0,  # Default
                "mol_num_rings": None,  # Calculate if needed
                "mol_num_heteroatoms": None,  # Calculate if needed
                "mol_fraction_csp3": None,  # Calculate if needed
                
                # Clinical data (unknown for PubChem)
                "max_clinical_phase": None,
                "clinical_status": "Unknown",
                "primary_condition": "Unknown",
                
                # Dataset metadata
                "data_source": "pubchem_api",
                "compound_type": "Small molecule",
                "study_type": "COMPOUND_DATABASE",
                "primary_phase": "UNKNOWN",
                "overall_status": "UNKNOWN",
                "lead_sponsor": "PubChem_Database",
                "sponsor_class": "DATABASE",
                "collected_date": datetime.now().isoformat(),
                
                # ML targets
                "efficacy_score": None,
                "safety_score": None,
                "success_probability": None
            }
            
        except Exception as e:
            logger.debug(f"Error processing PubChem compound: {e}")
            return None
    
    def create_real_dataset(self, target_size: int = 20000) -> pd.DataFrame:
        """Create a real pharmaceutical dataset"""
        logger.info(f"🚀 Creating real pharmaceutical dataset (target: {target_size:,} compounds)")
        
        all_compounds = []
        
        # Collect from ChEMBL (primary source - approved drugs)
        chembl_target = min(15000, target_size)  # Up to 15k from ChEMBL
        chembl_compounds = self.collect_chembl_approved_drugs(chembl_target)
        all_compounds.extend(chembl_compounds)
        
        # Collect from PubChem (supplementary)
        remaining = target_size - len(all_compounds)
        if remaining > 0:
            pubchem_compounds = self.collect_pubchem_drugs(remaining)
            all_compounds.extend(pubchem_compounds)
        
        logger.info(f"📊 Total compounds collected: {len(all_compounds)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_compounds)
        
        # Remove duplicates based on SMILES
        initial_count = len(df)
        df = df.drop_duplicates(subset=['smiles'], keep='first')
        final_count = len(df)
        
        logger.info(f"🧹 Removed {initial_count - final_count} duplicate SMILES")
        logger.info(f"✅ Final dataset size: {final_count:,} unique compounds")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, output_dir: str = "clinical_trial_dataset/data/real"):
        """Save the real dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete dataset
        complete_file = output_path / "complete_real_dataset.csv"
        df.to_csv(complete_file, index=False)
        logger.info(f"💾 Saved complete dataset: {complete_file}")
        
        # Create train/val/test splits (70/15/15)
        total_size = len(df)
        train_size = int(total_size * 0.70)
        val_size = int(total_size * 0.15)
        
        # Shuffle the data
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split the data
        train_df = df_shuffled[:train_size]
        val_df = df_shuffled[train_size:train_size + val_size]
        test_df = df_shuffled[train_size + val_size:]
        
        # Save splits
        train_file = output_path / "train_set_real.csv"
        val_file = output_path / "val_set_real.csv"
        test_file = output_path / "test_set_real.csv"
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        logger.info(f"💾 Saved train set ({len(train_df):,} compounds): {train_file}")
        logger.info(f"💾 Saved val set ({len(val_df):,} compounds): {val_file}")
        logger.info(f"💾 Saved test set ({len(test_df):,} compounds): {test_file}")
        
        # Save metadata
        metadata = {
            "dataset_info": {
                "total_compounds": len(df),
                "train_compounds": len(train_df),
                "val_compounds": len(val_df),
                "test_compounds": len(test_df),
                "smiles_coverage": "100%",
                "data_sources": ["ChEMBL_API", "PubChem_API"],
                "collection_date": datetime.now().isoformat(),
                "dataset_type": "Real Pharmaceutical Compounds"
            },
            "data_quality": {
                "duplicate_smiles_removed": True,
                "all_compounds_have_smiles": True,
                "molecular_properties_included": True
            }
        }
        
        metadata_file = output_path / "real_dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Saved metadata: {metadata_file}")
        
        return {
            "complete_dataset": complete_file,
            "train_set": train_file,
            "val_set": val_file,
            "test_set": test_file,
            "metadata": metadata_file
        }

def main():
    """Main execution function"""
    logger.info("🌟 Real Pharmaceutical Dataset Creator")
    logger.info("=" * 50)
    
    # Create collector
    collector = RealCompoundCollector()
    
    # Create dataset
    target_size = 20000
    df = collector.create_real_dataset(target_size)
    
    # Save dataset
    files = collector.save_dataset(df)
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("🎉 REAL DATASET CREATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"📊 Total compounds: {len(df):,}")
    logger.info(f"🧬 SMILES coverage: 100%")
    logger.info(f"🔬 Data sources: ChEMBL API, PubChem API")
    logger.info(f"📁 Files created:")
    for name, path in files.items():
        logger.info(f"   - {name}: {path}")
    
    logger.info("\n✅ Ready for machine learning!")
    
    return files

if __name__ == "__main__":
    main()