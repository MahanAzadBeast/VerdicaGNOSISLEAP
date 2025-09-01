"""
ChEMBL Bulk Data Collector
Collects large-scale compound data from ChEMBL to reach 20k+ compounds with SMILES
"""

import requests
import pandas as pd
import time
import logging
from typing import List, Dict, Optional
from tqdm import tqdm
from retrying import retry
import config

class ChEMBLBulkCollector:
    def __init__(self):
        self.base_url = config.CHEMBL_BASE_URL
        self.logger = logging.getLogger(__name__)
        self.collected_compounds = []
        
    @retry(wait_fixed=1000, stop_max_attempt_number=3)
    def fetch_compounds_batch(self, offset: int = 0, limit: int = 1000) -> Dict:
        """Fetch a batch of compounds from ChEMBL"""
        params = {
            "format": "json",
            "limit": limit,
            "offset": offset,
            # Filter for drug-like compounds with bioactivity data
            "molecule_type": "Small molecule",
            "max_phase__gte": 1,  # At least Phase 1 (clinical relevance)
        }
        
        url = f"{self.base_url}/molecule"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def extract_compound_record(self, molecule: Dict) -> Optional[Dict]:
        """Extract relevant information from a ChEMBL molecule"""
        try:
            # Get basic molecule info
            chembl_id = molecule.get("molecule_chembl_id")
            if not chembl_id:
                return None
            
            # Get SMILES from structure
            structures = molecule.get("molecule_structures", {})
            smiles = structures.get("canonical_smiles")
            
            # Skip if no SMILES
            if not smiles:
                return None
            
            # Get molecular properties
            properties = molecule.get("molecule_properties", {})
            
            # Get clinical development info
            max_phase = molecule.get("max_phase", 0)
            
            # Get drug names/synonyms
            pref_name = molecule.get("pref_name", "")
            synonyms = molecule.get("molecule_synonyms", [])
            drug_names = [pref_name] + [s.get("molecule_synonym") for s in synonyms if s.get("molecule_synonym")]
            drug_names = [name for name in drug_names if name]  # Remove empty names
            
            record = {
                # Identifiers
                "chembl_id": chembl_id,
                "primary_drug": pref_name or chembl_id,
                "all_drug_names": drug_names,
                
                # Chemical structure
                "smiles": smiles,
                "smiles_source": f"chembl_{chembl_id}",
                "mapping_status": "success",
                
                # Molecular properties (if available)
                "mol_molecular_weight": properties.get("mw_freebase"),
                "mol_logp": properties.get("alogp"),
                "mol_num_hbd": properties.get("hbd"),
                "mol_num_hba": properties.get("hba"),
                "mol_num_rotatable_bonds": properties.get("rtb"),
                "mol_tpsa": properties.get("psa"),
                "mol_num_aromatic_rings": properties.get("aromatic_rings"),
                "mol_num_heavy_atoms": properties.get("heavy_atoms"),
                
                # Clinical development info
                "max_clinical_phase": max_phase,
                "clinical_status": self._get_clinical_status(max_phase),
                
                # Metadata
                "data_source": "chembl_bulk",
                "collected_date": pd.Timestamp.now().isoformat(),
                "compound_type": molecule.get("molecule_type", "Small molecule"),
                
                # Placeholder fields for ML compatibility
                "primary_condition": "",
                "study_type": "COMPOUND_DATABASE",
                "primary_phase": f"PHASE{max_phase}" if max_phase > 0 else "PRECLINICAL",
                "enrollment_count": 0,
                "overall_status": "COMPOUND_RECORD",
                "lead_sponsor": "ChEMBL_Database",
                "sponsor_class": "DATABASE"
            }
            
            return record
            
        except Exception as e:
            self.logger.warning(f"Error extracting compound record: {e}")
            return None
    
    def _get_clinical_status(self, max_phase: int) -> str:
        """Convert max phase to clinical status"""
        phase_map = {
            0: "Preclinical",
            1: "Phase I",
            2: "Phase II", 
            3: "Phase III",
            4: "Approved/Phase IV"
        }
        return phase_map.get(max_phase, "Unknown")
    
    def collect_compounds(self, target_compounds: int = 20000) -> pd.DataFrame:
        """Collect compound data from ChEMBL to reach target"""
        self.logger.info(f"Starting ChEMBL compound collection, target: {target_compounds}")
        
        offset = 0
        batch_size = 1000
        
        with tqdm(desc="Collecting ChEMBL compounds", total=target_compounds) as pbar:
            while len(self.collected_compounds) < target_compounds:
                try:
                    # Fetch batch
                    batch_data = self.fetch_compounds_batch(offset=offset, limit=batch_size)
                    molecules = batch_data.get("molecules", [])
                    
                    if not molecules:
                        self.logger.info("No more molecules available")
                        break
                    
                    # Process each molecule
                    batch_records = []
                    for molecule in molecules:
                        record = self.extract_compound_record(molecule)
                        if record:
                            batch_records.append(record)
                    
                    self.collected_compounds.extend(batch_records)
                    
                    pbar.set_postfix({
                        'collected': len(self.collected_compounds),
                        'batch_valid': len(batch_records),
                        'smiles_rate': '100%'  # All ChEMBL compounds have SMILES
                    })
                    pbar.update(len(batch_records))
                    
                    self.logger.info(f"Collected {len(batch_records)} valid compounds, "
                                   f"Total: {len(self.collected_compounds)}")
                    
                    offset += batch_size
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                    # Break if we've reached target
                    if len(self.collected_compounds) >= target_compounds:
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error fetching ChEMBL batch at offset {offset}: {e}")
                    # Try to continue with next batch
                    offset += batch_size
                    if offset > target_compounds * 2:  # Safety limit
                        break
        
        df = pd.DataFrame(self.collected_compounds)
        self.logger.info(f"ChEMBL collection completed. Total compounds: {len(df)}")
        
        return df

# Test function
def test_chembl_collector():
    collector = ChEMBLBulkCollector()
    
    print("Testing ChEMBL bulk collection with 100 compounds...")
    test_data = collector.collect_compounds(target_compounds=100)
    print(f"Test completed. Shape: {test_data.shape}")
    print(f"Columns: {list(test_data.columns)}")
    print(f"Sample compounds: {test_data['primary_drug'].head().tolist()}")
    print(f"All have SMILES: {test_data['smiles'].notna().all()}")
    
    return test_data

if __name__ == "__main__":
    test_chembl_collector()