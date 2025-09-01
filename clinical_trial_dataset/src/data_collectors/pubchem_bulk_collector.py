"""
PubChem Bulk Data Collector
Collects additional compound data from PubChem to supplement ChEMBL data
"""

import requests
import pandas as pd
import time
import logging
from typing import List, Dict, Optional
from tqdm import tqdm
from retrying import retry
import pubchempy as pcp
import config

class PubChemBulkCollector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.collected_compounds = []
        
    def get_drug_compounds_from_pubchem(self, start_cid: int = 1, target_compounds: int = 5000) -> pd.DataFrame:
        """Collect drug-like compounds from PubChem by CID range"""
        self.logger.info(f"Starting PubChem compound collection, target: {target_compounds}")
        
        current_cid = start_cid
        max_attempts = target_compounds * 10  # Safety limit
        attempts = 0
        
        with tqdm(desc="Collecting PubChem compounds", total=target_compounds) as pbar:
            while len(self.collected_compounds) < target_compounds and attempts < max_attempts:
                try:
                    attempts += 1
                    
                    # Fetch compound by CID
                    compound = pcp.Compound.from_cid(current_cid)
                    
                    if compound and compound.canonical_smiles:
                        # Filter for drug-like properties
                        if self._is_drug_like(compound):
                            record = self._extract_pubchem_record(compound)
                            if record:
                                self.collected_compounds.append(record)
                                
                                pbar.set_postfix({
                                    'collected': len(self.collected_compounds),
                                    'current_cid': current_cid,
                                    'smiles_rate': '100%'
                                })
                                pbar.update(1)
                    
                    current_cid += 1
                    
                    # Rate limiting for PubChem
                    if attempts % 10 == 0:
                        time.sleep(1)
                    
                except Exception as e:
                    # Skip problematic CIDs and continue
                    current_cid += 1
                    if attempts % 1000 == 0:
                        self.logger.debug(f"Processed {attempts} CIDs, found {len(self.collected_compounds)} compounds")
        
        df = pd.DataFrame(self.collected_compounds)
        self.logger.info(f"PubChem collection completed. Total compounds: {len(df)}")
        
        return df
    
    def _is_drug_like(self, compound) -> bool:
        """Check if compound meets drug-like criteria (Lipinski's Rule of Five)"""
        try:
            # Get molecular properties
            mw = compound.molecular_weight
            logp = compound.xlogp
            hbd = compound.h_bond_donor_count
            hba = compound.h_bond_acceptor_count
            
            # Apply Lipinski's Rule of Five
            if (mw and mw <= 500 and 
                logp and logp <= 5 and
                hbd and hbd <= 5 and
                hba and hba <= 10):
                return True
            
            return False
            
        except:
            return False
    
    def _extract_pubchem_record(self, compound) -> Optional[Dict]:
        """Extract record from PubChem compound"""
        try:
            # Get synonyms/names
            synonyms = compound.synonyms[:5] if compound.synonyms else []
            primary_name = synonyms[0] if synonyms else f"PubChem_{compound.cid}"
            
            record = {
                # Identifiers
                "pubchem_cid": compound.cid,
                "primary_drug": primary_name,
                "all_drug_names": synonyms,
                
                # Chemical structure
                "smiles": compound.canonical_smiles,
                "smiles_source": f"pubchem_cid_{compound.cid}",
                "mapping_status": "success",
                
                # Molecular properties
                "mol_molecular_weight": compound.molecular_weight,
                "mol_logp": compound.xlogp,
                "mol_num_hbd": compound.h_bond_donor_count,
                "mol_num_hba": compound.h_bond_acceptor_count,
                "mol_num_rotatable_bonds": compound.rotatable_bond_count,
                "mol_tpsa": compound.tpsa,
                "mol_num_heavy_atoms": compound.heavy_atom_count,
                "mol_formal_charge": compound.charge,
                
                # Placeholder clinical info (since these are compounds, not trials)
                "max_clinical_phase": 0,
                "clinical_status": "Compound_Database",
                
                # Metadata
                "data_source": "pubchem_bulk",
                "collected_date": pd.Timestamp.now().isoformat(),
                "compound_type": "Small molecule",
                
                # ML compatibility fields
                "primary_condition": "",
                "study_type": "COMPOUND_DATABASE",
                "primary_phase": "COMPOUND",
                "enrollment_count": 0,
                "overall_status": "COMPOUND_RECORD",
                "lead_sponsor": "PubChem_Database",
                "sponsor_class": "DATABASE"
            }
            
            return record
            
        except Exception as e:
            self.logger.warning(f"Error extracting PubChem record for CID {compound.cid}: {e}")
            return None

# Test function
def test_pubchem_collector():
    collector = PubChemBulkCollector()
    
    print("Testing PubChem bulk collection with 50 compounds...")
    test_data = collector.get_drug_compounds_from_pubchem(start_cid=1, target_compounds=50)
    print(f"Test completed. Shape: {test_data.shape}")
    print(f"All have SMILES: {test_data['smiles'].notna().all()}")
    
    return test_data

if __name__ == "__main__":
    test_pubchem_collector()