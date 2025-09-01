"""
Drug Name to SMILES Mapper
Maps drug names to SMILES strings using multiple public APIs
"""

import requests
import pandas as pd
import time
import logging
from typing import Optional, Tuple, List
from tqdm import tqdm
from retrying import retry
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors
import config

class SMILESMapper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mapping_cache = {}
        self.failed_mappings = []
        
    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string using RDKit"""
        if not smiles or pd.isna(smiles):
            return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def clean_drug_name(self, drug_name: str) -> str:
        """Clean and standardize drug name for better matching"""
        if not drug_name:
            return ""
        
        # Remove common suffixes and prefixes
        drug_name = drug_name.strip().lower()
        
        # Remove dosage information
        import re
        drug_name = re.sub(r'\d+\s*(mg|g|ml|mcg|units?).*', '', drug_name)
        drug_name = re.sub(r'\([^)]*\)', '', drug_name)  # Remove parentheses content
        
        # Remove common pharmaceutical terms
        remove_terms = ['tablet', 'capsule', 'injection', 'solution', 'suspension', 
                       'oral', 'iv', 'intravenous', 'topical', 'cream', 'gel']
        
        for term in remove_terms:
            drug_name = drug_name.replace(term, '').strip()
        
        # Remove extra whitespace
        drug_name = ' '.join(drug_name.split())
        
        return drug_name
    
    @retry(wait_fixed=1000, stop_max_attempt_number=3)
    def get_smiles_pubchem(self, drug_name: str) -> Optional[Tuple[str, str]]:
        """Get SMILES from PubChem (free, public domain)"""
        try:
            # Try exact name match first
            compounds = pcp.get_compounds(drug_name, 'name')
            if compounds and len(compounds) > 0:
                smiles = compounds[0].canonical_smiles
                if self.validate_smiles(smiles):
                    return smiles, f"pubchem_name_{compounds[0].cid}"
            
            # Try synonym search with different search types
            search_types = ['name', 'synonym']
            for search_type in search_types:
                try:
                    compounds = pcp.get_compounds(drug_name, search_type)
                    if compounds:
                        for compound in compounds[:3]:  # Check top 3 matches
                            smiles = compound.canonical_smiles
                            if self.validate_smiles(smiles):
                                return smiles, f"pubchem_{search_type}_{compound.cid}"
                except:
                    continue
            
            return None, None
            
        except Exception as e:
            self.logger.debug(f"PubChem lookup failed for {drug_name}: {e}")
            return None, None
    
    @retry(wait_fixed=1000, stop_max_attempt_number=3)
    def get_smiles_chembl(self, drug_name: str) -> Optional[Tuple[str, str]]:
        """Get SMILES from ChEMBL (CC BY-SA 3.0 - commercial OK with attribution)"""
        try:
            # Search for molecule
            search_url = f"{config.CHEMBL_BASE_URL}/molecule/search"
            params = {"q": drug_name, "format": "json", "limit": 5}
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("molecules"):
                return None, None
            
            # Try each result
            for molecule in data["molecules"][:3]:
                chembl_id = molecule.get("molecule_chembl_id")
                if not chembl_id:
                    continue
                
                # Get detailed molecule info
                mol_url = f"{config.CHEMBL_BASE_URL}/molecule/{chembl_id}"
                mol_response = requests.get(mol_url, params={"format": "json"}, timeout=10)
                mol_response.raise_for_status()
                mol_data = mol_response.json()
                
                structures = mol_data.get("molecule_structures", {})
                smiles = structures.get("canonical_smiles")
                
                if smiles and self.validate_smiles(smiles):
                    return smiles, f"chembl_{chembl_id}"
            
            return None, None
            
        except Exception as e:
            self.logger.debug(f"ChEMBL lookup failed for {drug_name}: {e}")
            return None, None
    
    def get_smiles_drugbank_api(self, drug_name: str) -> Optional[Tuple[str, str]]:
        """
        Get SMILES from DrugBank API (requires API key for commercial use)
        This is a placeholder - you would need to license DrugBank for commercial use
        """
        # TODO: Implement DrugBank API if you get commercial license
        # For now, return None
        return None, None
    
    def manual_drug_mappings(self) -> dict:
        """Manual mappings for common drugs that might not be found automatically"""
        return {
            # Common drugs with known SMILES
            "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "acetaminophen": "CC(=O)NC1=CC=C(C=C1)O",
            "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "metformin": "CN(C)C(=N)N=C(N)N",
            "atorvastatin": "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
            
            # Brand name mappings
            "botox": None,  # Botulinum toxin - protein, no single SMILES
            "astragalus": None,  # Herbal extract - complex mixture
            "astragalus powder": None,  # Herbal extract - complex mixture
            
            # Common generic names
            "dexamethasone": "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
            "lidocaine": "CCN(CC)CC(=O)Nc1c(C)cccc1C",
            "insulin": None,  # Protein - too large for SMILES representation
            
            # Add more as needed
        }
    
    def get_smiles_for_drug(self, drug_name: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        Get SMILES for a single drug using multiple strategies
        Returns: (smiles, source, status)
        """
        if not drug_name:
            return None, None, "empty_name"
        
        # Check cache first
        cache_key = drug_name.lower().strip()
        if cache_key in self.mapping_cache:
            cached = self.mapping_cache[cache_key]
            return cached['smiles'], cached['source'], 'cached'
        
        # Clean the drug name
        clean_name = self.clean_drug_name(drug_name)
        
        # Check manual mappings
        manual_mappings = self.manual_drug_mappings()
        if clean_name in manual_mappings:
            smiles = manual_mappings[clean_name]
            if smiles is None:
                # Explicitly marked as unmappable
                self.mapping_cache[cache_key] = {'smiles': None, 'source': 'manual_unmappable'}
                return None, 'manual_unmappable', 'unmappable'
            else:
                self.mapping_cache[cache_key] = {'smiles': smiles, 'source': 'manual'}
                return smiles, 'manual', 'success'
        
        # Try different names
        names_to_try = [drug_name, clean_name]
        if drug_name != clean_name:
            names_to_try.append(drug_name.split()[0])  # Try first word only
        
        # Try each source
        sources = [
            ('pubchem', self.get_smiles_pubchem),
            ('chembl', self.get_smiles_chembl),
            # ('drugbank', self.get_smiles_drugbank_api),  # Uncomment if you license DrugBank
        ]
        
        for name_variant in names_to_try:
            for source_name, source_func in sources:
                try:
                    smiles, source_id = source_func(name_variant)
                    if smiles:
                        # Cache the result
                        self.mapping_cache[cache_key] = {
                            'smiles': smiles, 
                            'source': source_id or source_name
                        }
                        return smiles, source_id or source_name, 'success'
                    
                    # Rate limiting between API calls
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.debug(f"Error with {source_name} for {name_variant}: {e}")
                    continue
        
        # No SMILES found
        self.failed_mappings.append({
            'original_name': drug_name,
            'clean_name': clean_name,
            'reason': 'not_found'
        })
        
        return None, None, 'not_found'
    
    def calculate_molecular_features(self, smiles: str) -> dict:
        """Calculate molecular descriptors from SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            features = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'num_hbd': Descriptors.NumHDonors(mol),
                'num_hba': Descriptors.NumHAcceptors(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                'num_heavy_atoms': Descriptors.HeavyAtomCount(mol),
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
            }
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Error calculating molecular features for {smiles}: {e}")
            return {}
    
    def map_trials_to_smiles(self, trials_df: pd.DataFrame) -> pd.DataFrame:
        """Map all drugs in trials dataframe to SMILES"""
        self.logger.info(f"Mapping {len(trials_df)} trials to SMILES...")
        
        results = []
        unique_drugs = trials_df['primary_drug'].unique()
        
        self.logger.info(f"Found {len(unique_drugs)} unique drugs to map")
        
        # Create drug to SMILES mapping
        drug_smiles_map = {}
        
        with tqdm(total=len(unique_drugs), desc="Mapping drugs to SMILES") as pbar:
            for drug_name in unique_drugs:
                smiles, source, status = self.get_smiles_for_drug(drug_name)
                
                drug_smiles_map[drug_name] = {
                    'smiles': smiles,
                    'smiles_source': source,
                    'mapping_status': status
                }
                
                pbar.set_postfix({
                    'success_rate': f"{len([d for d in drug_smiles_map.values() if d['smiles']])/len(drug_smiles_map)*100:.1f}%"
                })
                pbar.update(1)
        
        # Apply mapping to trials dataframe
        for idx, trial in trials_df.iterrows():
            drug_name = trial['primary_drug']
            mapping_info = drug_smiles_map.get(drug_name, {})
            
            trial_copy = trial.copy()
            trial_copy['smiles'] = mapping_info.get('smiles')
            trial_copy['smiles_source'] = mapping_info.get('smiles_source')
            trial_copy['mapping_status'] = mapping_info.get('mapping_status')
            
            # Add molecular features if SMILES available
            if trial_copy['smiles']:
                mol_features = self.calculate_molecular_features(trial_copy['smiles'])
                for feature_name, feature_value in mol_features.items():
                    trial_copy[f'mol_{feature_name}'] = feature_value
            
            results.append(trial_copy)
        
        result_df = pd.DataFrame(results)
        
        # Log statistics
        total_trials = len(result_df)
        successful_mappings = len(result_df[result_df['smiles'].notna()])
        success_rate = successful_mappings / total_trials * 100
        
        self.logger.info(f"SMILES mapping completed:")
        self.logger.info(f"  Total trials: {total_trials}")
        self.logger.info(f"  Successful mappings: {successful_mappings}")
        self.logger.info(f"  Success rate: {success_rate:.1f}%")
        self.logger.info(f"  Failed mappings: {len(self.failed_mappings)}")
        
        # Save intermediate results
        processed_path = "data/processed/trials_with_smiles.csv"
        result_df.to_csv(processed_path, index=False)
        self.logger.info(f"Data with SMILES saved to {processed_path}")
        
        # Save failed mappings for analysis
        if self.failed_mappings:
            failed_df = pd.DataFrame(self.failed_mappings)
            failed_path = "data/processed/failed_smiles_mappings.csv"
            failed_df.to_csv(failed_path, index=False)
            self.logger.info(f"Failed mappings saved to {failed_path}")
        
        return result_df

# Test function
def test_smiles_mapper():
    mapper = SMILESMapper()
    
    # Test with known drugs
    test_drugs = ["aspirin", "metformin", "atorvastatin", "unknown_drug_12345"]
    
    print("Testing SMILES mapping...")
    for drug in test_drugs:
        smiles, source, status = mapper.get_smiles_for_drug(drug)
        print(f"{drug}: {smiles[:50] if smiles else 'None'} ({source}, {status})")
    
    return mapper

if __name__ == "__main__":
    test_smiles_mapper()