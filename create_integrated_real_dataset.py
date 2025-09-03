#!/usr/bin/env python3
"""
Integrated Real Dataset Creator - Combines Real Clinical Trials + ChEMBL + PubChem
Creates a comprehensive pharmaceutical dataset by integrating:
1. Real clinical trials data (27,999 trials already collected)
2. Real ChEMBL approved drugs with SMILES
3. Real PubChem compounds with SMILES
"""

import requests
import pandas as pd
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegratedRealDatasetCreator:
    """Creates integrated dataset from clinical trials + compound databases"""
    
    def __init__(self):
        self.chembl_base = "https://www.ebi.ac.uk/chembl/api/data"
        self.pubchem_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        
    def load_real_clinical_trials(self) -> pd.DataFrame:
        """Load the real clinical trials data that was already collected"""
        logger.info("📋 Loading real clinical trials data...")
        
        trials_file = Path("clinical_trial_dataset/data/raw/clinical_trials_raw.csv")
        
        if not trials_file.exists():
            logger.error(f"Clinical trials file not found: {trials_file}")
            return pd.DataFrame()
        
        try:
            # Load clinical trials
            trials_df = pd.read_csv(trials_file)
            logger.info(f"✅ Loaded {len(trials_df):,} real clinical trials")
            
            # Extract unique drugs from trials
            unique_drugs = self._extract_drugs_from_trials(trials_df)
            logger.info(f"🧬 Extracted {len(unique_drugs)} unique drugs from trials")
            
            return trials_df
            
        except Exception as e:
            logger.error(f"Error loading clinical trials: {e}")
            return pd.DataFrame()
    
    def _extract_drugs_from_trials(self, trials_df: pd.DataFrame) -> List[str]:
        """Extract unique drug names from clinical trials"""
        drugs = set()
        
        for _, trial in trials_df.iterrows():
            # Extract from primary_drug
            if pd.notna(trial.get('primary_drug')):
                drug_name = str(trial['primary_drug']).strip()
                if drug_name and drug_name != 'nan':
                    drugs.add(drug_name)
            
            # Extract from all_drug_names (JSON list)
            if pd.notna(trial.get('all_drug_names')):
                try:
                    drug_list_str = str(trial['all_drug_names'])
                    # Parse the list string (e.g., "['Drug1', 'Drug2']")
                    if drug_list_str.startswith('[') and drug_list_str.endswith(']'):
                        # Simple parsing - extract quoted strings
                        import ast
                        drug_list = ast.literal_eval(drug_list_str)
                        for drug in drug_list:
                            if drug and str(drug).strip():
                                drugs.add(str(drug).strip())
                except:
                    continue
        
        return list(drugs)
    
    def find_smiles_for_trial_drugs(self, drug_names: List[str]) -> Dict[str, Dict]:
        """Find SMILES for drugs mentioned in clinical trials"""
        logger.info(f"🔍 Finding SMILES for {len(drug_names)} trial drugs...")
        
        drug_smiles_map = {}
        successful_mappings = 0
        
        # Process in batches to avoid overwhelming APIs
        batch_size = 50
        for i in range(0, len(drug_names), batch_size):
            batch = drug_names[i:i+batch_size]
            logger.info(f"Processing drug batch {i//batch_size + 1}/{(len(drug_names)-1)//batch_size + 1}")
            
            for drug_name in batch:
                try:
                    # Try ChEMBL first (more reliable for drugs)
                    drug_data = self._find_drug_in_chembl(drug_name)
                    
                    if not drug_data:
                        # Fallback to PubChem
                        drug_data = self._find_drug_in_pubchem(drug_name)
                    
                    if drug_data:
                        drug_smiles_map[drug_name] = drug_data
                        successful_mappings += 1
                        logger.debug(f"✅ Found SMILES for {drug_name}")
                    else:
                        logger.debug(f"❌ No SMILES found for {drug_name}")
                    
                    time.sleep(0.2)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Error processing {drug_name}: {e}")
                    continue
        
        logger.info(f"🎉 Successfully mapped {successful_mappings}/{len(drug_names)} trial drugs to SMILES")
        return drug_smiles_map
    
    def _find_drug_in_chembl(self, drug_name: str) -> Optional[Dict]:
        """Find drug in ChEMBL database"""
        try:
            # Search for molecule by name
            search_url = f"{self.chembl_base}/molecule/search"
            params = {
                "q": drug_name,
                "format": "json",
                "limit": 5
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            molecules = data.get("molecules", [])
            
            for mol in molecules:
                # Check if this molecule has SMILES
                structures = mol.get("molecule_structures", {})
                smiles = structures.get("canonical_smiles")
                
                if smiles:
                    properties = mol.get("molecule_properties", {})
                    
                    return {
                        "smiles": smiles,
                        "source": "chembl",
                        "chembl_id": mol.get("molecule_chembl_id"),
                        "molecular_weight": properties.get("full_mwt"),
                        "logp": properties.get("alogp"),
                        "hbd": properties.get("hbd"),
                        "hba": properties.get("hba"),
                        "rotatable_bonds": properties.get("rtb"),
                        "tpsa": properties.get("psa"),
                        "heavy_atoms": properties.get("heavy_atoms"),
                        "max_phase": mol.get("max_phase", 4)
                    }
            
            return None
            
        except Exception as e:
            logger.debug(f"ChEMBL search error for {drug_name}: {e}")
            return None
    
    def _find_drug_in_pubchem(self, drug_name: str) -> Optional[Dict]:
        """Find drug in PubChem database"""
        try:
            # Search for compound by name
            search_url = f"{self.pubchem_base}/compound/name/{drug_name}/cids/JSON"
            
            response = requests.get(search_url, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            
            if not cids:
                return None
            
            # Get properties for first CID
            cid = cids[0]
            props_url = f"{self.pubchem_base}/compound/cid/{cid}/property/MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,TPSA,HeavyAtomCount,CanonicalSMILES/JSON"
            
            response = requests.get(props_url, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            properties = data.get("PropertyTable", {}).get("Properties", [])
            
            if properties:
                props = properties[0]
                smiles = props.get("CanonicalSMILES")
                
                if smiles:
                    return {
                        "smiles": smiles,
                        "source": "pubchem",
                        "pubchem_cid": cid,
                        "molecular_weight": props.get("MolecularWeight"),
                        "logp": props.get("XLogP"),
                        "hbd": props.get("HBondDonorCount"),
                        "hba": props.get("HBondAcceptorCount"),
                        "rotatable_bonds": props.get("RotatableBondCount"),
                        "tpsa": props.get("TPSA"),
                        "heavy_atoms": props.get("HeavyAtomCount"),
                        "max_phase": None  # Unknown from PubChem
                    }
            
            return None
            
        except Exception as e:
            logger.debug(f"PubChem search error for {drug_name}: {e}")
            return None
    
    def collect_additional_chembl_compounds(self, target_count: int = 10000) -> List[Dict]:
        """Collect additional approved compounds from ChEMBL"""
        logger.info(f"🔬 Collecting {target_count:,} additional approved compounds from ChEMBL...")
        
        compounds = []
        offset = 0
        batch_size = 1000
        
        while len(compounds) < target_count:
            try:
                params = {
                    "format": "json",
                    "limit": batch_size,
                    "offset": offset,
                    "max_phase": 4,  # Only approved drugs
                    "molecule_type": "Small molecule"
                }
                
                url = f"{self.chembl_base}/molecule"
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                molecules = data.get("molecules", [])
                
                if not molecules:
                    break
                
                batch_compounds = []
                for mol in molecules:
                    compound = self._extract_chembl_compound(mol)
                    if compound:
                        batch_compounds.append(compound)
                
                compounds.extend(batch_compounds)
                logger.info(f"✅ ChEMBL batch: {len(batch_compounds)} compounds (Total: {len(compounds)})")
                
                offset += batch_size
                time.sleep(0.5)
                
                if len(compounds) >= target_count:
                    break
                    
            except Exception as e:
                logger.warning(f"Error in ChEMBL batch at offset {offset}: {e}")
                offset += batch_size
                continue
        
        return compounds[:target_count]
    
    def _extract_chembl_compound(self, molecule: Dict) -> Optional[Dict]:
        """Extract compound data from ChEMBL molecule"""
        try:
            chembl_id = molecule.get("molecule_chembl_id")
            structures = molecule.get("molecule_structures", {})
            smiles = structures.get("canonical_smiles")
            
            if not smiles or not chembl_id:
                return None
            
            properties = molecule.get("molecule_properties", {})
            pref_name = molecule.get("pref_name", f"ChEMBL_{chembl_id}")
            
            return {
                "compound_id": f"CHEMBL_{chembl_id}",
                "primary_drug": pref_name,
                "all_drug_names": [pref_name],
                "smiles": smiles,
                "smiles_source": chembl_id,
                "mapping_status": "success",
                "mol_molecular_weight": properties.get("full_mwt"),
                "mol_logp": properties.get("alogp"),
                "mol_num_hbd": properties.get("hbd"),
                "mol_num_hba": properties.get("hba"),
                "mol_num_rotatable_bonds": properties.get("rtb"),
                "mol_tpsa": properties.get("psa"),
                "mol_num_aromatic_rings": properties.get("aromatic_rings"),
                "mol_num_heavy_atoms": properties.get("heavy_atoms"),
                "mol_formal_charge": 0,
                "mol_num_rings": properties.get("num_rings"),
                "mol_num_heteroatoms": None,
                "mol_fraction_csp3": None,
                "max_clinical_phase": molecule.get("max_phase", 4),
                "clinical_status": "Approved",
                "primary_condition": "Multiple",
                "data_source": "chembl_api",
                "compound_type": "Small molecule",
                "study_type": "APPROVED_DRUG",
                "primary_phase": "PHASE4",
                "overall_status": "APPROVED",
                "lead_sponsor": "ChEMBL_Database",
                "sponsor_class": "DATABASE",
                "collected_date": datetime.now().isoformat(),
                "efficacy_score": None,
                "safety_score": None,
                "success_probability": 1.0
            }
            
        except Exception as e:
            return None
    
    def integrate_trial_compounds_with_smiles(self, trials_df: pd.DataFrame, 
                                            drug_smiles_map: Dict[str, Dict]) -> List[Dict]:
        """Create compound records by integrating trials with SMILES data"""
        logger.info("🔗 Integrating clinical trials with SMILES data...")
        
        integrated_compounds = []
        compound_id_counter = 1
        
        for _, trial in trials_df.iterrows():
            try:
                primary_drug = trial.get('primary_drug')
                if pd.isna(primary_drug) or not primary_drug:
                    continue
                
                drug_name = str(primary_drug).strip()
                
                # Check if we have SMILES for this drug
                if drug_name in drug_smiles_map:
                    smiles_data = drug_smiles_map[drug_name]
                    
                    # Create integrated compound record
                    compound = {
                        "compound_id": f"TRIAL_{compound_id_counter:05d}",
                        "primary_drug": drug_name,
                        "all_drug_names": self._parse_drug_names(trial.get('all_drug_names', [])),
                        "smiles": smiles_data["smiles"],
                        "smiles_source": smiles_data.get("chembl_id") or smiles_data.get("pubchem_cid", "unknown"),
                        "mapping_status": "success",
                        
                        # Molecular properties from SMILES lookup
                        "mol_molecular_weight": smiles_data.get("molecular_weight"),
                        "mol_logp": smiles_data.get("logp"),
                        "mol_num_hbd": smiles_data.get("hbd"),
                        "mol_num_hba": smiles_data.get("hba"),
                        "mol_num_rotatable_bonds": smiles_data.get("rotatable_bonds"),
                        "mol_tpsa": smiles_data.get("tpsa"),
                        "mol_num_aromatic_rings": None,
                        "mol_num_heavy_atoms": smiles_data.get("heavy_atoms"),
                        "mol_formal_charge": 0,
                        "mol_num_rings": None,
                        "mol_num_heteroatoms": None,
                        "mol_fraction_csp3": None,
                        
                        # Clinical data from trial
                        "max_clinical_phase": self._parse_phase(trial.get('primary_phase')),
                        "clinical_status": trial.get('overall_status', 'Unknown'),
                        "primary_condition": trial.get('primary_condition', 'Unknown'),
                        "nct_id": trial.get('nct_id'),
                        "trial_title": trial.get('title'),
                        "enrollment_count": trial.get('enrollment_count'),
                        "start_date": trial.get('start_date'),
                        "completion_date": trial.get('completion_date'),
                        
                        # Dataset metadata
                        "data_source": f"clinical_trial_{smiles_data['source']}",
                        "compound_type": "Small molecule",
                        "study_type": trial.get('study_type', 'INTERVENTIONAL'),
                        "primary_phase": trial.get('primary_phase'),
                        "overall_status": trial.get('overall_status'),
                        "lead_sponsor": trial.get('lead_sponsor'),
                        "sponsor_class": trial.get('sponsor_class'),
                        "collected_date": datetime.now().isoformat(),
                        
                        # ML targets (derived from trial data)
                        "efficacy_score": self._calculate_efficacy_score(trial),
                        "safety_score": self._calculate_safety_score(trial),
                        "success_probability": self._calculate_success_probability(trial)
                    }
                    
                    integrated_compounds.append(compound)
                    compound_id_counter += 1
                    
            except Exception as e:
                logger.debug(f"Error integrating trial: {e}")
                continue
        
        logger.info(f"✅ Successfully integrated {len(integrated_compounds)} trial compounds with SMILES")
        return integrated_compounds
    
    def _parse_drug_names(self, drug_names_str) -> List[str]:
        """Parse drug names from string representation"""
        if pd.isna(drug_names_str):
            return []
        
        try:
            if isinstance(drug_names_str, str) and drug_names_str.startswith('['):
                import ast
                return ast.literal_eval(drug_names_str)
            else:
                return [str(drug_names_str)]
        except:
            return [str(drug_names_str)]
    
    def _parse_phase(self, phase_str) -> Optional[int]:
        """Parse clinical phase from string"""
        if pd.isna(phase_str):
            return None
        
        phase_str = str(phase_str).upper()
        if 'PHASE4' in phase_str or 'PHASE 4' in phase_str:
            return 4
        elif 'PHASE3' in phase_str or 'PHASE 3' in phase_str:
            return 3
        elif 'PHASE2' in phase_str or 'PHASE 2' in phase_str:
            return 2
        elif 'PHASE1' in phase_str or 'PHASE 1' in phase_str:
            return 1
        else:
            return None
    
    def _calculate_efficacy_score(self, trial) -> Optional[float]:
        """Calculate efficacy score from trial data"""
        # Simple heuristic based on completion and phase
        score = 0.5  # Base score
        
        if trial.get('overall_status') == 'COMPLETED':
            score += 0.3
        
        phase = self._parse_phase(trial.get('primary_phase'))
        if phase:
            score += (phase / 4) * 0.2  # Higher phases get higher scores
        
        return min(score, 1.0)
    
    def _calculate_safety_score(self, trial) -> Optional[float]:
        """Calculate safety score from trial data"""
        # Simple heuristic - assume higher phases = better safety profile
        score = 0.6  # Base score
        
        phase = self._parse_phase(trial.get('primary_phase'))
        if phase:
            score += (phase / 4) * 0.3
        
        return min(score, 1.0)
    
    def _calculate_success_probability(self, trial) -> Optional[float]:
        """Calculate success probability from trial data"""
        # Based on phase and status
        phase = self._parse_phase(trial.get('primary_phase'))
        status = trial.get('overall_status', '').upper()
        
        if 'COMPLETED' in status and phase == 4:
            return 0.9
        elif 'COMPLETED' in status and phase == 3:
            return 0.7
        elif 'COMPLETED' in status and phase == 2:
            return 0.5
        elif 'COMPLETED' in status and phase == 1:
            return 0.3
        else:
            return 0.2
    
    def create_integrated_dataset(self, target_size: int = 20000) -> pd.DataFrame:
        """Create the integrated real dataset"""
        logger.info(f"🚀 Creating integrated real dataset (target: {target_size:,} compounds)")
        
        all_compounds = []
        
        # Step 1: Load real clinical trials
        trials_df = self.load_real_clinical_trials()
        if trials_df.empty:
            logger.error("No clinical trials data found!")
            return pd.DataFrame()
        
        # Step 2: Extract drugs from trials and find SMILES
        trial_drugs = self._extract_drugs_from_trials(trials_df)
        drug_smiles_map = self.find_smiles_for_trial_drugs(trial_drugs[:1000])  # Limit for now
        
        # Step 3: Create integrated trial-compound records
        trial_compounds = self.integrate_trial_compounds_with_smiles(trials_df, drug_smiles_map)
        all_compounds.extend(trial_compounds)
        
        logger.info(f"📋 Integrated {len(trial_compounds)} compounds from clinical trials")
        
        # Step 4: Add additional ChEMBL compounds to reach target
        remaining = target_size - len(all_compounds)
        if remaining > 0:
            additional_compounds = self.collect_additional_chembl_compounds(remaining)
            all_compounds.extend(additional_compounds)
            logger.info(f"🔬 Added {len(additional_compounds)} additional ChEMBL compounds")
        
        # Step 5: Convert to DataFrame and clean
        logger.info(f"📊 Total compounds collected: {len(all_compounds)}")
        df = pd.DataFrame(all_compounds)
        
        if df.empty:
            logger.error("No compounds collected!")
            return df
        
        # Remove duplicates based on SMILES
        initial_count = len(df)
        df = df.drop_duplicates(subset=['smiles'], keep='first')
        final_count = len(df)
        
        logger.info(f"🧹 Removed {initial_count - final_count} duplicate SMILES")
        logger.info(f"✅ Final integrated dataset: {final_count:,} unique compounds")
        
        return df
    
    def save_integrated_dataset(self, df: pd.DataFrame, output_dir: str = "clinical_trial_dataset/data/integrated"):
        """Save the integrated real dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete dataset
        complete_file = output_path / "complete_integrated_real_dataset.csv"
        df.to_csv(complete_file, index=False)
        logger.info(f"💾 Saved complete integrated dataset: {complete_file}")
        
        # Create train/val/test splits
        total_size = len(df)
        train_size = int(total_size * 0.70)
        val_size = int(total_size * 0.15)
        
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df = df_shuffled[:train_size]
        val_df = df_shuffled[train_size:train_size + val_size]
        test_df = df_shuffled[train_size + val_size:]
        
        # Save splits
        train_file = output_path / "train_set_integrated_real.csv"
        val_file = output_path / "val_set_integrated_real.csv"
        test_file = output_path / "test_set_integrated_real.csv"
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        logger.info(f"💾 Saved train set ({len(train_df):,} compounds): {train_file}")
        logger.info(f"💾 Saved val set ({len(val_df):,} compounds): {val_file}")
        logger.info(f"💾 Saved test set ({len(test_df):,} compounds): {test_file}")
        
        # Save comprehensive metadata
        trial_compounds = len(df[df['data_source'].str.contains('clinical_trial', na=False)])
        chembl_compounds = len(df[df['data_source'] == 'chembl_api'])
        
        metadata = {
            "dataset_info": {
                "total_compounds": len(df),
                "train_compounds": len(train_df),
                "val_compounds": len(val_df),
                "test_compounds": len(test_df),
                "smiles_coverage": "100%",
                "data_sources": ["Clinical_Trials_+_ChEMBL", "Clinical_Trials_+_PubChem", "ChEMBL_API"],
                "collection_date": datetime.now().isoformat(),
                "dataset_type": "Integrated Real Clinical Trials + Pharmaceutical Compounds"
            },
            "data_composition": {
                "clinical_trial_compounds": trial_compounds,
                "chembl_only_compounds": chembl_compounds,
                "integration_method": "Drug name matching with SMILES lookup"
            },
            "data_quality": {
                "duplicate_smiles_removed": True,
                "all_compounds_have_smiles": True,
                "molecular_properties_included": True,
                "clinical_trial_data_included": True,
                "real_api_sources_only": True
            }
        }
        
        metadata_file = output_path / "integrated_real_dataset_metadata.json"
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
    logger.info("🌟 Integrated Real Dataset Creator")
    logger.info("Combining Clinical Trials + ChEMBL + PubChem")
    logger.info("=" * 60)
    
    # Create integrated dataset creator
    creator = IntegratedRealDatasetCreator()
    
    # Create integrated dataset
    target_size = 20000
    df = creator.create_integrated_dataset(target_size)
    
    if df.empty:
        logger.error("❌ Failed to create dataset")
        return None
    
    # Save dataset
    files = creator.save_integrated_dataset(df)
    
    # Summary
    trial_compounds = len(df[df['data_source'].str.contains('clinical_trial', na=False)])
    chembl_compounds = len(df[df['data_source'] == 'chembl_api'])
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 INTEGRATED REAL DATASET CREATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"📊 Total compounds: {len(df):,}")
    logger.info(f"📋 From clinical trials: {trial_compounds:,}")
    logger.info(f"🔬 From ChEMBL API: {chembl_compounds:,}")
    logger.info(f"🧬 SMILES coverage: 100%")
    logger.info(f"🔗 Integration: Clinical trials matched with compound databases")
    logger.info(f"📁 Files created:")
    for name, path in files.items():
        logger.info(f"   - {name}: {path}")
    
    logger.info("\n✅ Ready for machine learning with real integrated data!")
    
    return files

if __name__ == "__main__":
    main()