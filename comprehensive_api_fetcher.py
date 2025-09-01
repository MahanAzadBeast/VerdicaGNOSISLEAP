#!/usr/bin/env python3
"""
Comprehensive API Fetcher - Maximum Data Collection
Systematically fetches from ALL pharmaceutical APIs:
1. ClinicalTrials.gov - 100,000+ trials
2. ChEMBL - 50,000+ compounds  
3. PubChem - 20,000+ FDA drugs
4. Integrates everything into massive real dataset
"""

import requests
import pandas as pd
import time
import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveAPIFetcher:
    """Comprehensive fetcher for all pharmaceutical APIs"""
    
    def __init__(self):
        self.clinical_trials_url = "https://clinicaltrials.gov/api/v2/studies"
        self.chembl_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.pubchem_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        
        # Collection tracking
        self.collected_trials = []
        self.collected_compounds = []
        self.collected_drugs = []
        
        # API rate limiting
        self.clinical_trials_delay = 0.5
        self.chembl_delay = 0.3
        self.pubchem_delay = 0.2
    
    def get_api_status(self):
        """Check status of all APIs"""
        logger.info("🔍 Checking API availability...")
        
        apis = [
            ("ClinicalTrials.gov", self.clinical_trials_url),
            ("ChEMBL", f"{self.chembl_url}/molecule"),
            ("PubChem", f"{self.pubchem_url}/compound/cid/2244/property/MolecularWeight/JSON")
        ]
        
        for name, url in apis:
            try:
                response = requests.get(url, timeout=10)
                status = "✅ ONLINE" if response.status_code == 200 else f"⚠️ STATUS {response.status_code}"
                logger.info(f"   {name}: {status}")
            except Exception as e:
                logger.warning(f"   {name}: ❌ ERROR - {e}")
    
    def fetch_maximum_clinical_trials(self, target: int = 100000) -> List[Dict]:
        """Fetch maximum clinical trials from API"""
        logger.info(f"🏥 FETCHING MAXIMUM CLINICAL TRIALS (target: {target:,})")
        
        all_trials = []
        
        # Strategy 1: Fetch by study types in parallel
        study_types = ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"]
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for study_type in study_types:
                future = executor.submit(self._fetch_trials_by_type, study_type, target // len(study_types))
                futures.append((future, study_type))
            
            for future, study_type in futures:
                try:
                    trials = future.result(timeout=1800)  # 30 minute timeout
                    all_trials.extend(trials)
                    logger.info(f"✅ {study_type}: {len(trials):,} trials collected")
                except Exception as e:
                    logger.error(f"❌ {study_type} failed: {e}")
        
        # Strategy 2: Fetch by specific conditions if we need more
        if len(all_trials) < target:
            remaining = target - len(all_trials)
            condition_trials = self._fetch_trials_by_conditions(remaining)
            all_trials.extend(condition_trials)
        
        # Strategy 3: Fetch recent trials if still need more
        if len(all_trials) < target:
            remaining = target - len(all_trials)
            recent_trials = self._fetch_recent_trials(remaining)
            all_trials.extend(recent_trials)
        
        # Remove duplicates
        seen_nct_ids = set()
        unique_trials = []
        for trial in all_trials:
            nct_id = trial.get('nct_id')
            if nct_id and nct_id not in seen_nct_ids:
                seen_nct_ids.add(nct_id)
                unique_trials.append(trial)
        
        logger.info(f"🎉 Clinical trials fetched: {len(unique_trials):,} unique trials")
        return unique_trials
    
    def _fetch_trials_by_type(self, study_type: str, target: int) -> List[Dict]:
        """Fetch trials by study type"""
        logger.info(f"📊 Fetching {study_type} trials (target: {target:,})")
        
        trials = []
        page_token = None
        page_size = 1000
        
        while len(trials) < target:
            try:
                params = {
                    "format": "json",
                    "pageSize": min(page_size, target - len(trials)),
                    "query.studyType": study_type
                }
                
                if page_token:
                    params["pageToken"] = page_token
                
                response = requests.get(self.clinical_trials_url, params=params, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                studies = data.get("studies", [])
                
                if not studies:
                    break
                
                # Process studies
                batch_trials = []
                for study in studies:
                    trial_data = self._extract_comprehensive_trial_data(study)
                    if trial_data:
                        batch_trials.append(trial_data)
                
                trials.extend(batch_trials)
                
                if len(trials) % 5000 == 0:
                    logger.info(f"   {study_type} progress: {len(trials):,}/{target:,}")
                
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                
                time.sleep(self.clinical_trials_delay)
                
            except Exception as e:
                logger.warning(f"Error in {study_type} batch: {e}")
                time.sleep(2)
                continue
        
        return trials
    
    def _fetch_trials_by_conditions(self, target: int) -> List[Dict]:
        """Fetch trials by medical conditions"""
        logger.info(f"🏥 Fetching trials by conditions (target: {target:,})")
        
        conditions = [
            "cancer", "diabetes", "hypertension", "depression", "alzheimer",
            "arthritis", "asthma", "covid-19", "heart disease", "stroke",
            "obesity", "epilepsy", "parkinson", "multiple sclerosis", "hepatitis"
        ]
        
        all_trials = []
        trials_per_condition = target // len(conditions)
        
        for condition in conditions:
            if len(all_trials) >= target:
                break
                
            try:
                condition_trials = self._fetch_trials_by_keyword(condition, trials_per_condition)
                all_trials.extend(condition_trials)
                logger.info(f"   {condition}: {len(condition_trials)} trials")
                
            except Exception as e:
                logger.warning(f"Error fetching {condition} trials: {e}")
                continue
        
        return all_trials
    
    def _fetch_recent_trials(self, target: int) -> List[Dict]:
        """Fetch recent trials by date"""
        logger.info(f"📅 Fetching recent trials (target: {target:,})")
        
        all_trials = []
        current_year = datetime.now().year
        
        for year in range(current_year, current_year - 15, -1):  # Last 15 years
            if len(all_trials) >= target:
                break
                
            try:
                year_trials = self._fetch_trials_by_year(year, (target - len(all_trials)) // (current_year - year + 1))
                all_trials.extend(year_trials)
                logger.info(f"   {year}: {len(year_trials)} trials")
                
            except Exception as e:
                logger.warning(f"Error fetching {year} trials: {e}")
                continue
        
        return all_trials
    
    def _fetch_trials_by_keyword(self, keyword: str, target: int) -> List[Dict]:
        """Fetch trials by keyword"""
        trials = []
        page_token = None
        page_size = 500
        
        while len(trials) < target:
            try:
                params = {
                    "format": "json",
                    "pageSize": min(page_size, target - len(trials)),
                    "query.condition": keyword
                }
                
                if page_token:
                    params["pageToken"] = page_token
                
                response = requests.get(self.clinical_trials_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                studies = data.get("studies", [])
                
                if not studies:
                    break
                
                batch_trials = []
                for study in studies:
                    trial_data = self._extract_comprehensive_trial_data(study)
                    if trial_data:
                        batch_trials.append(trial_data)
                
                trials.extend(batch_trials)
                
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                
                time.sleep(self.clinical_trials_delay)
                
            except Exception as e:
                logger.debug(f"Error in {keyword} trials: {e}")
                break
        
        return trials
    
    def _fetch_trials_by_year(self, year: int, target: int) -> List[Dict]:
        """Fetch trials by year"""
        trials = []
        page_token = None
        
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        while len(trials) < target:
            try:
                params = {
                    "format": "json",
                    "pageSize": min(1000, target - len(trials)),
                    "query.startDate": f"{start_date}:{end_date}"
                }
                
                if page_token:
                    params["pageToken"] = page_token
                
                response = requests.get(self.clinical_trials_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                studies = data.get("studies", [])
                
                if not studies:
                    break
                
                batch_trials = []
                for study in studies:
                    trial_data = self._extract_comprehensive_trial_data(study)
                    if trial_data:
                        batch_trials.append(trial_data)
                
                trials.extend(batch_trials)
                
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                
                time.sleep(self.clinical_trials_delay)
                
            except Exception as e:
                logger.debug(f"Error in {year} trials: {e}")
                break
        
        return trials
    
    def _extract_comprehensive_trial_data(self, study: Dict) -> Optional[Dict]:
        """Extract comprehensive trial data"""
        try:
            protocol = study.get("protocolSection", {})
            
            # Basic identification
            identification = protocol.get("identificationModule", {})
            nct_id = identification.get("nctId")
            title = identification.get("briefTitle", "")
            
            if not nct_id:
                return None
            
            # Study design
            design = protocol.get("designModule", {})
            study_type = design.get("studyType")
            phases = design.get("phases", [])
            
            # Interventions
            arms_interventions = protocol.get("armsInterventionsModule", {})
            interventions = arms_interventions.get("interventions", [])
            
            # Extract drugs
            drugs = []
            for intervention in interventions:
                if intervention.get("type") == "DRUG":
                    drug_name = intervention.get("name", "").strip()
                    if drug_name:
                        drugs.append(drug_name)
                        # Also get other names
                        other_names = intervention.get("otherNames", [])
                        drugs.extend(other_names)
            
            # Skip if no drugs
            if not drugs:
                return None
            
            drugs = list(set(drugs))  # Remove duplicates
            
            # Conditions
            conditions_module = protocol.get("conditionsModule", {})
            conditions = conditions_module.get("conditions", [])
            
            # Status
            status_module = protocol.get("statusModule", {})
            overall_status = status_module.get("overallStatus")
            why_stopped = status_module.get("whyStopped", "")
            
            # Outcomes
            outcomes_module = protocol.get("outcomesModule", {})
            primary_outcomes = outcomes_module.get("primaryOutcomes", [])
            secondary_outcomes = outcomes_module.get("secondaryOutcomes", [])
            
            # Sponsor
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            lead_sponsor = sponsor_module.get("leadSponsor", {}).get("name")
            
            return {
                "nct_id": nct_id,
                "title": title,
                "primary_drug": drugs[0] if drugs else None,
                "all_drug_names": drugs,
                "primary_condition": conditions[0] if conditions else None,
                "all_conditions": conditions,
                "phases": phases,
                "primary_phase": phases[0] if phases else None,
                "study_type": study_type,
                "overall_status": overall_status,
                "why_stopped": why_stopped,
                "primary_outcomes": primary_outcomes,
                "secondary_outcomes": secondary_outcomes,
                "lead_sponsor": lead_sponsor,
                "collected_date": datetime.now().isoformat(),
                "api_version": "v2_comprehensive"
            }
            
        except Exception as e:
            logger.debug(f"Error extracting trial: {e}")
            return None
    
    def fetch_maximum_chembl_compounds(self, target: int = 50000) -> List[Dict]:
        """Fetch maximum compounds from ChEMBL API"""
        logger.info(f"🔬 FETCHING MAXIMUM CHEMBL COMPOUNDS (target: {target:,})")
        
        all_compounds = []
        
        # Fetch by different phases in parallel
        phases = [4, 3, 2, 1]  # Approved first, then clinical phases
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for phase in phases:
                phase_target = target // len(phases)
                future = executor.submit(self._fetch_chembl_by_phase, phase, phase_target)
                futures.append((future, phase))
            
            for future, phase in futures:
                try:
                    compounds = future.result(timeout=1800)  # 30 minute timeout
                    all_compounds.extend(compounds)
                    logger.info(f"✅ ChEMBL Phase {phase}: {len(compounds):,} compounds")
                except Exception as e:
                    logger.error(f"❌ ChEMBL Phase {phase} failed: {e}")
        
        # Remove duplicates by ChEMBL ID
        seen_ids = set()
        unique_compounds = []
        for compound in all_compounds:
            chembl_id = compound.get('chembl_id')
            if chembl_id and chembl_id not in seen_ids:
                seen_ids.add(chembl_id)
                unique_compounds.append(compound)
        
        logger.info(f"🎉 ChEMBL compounds fetched: {len(unique_compounds):,} unique compounds")
        return unique_compounds
    
    def _fetch_chembl_by_phase(self, phase: int, target: int) -> List[Dict]:
        """Fetch ChEMBL compounds by phase"""
        logger.info(f"🔬 Fetching ChEMBL Phase {phase} compounds (target: {target:,})")
        
        compounds = []
        offset = 0
        batch_size = 1000
        
        while len(compounds) < target:
            try:
                params = {
                    "format": "json",
                    "limit": batch_size,
                    "offset": offset,
                    "max_phase": phase,
                    "molecule_type": "Small molecule"
                }
                
                url = f"{self.chembl_url}/molecule"
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                molecules = data.get("molecules", [])
                
                if not molecules:
                    break
                
                batch_compounds = []
                for mol in molecules:
                    # Only include exact phase matches
                    if mol.get("max_phase") == phase:
                        compound = self._extract_chembl_compound(mol)
                        if compound:
                            batch_compounds.append(compound)
                
                compounds.extend(batch_compounds)
                
                if len(compounds) % 2000 == 0:
                    logger.info(f"   Phase {phase} progress: {len(compounds):,}/{target:,}")
                
                offset += batch_size
                time.sleep(self.chembl_delay)
                
            except Exception as e:
                logger.warning(f"Error in ChEMBL Phase {phase} batch: {e}")
                offset += batch_size
                time.sleep(2)
                continue
        
        return compounds
    
    def _extract_chembl_compound(self, molecule: Dict) -> Optional[Dict]:
        """Extract ChEMBL compound data"""
        try:
            chembl_id = molecule.get("molecule_chembl_id")
            structures = molecule.get("molecule_structures", {})
            smiles = structures.get("canonical_smiles")
            pref_name = molecule.get("pref_name")
            
            if not all([chembl_id, smiles, pref_name]):
                return None
            
            properties = molecule.get("molecule_properties", {})
            max_phase = molecule.get("max_phase")
            
            return {
                "compound_id": f"CHEMBL_{chembl_id}",
                "chembl_id": chembl_id,
                "primary_drug": pref_name,
                "smiles": smiles,
                "smiles_source": chembl_id,
                "max_clinical_phase": max_phase,
                "molecular_weight": properties.get("full_mwt"),
                "logp": properties.get("alogp"),
                "hbd": properties.get("hbd"),
                "hba": properties.get("hba"),
                "rotatable_bonds": properties.get("rtb"),
                "tpsa": properties.get("psa"),
                "aromatic_rings": properties.get("aromatic_rings"),
                "heavy_atoms": properties.get("heavy_atoms"),
                "formal_charge": properties.get("formal_charge", 0),
                "num_rings": properties.get("num_rings"),
                "data_source": "chembl_api_comprehensive",
                "collected_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.debug(f"Error extracting ChEMBL compound: {e}")
            return None
    
    def fetch_maximum_pubchem_drugs(self, target: int = 20000) -> List[Dict]:
        """Fetch maximum FDA approved drugs from PubChem"""
        logger.info(f"💊 FETCHING MAXIMUM PUBCHEM DRUGS (target: {target:,})")
        
        all_drugs = []
        
        # Different search strategies
        search_strategies = [
            ("FDA approved", target // 4),
            ("prescription drug", target // 4),
            ("pharmaceutical", target // 4),
            ("therapeutic", target // 4)
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for search_term, term_target in search_strategies:
                future = executor.submit(self._fetch_pubchem_by_term, search_term, term_target)
                futures.append((future, search_term))
            
            for future, search_term in futures:
                try:
                    drugs = future.result(timeout=1800)  # 30 minute timeout
                    all_drugs.extend(drugs)
                    logger.info(f"✅ PubChem '{search_term}': {len(drugs):,} drugs")
                except Exception as e:
                    logger.error(f"❌ PubChem '{search_term}' failed: {e}")
        
        # Remove duplicates by CID
        seen_cids = set()
        unique_drugs = []
        for drug in all_drugs:
            cid = drug.get('pubchem_cid')
            if cid and cid not in seen_cids:
                seen_cids.add(cid)
                unique_drugs.append(drug)
        
        logger.info(f"🎉 PubChem drugs fetched: {len(unique_drugs):,} unique drugs")
        return unique_drugs
    
    def _fetch_pubchem_by_term(self, search_term: str, target: int) -> List[Dict]:
        """Fetch PubChem compounds by search term"""
        logger.info(f"💊 Fetching PubChem '{search_term}' compounds")
        
        try:
            # Search for CIDs
            search_url = f"{self.pubchem_url}/compound/name/{search_term}/cids/JSON"
            response = requests.get(search_url, timeout=30)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            
            if not cids:
                return []
            
            # Limit to target
            cids = cids[:target]
            
            # Process in batches
            drugs = []
            batch_size = 100
            
            for i in range(0, len(cids), batch_size):
                batch_cids = cids[i:i+batch_size]
                batch_drugs = self._fetch_pubchem_properties_batch(batch_cids)
                drugs.extend(batch_drugs)
                
                if len(drugs) % 1000 == 0:
                    logger.info(f"   '{search_term}' progress: {len(drugs):,}")
                
                time.sleep(self.pubchem_delay)
            
            return drugs
            
        except Exception as e:
            logger.error(f"Error fetching PubChem '{search_term}': {e}")
            return []
    
    def _fetch_pubchem_properties_batch(self, cids: List[int]) -> List[Dict]:
        """Fetch properties for a batch of PubChem CIDs"""
        if not cids:
            return []
        
        try:
            cid_list = ",".join(map(str, cids))
            props_url = f"{self.pubchem_url}/compound/cid/{cid_list}/property/MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,TPSA,HeavyAtomCount,CanonicalSMILES,IUPACName/JSON"
            
            response = requests.get(props_url, timeout=30)
            if response.status_code != 200:
                return []
            
            data = response.json()
            properties_list = data.get("PropertyTable", {}).get("Properties", [])
            
            drugs = []
            for props in properties_list:
                drug = self._extract_pubchem_drug(props)
                if drug:
                    drugs.append(drug)
            
            return drugs
            
        except Exception as e:
            logger.debug(f"Error fetching PubChem batch: {e}")
            return []
    
    def _extract_pubchem_drug(self, props: Dict) -> Optional[Dict]:
        """Extract PubChem drug data"""
        try:
            cid = props.get("CID")
            smiles = props.get("CanonicalSMILES")
            
            if not smiles or not cid:
                return None
            
            compound_name = props.get("IUPACName", f"PubChem_CID_{cid}")
            
            return {
                "compound_id": f"PUBCHEM_{cid}",
                "pubchem_cid": cid,
                "primary_drug": compound_name,
                "smiles": smiles,
                "smiles_source": f"CID_{cid}",
                "molecular_weight": props.get("MolecularWeight"),
                "logp": props.get("XLogP"),
                "hbd": props.get("HBondDonorCount"),
                "hba": props.get("HBondAcceptorCount"),
                "rotatable_bonds": props.get("RotatableBondCount"),
                "tpsa": props.get("TPSA"),
                "heavy_atoms": props.get("HeavyAtomCount"),
                "data_source": "pubchem_api_comprehensive",
                "collected_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.debug(f"Error extracting PubChem drug: {e}")
            return None
    
    def create_comprehensive_dataset(self) -> pd.DataFrame:
        """Create comprehensive dataset from all APIs"""
        logger.info("🚀 CREATING COMPREHENSIVE DATASET FROM ALL APIS")
        logger.info("=" * 80)
        
        # Step 1: Fetch clinical trials
        logger.info("\n📊 STEP 1: FETCHING CLINICAL TRIALS")
        trials = self.fetch_maximum_clinical_trials(100000)
        self.collected_trials = trials
        
        # Step 2: Fetch ChEMBL compounds
        logger.info("\n📊 STEP 2: FETCHING CHEMBL COMPOUNDS")
        chembl_compounds = self.fetch_maximum_chembl_compounds(50000)
        self.collected_compounds.extend(chembl_compounds)
        
        # Step 3: Fetch PubChem drugs
        logger.info("\n📊 STEP 3: FETCHING PUBCHEM DRUGS")
        pubchem_drugs = self.fetch_maximum_pubchem_drugs(20000)
        self.collected_compounds.extend(pubchem_drugs)
        
        # Step 4: Extract unique drugs from trials and find SMILES
        logger.info("\n📊 STEP 4: MATCHING TRIAL DRUGS TO SMILES")
        trial_drugs = self._extract_unique_trial_drugs(trials)
        trial_compounds = self._match_trial_drugs_to_smiles(trial_drugs, chembl_compounds + pubchem_drugs)
        self.collected_compounds.extend(trial_compounds)
        
        # Step 5: Create final dataset
        logger.info("\n📊 STEP 5: CREATING FINAL COMPREHENSIVE DATASET")
        if not self.collected_compounds:
            logger.error("❌ No compounds collected!")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.collected_compounds)
        
        # Remove duplicates by SMILES
        initial_count = len(df)
        df = df.drop_duplicates(subset=['smiles'], keep='first')
        final_count = len(df)
        
        logger.info(f"🧹 Removed {initial_count - final_count} duplicate SMILES")
        logger.info(f"✅ Final comprehensive dataset: {final_count:,} unique compounds")
        
        return df
    
    def _extract_unique_trial_drugs(self, trials: List[Dict]) -> List[str]:
        """Extract unique drug names from trials"""
        unique_drugs = set()
        
        for trial in trials:
            drugs = trial.get('all_drug_names', [])
            for drug in drugs:
                if drug and isinstance(drug, str) and len(drug.strip()) > 2:
                    unique_drugs.add(drug.strip())
        
        return list(unique_drugs)
    
    def _match_trial_drugs_to_smiles(self, trial_drugs: List[str], compound_database: List[Dict]) -> List[Dict]:
        """Match trial drugs to SMILES from compound database"""
        logger.info(f"🔗 Matching {len(trial_drugs):,} trial drugs to SMILES...")
        
        # Create lookup dictionary
        drug_lookup = {}
        for compound in compound_database:
            drug_name = compound.get('primary_drug', '').lower()
            if drug_name:
                drug_lookup[drug_name] = compound
        
        matched_compounds = []
        matched_count = 0
        
        for drug_name in trial_drugs:
            drug_lower = drug_name.lower()
            
            # Exact match
            if drug_lower in drug_lookup:
                compound = drug_lookup[drug_lower].copy()
                compound['data_source'] = 'clinical_trial_matched'
                compound['trial_drug_name'] = drug_name
                matched_compounds.append(compound)
                matched_count += 1
            
            # Partial match
            else:
                for db_drug, compound in drug_lookup.items():
                    if drug_lower in db_drug or db_drug in drug_lower:
                        matched_compound = compound.copy()
                        matched_compound['data_source'] = 'clinical_trial_partial_match'
                        matched_compound['trial_drug_name'] = drug_name
                        matched_compounds.append(matched_compound)
                        matched_count += 1
                        break
        
        logger.info(f"✅ Matched {matched_count:,}/{len(trial_drugs):,} trial drugs to SMILES")
        return matched_compounds
    
    def save_comprehensive_dataset(self, df: pd.DataFrame, output_dir: str = "clinical_trial_dataset/data/api_comprehensive"):
        """Save the comprehensive API dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete dataset
        complete_file = output_path / "complete_api_comprehensive_dataset.csv"
        df.to_csv(complete_file, index=False)
        logger.info(f"💾 Saved complete dataset: {complete_file}")
        
        # Create train/val/test splits
        total_size = len(df)
        train_size = int(total_size * 0.70)
        val_size = int(total_size * 0.15)
        
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df = df_shuffled[:train_size]
        val_df = df_shuffled[train_size:train_size + val_size]
        test_df = df_shuffled[train_size + val_size:]
        
        # Save splits
        train_file = output_path / "train_set_api_comprehensive.csv"
        val_file = output_path / "val_set_api_comprehensive.csv"
        test_file = output_path / "test_set_api_comprehensive.csv"
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        logger.info(f"💾 Saved train set ({len(train_df):,} compounds): {train_file}")
        logger.info(f"💾 Saved val set ({len(val_df):,} compounds): {val_file}")
        logger.info(f"💾 Saved test set ({len(test_df):,} compounds): {test_file}")
        
        # Save comprehensive metadata
        clinical_trial_compounds = len(df[df['data_source'].str.contains('clinical_trial', na=False)])
        chembl_compounds = len(df[df['data_source'].str.contains('chembl', na=False)])
        pubchem_compounds = len(df[df['data_source'].str.contains('pubchem', na=False)])
        
        metadata = {
            "dataset_info": {
                "total_compounds": len(df),
                "train_compounds": len(train_df),
                "val_compounds": len(val_df),
                "test_compounds": len(test_df),
                "smiles_coverage": "100%",
                "collection_date": datetime.now().isoformat(),
                "dataset_type": "Comprehensive API Pharmaceutical Dataset"
            },
            "data_sources": {
                "clinical_trials_collected": len(self.collected_trials),
                "clinical_trial_compounds": clinical_trial_compounds,
                "chembl_compounds": chembl_compounds,
                "pubchem_compounds": pubchem_compounds
            },
            "api_endpoints": {
                "clinical_trials": "https://clinicaltrials.gov/api/v2/studies",
                "chembl": "https://www.ebi.ac.uk/chembl/api/data",
                "pubchem": "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
            },
            "data_quality": {
                "duplicate_smiles_removed": True,
                "all_compounds_have_smiles": True,
                "comprehensive_api_collection": True,
                "maximum_data_extraction": True
            }
        }
        
        metadata_file = output_path / "api_comprehensive_metadata.json"
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
    logger.info("🌟 COMPREHENSIVE API FETCHER")
    logger.info("🔗 Maximum data extraction from ALL pharmaceutical APIs")
    logger.info("🎯 Target: 100,000+ trials + 70,000+ compounds")
    logger.info("=" * 80)
    
    # Create comprehensive fetcher
    fetcher = ComprehensiveAPIFetcher()
    
    # Check API status
    fetcher.get_api_status()
    
    # Create comprehensive dataset
    logger.info("\n🚀 Starting comprehensive API data collection...")
    df = fetcher.create_comprehensive_dataset()
    
    if df.empty:
        logger.error("❌ Failed to create comprehensive dataset")
        return None
    
    # Save dataset
    files = fetcher.save_comprehensive_dataset(df)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("🎉 COMPREHENSIVE API COLLECTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"📊 Total compounds: {len(df):,}")
    logger.info(f"📋 Clinical trials collected: {len(fetcher.collected_trials):,}")
    logger.info(f"🧬 SMILES coverage: 100%")
    logger.info(f"🔗 API sources: ClinicalTrials.gov + ChEMBL + PubChem")
    logger.info(f"📁 Files created:")
    for name, path in files.items():
        logger.info(f"   - {name}: {path}")
    
    logger.info("\n🚀 MAXIMUM PHARMACEUTICAL DATA COLLECTED FROM ALL APIS!")
    
    return files

if __name__ == "__main__":
    main()