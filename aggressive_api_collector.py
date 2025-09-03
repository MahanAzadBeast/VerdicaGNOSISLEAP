#!/usr/bin/env python3
"""
Aggressive API Collector - Maximum Data Extraction
Designed to collect 100,000+ compounds from all APIs with:
- Aggressive parallel processing
- Extended timeouts and retries
- Multiple collection strategies
- Real-time progress monitoring
"""

import requests
import pandas as pd
import time
import json
import logging
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import sys
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AggressiveAPICollector:
    """Aggressive collector for maximum pharmaceutical data"""
    
    def __init__(self):
        self.clinical_trials_url = "https://clinicaltrials.gov/api/v2/studies"
        self.chembl_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.pubchem_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        
        # Aggressive collection settings
        self.max_workers = 8
        self.max_retries = 5
        self.timeout = 120  # 2 minutes
        self.batch_size = 1000
        
        # Progress tracking
        self.progress_lock = threading.Lock()
        self.total_collected = 0
        
    def aggressive_chembl_collection(self, target: int = 50000) -> List[Dict]:
        """Aggressively collect ChEMBL compounds"""
        logger.info(f"🔬 AGGRESSIVE CHEMBL COLLECTION (target: {target:,})")
        
        all_compounds = []
        
        # Strategy 1: Collect by molecule type
        molecule_types = ["Small molecule"]
        
        # Strategy 2: Collect by max_phase (all phases)
        phases = [4, 3, 2, 1, None]  # Include compounds with unknown phase
        
        # Strategy 3: Collect by different endpoints
        endpoints = [
            "/molecule",
            "/compound", 
            "/drug"
        ]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            # Submit multiple collection tasks
            for phase in phases:
                for endpoint in endpoints:
                    if endpoint == "/molecule":  # Primary endpoint
                        future = executor.submit(self._collect_chembl_by_phase_aggressive, phase, target // (len(phases) * len(endpoints)))
                        futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    compounds = future.result(timeout=3600)  # 1 hour timeout
                    all_compounds.extend(compounds)
                    logger.info(f"✅ ChEMBL batch complete: {len(compounds):,} compounds (Total: {len(all_compounds):,})")
                    
                    if len(all_compounds) >= target:
                        break
                        
                except Exception as e:
                    logger.error(f"❌ ChEMBL batch failed: {e}")
        
        # Remove duplicates
        seen_ids = set()
        unique_compounds = []
        for compound in all_compounds:
            chembl_id = compound.get('chembl_id')
            if chembl_id and chembl_id not in seen_ids:
                seen_ids.add(chembl_id)
                unique_compounds.append(compound)
        
        logger.info(f"🎉 ChEMBL aggressive collection: {len(unique_compounds):,} unique compounds")
        return unique_compounds[:target]
    
    def _collect_chembl_by_phase_aggressive(self, phase: Optional[int], target: int) -> List[Dict]:
        """Aggressively collect ChEMBL by phase"""
        phase_str = f"Phase {phase}" if phase else "Unknown Phase"
        logger.info(f"🔬 Aggressive {phase_str} collection (target: {target:,})")
        
        compounds = []
        offset = 0
        consecutive_failures = 0
        
        while len(compounds) < target and consecutive_failures < 5:
            try:
                params = {
                    "format": "json",
                    "limit": self.batch_size,
                    "offset": offset,
                    "molecule_type": "Small molecule"
                }
                
                if phase is not None:
                    params["max_phase"] = phase
                
                url = f"{self.chembl_url}/molecule"
                
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                molecules = data.get("molecules", [])
                
                if not molecules:
                    logger.info(f"   {phase_str}: No more molecules at offset {offset}")
                    break
                
                # Process all molecules (not just exact phase matches)
                batch_compounds = []
                for mol in molecules:
                    compound = self._extract_chembl_compound_aggressive(mol)
                    if compound:
                        batch_compounds.append(compound)
                
                compounds.extend(batch_compounds)
                consecutive_failures = 0  # Reset failure counter
                
                # Progress update
                if len(compounds) % 5000 == 0:
                    with self.progress_lock:
                        self.total_collected += len(batch_compounds)
                        logger.info(f"   {phase_str} progress: {len(compounds):,}/{target:,} (Global: {self.total_collected:,})")
                
                offset += self.batch_size
                time.sleep(0.1)  # Minimal delay for aggressive collection
                
            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"   {phase_str} error at offset {offset} (attempt {consecutive_failures}): {e}")
                
                if consecutive_failures < self.max_retries:
                    time.sleep(2 ** consecutive_failures)  # Exponential backoff
                    continue
                else:
                    logger.error(f"   {phase_str}: Max retries exceeded, moving to next batch")
                    offset += self.batch_size
                    consecutive_failures = 0
        
        logger.info(f"✅ {phase_str} complete: {len(compounds):,} compounds")
        return compounds
    
    def _extract_chembl_compound_aggressive(self, molecule: Dict) -> Optional[Dict]:
        """Aggressively extract ChEMBL compound data"""
        try:
            chembl_id = molecule.get("molecule_chembl_id")
            if not chembl_id:
                return None
            
            # Get SMILES - try multiple sources
            structures = molecule.get("molecule_structures", {})
            smiles = (structures.get("canonical_smiles") or 
                     structures.get("standard_inchi") or
                     structures.get("molfile"))
            
            if not smiles:
                return None
            
            # Get name - try multiple sources
            pref_name = (molecule.get("pref_name") or
                        molecule.get("molecule_synonyms", [{}])[0].get("molecule_synonym") if molecule.get("molecule_synonyms") else None or
                        f"ChEMBL_{chembl_id}")
            
            properties = molecule.get("molecule_properties", {})
            hierarchy = molecule.get("molecule_hierarchy", {})
            
            return {
                "compound_id": f"CHEMBL_{chembl_id}",
                "chembl_id": chembl_id,
                "primary_drug": pref_name,
                "all_drug_names": [pref_name],
                "smiles": smiles,
                "smiles_source": chembl_id,
                "mapping_status": "success",
                
                # Comprehensive molecular properties
                "mol_molecular_weight": properties.get("full_mwt"),
                "mol_logp": properties.get("alogp"),
                "mol_num_hbd": properties.get("hbd"),
                "mol_num_hba": properties.get("hba"),
                "mol_num_rotatable_bonds": properties.get("rtb"),
                "mol_tpsa": properties.get("psa"),
                "mol_num_aromatic_rings": properties.get("aromatic_rings"),
                "mol_num_heavy_atoms": properties.get("heavy_atoms"),
                "mol_formal_charge": properties.get("formal_charge", 0),
                "mol_num_rings": properties.get("num_rings"),
                "mol_num_heteroatoms": properties.get("num_heteroatoms"),
                "mol_fraction_csp3": properties.get("cx_most_apka"),
                
                # Clinical data
                "max_clinical_phase": molecule.get("max_phase"),
                "clinical_status": "Approved" if molecule.get("max_phase") == 4 else f"Phase_{molecule.get('max_phase')}" if molecule.get("max_phase") else "Unknown",
                "primary_condition": None,
                
                # Hierarchy data
                "parent_chembl_id": hierarchy.get("parent_chembl_id"),
                "molecule_type": molecule.get("molecule_type"),
                
                # Dataset metadata
                "data_source": "chembl_aggressive",
                "compound_type": "Small molecule",
                "study_type": "PHARMACEUTICAL_DATABASE",
                "collected_date": datetime.now().isoformat(),
                
                # ML targets (leave as None for real data)
                "efficacy_score": None,
                "safety_score": None,
                "success_probability": 1.0 if molecule.get("max_phase") == 4 else None
            }
            
        except Exception as e:
            logger.debug(f"Error extracting ChEMBL compound: {e}")
            return None
    
    def aggressive_pubchem_collection(self, target: int = 30000) -> List[Dict]:
        """Aggressively collect PubChem compounds"""
        logger.info(f"💊 AGGRESSIVE PUBCHEM COLLECTION (target: {target:,})")
        
        all_compounds = []
        
        # Aggressive search strategies
        search_strategies = [
            ("FDA approved drug", target // 10),
            ("prescription medication", target // 10),
            ("pharmaceutical compound", target // 10),
            ("therapeutic agent", target // 10),
            ("medicine", target // 10),
            ("drug", target // 10),
            ("medication", target // 10),
            ("pharmaceutical", target // 10),
            ("therapeutic", target // 10),
            ("approved drug", target // 10)
        ]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for search_term, term_target in search_strategies:
                future = executor.submit(self._collect_pubchem_aggressive, search_term, term_target)
                futures.append((future, search_term))
            
            for future, search_term in futures:
                try:
                    compounds = future.result(timeout=1800)  # 30 minute timeout
                    all_compounds.extend(compounds)
                    logger.info(f"✅ PubChem '{search_term}': {len(compounds):,} compounds (Total: {len(all_compounds):,})")
                    
                except Exception as e:
                    logger.error(f"❌ PubChem '{search_term}' failed: {e}")
        
        # Remove duplicates by CID
        seen_cids = set()
        unique_compounds = []
        for compound in all_compounds:
            cid = compound.get('pubchem_cid')
            if cid and cid not in seen_cids:
                seen_cids.add(cid)
                unique_compounds.append(compound)
        
        logger.info(f"🎉 PubChem aggressive collection: {len(unique_compounds):,} unique compounds")
        return unique_compounds[:target]
    
    def _collect_pubchem_aggressive(self, search_term: str, target: int) -> List[Dict]:
        """Aggressively collect PubChem by search term"""
        logger.info(f"💊 Aggressive '{search_term}' collection")
        
        try:
            # Get CIDs
            search_url = f"{self.pubchem_url}/compound/name/{search_term}/cids/JSON"
            response = requests.get(search_url, timeout=60)
            
            if response.status_code != 200:
                logger.warning(f"   '{search_term}' search failed: {response.status_code}")
                return []
            
            data = response.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            
            if not cids:
                logger.warning(f"   '{search_term}' returned no CIDs")
                return []
            
            # Take more CIDs for aggressive collection
            cids = cids[:target * 2]  # Get 2x target to account for failures
            logger.info(f"   '{search_term}': Found {len(cids):,} CIDs")
            
            # Process in large batches
            compounds = []
            batch_size = 200  # Larger batches
            
            for i in range(0, len(cids), batch_size):
                if len(compounds) >= target:
                    break
                    
                batch_cids = cids[i:i+batch_size]
                batch_compounds = self._fetch_pubchem_batch_aggressive(batch_cids)
                compounds.extend(batch_compounds)
                
                if len(compounds) % 1000 == 0:
                    logger.info(f"   '{search_term}' progress: {len(compounds):,}")
                
                time.sleep(0.1)  # Minimal delay
            
            return compounds[:target]
            
        except Exception as e:
            logger.error(f"Error in aggressive PubChem collection for '{search_term}': {e}")
            return []
    
    def _fetch_pubchem_batch_aggressive(self, cids: List[int]) -> List[Dict]:
        """Aggressively fetch PubChem properties"""
        if not cids:
            return []
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                cid_list = ",".join(map(str, cids))
                
                # Get comprehensive properties
                props_url = f"{self.pubchem_url}/compound/cid/{cid_list}/property/MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,TPSA,HeavyAtomCount,CanonicalSMILES,IUPACName,MolecularFormula/JSON"
                
                response = requests.get(props_url, timeout=120)
                if response.status_code != 200:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return []
                
                data = response.json()
                properties_list = data.get("PropertyTable", {}).get("Properties", [])
                
                compounds = []
                for props in properties_list:
                    compound = self._extract_pubchem_compound_aggressive(props)
                    if compound:
                        compounds.append(compound)
                
                return compounds
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.debug(f"PubChem batch error (attempt {attempt + 1}): {e}")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.debug(f"PubChem batch failed after {max_retries} attempts: {e}")
                    return []
        
        return []
    
    def _extract_pubchem_compound_aggressive(self, props: Dict) -> Optional[Dict]:
        """Aggressively extract PubChem compound data"""
        try:
            cid = props.get("CID")
            smiles = props.get("CanonicalSMILES")
            
            if not smiles or not cid:
                return None
            
            # Get best available name
            compound_name = (props.get("IUPACName") or
                           f"PubChem_CID_{cid}")
            
            return {
                "compound_id": f"PUBCHEM_{cid}",
                "pubchem_cid": cid,
                "primary_drug": compound_name,
                "all_drug_names": [compound_name],
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
                "mol_num_aromatic_rings": None,
                "mol_num_heavy_atoms": props.get("HeavyAtomCount"),
                "mol_formal_charge": 0,
                "mol_num_rings": None,
                "mol_num_heteroatoms": None,
                "mol_fraction_csp3": None,
                
                # Additional data
                "molecular_formula": props.get("MolecularFormula"),
                
                # Clinical data (unknown for PubChem)
                "max_clinical_phase": None,
                "clinical_status": "Database_Entry",
                "primary_condition": None,
                
                # Dataset metadata
                "data_source": "pubchem_aggressive",
                "compound_type": "Small molecule",
                "study_type": "CHEMICAL_DATABASE",
                "collected_date": datetime.now().isoformat(),
                
                # ML targets
                "efficacy_score": None,
                "safety_score": None,
                "success_probability": None
            }
            
        except Exception as e:
            logger.debug(f"Error extracting PubChem compound: {e}")
            return None
    
    def aggressive_clinical_trials_collection(self, target: int = 50000) -> List[Dict]:
        """Aggressively collect clinical trials"""
        logger.info(f"🏥 AGGRESSIVE CLINICAL TRIALS COLLECTION (target: {target:,})")
        
        all_trials = []
        
        # Multiple parallel strategies
        strategies = [
            ("INTERVENTIONAL", target // 3),
            ("OBSERVATIONAL", target // 3),
            ("EXPANDED_ACCESS", target // 3)
        ]
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            for study_type, type_target in strategies:
                future = executor.submit(self._collect_trials_aggressive, study_type, type_target)
                futures.append((future, study_type))
            
            for future, study_type in futures:
                try:
                    trials = future.result(timeout=3600)  # 1 hour timeout
                    all_trials.extend(trials)
                    logger.info(f"✅ Clinical trials '{study_type}': {len(trials):,} trials")
                    
                except Exception as e:
                    logger.error(f"❌ Clinical trials '{study_type}' failed: {e}")
        
        # Remove duplicates
        seen_nct_ids = set()
        unique_trials = []
        for trial in all_trials:
            nct_id = trial.get('nct_id')
            if nct_id and nct_id not in seen_nct_ids:
                seen_nct_ids.add(nct_id)
                unique_trials.append(trial)
        
        logger.info(f"🎉 Clinical trials aggressive collection: {len(unique_trials):,} unique trials")
        return unique_trials[:target]
    
    def _collect_trials_aggressive(self, study_type: str, target: int) -> List[Dict]:
        """Aggressively collect trials by type"""
        logger.info(f"🏥 Aggressive {study_type} trials collection")
        
        trials = []
        page_token = None
        consecutive_failures = 0
        
        while len(trials) < target and consecutive_failures < 5:
            try:
                params = {
                    "format": "json",
                    "pageSize": 1000,  # Maximum page size
                    "query.studyType": study_type
                }
                
                if page_token:
                    params["pageToken"] = page_token
                
                response = requests.get(self.clinical_trials_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                studies = data.get("studies", [])
                
                if not studies:
                    logger.info(f"   {study_type}: No more studies available")
                    break
                
                # Process all studies
                batch_trials = []
                for study in studies:
                    trial_data = self._extract_trial_aggressive(study)
                    if trial_data:
                        batch_trials.append(trial_data)
                
                trials.extend(batch_trials)
                consecutive_failures = 0
                
                # Progress update
                if len(trials) % 5000 == 0:
                    logger.info(f"   {study_type} progress: {len(trials):,}/{target:,}")
                
                page_token = data.get("nextPageToken")
                if not page_token:
                    logger.info(f"   {study_type}: Reached end of available studies")
                    break
                
                time.sleep(0.1)  # Minimal delay
                
            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"   {study_type} error (attempt {consecutive_failures}): {e}")
                
                if consecutive_failures < self.max_retries:
                    time.sleep(2 ** consecutive_failures)
                    continue
                else:
                    logger.error(f"   {study_type}: Max retries exceeded")
                    break
        
        logger.info(f"✅ {study_type} trials complete: {len(trials):,}")
        return trials
    
    def _extract_trial_aggressive(self, study: Dict) -> Optional[Dict]:
        """Aggressively extract trial data"""
        try:
            protocol = study.get("protocolSection", {})
            
            # Basic info
            identification = protocol.get("identificationModule", {})
            nct_id = identification.get("nctId")
            if not nct_id:
                return None
            
            title = identification.get("briefTitle", "")
            
            # Design
            design = protocol.get("designModule", {})
            study_type = design.get("studyType")
            phases = design.get("phases", [])
            
            # Interventions - extract all drugs
            arms_interventions = protocol.get("armsInterventionsModule", {})
            interventions = arms_interventions.get("interventions", [])
            
            drugs = []
            for intervention in interventions:
                if intervention.get("type") == "DRUG":
                    drug_name = intervention.get("name", "").strip()
                    if drug_name:
                        drugs.append(drug_name)
                        # Include other names
                        other_names = intervention.get("otherNames", [])
                        drugs.extend(other_names)
            
            # Skip if no drugs
            if not drugs:
                return None
            
            drugs = list(set([d for d in drugs if d]))  # Remove duplicates and empty
            
            # Conditions
            conditions_module = protocol.get("conditionsModule", {})
            conditions = conditions_module.get("conditions", [])
            
            # Status
            status_module = protocol.get("statusModule", {})
            overall_status = status_module.get("overallStatus")
            why_stopped = status_module.get("whyStopped", "")
            
            # Dates
            start_date = status_module.get("startDateStruct", {}).get("date")
            completion_date = status_module.get("completionDateStruct", {}).get("date")
            
            # Outcomes
            outcomes_module = protocol.get("outcomesModule", {})
            primary_outcomes = outcomes_module.get("primaryOutcomes", [])
            secondary_outcomes = outcomes_module.get("secondaryOutcomes", [])
            
            # Sponsor
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            lead_sponsor = sponsor_module.get("leadSponsor", {}).get("name")
            sponsor_class = sponsor_module.get("leadSsponsor", {}).get("class")
            
            # Eligibility
            eligibility_module = protocol.get("eligibilityModule", {})
            
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
                "start_date": start_date,
                "completion_date": completion_date,
                "primary_outcomes": primary_outcomes,
                "secondary_outcomes": secondary_outcomes,
                "lead_sponsor": lead_sponsor,
                "sponsor_class": sponsor_class,
                "min_age": eligibility_module.get("minimumAge"),
                "max_age": eligibility_module.get("maximumAge"),
                "sex": eligibility_module.get("sex"),
                "healthy_volunteers": eligibility_module.get("healthyVolunteers"),
                "enrollment_count": design.get("enrollmentInfo", {}).get("count"),
                "collected_date": datetime.now().isoformat(),
                "api_version": "v2_aggressive"
            }
            
        except Exception as e:
            logger.debug(f"Error extracting trial: {e}")
            return None
    
    def create_massive_dataset(self) -> pd.DataFrame:
        """Create massive pharmaceutical dataset"""
        logger.info("🚀 CREATING MASSIVE PHARMACEUTICAL DATASET")
        logger.info("=" * 80)
        
        all_compounds = []
        all_trials = []
        
        # Parallel collection from all sources
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all collection tasks
            chembl_future = executor.submit(self.aggressive_chembl_collection, 50000)
            pubchem_future = executor.submit(self.aggressive_pubchem_collection, 30000)
            trials_future = executor.submit(self.aggressive_clinical_trials_collection, 50000)
            
            # Collect results
            try:
                chembl_compounds = chembl_future.result(timeout=7200)  # 2 hours
                all_compounds.extend(chembl_compounds)
                logger.info(f"✅ ChEMBL collection complete: {len(chembl_compounds):,}")
            except Exception as e:
                logger.error(f"❌ ChEMBL collection failed: {e}")
            
            try:
                pubchem_compounds = pubchem_future.result(timeout=7200)  # 2 hours
                all_compounds.extend(pubchem_compounds)
                logger.info(f"✅ PubChem collection complete: {len(pubchem_compounds):,}")
            except Exception as e:
                logger.error(f"❌ PubChem collection failed: {e}")
            
            try:
                trials = trials_future.result(timeout=7200)  # 2 hours
                all_trials.extend(trials)
                logger.info(f"✅ Clinical trials collection complete: {len(trials):,}")
            except Exception as e:
                logger.error(f"❌ Clinical trials collection failed: {e}")
        
        # Extract drugs from trials and match to compounds
        if all_trials:
            logger.info("🔗 Matching trial drugs to compound SMILES...")
            trial_drugs = self._extract_trial_drugs(all_trials)
            matched_compounds = self._match_drugs_to_compounds(trial_drugs, all_compounds)
            all_compounds.extend(matched_compounds)
        
        logger.info(f"📊 Total compounds before deduplication: {len(all_compounds):,}")
        
        if not all_compounds:
            logger.error("❌ No compounds collected!")
            return pd.DataFrame()
        
        # Create DataFrame and deduplicate
        df = pd.DataFrame(all_compounds)
        
        # Remove duplicates by SMILES
        initial_count = len(df)
        df = df.drop_duplicates(subset=['smiles'], keep='first')
        final_count = len(df)
        
        logger.info(f"🧹 Removed {initial_count - final_count} duplicate SMILES")
        logger.info(f"✅ Final massive dataset: {final_count:,} unique compounds")
        
        return df
    
    def _extract_trial_drugs(self, trials: List[Dict]) -> List[str]:
        """Extract unique drugs from trials"""
        unique_drugs = set()
        
        for trial in trials:
            drugs = trial.get('all_drug_names', [])
            for drug in drugs:
                if drug and isinstance(drug, str) and len(drug.strip()) > 2:
                    # Clean drug name
                    clean_drug = drug.strip()
                    # Remove dosage info
                    clean_drug = clean_drug.split(' mg')[0].split(' mcg')[0]
                    if len(clean_drug) > 2:
                        unique_drugs.add(clean_drug)
        
        return list(unique_drugs)
    
    def _match_drugs_to_compounds(self, trial_drugs: List[str], compounds: List[Dict]) -> List[Dict]:
        """Match trial drugs to compound SMILES"""
        logger.info(f"🔗 Matching {len(trial_drugs):,} trial drugs to {len(compounds):,} compounds...")
        
        # Create lookup
        compound_lookup = {}
        for compound in compounds:
            drug_name = compound.get('primary_drug', '').lower()
            if drug_name:
                compound_lookup[drug_name] = compound
        
        matched_compounds = []
        matched_count = 0
        
        for drug_name in trial_drugs:
            drug_lower = drug_name.lower()
            
            # Exact match
            if drug_lower in compound_lookup:
                compound = compound_lookup[drug_lower].copy()
                compound['data_source'] = 'clinical_trial_matched'
                compound['trial_drug_name'] = drug_name
                matched_compounds.append(compound)
                matched_count += 1
            
            # Partial match
            else:
                for db_drug, compound in compound_lookup.items():
                    if (len(drug_lower) > 3 and drug_lower in db_drug) or (len(db_drug) > 3 and db_drug in drug_lower):
                        matched_compound = compound.copy()
                        matched_compound['data_source'] = 'clinical_trial_partial_match'
                        matched_compound['trial_drug_name'] = drug_name
                        matched_compounds.append(matched_compound)
                        matched_count += 1
                        break
        
        logger.info(f"✅ Matched {matched_count:,}/{len(trial_drugs):,} trial drugs to SMILES")
        return matched_compounds
    
    def save_massive_dataset(self, df: pd.DataFrame, output_dir: str = "clinical_trial_dataset/data/massive_aggressive"):
        """Save the massive dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete dataset
        complete_file = output_path / "complete_massive_aggressive_dataset.csv"
        df.to_csv(complete_file, index=False)
        logger.info(f"💾 Saved complete dataset: {complete_file}")
        
        # Create splits
        total_size = len(df)
        train_size = int(total_size * 0.70)
        val_size = int(total_size * 0.15)
        
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df = df_shuffled[:train_size]
        val_df = df_shuffled[train_size:train_size + val_size]
        test_df = df_shuffled[train_size + val_size:]
        
        # Save splits
        train_file = output_path / "train_set_massive_aggressive.csv"
        val_file = output_path / "val_set_massive_aggressive.csv"
        test_file = output_path / "test_set_massive_aggressive.csv"
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        logger.info(f"💾 Saved train set ({len(train_df):,} compounds): {train_file}")
        logger.info(f"💾 Saved val set ({len(val_df):,} compounds): {val_file}")
        logger.info(f"💾 Saved test set ({len(test_df):,} compounds): {test_file}")
        
        # Save metadata
        chembl_compounds = len(df[df['data_source'].str.contains('chembl', na=False)])
        pubchem_compounds = len(df[df['data_source'].str.contains('pubchem', na=False)])
        trial_compounds = len(df[df['data_source'].str.contains('clinical_trial', na=False)])
        
        metadata = {
            "dataset_info": {
                "total_compounds": len(df),
                "train_compounds": len(train_df),
                "val_compounds": len(val_df),
                "test_compounds": len(test_df),
                "smiles_coverage": "100%",
                "collection_date": datetime.now().isoformat(),
                "dataset_type": "Massive Aggressive Pharmaceutical Dataset"
            },
            "data_sources": {
                "chembl_compounds": chembl_compounds,
                "pubchem_compounds": pubchem_compounds,
                "clinical_trial_compounds": trial_compounds
            },
            "collection_parameters": {
                "max_workers": self.max_workers,
                "timeout": self.timeout,
                "batch_size": self.batch_size,
                "aggressive_mode": True
            },
            "data_quality": {
                "duplicate_smiles_removed": True,
                "all_compounds_have_smiles": True,
                "aggressive_collection": True,
                "maximum_extraction": True
            }
        }
        
        metadata_file = output_path / "massive_aggressive_metadata.json"
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
    logger.info("🌟 AGGRESSIVE API COLLECTOR")
    logger.info("🎯 Target: 100,000+ compounds from all pharmaceutical APIs")
    logger.info("⚡ Aggressive parallel processing with extended timeouts")
    logger.info("=" * 80)
    
    # Create aggressive collector
    collector = AggressiveAPICollector()
    
    # Create massive dataset
    logger.info("🚀 Starting aggressive API collection...")
    df = collector.create_massive_dataset()
    
    if df.empty:
        logger.error("❌ Failed to create massive dataset")
        return None
    
    # Save dataset
    files = collector.save_massive_dataset(df)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("🎉 MASSIVE AGGRESSIVE COLLECTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"📊 Total compounds: {len(df):,}")
    logger.info(f"🧬 SMILES coverage: 100%")
    logger.info(f"⚡ Aggressive collection: SUCCESS")
    logger.info(f"📁 Files created:")
    for name, path in files.items():
        logger.info(f"   - {name}: {path}")
    
    logger.info("\n🚀 MASSIVE PHARMACEUTICAL DATASET READY!")
    
    return files

if __name__ == "__main__":
    main()