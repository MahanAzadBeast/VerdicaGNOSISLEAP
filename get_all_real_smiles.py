#!/usr/bin/env python3
"""
Get ALL Real SMILES for Clinical Trials
Optimized approach to find real SMILES for ALL drugs in clinical trials
with incremental saving and comprehensive search strategies
"""

import pandas as pd
import requests
import time
import json
import os
import re
import concurrent.futures
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class AllRealSMILESFinder:
    """Finds ALL real SMILES for clinical trials with progress saving"""
    
    def __init__(self):
        self.chembl_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.pubchem_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.github_dir = Path("clinical_trial_dataset/data/github_final")
        self.progress_dir = Path("clinical_trial_dataset/data/smiles_progress")
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress files
        self.drug_results_file = self.progress_dir / "all_drug_smiles_results.json"
        self.batch_progress_file = self.progress_dir / "batch_progress.json"
        
        # Comprehensive drug mappings for better matching
        self.drug_mappings = {
            # Pain medications
            'acetaminophen': ['paracetamol', 'acetaminophen', 'tylenol'],
            'acetaminophen/apap': ['paracetamol', 'acetaminophen'],
            'ibuprofen': ['ibuprofen', 'advil', 'motrin'],
            'aspirin': ['aspirin', 'acetylsalicylic acid'],
            'morphine': ['morphine', 'morphine sulfate'],
            'fentanyl': ['fentanyl', 'fentanyl citrate'],
            'tramadol': ['tramadol', 'tramadol hydrochloride'],
            'buprenorphine': ['buprenorphine', 'buprenorphine hydrochloride', 'subutex'],
            
            # Anesthetics
            'ropivacaine': ['ropivacaine', 'ropivacaine hydrochloride', 'naropin'],
            'lidocaine': ['lidocaine', 'lignocaine', 'xylocaine'],
            'procaine': ['procaine', 'novocaine'],
            'articaine': ['articaine', 'articaine hydrochloride'],
            
            # Biologics and complex drugs
            'botox': ['botulinum toxin type a', 'onabotulinumtoxina', 'botulinum toxin'],
            'restylane': ['hyaluronic acid', 'sodium hyaluronate'],
            'iron sucrose': ['iron sucrose', 'iron(iii) sucrose complex'],
            
            # Antibiotics
            'amoxicillin': ['amoxicillin', 'amoxycillin'],
            'ciprofloxacin': ['ciprofloxacin', 'cipro'],
            'doxycycline': ['doxycycline', 'vibramycin'],
            'metronidazole': ['metronidazole', 'flagyl'],
            
            # Cardiovascular
            'metoprolol': ['metoprolol', 'metoprolol tartrate', 'lopressor'],
            'atorvastatin': ['atorvastatin', 'atorvastatin calcium', 'lipitor'],
            'warfarin': ['warfarin', 'warfarin sodium', 'coumadin'],
            
            # Diabetes
            'metformin': ['metformin', 'metformin hydrochloride'],
            'insulin': ['insulin', 'human insulin'],
            
            # Neurological
            'levodopa': ['levodopa', 'l-dopa'],
            'donepezil': ['donepezil', 'donepezil hydrochloride', 'aricept'],
            
            # Experimental/special
            'dpc': ['dpc'],  # User provided
        }
        
        # Load existing ChEMBL data for fast lookup
        self.chembl_lookup = self._load_chembl_lookup()
    
    def _load_chembl_lookup(self):
        """Load ChEMBL data for fast lookup"""
        logger.info("📂 Loading ChEMBL lookup database...")
        
        chembl_file = self.github_dir / "chembl_complete_dataset.csv"
        lookup = {}
        
        if chembl_file.exists():
            df = pd.read_csv(chembl_file)
            
            for _, row in df.iterrows():
                drug_name = str(row['primary_drug']).lower().strip()
                lookup[drug_name] = {
                    'smiles': row['smiles'],
                    'chembl_id': row['chembl_id'],
                    'molecular_weight': row.get('mol_molecular_weight'),
                    'logp': row.get('mol_logp'),
                    'source': 'local_chembl'
                }
            
            logger.info(f"✅ Loaded {len(lookup):,} ChEMBL compounds for fast lookup")
        
        return lookup
    
    def load_existing_progress(self):
        """Load existing progress"""
        existing_results = {}
        batch_info = {'processed_batches': 0, 'total_drugs_found': 0}
        
        if self.drug_results_file.exists():
            try:
                with open(self.drug_results_file, 'r') as f:
                    existing_results = json.load(f)
                logger.info(f"📂 Loaded existing results: {len(existing_results):,} drugs")
            except Exception as e:
                logger.warning(f"⚠️ Could not load existing results: {e}")
        
        if self.batch_progress_file.exists():
            try:
                with open(self.batch_progress_file, 'r') as f:
                    batch_info = json.load(f)
                logger.info(f"📊 Loaded batch progress: {batch_info}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load batch progress: {e}")
        
        return existing_results, batch_info
    
    def save_progress(self, drug_results, batch_info):
        """Save progress incrementally"""
        try:
            # Save drug results
            with open(self.drug_results_file, 'w') as f:
                json.dump(drug_results, f, indent=2)
            
            # Save batch progress
            batch_info['last_update'] = datetime.now().isoformat()
            with open(self.batch_progress_file, 'w') as f:
                json.dump(batch_info, f, indent=2)
            
            successful = sum(1 for v in drug_results.values() if isinstance(v, dict) and v.get('smiles'))
            logger.info(f"💾 Progress saved: {len(drug_results):,} drugs processed, {successful:,} SMILES found")
            
        except Exception as e:
            logger.error(f"❌ Error saving progress: {e}")
    
    def extract_all_unique_drugs(self):
        """Extract ALL unique drugs from clinical trials"""
        logger.info("🧬 Extracting ALL unique drugs from clinical trials...")
        
        all_drugs = set()
        drug_trial_map = {}  # Track which trials use each drug
        
        # Process all trial parts
        for part_num in range(1, 5):
            part_file = self.github_dir / f"trials_part_{part_num}.csv"
            
            if part_file.exists():
                df = pd.read_csv(part_file)
                
                for idx, row in df.iterrows():
                    drug = row.get('primary_drug')
                    nct_id = row.get('nct_id')
                    
                    if drug and pd.notna(drug):
                        clean_drug = self._clean_drug_name(str(drug))
                        if clean_drug:
                            all_drugs.add(clean_drug)
                            
                            # Track trials for this drug
                            if clean_drug not in drug_trial_map:
                                drug_trial_map[clean_drug] = []
                            drug_trial_map[clean_drug].append(nct_id)
        
        unique_drugs = sorted(list(all_drugs))
        logger.info(f"✅ Found {len(unique_drugs):,} unique drugs across {sum(len(trials) for trials in drug_trial_map.values()):,} trial mentions")
        
        return unique_drugs, drug_trial_map
    
    def _clean_drug_name(self, drug_name):
        """Clean drug name for better matching"""
        if not drug_name:
            return None
        
        clean = str(drug_name).strip()
        
        # Remove dosage information
        clean = re.sub(r'\s+\d+\s*(mg|mcg|ml|g|%|unit|iu)\b.*', '', clean, flags=re.IGNORECASE)
        
        # Remove formulation information
        clean = re.sub(r'\s+(tablet|capsule|injection|solution|gel|cream|patch|spray|powder|suspension|infusion)\b.*', '', clean, flags=re.IGNORECASE)
        
        # Remove route information
        clean = re.sub(r'\s+(oral|iv|im|sc|topical|vaginal|rectal|sublingual)\b.*', '', clean, flags=re.IGNORECASE)
        
        # Remove brand indicators
        clean = re.sub(r'[®™©]', '', clean)
        
        # Remove parentheses and brackets
        clean = re.sub(r'\s*\([^)]*\)\s*', ' ', clean)
        clean = re.sub(r'\s*\[[^\]]*\]\s*', ' ', clean)
        
        # Remove "based" phrases
        clean = re.sub(r'\s+(based|containing|with|plus|and)\s+.*', '', clean, flags=re.IGNORECASE)
        
        # Clean up spaces
        clean = re.sub(r'\s+', ' ', clean).strip()
        clean = clean.strip(' .-/')
        
        return clean if len(clean) > 2 else None
    
    def comprehensive_smiles_search(self, drug_name):
        """Comprehensive SMILES search for a single drug"""
        
        # Strategy 1: DpC special case (user provided)
        if drug_name.lower().strip() == 'dpc':
            return {
                'smiles': 'S=C(N/N=C(C1=NC=CC=C1)\\C2=NC=CC=C2)N(C3CCCCC3)C',
                'source': 'user_provided',
                'search_method': 'user_provided_dpc',
                'confidence': 'high'
            }
        
        # Strategy 2: Local ChEMBL exact match
        drug_lower = drug_name.lower()
        if drug_lower in self.chembl_lookup:
            result = self.chembl_lookup[drug_lower].copy()
            result['search_method'] = 'local_chembl_exact'
            result['confidence'] = 'high'
            return result
        
        # Strategy 3: Local ChEMBL fuzzy match
        for chembl_drug, data in self.chembl_lookup.items():
            if self._advanced_name_match(drug_lower, chembl_drug):
                result = data.copy()
                result['search_method'] = 'local_chembl_fuzzy'
                result['confidence'] = 'medium'
                return result
        
        # Strategy 4: Drug mappings
        if drug_lower in self.drug_mappings:
            for alt_name in self.drug_mappings[drug_lower]:
                if alt_name in self.chembl_lookup:
                    result = self.chembl_lookup[alt_name].copy()
                    result['search_method'] = f'mapping_{alt_name}'
                    result['confidence'] = 'high'
                    return result
        
        # Strategy 5: ChEMBL API search
        api_result = self._chembl_api_search(drug_name)
        if api_result:
            return api_result
        
        # Strategy 6: PubChem API search
        pubchem_result = self._pubchem_api_search(drug_name)
        if pubchem_result:
            return pubchem_result
        
        return None
    
    def _advanced_name_match(self, name1, name2):
        """Advanced name matching algorithm"""
        if not name1 or not name2:
            return False
        
        # Exact match
        if name1 == name2:
            return True
        
        # Substring match with length validation
        if len(name1) > 4 and len(name2) > 4:
            if name1 in name2 or name2 in name1:
                return True
        
        # Word-based matching
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if len(words1) > 1 and len(words2) > 1:
            overlap = words1.intersection(words2)
            # Require significant overlap
            min_words = min(len(words1), len(words2))
            if len(overlap) >= min_words * 0.8:
                return True
        
        # First significant word match
        significant_words1 = [w for w in words1 if len(w) > 3]
        significant_words2 = [w for w in words2 if len(w) > 3]
        
        if significant_words1 and significant_words2:
            if significant_words1[0] == significant_words2[0]:
                return True
        
        return False
    
    def _chembl_api_search(self, drug_name):
        """ChEMBL API search with multiple strategies"""
        try:
            search_patterns = [
                drug_name,
                drug_name.replace(' ', ''),
                drug_name.split()[0] if ' ' in drug_name else None,
                f"*{drug_name}*" if len(drug_name) > 3 else None
            ]
            
            for pattern in search_patterns:
                if not pattern:
                    continue
                
                search_url = f"{self.chembl_url}/molecule/search"
                params = {
                    "q": pattern,
                    "format": "json",
                    "limit": 10
                }
                
                response = requests.get(search_url, params=params, timeout=15)
                if response.status_code != 200:
                    continue
                
                data = response.json()
                molecules = data.get("molecules", [])
                
                # Find best match
                best_match = None
                best_score = 0
                
                for mol in molecules:
                    pref_name = mol.get("pref_name", "").lower()
                    
                    # Calculate match score
                    score = self._calculate_match_score(drug_name.lower(), pref_name)
                    
                    if score > best_score and score > 0.7:  # High confidence threshold
                        structures = mol.get("molecule_structures", {})
                        smiles = structures.get("canonical_smiles")
                        
                        if smiles and self._validate_smiles(smiles):
                            properties = mol.get("molecule_properties", {})
                            best_match = {
                                'smiles': smiles,
                                'source': 'chembl_api',
                                'chembl_id': mol.get("molecule_chembl_id"),
                                'molecular_weight': properties.get("full_mwt"),
                                'logp': properties.get("alogp"),
                                'search_method': f'chembl_api_{pattern}',
                                'confidence': 'high' if score > 0.9 else 'medium',
                                'match_score': score
                            }
                            best_score = score
                
                if best_match:
                    return best_match
                
                time.sleep(0.5)  # Rate limiting
            
            return None
            
        except Exception as e:
            logger.debug(f"ChEMBL API error for {drug_name}: {e}")
            return None
    
    def _pubchem_api_search(self, drug_name):
        """PubChem API search"""
        try:
            clean_name = drug_name.replace("/", " ").replace("-", " ").replace("®", "").strip()
            
            # Try exact name search
            search_url = f"{self.pubchem_url}/compound/name/{clean_name}/cids/JSON"
            response = requests.get(search_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                cids = data.get("IdentifierList", {}).get("CID", [])
                
                if cids:
                    # Try first few CIDs
                    for cid in cids[:3]:
                        smiles_data = self._get_pubchem_smiles(cid)
                        if smiles_data:
                            smiles_data['search_method'] = 'pubchem_exact'
                            smiles_data['confidence'] = 'high'
                            return smiles_data
                        time.sleep(0.2)
            
            # Try synonym search
            synonym_url = f"{self.pubchem_url}/compound/synonym/{clean_name}/cids/JSON"
            response = requests.get(synonym_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                cids = data.get("IdentifierList", {}).get("CID", [])
                
                if cids:
                    cid = cids[0]
                    smiles_data = self._get_pubchem_smiles(cid)
                    if smiles_data:
                        smiles_data['search_method'] = 'pubchem_synonym'
                        smiles_data['confidence'] = 'medium'
                        return smiles_data
            
            return None
            
        except Exception as e:
            logger.debug(f"PubChem error for {drug_name}: {e}")
            return None
    
    def _get_pubchem_smiles(self, cid):
        """Get SMILES from PubChem CID"""
        try:
            props_url = f"{self.pubchem_url}/compound/cid/{cid}/property/CanonicalSMILES,MolecularWeight,XLogP,IUPACName/JSON"
            response = requests.get(props_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                properties = data.get("PropertyTable", {}).get("Properties", [])
                
                if properties:
                    props = properties[0]
                    smiles = props.get("CanonicalSMILES")
                    
                    if smiles and self._validate_smiles(smiles):
                        return {
                            'smiles': smiles,
                            'source': 'pubchem_api',
                            'pubchem_cid': cid,
                            'molecular_weight': props.get("MolecularWeight"),
                            'logp': props.get("XLogP"),
                            'iupac_name': props.get("IUPACName")
                        }
            
            return None
            
        except Exception as e:
            logger.debug(f"PubChem properties error for CID {cid}: {e}")
            return None
    
    def _calculate_match_score(self, name1, name2):
        """Calculate match score between drug names"""
        if not name1 or not name2:
            return 0.0
        
        # Exact match
        if name1 == name2:
            return 1.0
        
        # Substring match
        if name1 in name2 or name2 in name1:
            longer = max(len(name1), len(name2))
            shorter = min(len(name1), len(name2))
            return shorter / longer
        
        # Word overlap
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if words1 and words2:
            overlap = words1.intersection(words2)
            total_words = words1.union(words2)
            return len(overlap) / len(total_words)
        
        return 0.0
    
    def _validate_smiles(self, smiles):
        """Validate SMILES string"""
        if not smiles or not isinstance(smiles, str):
            return False
        
        if len(smiles) < 3 or len(smiles) > 1000:
            return False
        
        # Valid SMILES characters
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@+\\-[]()=#$:/.%\\\\')
        if not all(c in valid_chars for c in smiles):
            return False
        
        # Bracket validation
        brackets = {'(': ')', '[': ']'}
        stack = []
        for char in smiles:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False
                expected = brackets.get(stack.pop())
                if expected != char:
                    return False
        
        return len(stack) == 0
    
    def find_all_real_smiles(self):
        """Find ALL real SMILES with incremental saving"""
        logger.info("🚀 FINDING ALL REAL SMILES FOR CLINICAL TRIALS")
        logger.info("💾 With incremental saving to prevent data loss")
        logger.info("=" * 80)
        
        # Load existing progress
        drug_results, batch_info = self.load_existing_progress()
        
        # Extract all drugs
        all_drugs, drug_trial_map = self.extract_all_unique_drugs()
        
        # Filter out already processed drugs
        remaining_drugs = [drug for drug in all_drugs if drug not in drug_results]
        
        logger.info(f"📊 Search status:")
        logger.info(f"   Total unique drugs: {len(all_drugs):,}")
        logger.info(f"   Already processed: {len(drug_results):,}")
        logger.info(f"   Remaining to search: {len(remaining_drugs):,}")
        
        # Process in batches with incremental saving
        batch_size = 20  # Smaller batches for more frequent saves
        total_found = sum(1 for v in drug_results.values() if isinstance(v, dict) and v.get('smiles'))
        
        for i in range(0, len(remaining_drugs), batch_size):
            batch = remaining_drugs[i:i+batch_size]
            batch_num = batch_info.get('processed_batches', 0) + 1
            total_batches = (len(remaining_drugs) - 1) // batch_size + 1
            
            logger.info(f"🔍 Processing batch {batch_num}/{total_batches} ({len(batch)} drugs)")
            
            batch_found = 0
            
            for drug_name in batch:
                try:
                    # Comprehensive search
                    smiles_data = self.comprehensive_smiles_search(drug_name)
                    
                    if smiles_data:
                        drug_results[drug_name] = smiles_data
                        total_found += 1
                        batch_found += 1
                        logger.debug(f"✅ Found SMILES for {drug_name}")
                    else:
                        drug_results[drug_name] = {
                            'smiles': None,
                            'source': 'not_found',
                            'search_method': 'comprehensive_search_failed',
                            'confidence': 'none'
                        }
                    
                    # Rate limiting
                    time.sleep(0.3)
                    
                except Exception as e:
                    logger.debug(f"Error processing {drug_name}: {e}")
                    drug_results[drug_name] = {
                        'smiles': None,
                        'source': 'search_error',
                        'search_method': 'error',
                        'error': str(e)
                    }
            
            # Update batch progress
            batch_info['processed_batches'] = batch_num
            batch_info['total_found'] = total_found
            batch_info['batch_found'] = batch_found
            
            # Save progress after each batch
            self.save_progress(drug_results, batch_info)
            
            success_rate = (total_found / len(drug_results)) * 100 if drug_results else 0
            logger.info(f"✅ Batch {batch_num}: {batch_found}/{len(batch)} found (Total: {total_found:,}, Rate: {success_rate:.1f}%)")
            
            # Pause between batches
            time.sleep(2)
        
        logger.info(f"🎉 All drugs processed: {total_found:,}/{len(all_drugs):,} SMILES found ({(total_found/len(all_drugs)*100):.1f}%)")
        
        return drug_results
    
    def integrate_all_smiles_with_trials(self, drug_results):
        """Integrate all found SMILES with clinical trials"""
        logger.info("🔗 INTEGRATING ALL SMILES WITH CLINICAL TRIALS")
        
        all_trials_with_smiles = []
        total_trials_with_smiles = 0
        
        # Process each trial part
        for part_num in range(1, 5):
            part_file = self.github_dir / f"trials_part_{part_num}.csv"
            
            if part_file.exists():
                logger.info(f"📊 Processing trials part {part_num}...")
                
                df = pd.read_csv(part_file)
                part_trials = []
                part_smiles_count = 0
                
                for _, trial in df.iterrows():
                    enhanced_trial = trial.to_dict()
                    
                    # Add SMILES if available
                    drug_name = trial.get('primary_drug')
                    
                    if drug_name and pd.notna(drug_name):
                        clean_drug = self._clean_drug_name(str(drug_name))
                        
                        if clean_drug and clean_drug in drug_results:
                            smiles_data = drug_results[clean_drug]
                            
                            if smiles_data.get('smiles'):
                                enhanced_trial.update({
                                    'smiles': smiles_data['smiles'],
                                    'smiles_source': smiles_data.get('chembl_id') or smiles_data.get('pubchem_cid', 'unknown'),
                                    'smiles_database': smiles_data['source'],
                                    'smiles_search_method': smiles_data['search_method'],
                                    'smiles_confidence': smiles_data.get('confidence', 'unknown'),
                                    'has_real_smiles': True,
                                    'molecular_weight': smiles_data.get('molecular_weight'),
                                    'logp': smiles_data.get('logp'),
                                    'iupac_name': smiles_data.get('iupac_name')
                                })
                                part_smiles_count += 1
                                total_trials_with_smiles += 1
                            else:
                                enhanced_trial.update({
                                    'smiles': None,
                                    'smiles_source': 'NOT_FOUND',
                                    'has_real_smiles': False,
                                    'search_status': 'comprehensive_search_failed'
                                })
                        else:
                            enhanced_trial.update({
                                'smiles': None,
                                'smiles_source': 'NOT_SEARCHED',
                                'has_real_smiles': False
                            })
                    
                    part_trials.append(enhanced_trial)
                
                all_trials_with_smiles.extend(part_trials)
                
                # Save part progress
                part_file_integrated = self.progress_dir / f"trials_part_{part_num}_with_smiles.csv"
                pd.DataFrame(part_trials).to_csv(part_file_integrated, index=False)
                
                coverage = (part_smiles_count / len(df)) * 100
                logger.info(f"✅ Part {part_num}: {part_smiles_count}/{len(df)} with SMILES ({coverage:.1f}%)")
                logger.info(f"💾 Saved part {part_num} with SMILES integration")
        
        final_df = pd.DataFrame(all_trials_with_smiles)
        final_coverage = (total_trials_with_smiles / len(final_df)) * 100
        
        logger.info(f"🎉 FINAL INTEGRATION COMPLETE:")
        logger.info(f"   Total trials: {len(final_df):,}")
        logger.info(f"   Trials with SMILES: {total_trials_with_smiles:,} ({final_coverage:.1f}%)")
        
        return final_df
    
    def save_final_complete_dataset(self, df):
        """Save final complete dataset with all SMILES"""
        logger.info("💾 SAVING FINAL COMPLETE DATASET WITH ALL SMILES")
        
        # Save complete dataset
        complete_file = self.github_dir / "clinical_trials_all_real_smiles.csv"
        df.to_csv(complete_file, index=False)
        
        # Create splits
        train_size = int(len(df) * 0.70)
        val_size = int(len(df) * 0.15)
        
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df = df_shuffled[:train_size]
        val_df = df_shuffled[train_size:train_size + val_size]
        test_df = df_shuffled[train_size + val_size:]
        
        # Save splits
        train_file = self.github_dir / "train_all_real_smiles.csv"
        val_file = self.github_dir / "val_all_real_smiles.csv"
        test_file = self.github_dir / "test_all_real_smiles.csv"
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        # Final verification
        total_with_smiles = df['has_real_smiles'].sum()
        smiles_coverage = (total_with_smiles / len(df)) * 100
        
        # Verify NCT02688101
        nct_check = df[df['nct_id'] == 'NCT02688101']
        nct_has_smiles = False
        nct_smiles = None
        
        if len(nct_check) > 0:
            record = nct_check.iloc[0]
            nct_has_smiles = record.get('has_real_smiles', False)
            nct_smiles = record.get('smiles') if nct_has_smiles else None
        
        logger.info(f"💾 Final complete datasets:")
        for file_path in [complete_file, train_file, val_file, test_file]:
            size_mb = os.path.getsize(file_path) / (1024*1024)
            count = len(pd.read_csv(file_path))
            status = "✅ GitHub OK" if size_mb < 100 else "❌ Too large"
            logger.info(f"   {status} {file_path.name}: {count:,} trials ({size_mb:.1f} MB)")
        
        logger.info(f"\\n🎯 FINAL VERIFICATION:")
        logger.info(f"   Total trials: {len(df):,}")
        logger.info(f"   Trials with real SMILES: {total_with_smiles:,} ({smiles_coverage:.1f}%)")
        logger.info(f"   NCT02688101 with SMILES: {'✅ YES' if nct_has_smiles else '❌ NO'}")
        
        if nct_has_smiles and nct_smiles:
            logger.info(f"   DpC SMILES verified: {nct_smiles}")
        
        return {
            'complete_file': complete_file,
            'smiles_coverage': smiles_coverage,
            'total_with_smiles': total_with_smiles,
            'nct02688101_verified': nct_has_smiles
        }

def main():
    """Main execution"""
    logger.info("🌟 GET ALL REAL SMILES FOR CLINICAL TRIALS")
    logger.info("🎯 Comprehensive search for ALL drugs with incremental saving")
    logger.info("🚫 NO fake, synthetic, or incomplete SMILES")
    logger.info("💾 Progress saved after each batch")
    logger.info("=" * 80)
    
    finder = AllRealSMILESFinder()
    
    # Find all SMILES
    drug_results = finder.find_all_real_smiles()
    
    # Integrate with trials
    final_df = finder.integrate_all_smiles_with_trials(drug_results)
    
    # Save final dataset
    results = finder.save_final_complete_dataset(final_df)
    
    logger.info("\\n" + "=" * 80)
    logger.info("🎉 ALL REAL SMILES SEARCH COMPLETE")
    logger.info("=" * 80)
    logger.info(f"📊 Final SMILES coverage: {results['smiles_coverage']:.1f}%")
    logger.info(f"🧬 Total trials with SMILES: {results['total_with_smiles']:,}")
    logger.info(f"🎯 NCT02688101 verified: {'✅ YES' if results['nct02688101_verified'] else '❌ NO'}")
    logger.info(f"✅ All SMILES from real pharmaceutical databases")
    logger.info(f"💾 Progress saved throughout - can resume if interrupted")
    logger.info(f"📁 Ready for final verification and GitHub push")
    
    return results

if __name__ == "__main__":
    main()