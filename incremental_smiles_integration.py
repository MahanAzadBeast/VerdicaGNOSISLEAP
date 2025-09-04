#!/usr/bin/env python3
"""
Incremental SMILES Integration with Progress Saving
Saves progress after each batch to prevent data loss on restarts
"""

import pandas as pd
import requests
import time
import json
import os
import re
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class IncrementalSMILESIntegrator:
    """SMILES integrator with incremental progress saving"""
    
    def __init__(self):
        self.chembl_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.pubchem_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.github_dir = Path("clinical_trial_dataset/data/github_final")
        self.progress_dir = Path("clinical_trial_dataset/data/smiles_progress")
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking files
        self.progress_file = self.progress_dir / "smiles_search_progress.json"
        self.completed_drugs_file = self.progress_dir / "completed_drug_searches.json"
        
    def load_existing_progress(self):
        """Load existing progress to resume from where we left off"""
        logger.info("📂 Loading existing progress...")
        
        completed_drugs = {}
        progress_info = {}
        
        # Load completed drug searches
        if self.completed_drugs_file.exists():
            try:
                with open(self.completed_drugs_file, 'r') as f:
                    completed_drugs = json.load(f)
                logger.info(f"✅ Loaded {len(completed_drugs):,} completed drug searches")
            except Exception as e:
                logger.warning(f"⚠️ Could not load completed drugs: {e}")
        
        # Load progress info
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress_info = json.load(f)
                logger.info(f"✅ Loaded progress info from {progress_info.get('last_update', 'unknown')}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load progress info: {e}")
        
        return completed_drugs, progress_info
    
    def save_progress(self, completed_drugs, progress_info):
        """Save current progress"""
        try:
            # Save completed drug searches
            with open(self.completed_drugs_file, 'w') as f:
                json.dump(completed_drugs, f, indent=2)
            
            # Save progress info
            progress_info['last_update'] = datetime.now().isoformat()
            with open(self.progress_file, 'w') as f:
                json.dump(progress_info, f, indent=2)
            
            logger.info(f"💾 Progress saved: {len(completed_drugs):,} drugs completed")
            
        except Exception as e:
            logger.error(f"❌ Error saving progress: {e}")
    
    def extract_all_unique_drugs(self):
        """Extract all unique drugs from clinical trials"""
        logger.info("🧬 Extracting all unique drugs from clinical trials...")
        
        all_drugs = set()
        
        # Load all trial parts
        for part_num in range(1, 5):
            part_file = self.github_dir / f"trials_part_{part_num}.csv"
            
            if part_file.exists():
                df = pd.read_csv(part_file)
                
                for drug in df['primary_drug'].dropna():
                    if isinstance(drug, str) and len(drug.strip()) > 2:
                        clean_drug = self._clean_drug_name(drug.strip())
                        if clean_drug:
                            all_drugs.add(clean_drug)
        
        unique_drugs = sorted(list(all_drugs))
        logger.info(f"✅ Found {len(unique_drugs):,} unique drugs to process")
        
        return unique_drugs
    
    def _clean_drug_name(self, drug_name):
        """Clean drug name for searching"""
        if not drug_name:
            return None
        
        clean = str(drug_name).strip()
        
        # Remove dosage and formulation
        clean = re.sub(r'\s+\d+\s*(mg|mcg|ml|g|%|unit)\b.*', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\s+(tablet|capsule|injection|solution|gel|cream|patch)\b.*', '', clean, flags=re.IGNORECASE)
        
        # Remove brand markers
        clean = re.sub(r'[®™©]', '', clean)
        clean = re.sub(r'\s*\([^)]*\)\s*', ' ', clean)
        
        # Clean spaces
        clean = ' '.join(clean.split()).strip()
        
        return clean if len(clean) > 2 else None
    
    def search_smiles_for_drug(self, drug_name):
        """Search for SMILES for a single drug"""
        try:
            # Try ChEMBL first
            chembl_result = self._search_chembl_simple(drug_name)
            if chembl_result:
                return chembl_result
            
            # Try PubChem
            pubchem_result = self._search_pubchem_simple(drug_name)
            if pubchem_result:
                return pubchem_result
            
            # Try alternative names
            alternatives = self._get_alternative_names(drug_name)
            for alt_name in alternatives:
                chembl_result = self._search_chembl_simple(alt_name)
                if chembl_result:
                    chembl_result['search_method'] = f'alternative_{alt_name}'
                    return chembl_result
            
            return None
            
        except Exception as e:
            logger.debug(f"Error searching {drug_name}: {e}")
            return None
    
    def _search_chembl_simple(self, drug_name):
        """Simple ChEMBL search"""
        try:
            search_url = f"{self.chembl_url}/molecule/search"
            params = {"q": drug_name, "format": "json", "limit": 5}
            
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            molecules = data.get("molecules", [])
            
            for mol in molecules:
                pref_name = mol.get("pref_name", "").lower()
                if self._names_match(drug_name.lower(), pref_name):
                    structures = mol.get("molecule_structures", {})
                    smiles = structures.get("canonical_smiles")
                    
                    if smiles and len(smiles) > 5:
                        return {
                            'smiles': smiles,
                            'source': 'chembl',
                            'chembl_id': mol.get("molecule_chembl_id"),
                            'search_method': 'chembl_api'
                        }
            
            return None
            
        except Exception as e:
            logger.debug(f"ChEMBL search error: {e}")
            return None
    
    def _search_pubchem_simple(self, drug_name):
        """Simple PubChem search"""
        try:
            clean_name = drug_name.replace("/", " ").strip()
            search_url = f"{self.pubchem_url}/compound/name/{clean_name}/cids/JSON"
            
            response = requests.get(search_url, timeout=8)
            if response.status_code != 200:
                return None
            
            data = response.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            
            if cids:
                cid = cids[0]
                props_url = f"{self.pubchem_url}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
                
                props_response = requests.get(props_url, timeout=8)
                if props_response.status_code == 200:
                    props_data = props_response.json()
                    properties = props_data.get("PropertyTable", {}).get("Properties", [])
                    
                    if properties:
                        smiles = properties[0].get("CanonicalSMILES")
                        if smiles and len(smiles) > 5:
                            return {
                                'smiles': smiles,
                                'source': 'pubchem',
                                'pubchem_cid': cid,
                                'search_method': 'pubchem_api'
                            }
            
            return None
            
        except Exception as e:
            logger.debug(f"PubChem search error: {e}")
            return None
    
    def _get_alternative_names(self, drug_name):
        """Get alternative names for drug"""
        alternatives = []
        
        # Common mappings
        mappings = {
            'acetaminophen': 'paracetamol',
            'epinephrine': 'adrenaline',
            'botox': 'botulinum toxin',
            'ropivacaine': 'ropivacaine hydrochloride'
        }
        
        drug_lower = drug_name.lower()
        
        if drug_lower in mappings:
            alternatives.append(mappings[drug_lower])
        
        # Try without common suffixes
        suffixes = ['hydrochloride', 'sulfate', 'sodium']
        for suffix in suffixes:
            if drug_lower.endswith(f' {suffix}'):
                alternatives.append(drug_lower.replace(f' {suffix}', ''))
        
        return alternatives
    
    def _names_match(self, name1, name2):
        """Check if names match"""
        if name1 == name2:
            return True
        
        if len(name1) > 4 and len(name2) > 4:
            if name1 in name2 or name2 in name1:
                return True
        
        return False
    
    def incremental_smiles_search(self):
        """Perform incremental SMILES search with progress saving"""
        logger.info("🚀 INCREMENTAL SMILES SEARCH WITH PROGRESS SAVING")
        logger.info("=" * 70)
        
        # Load existing progress
        completed_drugs, progress_info = self.load_existing_progress()
        
        # Extract all unique drugs
        all_drugs = self.extract_all_unique_drugs()
        
        # Filter out already completed drugs
        remaining_drugs = [drug for drug in all_drugs if drug not in completed_drugs]
        
        logger.info(f"📊 Search status:")
        logger.info(f"   Total drugs: {len(all_drugs):,}")
        logger.info(f"   Already completed: {len(completed_drugs):,}")
        logger.info(f"   Remaining to search: {len(remaining_drugs):,}")
        
        # Process in small batches with progress saving
        batch_size = 50
        successful_searches = len([d for d in completed_drugs.values() if d.get('smiles')])
        
        for i in range(0, len(remaining_drugs), batch_size):
            batch = remaining_drugs[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(remaining_drugs) - 1) // batch_size + 1
            
            logger.info(f"🔍 Processing batch {batch_num}/{total_batches} ({len(batch)} drugs)")
            
            batch_found = 0
            
            for drug_name in batch:
                # Special case: DpC
                if drug_name.lower() == 'dpc':
                    completed_drugs[drug_name] = {
                        'smiles': 'S=C(N/N=C(C1=NC=CC=C1)\\C2=NC=CC=C2)N(C3CCCCC3)C',
                        'source': 'user_provided',
                        'search_method': 'user_provided_dpc'
                    }
                    successful_searches += 1
                    batch_found += 1
                    continue
                
                # Search for SMILES
                smiles_data = self.search_smiles_for_drug(drug_name)
                
                if smiles_data:
                    completed_drugs[drug_name] = smiles_data
                    successful_searches += 1
                    batch_found += 1
                else:
                    completed_drugs[drug_name] = {
                        'smiles': None,
                        'source': 'not_found',
                        'search_method': 'comprehensive_search_failed'
                    }
                
                # Rate limiting
                time.sleep(0.5)
            
            # Save progress after each batch
            progress_info['batches_completed'] = batch_num
            progress_info['total_batches'] = total_batches
            progress_info['successful_searches'] = successful_searches
            progress_info['batch_found'] = batch_found
            
            self.save_progress(completed_drugs, progress_info)
            
            logger.info(f"✅ Batch {batch_num}: {batch_found}/{len(batch)} found (Total: {successful_searches:,})")
            
            # Don't overwhelm APIs
            time.sleep(2)
        
        logger.info(f"🎉 Incremental search complete: {successful_searches:,}/{len(all_drugs):,} drugs found")
        
        return completed_drugs
    
    def integrate_smiles_with_trials(self, completed_drugs):
        """Integrate found SMILES with clinical trials"""
        logger.info("🔗 INTEGRATING SMILES WITH CLINICAL TRIALS")
        
        all_integrated_trials = []
        
        # Process each trial part
        for part_num in range(1, 5):
            part_file = self.github_dir / f"trials_part_{part_num}.csv"
            
            if part_file.exists():
                logger.info(f"📊 Processing trials part {part_num}...")
                
                df = pd.read_csv(part_file)
                integrated_part = []
                part_smiles_count = 0
                
                for _, trial in df.iterrows():
                    integrated_trial = trial.to_dict()
                    
                    # Add SMILES if available
                    drug_name = trial.get('primary_drug')
                    
                    if drug_name and pd.notna(drug_name):
                        clean_drug = self._clean_drug_name(str(drug_name))
                        
                        if clean_drug and clean_drug in completed_drugs:
                            smiles_data = completed_drugs[clean_drug]
                            
                            if smiles_data.get('smiles'):
                                integrated_trial.update({
                                    'smiles': smiles_data['smiles'],
                                    'smiles_source': smiles_data.get('chembl_id') or smiles_data.get('pubchem_cid', 'unknown'),
                                    'smiles_database': smiles_data['source'],
                                    'smiles_search_method': smiles_data['search_method'],
                                    'has_smiles': True
                                })
                                part_smiles_count += 1
                            else:
                                integrated_trial.update({
                                    'smiles': None,
                                    'smiles_source': 'NOT_FOUND',
                                    'smiles_database': 'none',
                                    'smiles_search_method': 'search_failed',
                                    'has_smiles': False
                                })
                        else:
                            integrated_trial.update({
                                'smiles': None,
                                'smiles_source': 'NOT_SEARCHED',
                                'has_smiles': False
                            })
                    
                    integrated_part.append(integrated_trial)
                
                all_integrated_trials.extend(integrated_part)
                
                # Save part progress
                part_file_integrated = self.progress_dir / f"integrated_part_{part_num}.csv"
                pd.DataFrame(integrated_part).to_csv(part_file_integrated, index=False)
                
                coverage = (part_smiles_count / len(df)) * 100
                logger.info(f"✅ Part {part_num}: {part_smiles_count}/{len(df)} with SMILES ({coverage:.1f}%)")
                logger.info(f"💾 Saved part {part_num} progress")
        
        return pd.DataFrame(all_integrated_trials)
    
    def save_final_integrated_dataset(self, df):
        """Save final integrated dataset"""
        logger.info("💾 SAVING FINAL INTEGRATED DATASET")
        
        # Save complete dataset
        complete_file = self.github_dir / "clinical_trials_final_smiles.csv"
        df.to_csv(complete_file, index=False)
        
        # Create splits
        train_size = int(len(df) * 0.70)
        val_size = int(len(df) * 0.15)
        
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df = df_shuffled[:train_size]
        val_df = df_shuffled[train_size:train_size + val_size]
        test_df = df_shuffled[train_size + val_size:]
        
        # Save splits
        train_file = self.github_dir / "train_final_smiles.csv"
        val_file = self.github_dir / "val_final_smiles.csv"
        test_file = self.github_dir / "test_final_smiles.csv"
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        # Final statistics
        total_with_smiles = df['has_smiles'].sum()
        smiles_coverage = (total_with_smiles / len(df)) * 100
        
        # Verify NCT02688101
        nct_check = df[df['nct_id'] == 'NCT02688101']
        nct_has_smiles = False
        if len(nct_check) > 0:
            nct_record = nct_check.iloc[0]
            nct_has_smiles = nct_record.get('has_smiles', False)
        
        logger.info(f"💾 Final datasets saved:")
        for file_path in [complete_file, train_file, val_file, test_file]:
            size_mb = os.path.getsize(file_path) / (1024*1024)
            count = len(pd.read_csv(file_path))
            status = "✅ GitHub OK" if size_mb < 50 else "⚠️ Large" if size_mb < 100 else "❌ Too large"
            logger.info(f"   {status} {file_path.name}: {count:,} trials ({size_mb:.1f} MB)")
        
        logger.info(f"\\n📊 FINAL SMILES INTEGRATION RESULTS:")
        logger.info(f"   Total trials: {len(df):,}")
        logger.info(f"   Trials with SMILES: {total_with_smiles:,} ({smiles_coverage:.1f}%)")
        logger.info(f"   NCT02688101 with SMILES: {'✅ YES' if nct_has_smiles else '❌ NO'}")
        
        return {
            'coverage': smiles_coverage,
            'total_with_smiles': total_with_smiles,
            'files': [complete_file, train_file, val_file, test_file]
        }

def main():
    """Main execution with incremental progress"""
    logger.info("🌟 INCREMENTAL SMILES INTEGRATION")
    logger.info("💾 With progress saving to prevent data loss")
    logger.info("🔄 Resumes from where it left off")
    logger.info("=" * 70)
    
    integrator = IncrementalSMILESIntegrator()
    
    # Perform incremental search
    completed_drugs = integrator.incremental_smiles_search()
    
    # Integrate with trials
    integrated_df = integrator.integrate_smiles_with_trials(completed_drugs)
    
    # Save final dataset
    results = integrator.save_final_integrated_dataset(integrated_df)
    
    logger.info("\\n" + "=" * 70)
    logger.info("🎉 INCREMENTAL SMILES INTEGRATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"📊 Final SMILES coverage: {results['coverage']:.1f}%")
    logger.info(f"💾 Progress saved throughout process")
    logger.info(f"🔄 Can resume if interrupted")
    logger.info(f"✅ No more restarts needed")
    
    return results

if __name__ == "__main__":
    main()