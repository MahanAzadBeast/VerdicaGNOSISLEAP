#!/usr/bin/env python3
"""
Clinical Trials Dataset Expander
Connects to ClinicalTrials.gov API to collect MASSIVE real clinical trials dataset
Target: 50,000+ trials before SMILES matching and integration
"""

import requests
import pandas as pd
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import sys
import urllib.parse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClinicalTrialsExpander:
    """Expands clinical trials dataset using ClinicalTrials.gov API"""
    
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.collected_trials = []
        
    def get_total_available_studies(self) -> int:
        """Get total number of studies available in the API"""
        logger.info("🔍 Checking total available studies in ClinicalTrials.gov...")
        
        try:
            params = {
                "format": "json",
                "pageSize": 1,  # Just get count
                "countTotal": True
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            total_count = data.get("totalCount", 0)
            
            logger.info(f"📊 Total available studies: {total_count:,}")
            return total_count
            
        except Exception as e:
            logger.error(f"Error getting total count: {e}")
            return 0
    
    def collect_massive_trials_dataset(self, target_trials: int = 50000) -> pd.DataFrame:
        """Collect massive clinical trials dataset"""
        logger.info(f"🚀 Collecting MASSIVE clinical trials dataset (target: {target_trials:,})")
        
        # Check what's available
        total_available = self.get_total_available_studies()
        if total_available == 0:
            logger.error("❌ Could not determine available studies")
            return pd.DataFrame()
        
        # Adjust target based on availability
        actual_target = min(target_trials, total_available)
        logger.info(f"🎯 Adjusted target: {actual_target:,} trials")
        
        all_trials = []
        
        # Strategy 1: Collect by study type
        interventional_trials = self.collect_trials_by_type("INTERVENTIONAL", actual_target // 2)
        all_trials.extend(interventional_trials)
        
        # Strategy 2: Collect observational studies
        remaining = actual_target - len(all_trials)
        if remaining > 0:
            observational_trials = self.collect_trials_by_type("OBSERVATIONAL", remaining)
            all_trials.extend(observational_trials)
        
        # Strategy 3: Collect expanded access studies
        remaining = actual_target - len(all_trials)
        if remaining > 0:
            expanded_trials = self.collect_trials_by_type("EXPANDED_ACCESS", remaining)
            all_trials.extend(expanded_trials)
        
        # Strategy 4: Collect by recent years if still need more
        remaining = actual_target - len(all_trials)
        if remaining > 0:
            recent_trials = self.collect_trials_by_date_range(remaining)
            all_trials.extend(recent_trials)
        
        logger.info(f"📊 Total trials collected: {len(all_trials):,}")
        
        # Convert to DataFrame and clean
        if all_trials:
            df = pd.DataFrame(all_trials)
            df = self.clean_trials_data(df)
            return df
        else:
            return pd.DataFrame()
    
    def collect_trials_by_type(self, study_type: str, target: int) -> List[Dict]:
        """Collect trials by study type"""
        logger.info(f"🔬 Collecting {study_type} trials (target: {target:,})")
        
        trials = []
        page_token = None
        page_size = 1000  # Maximum allowed
        
        while len(trials) < target:
            try:
                params = {
                    "format": "json",
                    "pageSize": min(page_size, target - len(trials)),
                    "query.studyType": study_type
                }
                
                if page_token:
                    params["pageToken"] = page_token
                
                logger.info(f"📥 Fetching {study_type} page (collected: {len(trials):,})")
                
                response = requests.get(self.base_url, params=params, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                studies = data.get("studies", [])
                
                if not studies:
                    logger.info(f"✅ No more {study_type} studies available")
                    break
                
                # Process studies
                batch_trials = []
                for study in studies:
                    trial_data = self.extract_trial_data(study)
                    if trial_data:
                        batch_trials.append(trial_data)
                
                trials.extend(batch_trials)
                logger.info(f"✅ {study_type}: {len(batch_trials)} valid trials (Total: {len(trials):,})")
                
                # Get next page token
                page_token = data.get("nextPageToken")
                if not page_token:
                    logger.info(f"✅ Reached end of {study_type} studies")
                    break
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error collecting {study_type} trials: {e}")
                time.sleep(2)
                continue
        
        logger.info(f"🎉 {study_type} collection complete: {len(trials):,} trials")
        return trials
    
    def collect_trials_by_date_range(self, target: int) -> List[Dict]:
        """Collect trials by date ranges to get more recent studies"""
        logger.info(f"📅 Collecting recent trials by date (target: {target:,})")
        
        trials = []
        current_year = datetime.now().year
        
        # Collect from recent years first
        for year in range(current_year, current_year - 10, -1):  # Last 10 years
            if len(trials) >= target:
                break
                
            year_trials = self.collect_trials_by_year(year, (target - len(trials)) // (current_year - year + 1))
            trials.extend(year_trials)
            
            logger.info(f"✅ Year {year}: {len(year_trials)} trials (Total: {len(trials):,})")
        
        return trials
    
    def collect_trials_by_year(self, year: int, target: int) -> List[Dict]:
        """Collect trials started in a specific year"""
        logger.info(f"📅 Collecting trials from {year} (target: {target:,})")
        
        trials = []
        page_token = None
        page_size = 1000
        
        # Date range for the year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        while len(trials) < target:
            try:
                params = {
                    "format": "json",
                    "pageSize": min(page_size, target - len(trials)),
                    "query.startDate": f"{start_date}:{end_date}"
                }
                
                if page_token:
                    params["pageToken"] = page_token
                
                response = requests.get(self.base_url, params=params, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                studies = data.get("studies", [])
                
                if not studies:
                    break
                
                # Process studies
                batch_trials = []
                for study in studies:
                    trial_data = self.extract_trial_data(study)
                    if trial_data:
                        batch_trials.append(trial_data)
                
                trials.extend(batch_trials)
                
                # Get next page token
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                
                time.sleep(0.3)
                
            except Exception as e:
                logger.debug(f"Error collecting {year} trials: {e}")
                break
        
        return trials
    
    def collect_drug_specific_trials(self, target: int = 10000) -> List[Dict]:
        """Collect trials for specific drug categories"""
        logger.info(f"💊 Collecting drug-specific trials (target: {target:,})")
        
        # Common drug categories and terms
        drug_terms = [
            "cancer treatment", "chemotherapy", "immunotherapy", "targeted therapy",
            "diabetes", "insulin", "metformin", "cardiovascular", "hypertension",
            "antibiotics", "antiviral", "vaccine", "monoclonal antibody",
            "pain management", "analgesic", "opioid", "nsaid",
            "psychiatric", "antidepressant", "antipsychotic", "mood stabilizer",
            "neurological", "alzheimer", "parkinson", "epilepsy",
            "respiratory", "asthma", "copd", "bronchodilator",
            "oncology", "breast cancer", "lung cancer", "prostate cancer",
            "inflammatory", "arthritis", "autoimmune", "immunosuppressive"
        ]
        
        all_trials = []
        trials_per_term = target // len(drug_terms)
        
        for term in drug_terms:
            if len(all_trials) >= target:
                break
                
            try:
                term_trials = self.collect_trials_by_condition(term, trials_per_term)
                all_trials.extend(term_trials)
                logger.info(f"✅ '{term}': {len(term_trials)} trials (Total: {len(all_trials):,})")
                
                time.sleep(1)  # Rate limiting between terms
                
            except Exception as e:
                logger.warning(f"Error collecting trials for '{term}': {e}")
                continue
        
        return all_trials[:target]
    
    def collect_trials_by_condition(self, condition: str, target: int) -> List[Dict]:
        """Collect trials by medical condition"""
        trials = []
        page_token = None
        page_size = 500
        
        while len(trials) < target:
            try:
                params = {
                    "format": "json",
                    "pageSize": min(page_size, target - len(trials)),
                    "query.condition": condition
                }
                
                if page_token:
                    params["pageToken"] = page_token
                
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                studies = data.get("studies", [])
                
                if not studies:
                    break
                
                # Process studies
                batch_trials = []
                for study in studies:
                    trial_data = self.extract_trial_data(study)
                    if trial_data:
                        batch_trials.append(trial_data)
                
                trials.extend(batch_trials)
                
                # Get next page token
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                
                time.sleep(0.3)
                
            except Exception as e:
                logger.debug(f"Error collecting condition '{condition}' trials: {e}")
                break
        
        return trials
    
    def extract_trial_data(self, study: Dict) -> Optional[Dict]:
        """Extract comprehensive trial data from API response"""
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
            primary_phase = phases[0] if phases else None
            
            # Interventions (drugs)
            arms_interventions = protocol.get("armsInterventionsModule", {})
            interventions = arms_interventions.get("interventions", [])
            
            # Extract drug information
            primary_drug = None
            all_drug_names = []
            
            for intervention in interventions:
                if intervention.get("type") == "DRUG":
                    drug_name = intervention.get("name", "").strip()
                    if drug_name:
                        if not primary_drug:
                            primary_drug = drug_name
                        all_drug_names.append(drug_name)
                        
                        # Also check other names
                        other_names = intervention.get("otherNames", [])
                        all_drug_names.extend(other_names)
            
            # Skip if no drug interventions
            if not primary_drug:
                return None
            
            # Remove duplicates from drug names
            all_drug_names = list(set(all_drug_names))
            
            # Conditions
            conditions_module = protocol.get("conditionsModule", {})
            conditions = conditions_module.get("conditions", [])
            primary_condition = conditions[0] if conditions else None
            
            # Status
            status_module = protocol.get("statusModule", {})
            overall_status = status_module.get("overallStatus")
            start_date = status_module.get("startDateStruct", {}).get("date")
            completion_date = status_module.get("completionDateStruct", {}).get("date")
            
            # Eligibility
            eligibility_module = protocol.get("eligibilityModule", {})
            min_age = eligibility_module.get("minimumAge")
            max_age = eligibility_module.get("maximumAge")
            sex = eligibility_module.get("sex")
            healthy_volunteers = eligibility_module.get("healthyVolunteers")
            
            # Sponsor
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            lead_sponsor = sponsor_module.get("leadSponsor", {}).get("name")
            sponsor_class = sponsor_module.get("leadSponsor", {}).get("class")
            
            # Outcomes
            outcomes_module = protocol.get("outcomesModule", {})
            primary_outcomes = outcomes_module.get("primaryOutcomes", [])
            secondary_outcomes = outcomes_module.get("secondaryOutcomes", [])
            
            # Enrollment
            enrollment_count = design.get("enrollmentInfo", {}).get("count")
            enrollment_type = design.get("enrollmentInfo", {}).get("type")
            
            return {
                "nct_id": nct_id,
                "title": title,
                "primary_drug": primary_drug,
                "all_drug_names": all_drug_names,
                "primary_condition": primary_condition,
                "all_conditions": conditions,
                "phases": phases,
                "primary_phase": primary_phase,
                "study_type": study_type,
                "allocation": design.get("allocation"),
                "intervention_model": design.get("interventionModel"),
                "masking": design.get("maskingInfo", {}).get("masking"),
                "primary_purpose": design.get("primaryPurpose"),
                "enrollment_count": enrollment_count,
                "enrollment_type": enrollment_type,
                "min_age": min_age,
                "max_age": max_age,
                "sex": sex,
                "healthy_volunteers": healthy_volunteers,
                "primary_outcomes": primary_outcomes,
                "secondary_outcomes": secondary_outcomes,
                "start_date": start_date,
                "completion_date": completion_date,
                "overall_status": overall_status,
                "lead_sponsor": lead_sponsor,
                "sponsor_class": sponsor_class,
                "collected_date": datetime.now().isoformat(),
                "api_version": "v2"
            }
            
        except Exception as e:
            logger.debug(f"Error extracting trial data: {e}")
            return None
    
    def clean_trials_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize trials data"""
        logger.info("🧹 Cleaning trials data...")
        
        initial_count = len(df)
        
        # Remove duplicates by NCT ID
        df = df.drop_duplicates(subset=['nct_id'], keep='first')
        logger.info(f"🧹 Removed {initial_count - len(df)} duplicate NCT IDs")
        
        # Remove trials without drug information
        df = df.dropna(subset=['primary_drug'])
        df = df[df['primary_drug'].str.strip() != '']
        
        # Clean drug names
        df['primary_drug'] = df['primary_drug'].str.strip()
        
        # Standardize phases
        df['primary_phase'] = df['primary_phase'].fillna('UNKNOWN')
        
        # Clean status
        df['overall_status'] = df['overall_status'].fillna('UNKNOWN')
        
        final_count = len(df)
        logger.info(f"✅ Cleaned dataset: {final_count:,} trials with drug interventions")
        
        return df
    
    def analyze_collected_trials(self, df: pd.DataFrame):
        """Analyze the collected trials dataset"""
        logger.info("📊 ANALYZING COLLECTED TRIALS DATASET")
        logger.info("=" * 60)
        
        logger.info(f"📊 Total trials: {len(df):,}")
        
        # Unique drugs analysis
        unique_drugs = df['primary_drug'].nunique()
        total_drug_mentions = df['primary_drug'].count()
        logger.info(f"🧬 Unique drugs: {unique_drugs:,}")
        logger.info(f"🧬 Total drug mentions: {total_drug_mentions:,}")
        
        # Phase analysis
        logger.info("\n📊 Trial phases:")
        phase_counts = df['primary_phase'].value_counts()
        for phase, count in phase_counts.head(10).items():
            logger.info(f"   {phase}: {count:,}")
        
        # Status analysis
        logger.info("\n📊 Trial status:")
        status_counts = df['overall_status'].value_counts()
        for status, count in status_counts.head(10).items():
            logger.info(f"   {status}: {count:,}")
        
        # Study type analysis
        logger.info("\n📊 Study types:")
        type_counts = df['study_type'].value_counts()
        for study_type, count in type_counts.head(10).items():
            logger.info(f"   {study_type}: {count:,}")
        
        # Top drugs
        logger.info("\n🧬 Top 20 drugs by trial count:")
        top_drugs = df['primary_drug'].value_counts().head(20)
        for drug, count in top_drugs.items():
            logger.info(f"   {drug}: {count} trials")
        
        logger.info("=" * 60)
    
    def save_expanded_trials(self, df: pd.DataFrame, output_dir: str = "clinical_trial_dataset/data/expanded"):
        """Save the expanded trials dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main dataset
        trials_file = output_path / "expanded_clinical_trials.csv"
        df.to_csv(trials_file, index=False)
        logger.info(f"💾 Saved expanded trials: {trials_file}")
        
        # Save drug-focused subset
        drug_trials = df[df['primary_drug'].notna()].copy()
        drug_file = output_path / "drug_focused_trials.csv"
        drug_trials.to_csv(drug_file, index=False)
        logger.info(f"💾 Saved drug-focused trials: {drug_file}")
        
        # Save unique drugs list
        unique_drugs = drug_trials['primary_drug'].unique()
        drugs_file = output_path / "unique_trial_drugs.txt"
        with open(drugs_file, 'w') as f:
            for drug in sorted(unique_drugs):
                f.write(f"{drug}\n")
        logger.info(f"💾 Saved unique drugs list: {drugs_file}")
        
        # Save metadata
        metadata = {
            "collection_info": {
                "total_trials": len(df),
                "unique_drugs": len(unique_drugs),
                "collection_date": datetime.now().isoformat(),
                "api_version": "v2",
                "source": "ClinicalTrials.gov API"
            },
            "data_quality": {
                "duplicates_removed": True,
                "drug_interventions_only": True,
                "comprehensive_extraction": True
            },
            "next_steps": [
                "Find SMILES for unique drugs",
                "Integrate with ChEMBL/PubChem",
                "Create comprehensive dataset"
            ]
        }
        
        metadata_file = output_path / "expanded_trials_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Saved metadata: {metadata_file}")
        
        return {
            "expanded_trials": trials_file,
            "drug_focused_trials": drug_file,
            "unique_drugs": drugs_file,
            "metadata": metadata_file
        }

def main():
    """Main execution function"""
    logger.info("🌟 CLINICAL TRIALS DATASET EXPANDER")
    logger.info("🔗 Connecting to ClinicalTrials.gov API")
    logger.info("🎯 Target: 50,000+ real clinical trials")
    logger.info("=" * 70)
    
    # Create expander
    expander = ClinicalTrialsExpander()
    
    # Collect massive trials dataset
    target_trials = 50000
    logger.info(f"🚀 Starting collection of {target_trials:,} clinical trials...")
    
    df = expander.collect_massive_trials_dataset(target_trials)
    
    if df.empty:
        logger.error("❌ Failed to collect trials")
        return None
    
    # Analyze collected data
    expander.analyze_collected_trials(df)
    
    # Save expanded dataset
    files = expander.save_expanded_trials(df)
    
    # Final summary
    unique_drugs = df['primary_drug'].nunique()
    
    logger.info("\n" + "=" * 70)
    logger.info("🎉 CLINICAL TRIALS EXPANSION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"📊 Total trials collected: {len(df):,}")
    logger.info(f"🧬 Unique drugs identified: {unique_drugs:,}")
    logger.info(f"📈 Expansion factor: {len(df)/943:.1f}x larger than original")
    logger.info(f"📁 Files created:")
    for name, path in files.items():
        logger.info(f"   - {name}: {path}")
    
    logger.info(f"\n🎯 NEXT STEPS:")
    logger.info(f"   1. Find SMILES for {unique_drugs:,} unique drugs")
    logger.info(f"   2. Expand ChEMBL dataset to 50,000+ compounds")
    logger.info(f"   3. Expand PubChem dataset to 20,000+ compounds") 
    logger.info(f"   4. Integrate all expanded datasets together")
    logger.info(f"   5. Create massive comprehensive pharmaceutical dataset")
    
    logger.info("\n🚀 READY FOR SMILES MATCHING AND INTEGRATION!")
    
    return files

if __name__ == "__main__":
    main()