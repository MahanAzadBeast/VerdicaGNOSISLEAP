#!/usr/bin/env python3
"""
Safety & Failed Trials Collector
Specifically targets clinical trials with:
1. Adverse events and side effects data
2. Failed, terminated, and withdrawn trials
3. Safety outcomes and toxicity information
4. Drug discontinuation reasons
5. Black box warnings and safety alerts
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SafetyAndFailedTrialsCollector:
    """Collects clinical trials focused on safety, adverse events, and failures"""
    
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.safety_trials = []
        self.failed_trials = []
        
    def collect_failed_and_terminated_trials(self, target: int = 20000) -> List[Dict]:
        """Collect trials that failed, were terminated, or withdrawn"""
        logger.info(f"💀 Collecting FAILED and TERMINATED trials (target: {target:,})")
        
        failed_statuses = [
            "TERMINATED",
            "WITHDRAWN", 
            "SUSPENDED",
            "WITHHELD"
        ]
        
        all_failed_trials = []
        
        for status in failed_statuses:
            if len(all_failed_trials) >= target:
                break
                
            status_target = min(target // len(failed_statuses), target - len(all_failed_trials))
            status_trials = self.collect_trials_by_status(status, status_target)
            all_failed_trials.extend(status_trials)
            
            logger.info(f"💀 {status}: {len(status_trials)} trials (Total: {len(all_failed_trials):,})")
        
        logger.info(f"🎉 Failed trials collection complete: {len(all_failed_trials):,}")
        return all_failed_trials
    
    def collect_trials_by_status(self, status: str, target: int) -> List[Dict]:
        """Collect trials by specific status"""
        logger.info(f"📊 Collecting {status} trials...")
        
        trials = []
        page_token = None
        page_size = 1000
        
        while len(trials) < target:
            try:
                params = {
                    "format": "json",
                    "pageSize": min(page_size, target - len(trials)),
                    "query.overallStatus": status
                }
                
                if page_token:
                    params["pageToken"] = page_token
                
                response = requests.get(self.base_url, params=params, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                studies = data.get("studies", [])
                
                if not studies:
                    break
                
                # Process studies with enhanced safety extraction
                batch_trials = []
                for study in studies:
                    trial_data = self.extract_safety_focused_trial_data(study)
                    if trial_data:
                        batch_trials.append(trial_data)
                
                trials.extend(batch_trials)
                logger.info(f"✅ {status}: {len(batch_trials)} trials with safety data (Total: {len(trials):,})")
                
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Error collecting {status} trials: {e}")
                time.sleep(2)
                continue
        
        return trials
    
    def collect_adverse_events_trials(self, target: int = 15000) -> List[Dict]:
        """Collect trials specifically mentioning adverse events"""
        logger.info(f"⚠️ Collecting ADVERSE EVENTS trials (target: {target:,})")
        
        adverse_event_terms = [
            "adverse events",
            "adverse reactions", 
            "side effects",
            "toxicity",
            "safety",
            "tolerability",
            "dose limiting toxicity",
            "maximum tolerated dose",
            "drug related adverse events",
            "serious adverse events",
            "treatment emergent adverse events"
        ]
        
        all_ae_trials = []
        trials_per_term = target // len(adverse_event_terms)
        
        for term in adverse_event_terms:
            if len(all_ae_trials) >= target:
                break
                
            try:
                term_trials = self.collect_trials_by_keyword(term, trials_per_term)
                all_ae_trials.extend(term_trials)
                logger.info(f"⚠️ '{term}': {len(term_trials)} trials (Total: {len(all_ae_trials):,})")
                
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Error collecting AE trials for '{term}': {e}")
                continue
        
        # Remove duplicates by NCT ID
        seen_nct_ids = set()
        unique_trials = []
        for trial in all_ae_trials:
            nct_id = trial.get('nct_id')
            if nct_id and nct_id not in seen_nct_ids:
                seen_nct_ids.add(nct_id)
                unique_trials.append(trial)
        
        logger.info(f"🎉 Adverse events trials: {len(unique_trials):,} unique trials")
        return unique_trials
    
    def collect_trials_by_keyword(self, keyword: str, target: int) -> List[Dict]:
        """Collect trials by keyword search"""
        trials = []
        page_token = None
        page_size = 500
        
        while len(trials) < target:
            try:
                params = {
                    "format": "json", 
                    "pageSize": min(page_size, target - len(trials)),
                    "query.term": keyword
                }
                
                if page_token:
                    params["pageToken"] = page_token
                
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                studies = data.get("studies", [])
                
                if not studies:
                    break
                
                batch_trials = []
                for study in studies:
                    trial_data = self.extract_safety_focused_trial_data(study)
                    if trial_data:
                        batch_trials.append(trial_data)
                
                trials.extend(batch_trials)
                
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                
                time.sleep(0.3)
                
            except Exception as e:
                logger.debug(f"Error collecting keyword '{keyword}' trials: {e}")
                break
        
        return trials
    
    def collect_safety_focused_phase1_trials(self, target: int = 10000) -> List[Dict]:
        """Collect Phase 1 trials (safety/dose-finding studies)"""
        logger.info(f"🔬 Collecting PHASE 1 safety trials (target: {target:,})")
        
        trials = []
        page_token = None
        page_size = 1000
        
        while len(trials) < target:
            try:
                params = {
                    "format": "json",
                    "pageSize": min(page_size, target - len(trials)),
                    "query.phase": "PHASE1"
                }
                
                if page_token:
                    params["pageToken"] = page_token
                
                response = requests.get(self.base_url, params=params, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                studies = data.get("studies", [])
                
                if not studies:
                    break
                
                batch_trials = []
                for study in studies:
                    trial_data = self.extract_safety_focused_trial_data(study)
                    if trial_data and self.has_safety_focus(trial_data):
                        batch_trials.append(trial_data)
                
                trials.extend(batch_trials)
                logger.info(f"🔬 Phase 1 safety: {len(batch_trials)} trials (Total: {len(trials):,})")
                
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Error collecting Phase 1 trials: {e}")
                time.sleep(2)
                continue
        
        return trials
    
    def has_safety_focus(self, trial_data: Dict) -> bool:
        """Check if trial has safety focus"""
        safety_keywords = [
            "safety", "tolerability", "adverse", "toxicity", "dose",
            "maximum tolerated", "dose limiting", "side effect"
        ]
        
        # Check title
        title = trial_data.get('title', '').lower()
        for keyword in safety_keywords:
            if keyword in title:
                return True
        
        # Check primary outcomes
        primary_outcomes = trial_data.get('primary_outcomes', [])
        for outcome in primary_outcomes:
            if isinstance(outcome, dict):
                measure = outcome.get('measure', '').lower()
                for keyword in safety_keywords:
                    if keyword in measure:
                        return True
        
        return False
    
    def extract_safety_focused_trial_data(self, study: Dict) -> Optional[Dict]:
        """Extract comprehensive trial data with focus on safety information"""
        try:
            protocol = study.get("protocolSection", {})
            
            # Basic identification
            identification = protocol.get("identificationModule", {})
            nct_id = identification.get("nctId")
            title = identification.get("briefTitle", "")
            official_title = identification.get("officialTitle", "")
            
            if not nct_id:
                return None
            
            # Study design
            design = protocol.get("designModule", {})
            study_type = design.get("studyType")
            phases = design.get("phases", [])
            primary_phase = phases[0] if phases else None
            
            # Interventions (focus on drugs)
            arms_interventions = protocol.get("armsInterventionsModule", {})
            interventions = arms_interventions.get("interventions", [])
            
            primary_drug = None
            all_drug_names = []
            
            for intervention in interventions:
                if intervention.get("type") == "DRUG":
                    drug_name = intervention.get("name", "").strip()
                    if drug_name:
                        if not primary_drug:
                            primary_drug = drug_name
                        all_drug_names.append(drug_name)
                        
                        # Include other names and descriptions
                        other_names = intervention.get("otherNames", [])
                        all_drug_names.extend(other_names)
            
            # Skip if no drug interventions
            if not primary_drug:
                return None
            
            all_drug_names = list(set(all_drug_names))
            
            # Conditions
            conditions_module = protocol.get("conditionsModule", {})
            conditions = conditions_module.get("conditions", [])
            keywords = conditions_module.get("keywords", [])
            
            # Status and dates
            status_module = protocol.get("statusModule", {})
            overall_status = status_module.get("overallStatus")
            why_stopped = status_module.get("whyStopped", "")  # CRITICAL for failed trials
            start_date = status_module.get("startDateStruct", {}).get("date")
            completion_date = status_module.get("completionDateStruct", {}).get("date")
            
            # Outcomes (CRITICAL for safety data)
            outcomes_module = protocol.get("outcomesModule", {})
            primary_outcomes = outcomes_module.get("primaryOutcomes", [])
            secondary_outcomes = outcomes_module.get("secondaryOutcomes", [])
            
            # Extract safety-specific outcomes
            safety_outcomes = []
            all_outcomes = primary_outcomes + secondary_outcomes
            
            for outcome in all_outcomes:
                if isinstance(outcome, dict):
                    measure = outcome.get("measure", "").lower()
                    description = outcome.get("description", "").lower()
                    
                    # Check if outcome is safety-related
                    safety_terms = ["safety", "adverse", "toxicity", "tolerability", "side effect"]
                    if any(term in measure or term in description for term in safety_terms):
                        safety_outcomes.append(outcome)
            
            # Eligibility
            eligibility_module = protocol.get("eligibilityModule", {})
            
            # Sponsor information
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            lead_sponsor = sponsor_module.get("leadSponsor", {}).get("name")
            sponsor_class = sponsor_module.get("leadSponsor", {}).get("class")
            
            # Enhanced data extraction for safety focus
            return {
                "nct_id": nct_id,
                "title": title,
                "official_title": official_title,
                "primary_drug": primary_drug,
                "all_drug_names": all_drug_names,
                "primary_condition": conditions[0] if conditions else None,
                "all_conditions": conditions,
                "keywords": keywords,
                "phases": phases,
                "primary_phase": primary_phase,
                "study_type": study_type,
                "allocation": design.get("allocation"),
                "intervention_model": design.get("interventionModel"),
                "masking": design.get("maskingInfo", {}).get("masking"),
                "primary_purpose": design.get("primaryPurpose"),
                "enrollment_count": design.get("enrollmentInfo", {}).get("count"),
                "enrollment_type": design.get("enrollmentInfo", {}).get("type"),
                "min_age": eligibility_module.get("minimumAge"),
                "max_age": eligibility_module.get("maximumAge"),
                "sex": eligibility_module.get("sex"),
                "healthy_volunteers": eligibility_module.get("healthyVolunteers"),
                
                # SAFETY-FOCUSED FIELDS
                "overall_status": overall_status,
                "why_stopped": why_stopped,  # Critical for understanding failures
                "primary_outcomes": primary_outcomes,
                "secondary_outcomes": secondary_outcomes,
                "safety_outcomes": safety_outcomes,  # Extracted safety-specific outcomes
                "is_safety_focused": len(safety_outcomes) > 0,
                "is_failed_trial": overall_status in ["TERMINATED", "WITHDRAWN", "SUSPENDED", "WITHHELD"],
                "failure_reason": why_stopped if overall_status in ["TERMINATED", "WITHDRAWN", "SUSPENDED"] else None,
                
                "start_date": start_date,
                "completion_date": completion_date,
                "lead_sponsor": lead_sponsor,
                "sponsor_class": sponsor_class,
                "collected_date": datetime.now().isoformat(),
                "api_version": "v2_safety_focused",
                
                # Safety classification
                "safety_classification": self.classify_safety_trial(title, primary_outcomes, secondary_outcomes, overall_status)
            }
            
        except Exception as e:
            logger.debug(f"Error extracting safety trial data: {e}")
            return None
    
    def classify_safety_trial(self, title: str, primary_outcomes: List, secondary_outcomes: List, status: str) -> str:
        """Classify the type of safety trial"""
        title_lower = title.lower()
        
        # Failed/terminated trials
        if status in ["TERMINATED", "WITHDRAWN", "SUSPENDED", "WITHHELD"]:
            return "FAILED_TRIAL"
        
        # Dose-finding studies
        if any(term in title_lower for term in ["dose", "maximum tolerated", "mtd", "dose escalation"]):
            return "DOSE_FINDING"
        
        # Safety/tolerability studies
        if any(term in title_lower for term in ["safety", "tolerability", "adverse"]):
            return "SAFETY_STUDY"
        
        # Toxicity studies
        if any(term in title_lower for term in ["toxicity", "toxic", "dlt"]):
            return "TOXICITY_STUDY"
        
        # Check outcomes for safety focus
        all_outcomes = primary_outcomes + secondary_outcomes
        for outcome in all_outcomes:
            if isinstance(outcome, dict):
                measure = outcome.get("measure", "").lower()
                if any(term in measure for term in ["safety", "adverse", "tolerability", "toxicity"]):
                    return "SAFETY_OUTCOME"
        
        return "OTHER"
    
    def collect_comprehensive_safety_dataset(self, target_total: int = 50000) -> pd.DataFrame:
        """Collect comprehensive safety and failed trials dataset"""
        logger.info(f"🚀 Collecting COMPREHENSIVE SAFETY DATASET (target: {target_total:,})")
        logger.info("🎯 Focus: Failed trials, adverse events, safety outcomes")
        logger.info("=" * 80)
        
        all_safety_trials = []
        
        # 1. Collect failed and terminated trials (40% of target)
        failed_target = int(target_total * 0.4)
        logger.info(f"\n📊 STEP 1: Failed/Terminated Trials (target: {failed_target:,})")
        failed_trials = self.collect_failed_and_terminated_trials(failed_target)
        all_safety_trials.extend(failed_trials)
        
        # 2. Collect adverse events trials (40% of target)
        ae_target = int(target_total * 0.4)
        logger.info(f"\n📊 STEP 2: Adverse Events Trials (target: {ae_target:,})")
        ae_trials = self.collect_adverse_events_trials(ae_target)
        all_safety_trials.extend(ae_trials)
        
        # 3. Collect safety-focused Phase 1 trials (20% of target)
        phase1_target = int(target_total * 0.2)
        logger.info(f"\n📊 STEP 3: Safety Phase 1 Trials (target: {phase1_target:,})")
        phase1_trials = self.collect_safety_focused_phase1_trials(phase1_target)
        all_safety_trials.extend(phase1_trials)
        
        logger.info(f"\n📊 Total safety trials collected: {len(all_safety_trials):,}")
        
        if not all_safety_trials:
            logger.error("❌ No safety trials collected!")
            return pd.DataFrame()
        
        # Convert to DataFrame and clean
        df = pd.DataFrame(all_safety_trials)
        df = self.clean_safety_trials_data(df)
        
        return df
    
    def clean_safety_trials_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and enhance safety trials data"""
        logger.info("🧹 Cleaning safety trials data...")
        
        initial_count = len(df)
        
        # Remove duplicates by NCT ID
        df = df.drop_duplicates(subset=['nct_id'], keep='first')
        logger.info(f"🧹 Removed {initial_count - len(df)} duplicate NCT IDs")
        
        # Ensure we have drug information
        df = df.dropna(subset=['primary_drug'])
        df = df[df['primary_drug'].str.strip() != '']
        
        # Clean and standardize
        df['primary_drug'] = df['primary_drug'].str.strip()
        df['overall_status'] = df['overall_status'].fillna('UNKNOWN')
        df['why_stopped'] = df['why_stopped'].fillna('')
        
        # Add safety analysis flags
        df['has_safety_outcomes'] = df['safety_outcomes'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        df['has_failure_reason'] = df['why_stopped'].str.len() > 0
        
        final_count = len(df)
        logger.info(f"✅ Cleaned safety dataset: {final_count:,} trials")
        
        return df
    
    def analyze_safety_dataset(self, df: pd.DataFrame):
        """Analyze the safety and failed trials dataset"""
        logger.info("\n📊 SAFETY & FAILED TRIALS ANALYSIS")
        logger.info("=" * 80)
        
        logger.info(f"📊 Total safety-focused trials: {len(df):,}")
        
        # Failure analysis
        failed_trials = df[df['is_failed_trial'] == True]
        logger.info(f"💀 Failed/terminated trials: {len(failed_trials):,}")
        
        # Safety classification
        logger.info("\n🔍 Safety trial classification:")
        safety_counts = df['safety_classification'].value_counts()
        for classification, count in safety_counts.items():
            logger.info(f"   {classification}: {count:,}")
        
        # Status breakdown
        logger.info("\n📊 Trial status breakdown:")
        status_counts = df['overall_status'].value_counts()
        for status, count in status_counts.head(10).items():
            logger.info(f"   {status}: {count:,}")
        
        # Failure reasons
        logger.info("\n💀 Top failure reasons:")
        failure_reasons = df[df['has_failure_reason']]['why_stopped'].value_counts()
        for reason, count in failure_reasons.head(15).items():
            if reason.strip():
                logger.info(f"   {reason}: {count}")
        
        # Unique drugs in safety trials
        unique_drugs = df['primary_drug'].nunique()
        logger.info(f"\n🧬 Unique drugs in safety trials: {unique_drugs:,}")
        
        # Safety outcomes
        safety_outcome_trials = df[df['has_safety_outcomes']].shape[0]
        logger.info(f"⚠️ Trials with safety outcomes: {safety_outcome_trials:,}")
        
        logger.info("=" * 80)
    
    def save_safety_dataset(self, df: pd.DataFrame, output_dir: str = "clinical_trial_dataset/data/safety"):
        """Save the safety and failed trials dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete safety dataset
        safety_file = output_path / "safety_and_failed_trials.csv"
        df.to_csv(safety_file, index=False)
        logger.info(f"💾 Saved safety dataset: {safety_file}")
        
        # Save failed trials subset
        failed_trials = df[df['is_failed_trial'] == True]
        failed_file = output_path / "failed_trials_only.csv"
        failed_trials.to_csv(failed_file, index=False)
        logger.info(f"💾 Saved failed trials: {failed_file}")
        
        # Save adverse events subset
        ae_trials = df[df['safety_classification'].isin(['SAFETY_STUDY', 'SAFETY_OUTCOME', 'TOXICITY_STUDY'])]
        ae_file = output_path / "adverse_events_trials.csv"
        ae_trials.to_csv(ae_file, index=False)
        logger.info(f"💾 Saved adverse events trials: {ae_file}")
        
        # Save unique drugs with safety data
        safety_drugs = df[['primary_drug', 'safety_classification', 'is_failed_trial', 'why_stopped']].drop_duplicates()
        drugs_file = output_path / "drugs_with_safety_data.csv"
        safety_drugs.to_csv(drugs_file, index=False)
        logger.info(f"💾 Saved drugs with safety data: {drugs_file}")
        
        # Save failure reasons analysis
        failure_analysis = df[df['has_failure_reason']]['why_stopped'].value_counts().reset_index()
        failure_analysis.columns = ['failure_reason', 'count']
        failure_file = output_path / "failure_reasons_analysis.csv"
        failure_analysis.to_csv(failure_file, index=False)
        logger.info(f"💾 Saved failure analysis: {failure_file}")
        
        # Save comprehensive metadata
        failed_count = len(df[df['is_failed_trial'] == True])
        safety_outcome_count = len(df[df['has_safety_outcomes']])
        
        metadata = {
            "dataset_info": {
                "total_safety_trials": len(df),
                "failed_trials": failed_count,
                "trials_with_safety_outcomes": safety_outcome_count,
                "unique_drugs": df['primary_drug'].nunique(),
                "collection_date": datetime.now().isoformat(),
                "focus": "Safety, adverse events, and failed trials"
            },
            "safety_classification": dict(df['safety_classification'].value_counts()),
            "trial_status": dict(df['overall_status'].value_counts()),
            "data_quality": {
                "has_failure_reasons": len(df[df['has_failure_reason']]),
                "has_safety_outcomes": safety_outcome_count,
                "comprehensive_safety_focus": True
            },
            "next_steps": [
                "Find SMILES for safety-relevant drugs",
                "Analyze drug toxicity patterns",
                "Integrate with compound databases",
                "Build safety prediction models"
            ]
        }
        
        metadata_file = output_path / "safety_dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Saved metadata: {metadata_file}")
        
        return {
            "safety_dataset": safety_file,
            "failed_trials": failed_file,
            "adverse_events": ae_file,
            "safety_drugs": drugs_file,
            "failure_analysis": failure_file,
            "metadata": metadata_file
        }

def main():
    """Main execution function"""
    logger.info("🌟 SAFETY & FAILED TRIALS COLLECTOR")
    logger.info("💀 Focus: Failed, terminated, and adverse events trials")
    logger.info("⚠️ Target: Comprehensive drug safety data")
    logger.info("=" * 80)
    
    # Create safety collector
    collector = SafetyAndFailedTrialsCollector()
    
    # Collect comprehensive safety dataset
    target_trials = 50000
    logger.info(f"🚀 Starting collection of {target_trials:,} safety-focused trials...")
    
    df = collector.collect_comprehensive_safety_dataset(target_trials)
    
    if df.empty:
        logger.error("❌ Failed to collect safety trials")
        return None
    
    # Analyze collected data
    collector.analyze_safety_dataset(df)
    
    # Save safety dataset
    files = collector.save_safety_dataset(df)
    
    # Final summary
    failed_trials = len(df[df['is_failed_trial'] == True])
    safety_outcomes = len(df[df['has_safety_outcomes']])
    unique_drugs = df['primary_drug'].nunique()
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 SAFETY & FAILED TRIALS COLLECTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"📊 Total safety trials: {len(df):,}")
    logger.info(f"💀 Failed/terminated trials: {failed_trials:,}")
    logger.info(f"⚠️ Trials with safety outcomes: {safety_outcomes:,}")
    logger.info(f"🧬 Unique drugs with safety data: {unique_drugs:,}")
    logger.info(f"📁 Files created:")
    for name, path in files.items():
        logger.info(f"   - {name}: {path}")
    
    logger.info(f"\n🎯 CRITICAL SAFETY DATA COLLECTED:")
    logger.info(f"   ✅ Drug failure reasons and patterns")
    logger.info(f"   ✅ Adverse events and side effects")
    logger.info(f"   ✅ Dose-limiting toxicities")
    logger.info(f"   ✅ Safety outcomes and tolerability")
    logger.info(f"   ✅ Terminated trial analysis")
    
    logger.info("\n🚀 READY FOR SAFETY-FOCUSED SMILES MATCHING!")
    
    return files

if __name__ == "__main__":
    main()