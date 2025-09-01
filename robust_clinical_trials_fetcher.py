#!/usr/bin/env python3
"""
Robust Clinical Trials Fetcher - Fixed API Implementation
Based on error analysis, creates a robust clinical trials collector that:
1. Uses correct API parameters (no invalid query.* parameters)
2. Implements proper pagination with nextPageToken
3. Filters data after retrieval instead of using broken API filters
4. Has comprehensive error handling and retry logic
"""

import requests
import pandas as pd
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RobustClinicalTrialsFetcher:
    """Robust fetcher for clinical trials with fixed API usage"""
    
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.output_dir = Path("clinical_trial_dataset/data/robust_trials")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Retry settings
        self.max_retries = 5
        self.base_delay = 2
        self.max_delay = 60
        
    def test_api_connectivity(self):
        """Test API with correct parameters"""
        logger.info("🔍 Testing ClinicalTrials.gov API connectivity...")
        
        try:
            # Use only known working parameters
            params = {
                "format": "json",
                "pageSize": 10
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                studies = data.get("studies", [])
                total_count = data.get("totalCount", 0)
                
                logger.info(f"✅ API Working: {len(studies)} studies returned")
                logger.info(f"📊 Total available studies: {total_count:,}")
                return True
            else:
                logger.error(f"❌ API Error: {response.status_code}")
                logger.error(f"Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            logger.error(f"❌ API Connection Error: {e}")
            return False
    
    def robust_fetch_all_studies(self, target_studies: int = 50000) -> List[Dict]:
        """Robustly fetch clinical studies using correct API approach"""
        logger.info(f"🏥 ROBUST CLINICAL TRIALS COLLECTION (target: {target_studies:,})")
        
        if not self.test_api_connectivity():
            logger.error("❌ API not accessible, aborting collection")
            return []
        
        all_studies = []
        page_token = None
        page_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while len(all_studies) < target_studies and consecutive_failures < max_consecutive_failures:
            try:
                page_count += 1
                logger.info(f"📄 Fetching page {page_count} (collected: {len(all_studies):,})")
                
                # Use only working parameters - NO query.* filters
                params = {
                    "format": "json",
                    "pageSize": 1000  # Maximum allowed
                }
                
                if page_token:
                    params["pageToken"] = page_token
                
                # Make request with retries
                response = self._make_robust_request(params)
                
                if not response:
                    consecutive_failures += 1
                    logger.warning(f"❌ Failed to get response (failure {consecutive_failures})")
                    continue
                
                data = response.json()
                studies = data.get("studies", [])
                
                if not studies:
                    logger.info("✅ No more studies available - collection complete")
                    break
                
                # Process studies and filter for drug interventions
                drug_studies = []
                for study in studies:
                    processed_study = self._extract_study_data(study)
                    if processed_study and self._has_drug_interventions(processed_study):
                        drug_studies.append(processed_study)
                
                all_studies.extend(drug_studies)
                consecutive_failures = 0  # Reset failure counter
                
                logger.info(f"✅ Page {page_count}: {len(drug_studies)} drug studies (Total: {len(all_studies):,})")
                
                # Save incremental progress
                if len(all_studies) % 5000 == 0:
                    self._save_incremental_progress(all_studies)
                
                # Get next page token
                page_token = data.get("nextPageToken")
                if not page_token:
                    logger.info("✅ Reached end of available studies")
                    break
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"❌ Page {page_count} error (failure {consecutive_failures}): {e}")
                
                if consecutive_failures < max_consecutive_failures:
                    wait_time = min(self.max_delay, self.base_delay * (2 ** consecutive_failures))
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error("❌ Max consecutive failures reached, stopping collection")
                    break
        
        logger.info(f"🎉 Clinical trials collection complete: {len(all_studies):,} studies with drug interventions")
        return all_studies
    
    def _make_robust_request(self, params: Dict) -> Optional[requests.Response]:
        """Make robust HTTP request with retries"""
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limited
                    wait_time = min(self.max_delay, self.base_delay * (2 ** attempt))
                    logger.warning(f"⚠️ Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                elif response.status_code == 400:
                    logger.error(f"❌ Bad Request (400): {response.text[:200]}")
                    return None  # Don't retry bad requests
                else:
                    logger.warning(f"⚠️ HTTP {response.status_code} (attempt {attempt + 1})")
                    time.sleep(self.base_delay * (attempt + 1))
                    
            except requests.exceptions.Timeout:
                logger.warning(f"⚠️ Timeout (attempt {attempt + 1})")
                time.sleep(self.base_delay * (attempt + 1))
            except requests.exceptions.ConnectionError:
                logger.warning(f"⚠️ Connection error (attempt {attempt + 1})")
                time.sleep(self.base_delay * (attempt + 1))
            except Exception as e:
                logger.warning(f"⚠️ Request error (attempt {attempt + 1}): {e}")
                time.sleep(self.base_delay * (attempt + 1))
        
        logger.error(f"❌ All {self.max_retries} attempts failed")
        return None
    
    def _extract_study_data(self, study: Dict) -> Optional[Dict]:
        """Extract comprehensive study data"""
        try:
            protocol = study.get("protocolSection", {})
            
            # Basic identification
            identification = protocol.get("identificationModule", {})
            nct_id = identification.get("nctId")
            
            if not nct_id:
                return None
            
            title = identification.get("briefTitle", "")
            official_title = identification.get("officialTitle", "")
            
            # Study design
            design = protocol.get("designModule", {})
            study_type = design.get("studyType")
            phases = design.get("phases", [])
            primary_purpose = design.get("primaryPurpose")
            
            # Interventions
            arms_interventions = protocol.get("armsInterventionsModule", {})
            interventions = arms_interventions.get("interventions", [])
            
            # Extract all drug interventions
            all_drugs = []
            primary_drug = None
            
            for intervention in interventions:
                if intervention.get("type") == "DRUG":
                    drug_name = intervention.get("name", "").strip()
                    if drug_name:
                        if not primary_drug:
                            primary_drug = drug_name
                        all_drugs.append(drug_name)
                        
                        # Include other names
                        other_names = intervention.get("otherNames", [])
                        all_drugs.extend([name.strip() for name in other_names if name.strip()])
            
            # Remove duplicates
            all_drugs = list(set(all_drugs))
            
            # Conditions
            conditions_module = protocol.get("conditionsModule", {})
            conditions = conditions_module.get("conditions", [])
            keywords = conditions_module.get("keywords", [])
            
            # Status
            status_module = protocol.get("statusModule", {})
            overall_status = status_module.get("overallStatus")
            why_stopped = status_module.get("whyStopped", "")
            start_date = status_module.get("startDateStruct", {}).get("date")
            completion_date = status_module.get("completionDateStruct", {}).get("date")
            
            # Outcomes
            outcomes_module = protocol.get("outcomesModule", {})
            primary_outcomes = outcomes_module.get("primaryOutcomes", [])
            secondary_outcomes = outcomes_module.get("secondaryOutcomes", [])
            
            # Eligibility
            eligibility_module = protocol.get("eligibilityModule", {})
            
            # Sponsor
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            lead_sponsor = sponsor_module.get("leadSponsor", {}).get("name")
            sponsor_class = sponsor_module.get("leadSponsor", {}).get("class")
            collaborators = sponsor_module.get("collaborators", [])
            
            # Enrollment
            enrollment_count = design.get("enrollmentInfo", {}).get("count")
            enrollment_type = design.get("enrollmentInfo", {}).get("type")
            
            return {
                # Identifiers
                "nct_id": nct_id,
                "title": title,
                "official_title": official_title,
                
                # Drugs
                "primary_drug": primary_drug,
                "all_drug_names": all_drugs,
                "drug_count": len(all_drugs),
                
                # Study design
                "study_type": study_type,
                "phases": phases,
                "primary_phase": phases[0] if phases else None,
                "primary_purpose": primary_purpose,
                
                # Conditions
                "primary_condition": conditions[0] if conditions else None,
                "all_conditions": conditions,
                "keywords": keywords,
                
                # Status
                "overall_status": overall_status,
                "why_stopped": why_stopped,
                "start_date": start_date,
                "completion_date": completion_date,
                
                # Outcomes
                "primary_outcomes": primary_outcomes,
                "secondary_outcomes": secondary_outcomes,
                "outcome_count": len(primary_outcomes) + len(secondary_outcomes),
                
                # Eligibility
                "min_age": eligibility_module.get("minimumAge"),
                "max_age": eligibility_module.get("maximumAge"),
                "sex": eligibility_module.get("sex"),
                "healthy_volunteers": eligibility_module.get("healthyVolunteers"),
                
                # Sponsor
                "lead_sponsor": lead_sponsor,
                "sponsor_class": sponsor_class,
                "collaborator_count": len(collaborators),
                
                # Enrollment
                "enrollment_count": enrollment_count,
                "enrollment_type": enrollment_type,
                
                # Collection metadata
                "collected_date": datetime.now().isoformat(),
                "api_version": "v2_robust",
                
                # Derived flags
                "is_interventional": study_type == "INTERVENTIONAL",
                "has_drug_interventions": len(all_drugs) > 0,
                "is_completed": overall_status == "COMPLETED",
                "is_failed": overall_status in ["TERMINATED", "WITHDRAWN", "SUSPENDED", "WITHHELD"],
                "has_safety_outcomes": any("safety" in str(outcome).lower() or "adverse" in str(outcome).lower() 
                                         for outcome in primary_outcomes + secondary_outcomes),
                "failure_reason": why_stopped if overall_status in ["TERMINATED", "WITHDRAWN", "SUSPENDED"] else None
            }
            
        except Exception as e:
            logger.debug(f"Error extracting study data: {e}")
            return None
    
    def _has_drug_interventions(self, study: Dict) -> bool:
        """Check if study has drug interventions"""
        return study.get("has_drug_interventions", False) and study.get("drug_count", 0) > 0
    
    def _save_incremental_progress(self, studies: List[Dict]):
        """Save incremental progress"""
        if not studies:
            return
        
        incremental_file = self.output_dir / "trials_incremental.csv"
        df = pd.DataFrame(studies)
        df.to_csv(incremental_file, index=False)
        
        logger.info(f"💾 Saved incremental progress: {len(studies):,} studies")
    
    def analyze_collected_data(self, studies: List[Dict]):
        """Analyze collected clinical trials data"""
        if not studies:
            logger.warning("❌ No studies to analyze")
            return
        
        logger.info("📊 CLINICAL TRIALS DATA ANALYSIS")
        logger.info("=" * 50)
        
        df = pd.DataFrame(studies)
        
        # Basic stats
        logger.info(f"📋 Total studies: {len(df):,}")
        logger.info(f"🧬 Total unique drugs: {df['primary_drug'].nunique():,}")
        logger.info(f"💊 Studies with drugs: {df['has_drug_interventions'].sum():,}")
        
        # Study types
        logger.info("\n📊 Study types:")
        study_types = df['study_type'].value_counts()
        for study_type, count in study_types.head().items():
            logger.info(f"   {study_type}: {count:,}")
        
        # Phases
        logger.info("\n🔬 Clinical phases:")
        phase_counts = df['primary_phase'].value_counts()
        for phase, count in phase_counts.head().items():
            logger.info(f"   {phase}: {count:,}")
        
        # Status
        logger.info("\n📈 Trial status:")
        status_counts = df['overall_status'].value_counts()
        for status, count in status_counts.head().items():
            logger.info(f"   {status}: {count:,}")
        
        # Failed trials
        failed_trials = df[df['is_failed']].shape[0]
        logger.info(f"\n💀 Failed trials: {failed_trials:,}")
        
        # Safety trials
        safety_trials = df[df['has_safety_outcomes']].shape[0]
        logger.info(f"⚠️ Safety-focused trials: {safety_trials:,}")
        
        # Top drugs
        logger.info("\n🧬 Top 10 drugs by trial count:")
        top_drugs = df['primary_drug'].value_counts().head(10)
        for drug, count in top_drugs.items():
            logger.info(f"   {drug}: {count} trials")
    
    def save_robust_dataset(self, studies: List[Dict]) -> Dict[str, Path]:
        """Save the robust clinical trials dataset"""
        if not studies:
            logger.error("❌ No studies to save")
            return {}
        
        logger.info("💾 SAVING ROBUST CLINICAL TRIALS DATASET")
        
        df = pd.DataFrame(studies)
        
        # Save complete dataset
        complete_file = self.output_dir / "complete_robust_clinical_trials.csv"
        df.to_csv(complete_file, index=False)
        logger.info(f"💾 Saved complete dataset: {complete_file} ({len(df):,} studies)")
        
        # Save drug-focused subset
        drug_studies = df[df['has_drug_interventions']].copy()
        drug_file = self.output_dir / "drug_intervention_trials.csv"
        drug_studies.to_csv(drug_file, index=False)
        logger.info(f"💾 Saved drug trials: {drug_file} ({len(drug_studies):,} studies)")
        
        # Save failed trials subset
        failed_studies = df[df['is_failed']].copy()
        if len(failed_studies) > 0:
            failed_file = self.output_dir / "failed_trials.csv"
            failed_studies.to_csv(failed_file, index=False)
            logger.info(f"💾 Saved failed trials: {failed_file} ({len(failed_studies):,} studies)")
        
        # Save safety trials subset
        safety_studies = df[df['has_safety_outcomes']].copy()
        if len(safety_studies) > 0:
            safety_file = self.output_dir / "safety_focused_trials.csv"
            safety_studies.to_csv(safety_file, index=False)
            logger.info(f"💾 Saved safety trials: {safety_file} ({len(safety_studies):,} studies)")
        
        # Extract unique drugs for SMILES matching
        unique_drugs = df[df['has_drug_interventions']]['primary_drug'].unique()
        drugs_file = self.output_dir / "unique_trial_drugs.txt"
        with open(drugs_file, 'w') as f:
            for drug in sorted(unique_drugs):
                f.write(f"{drug}\n")
        logger.info(f"💾 Saved unique drugs: {drugs_file} ({len(unique_drugs):,} drugs)")
        
        # Save comprehensive metadata
        metadata = {
            "dataset_info": {
                "total_studies": len(df),
                "drug_intervention_studies": len(drug_studies),
                "failed_studies": len(failed_studies),
                "safety_studies": len(safety_studies),
                "unique_drugs": len(unique_drugs),
                "collection_date": datetime.now().isoformat(),
                "api_version": "v2_robust_fixed"
            },
            "data_quality": {
                "api_errors_fixed": True,
                "robust_pagination": True,
                "comprehensive_extraction": True,
                "no_synthetic_data": True
            },
            "collection_parameters": {
                "base_url": self.base_url,
                "max_retries": self.max_retries,
                "rate_limiting": True,
                "incremental_saving": True
            }
        }
        
        metadata_file = self.output_dir / "robust_trials_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Saved metadata: {metadata_file}")
        
        return {
            "complete_dataset": complete_file,
            "drug_trials": drug_file,
            "failed_trials": failed_file if len(failed_studies) > 0 else None,
            "safety_trials": safety_file if len(safety_studies) > 0 else None,
            "unique_drugs": drugs_file,
            "metadata": metadata_file
        }

def main():
    """Main execution function"""
    logger.info("🌟 ROBUST CLINICAL TRIALS FETCHER")
    logger.info("🔧 Fixed API implementation with proper error handling")
    logger.info("🎯 Target: 50,000+ real clinical trials with drug interventions")
    logger.info("=" * 80)
    
    # Create robust fetcher
    fetcher = RobustClinicalTrialsFetcher()
    
    # Fetch clinical trials
    target_studies = 50000
    logger.info(f"🚀 Starting robust collection of {target_studies:,} clinical trials...")
    
    studies = fetcher.robust_fetch_all_studies(target_studies)
    
    if not studies:
        logger.error("❌ Failed to collect clinical trials")
        return None
    
    # Analyze collected data
    fetcher.analyze_collected_data(studies)
    
    # Save dataset
    files = fetcher.save_robust_dataset(studies)
    
    # Final summary
    drug_studies = len([s for s in studies if s.get("has_drug_interventions", False)])
    failed_studies = len([s for s in studies if s.get("is_failed", False)])
    unique_drugs = len(set(s.get("primary_drug") for s in studies if s.get("primary_drug")))
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 ROBUST CLINICAL TRIALS COLLECTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"📊 Total studies: {len(studies):,}")
    logger.info(f"💊 Drug intervention studies: {drug_studies:,}")
    logger.info(f"💀 Failed studies: {failed_studies:,}")
    logger.info(f"🧬 Unique drugs: {unique_drugs:,}")
    logger.info(f"🔧 API errors: FIXED")
    logger.info(f"📁 Files created:")
    for name, path in files.items():
        if path:
            logger.info(f"   - {name}: {path}")
    
    logger.info("\n✅ ROBUST CLINICAL TRIALS DATASET READY!")
    
    return files

if __name__ == "__main__":
    main()