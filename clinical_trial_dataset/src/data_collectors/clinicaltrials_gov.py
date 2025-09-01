"""
ClinicalTrials.gov API Data Collector
Fetches clinical trial data from the public API
"""

import requests
import pandas as pd
import time
import logging
from typing import List, Dict, Optional
from tqdm import tqdm
from retrying import retry
import config

class ClinicalTrialsCollector:
    def __init__(self):
        self.base_url = config.CLINICALTRIALS_BASE_URL
        self.logger = logging.getLogger(__name__)
        self.collected_trials = []
        
    @retry(wait_fixed=2000, stop_max_attempt_number=3)
    def fetch_trials_batch(self, page_token: str = None) -> Dict:
        """Fetch a batch of clinical trials with retry logic"""
        # Basic parameters that work
        params = {
            "format": "json",
            "pageSize": min(config.BATCH_SIZE, 100),  # Start with smaller batch
        }
        
        # Only add pageToken if we have one (for pagination)
        if page_token:
            params["pageToken"] = page_token
        
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def extract_trial_record(self, study: Dict) -> Optional[Dict]:
        """Extract relevant information from a single clinical trial"""
        try:
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            design = protocol.get("designModule", {})
            arms = protocol.get("armsInterventionsModule", {})
            eligibility = protocol.get("eligibilityModule", {})
            outcomes = protocol.get("outcomesModule", {})
            status = protocol.get("statusModule", {})
            sponsors = protocol.get("sponsorCollaboratorsModule", {})
            
            # Extract drug interventions
            interventions = arms.get("interventions", [])
            drug_interventions = [i for i in interventions 
                                if i.get("type", "").upper() == "DRUG"]
            
            # If no drug interventions, skip this study
            if not drug_interventions:
                return None
                
            primary_drug = drug_interventions[0].get("name", "").strip()
            if not primary_drug:
                return None
            
            # Extract conditions - use empty list if none
            conditions = identification.get("conditions", [])
            
            # Extract phases - use empty list if none  
            phases = design.get("phases", [])
            
            record = {
                # Identifiers
                "nct_id": identification.get("nctId", ""),
                "title": identification.get("briefTitle", ""),
                
                # Drug information
                "primary_drug": primary_drug,
                "all_drug_names": [d.get("name", "") for d in drug_interventions],
                
                # Disease/Condition
                "primary_condition": conditions[0] if conditions else "",
                "all_conditions": conditions,
                
                # Trial design
                "phases": phases,
                "primary_phase": phases[0] if phases else "",
                "study_type": design.get("studyType", ""),
                "allocation": design.get("designInfo", {}).get("allocation", ""),
                "intervention_model": design.get("designInfo", {}).get("interventionModel", ""),
                "masking": design.get("designInfo", {}).get("masking", {}).get("description", ""),
                "primary_purpose": design.get("designInfo", {}).get("primaryPurpose", ""),
                
                # Enrollment
                "enrollment_count": design.get("enrollmentInfo", {}).get("count", 0),
                "enrollment_type": design.get("enrollmentInfo", {}).get("type", ""),
                
                # Eligibility
                "eligibility_criteria": eligibility.get("eligibilityCriteria", ""),
                "min_age": eligibility.get("minimumAge", ""),
                "max_age": eligibility.get("maximumAge", ""),
                "sex": eligibility.get("sex", ""),
                "healthy_volunteers": eligibility.get("healthyVolunteers", ""),
                
                # Outcomes
                "primary_outcomes": outcomes.get("primaryOutcomes", []),
                "secondary_outcomes": outcomes.get("secondaryOutcomes", []),
                
                # Timeline
                "start_date": status.get("startDateStruct", {}).get("date", ""),
                "completion_date": status.get("completionDateStruct", {}).get("date", ""),
                "first_posted_date": status.get("studyFirstPostDateStruct", {}).get("date", ""),
                
                # Status
                "overall_status": status.get("overallStatus", ""),
                "why_stopped": status.get("whyStopped", ""),
                "has_results": status.get("hasResults", False),
                
                # Sponsor
                "lead_sponsor": sponsors.get("leadSponsor", {}).get("name", ""),
                "sponsor_class": sponsors.get("leadSponsor", {}).get("class", ""),
                "collaborators": [c.get("name", "") for c in sponsors.get("collaborators", [])],
                
                # Location (first location for now)
                "locations": protocol.get("contactsLocationsModule", {}).get("locations", []),
                
                # Data collection metadata
                "collected_date": pd.Timestamp.now().isoformat(),
                "api_version": "v2"
            }
            
            return record
            
        except Exception as e:
            self.logger.warning(f"Error extracting trial record: {e}")
            return None
    
    def collect_all_trials(self, max_records: int = config.TARGET_RECORDS) -> pd.DataFrame:
        """Collect clinical trials data up to max_records"""
        self.logger.info(f"Starting data collection, target: {max_records} records")
        
        page_token = None
        total_collected = 0
        
        with tqdm(desc="Collecting trials") as pbar:
            while total_collected < max_records:
                try:
                    # Fetch batch
                    batch_data = self.fetch_trials_batch(page_token)
                    studies = batch_data.get("studies", [])
                    
                    if not studies:
                        self.logger.info("No more studies available")
                        break
                    
                    # Process each study
                    batch_records = []
                    for study in studies:
                        record = self.extract_trial_record(study)
                        if record:
                            batch_records.append(record)
                    
                    self.collected_trials.extend(batch_records)
                    total_collected = len(self.collected_trials)
                    
                    pbar.set_postfix({
                        'collected': total_collected,
                        'batch_valid': len(batch_records)
                    })
                    pbar.update(len(batch_records))
                    
                    self.logger.info(f"Collected {len(batch_records)} valid records, "
                                   f"Total: {total_collected}")
                    
                    # Get next page token for pagination
                    page_token = batch_data.get("nextPageToken")
                    if not page_token:
                        self.logger.info("No more pages available")
                        break
                    
                    # Rate limiting
                    time.sleep(config.DELAY_BETWEEN_REQUESTS)
                        
                except Exception as e:
                    self.logger.error(f"Error fetching batch: {e}")
                    break
        
        df = pd.DataFrame(self.collected_trials)
        self.logger.info(f"Collection completed. Total records: {len(df)}")
        
        # Save raw data as CSV for now (easier with mixed data types)
        raw_path = "data/raw/clinical_trials_raw.csv"
        df.to_csv(raw_path, index=False)
        self.logger.info(f"Raw data saved to {raw_path}")
        
        return df

# Test function
def test_collector():
    collector = ClinicalTrialsCollector()
    
    # Test with small batch first
    print("Testing with 10 records...")
    test_data = collector.collect_all_trials(max_records=10)
    print(f"Test completed. Shape: {test_data.shape}")
    print(f"Columns: {list(test_data.columns)}")
    print(f"Sample drug names: {test_data['primary_drug'].head().tolist()}")
    
    return test_data

if __name__ == "__main__":
    test_collector()