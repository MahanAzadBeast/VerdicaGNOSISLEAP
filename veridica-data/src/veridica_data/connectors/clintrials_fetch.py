"""
ClinicalTrials.gov Data Connector
Fetches real clinical trial data from ClinicalTrials.gov API v2
"""

import requests
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import json

from ..utils.rate_limit import qps_limiter
from ..utils.mapping import DrugNameMapper

logger = logging.getLogger(__name__)


class ClinicalTrialsFetcher:
    """
    ClinicalTrials.gov API v2 data fetcher
    """
    
    def __init__(self, base_url: str = "https://clinicaltrials.gov/api/v2", max_qps: float = 3):
        self.base_url = base_url
        self.max_qps = max_qps
        self.mapper = DrugNameMapper()
        
        # API endpoints
        self.studies_url = f"{base_url}/studies"
        
    @qps_limiter(max_qps=3)
    def fetch_studies_batch(self, page_token: str = None, page_size: int = 1000) -> Dict[str, Any]:
        """
        Fetch batch of clinical studies from ClinicalTrials.gov
        
        Args:
            page_token: Pagination token for next batch
            page_size: Number of studies per batch
            
        Returns:
            API response with studies data
        """
        params = {
            'format': 'json',
            'pageSize': page_size
        }
        
        if page_token:
            params['pageToken'] = page_token
        
        try:
            response = requests.get(self.studies_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Fetched batch: {len(data.get('studies', []))} studies")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching studies batch: {e}")
            return {}
    
    def extract_trial_record(self, study: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract structured trial record from raw study data
        
        Args:
            study: Raw study data from API
            
        Returns:
            Structured trial record or None if not drug-related
        """
        try:
            protocol_section = study.get('protocolSection', {})
            identification_module = protocol_section.get('identificationModule', {})
            status_module = protocol_section.get('statusModule', {})
            design_module = protocol_section.get('designModule', {})
            arms_module = protocol_section.get('armsInterventionsModule', {})
            
            # Extract basic trial information
            nct_id = identification_module.get('nctId')
            if not nct_id:
                return None
            
            # Extract interventions (drugs)
            interventions = arms_module.get('interventions', [])
            drug_interventions = [
                intervention for intervention in interventions 
                if intervention.get('type', '').lower() in ['drug', 'biological']
            ]
            
            if not drug_interventions:
                return None  # Skip non-drug studies
            
            # Extract drug names
            primary_drugs = []
            all_drug_names = []
            
            for intervention in drug_interventions:
                name = intervention.get('name', '')
                if name:
                    cleaned_name = self.mapper.clean_drug_name(name)
                    if cleaned_name:
                        all_drug_names.append(cleaned_name)
                        if not primary_drugs:
                            primary_drugs.append(cleaned_name)
                
                # Also check other names
                other_names = intervention.get('otherNames', [])
                for other_name in other_names:
                    cleaned = self.mapper.clean_drug_name(other_name)
                    if cleaned:
                        all_drug_names.append(cleaned)
            
            if not primary_drugs:
                return None
            
            # Extract trial phases
            phases = design_module.get('phases', [])
            phase_numbers = []
            
            for phase in phases:
                if 'PHASE1' in phase:
                    phase_numbers.append(1)
                elif 'PHASE2' in phase:
                    phase_numbers.append(2)
                elif 'PHASE3' in phase:
                    phase_numbers.append(3)
                elif 'PHASE4' in phase:
                    phase_numbers.append(4)
            
            max_phase = max(phase_numbers) if phase_numbers else 0
            
            # Extract dates
            start_date = status_module.get('startDateStruct', {}).get('date')
            completion_date = status_module.get('completionDateStruct', {}).get('date')
            last_update = status_module.get('lastUpdatePostDateStruct', {}).get('date')
            
            # Extract status
            overall_status = status_module.get('overallStatus', '')
            
            # Create trial record
            trial_record = {
                'nct_id': nct_id,
                'title': identification_module.get('briefTitle', ''),
                'primary_drug': primary_drugs[0],
                'all_drug_names': list(set(all_drug_names)),  # Remove duplicates
                'max_phase': max_phase,
                'phases': phase_numbers,
                'overall_status': overall_status,
                'start_date': start_date,
                'completion_date': completion_date,
                'last_update_date': last_update,
                'study_type': design_module.get('studyType', ''),
                'intervention_count': len(drug_interventions),
                'conditions': [
                    condition.get('name', '') 
                    for condition in protocol_section.get('conditionsModule', {}).get('conditions', [])
                ],
                'enrollment': design_module.get('enrollmentInfo', {}).get('count'),
                'data_source': 'clinicaltrials_gov',
                'collected_date': datetime.now().isoformat()
            }
            
            return trial_record
            
        except Exception as e:
            logger.error(f"Error extracting trial record: {e}")
            return None
    
    def collect_clinical_trials(self, max_trials: int = 50000, save_progress: bool = True) -> pd.DataFrame:
        """
        Collect clinical trials data from ClinicalTrials.gov
        
        Args:
            max_trials: Maximum number of trials to collect
            save_progress: Whether to save progress incrementally
            
        Returns:
            DataFrame with clinical trials data
        """
        logger.info(f"🏥 COLLECTING CLINICAL TRIALS DATA")
        logger.info(f"🎯 Target: {max_trials:,} drug trials from ClinicalTrials.gov")
        logger.info("🚫 100% REAL clinical trial data - NO synthetic trials")
        logger.info("=" * 60)
        
        all_trials = []
        page_token = None
        batch_count = 0
        
        while len(all_trials) < max_trials:
            try:
                # Fetch batch
                batch_data = self.fetch_studies_batch(page_token=page_token)
                
                if not batch_data or 'studies' not in batch_data:
                    logger.warning("No more studies available")
                    break
                
                studies = batch_data['studies']
                batch_count += 1
                
                logger.info(f"📦 Processing batch {batch_count}: {len(studies)} studies")
                
                # Process each study
                batch_trials = []
                for study in studies:
                    trial_record = self.extract_trial_record(study)
                    if trial_record:
                        batch_trials.append(trial_record)
                
                all_trials.extend(batch_trials)
                
                logger.info(f"✅ Batch {batch_count}: {len(batch_trials)} drug trials extracted")
                logger.info(f"📊 Total collected: {len(all_trials):,} drug trials")
                
                # Save progress incrementally
                if save_progress and len(all_trials) % 5000 == 0:
                    temp_df = pd.DataFrame(all_trials)
                    temp_file = f"data/clinical_trials_progress_{len(all_trials)}.csv"
                    temp_df.to_csv(temp_file, index=False)
                    logger.info(f"💾 Progress saved: {temp_file}")
                
                # Check for next page
                next_page_token = batch_data.get('nextPageToken')
                if not next_page_token:
                    logger.info("📄 Reached end of available studies")
                    break
                
                page_token = next_page_token
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in batch {batch_count}: {e}")
                break
        
        # Create final DataFrame
        trials_df = pd.DataFrame(all_trials)
        
        if not trials_df.empty:
            logger.info(f"🎉 CLINICAL TRIALS COLLECTION COMPLETE")
            logger.info(f"📊 Total drug trials: {len(trials_df):,}")
            logger.info(f"🧬 Unique drugs: {trials_df['primary_drug'].nunique():,}")
            
            # Generate collection report
            self._generate_collection_report(trials_df)
        else:
            logger.error("❌ No clinical trials collected")
        
        return trials_df
    
    def _generate_collection_report(self, df: pd.DataFrame) -> None:
        """Generate collection quality report"""
        logger.info("📊 CLINICAL TRIALS COLLECTION REPORT")
        logger.info("=" * 50)
        
        # Basic statistics
        logger.info(f"Total trials: {len(df):,}")
        logger.info(f"Unique NCT IDs: {df['nct_id'].nunique():,}")
        logger.info(f"Unique primary drugs: {df['primary_drug'].nunique():,}")
        
        # Phase distribution
        phase_counts = df['max_phase'].value_counts().sort_index()
        logger.info(f"\\nPhase distribution:")
        for phase, count in phase_counts.items():
            phase_name = f"Phase {int(phase)}" if phase > 0 else "Preclinical/Early"
            logger.info(f"  {phase_name}: {count:,} trials")
        
        # Status distribution
        status_counts = df['overall_status'].value_counts()
        logger.info(f"\\nStatus distribution:")
        for status, count in status_counts.head(10).items():
            logger.info(f"  {status}: {count:,} trials")
        
        # Temporal distribution
        if 'start_date' in df.columns:
            df_with_dates = df[df['start_date'].notna()].copy()
            if not df_with_dates.empty:
                df_with_dates['start_year'] = pd.to_datetime(df_with_dates['start_date']).dt.year
                year_counts = df_with_dates['start_year'].value_counts().sort_index()
                logger.info(f"\\nTrial start years (recent):")
                for year, count in year_counts.tail(10).items():
                    logger.info(f"  {int(year)}: {count:,} trials")
    
    def build_clinical_table(self, output_file: str = "data/clinical.parquet") -> pd.DataFrame:
        """
        Build clinical trials table
        
        Args:
            output_file: Output file path
            
        Returns:
            Clinical trials DataFrame
        """
        logger.info("🏥 BUILDING CLINICAL TRIALS TABLE")
        
        # Collect clinical trials data
        trials_df = self.collect_clinical_trials()
        
        if trials_df.empty:
            logger.error("❌ No clinical trials data collected")
            return pd.DataFrame()
        
        # Aggregate by drug for master table joining
        logger.info("🔗 Aggregating trials by drug...")
        
        drug_aggregates = []
        
        for drug, group in trials_df.groupby('primary_drug'):
            try:
                # Calculate aggregated clinical metrics
                max_phase = group['max_phase'].max()
                trial_count = len(group)
                
                # Status aggregation
                status_counts = group['overall_status'].value_counts()
                most_recent_status = status_counts.index[0] if not status_counts.empty else 'Unknown'
                
                # Temporal information
                start_dates = pd.to_datetime(group['start_date'], errors='coerce')
                completion_dates = pd.to_datetime(group['completion_date'], errors='coerce')
                
                first_trial_date = start_dates.min() if start_dates.notna().any() else None
                last_trial_date = start_dates.max() if start_dates.notna().any() else None
                
                # Create aggregated record
                drug_record = {
                    'primary_drug': drug,
                    'max_clinical_phase': max_phase,
                    'trial_activity_count': trial_count,
                    'trial_status_most_recent': most_recent_status,
                    'first_trial_date': first_trial_date,
                    'last_trial_date': last_trial_date,
                    'active_trials': len(group[group['overall_status'].str.contains('Active|Recruiting', na=False)]),
                    'completed_trials': len(group[group['overall_status'].str.contains('Completed', na=False)]),
                    'terminated_trials': len(group[group['overall_status'].str.contains('Terminated|Withdrawn', na=False)]),
                    'conditions_studied': list(set([
                        condition for conditions_list in group['conditions'] 
                        for condition in conditions_list if condition
                    ])),
                    'data_source': 'clinicaltrials_gov',
                    'created_at': datetime.now(),
                    'last_updated': datetime.now()
                }
                
                drug_aggregates.append(drug_record)
                
            except Exception as e:
                logger.error(f"Error aggregating drug {drug}: {e}")
                continue
        
        # Create clinical DataFrame
        clinical_df = pd.DataFrame(drug_aggregates)
        
        logger.info(f"✅ Clinical aggregation complete: {len(clinical_df):,} drugs")
        
        # Save clinical table
        try:
            clinical_df.to_parquet(output_file, compression='snappy')
            logger.info(f"💾 Clinical table saved: {output_file}")
        except Exception as e:
            logger.error(f"Error saving clinical table: {e}")
        
        return clinical_df