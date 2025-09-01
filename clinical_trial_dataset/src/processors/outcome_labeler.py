"""
Clinical Trial Outcome Labeler
Creates binary and multi-class outcome labels from trial status information
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List
import re
from datetime import datetime, timedelta

class OutcomeLabeler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.outcome_stats = {}
        
    def parse_why_stopped(self, why_stopped: str) -> Dict[str, bool]:
        """Parse the 'why_stopped' field to identify failure reasons"""
        if not why_stopped or pd.isna(why_stopped):
            return {'safety': False, 'efficacy': False, 'recruitment': False, 'business': False, 'other': False}
        
        why_stopped = str(why_stopped).lower()
        
        # Safety-related keywords
        safety_keywords = [
            'safety', 'adverse', 'toxicity', 'toxic', 'side effect', 'harm', 
            'serious adverse event', 'sae', 'death', 'safety concern'
        ]
        
        # Efficacy-related keywords
        efficacy_keywords = [
            'efficacy', 'lack of efficacy', 'insufficient efficacy', 'futility',
            'no improvement', 'ineffective', 'failed to meet', 'primary endpoint',
            'not effective', 'lack of response'
        ]
        
        # Recruitment-related keywords
        recruitment_keywords = [
            'recruitment', 'enrollment', 'accrual', 'slow recruitment', 
            'poor recruitment', 'unable to recruit', 'low enrollment'
        ]
        
        # Business-related keywords
        business_keywords = [
            'funding', 'sponsor', 'business', 'financial', 'strategic',
            'portfolio', 'commercial', 'resource', 'priority'
        ]
        
        return {
            'safety': any(keyword in why_stopped for keyword in safety_keywords),
            'efficacy': any(keyword in why_stopped for keyword in efficacy_keywords),
            'recruitment': any(keyword in why_stopped for keyword in recruitment_keywords),
            'business': any(keyword in why_stopped for keyword in business_keywords),
            'other': not any(keyword in why_stopped for keyword in 
                           safety_keywords + efficacy_keywords + recruitment_keywords + business_keywords)
        }
    
    def determine_trial_success(self, row: pd.Series) -> Dict[str, any]:
        """Determine if a trial was successful based on multiple factors"""
        status = str(row['overall_status']).upper()
        why_stopped = str(row.get('why_stopped', '')).upper()
        has_results = row.get('has_results', False)
        completion_date = row.get('completion_date', '')
        
        # Parse failure reasons
        failure_reasons = self.parse_why_stopped(row.get('why_stopped', ''))
        
        # Primary outcome determination
        if status == 'COMPLETED':
            if has_results:
                outcome = 'SUCCESS_WITH_RESULTS'
                binary_outcome = 1
                confidence = 0.9
            else:
                # Completed but no results posted yet - assume success but lower confidence
                outcome = 'SUCCESS_NO_RESULTS'
                binary_outcome = 1
                confidence = 0.7
        
        elif status in ['ACTIVE_NOT_RECRUITING', 'RECRUITING']:
            # Still ongoing trials - label as unknown for now
            outcome = 'ONGOING'
            binary_outcome = -1  # Unknown
            confidence = 0.0
            
        elif status in ['TERMINATED', 'SUSPENDED', 'WITHDRAWN']:
            # Failed trials - determine reason
            if failure_reasons['safety']:
                outcome = 'FAILURE_SAFETY'
            elif failure_reasons['efficacy']:
                outcome = 'FAILURE_EFFICACY'
            elif failure_reasons['recruitment']:
                outcome = 'FAILURE_RECRUITMENT'
            elif failure_reasons['business']:
                outcome = 'FAILURE_BUSINESS'
            else:
                outcome = 'FAILURE_OTHER'
            
            binary_outcome = 0
            confidence = 0.8
            
        else:
            # Unknown status
            outcome = 'UNKNOWN'
            binary_outcome = -1
            confidence = 0.0
        
        return {
            'outcome_label': outcome,
            'binary_outcome': binary_outcome,
            'outcome_confidence': confidence,
            'failure_reason_safety': failure_reasons['safety'],
            'failure_reason_efficacy': failure_reasons['efficacy'],
            'failure_reason_recruitment': failure_reasons['recruitment'],
            'failure_reason_business': failure_reasons['business'],
            'failure_reason_other': failure_reasons['other']
        }
    
    def calculate_trial_duration(self, row: pd.Series) -> Dict[str, float]:
        """Calculate various duration metrics"""
        try:
            start_date = pd.to_datetime(row.get('start_date', ''), errors='coerce')
            completion_date = pd.to_datetime(row.get('completion_date', ''), errors='coerce')
            first_posted = pd.to_datetime(row.get('first_posted_date', ''), errors='coerce')
            
            durations = {}
            
            if pd.notna(start_date) and pd.notna(completion_date):
                duration = (completion_date - start_date).days
                durations['trial_duration_days'] = duration
                durations['trial_duration_years'] = duration / 365.25
            else:
                durations['trial_duration_days'] = np.nan
                durations['trial_duration_years'] = np.nan
            
            if pd.notna(first_posted) and pd.notna(start_date):
                planning_time = (start_date - first_posted).days
                durations['planning_duration_days'] = max(0, planning_time)  # Can't be negative
            else:
                durations['planning_duration_days'] = np.nan
            
            # Expected duration based on phase (industry averages)
            phase_durations = {'PHASE1': 365, 'PHASE2': 730, 'PHASE3': 1095}  # days
            primary_phase = row.get('primary_phase', '')
            expected_duration = phase_durations.get(primary_phase, 730)
            durations['expected_duration_days'] = expected_duration
            
            if pd.notna(durations.get('trial_duration_days')):
                durations['duration_vs_expected'] = durations['trial_duration_days'] / expected_duration
            else:
                durations['duration_vs_expected'] = np.nan
            
            return durations
            
        except Exception as e:
            self.logger.warning(f"Error calculating durations: {e}")
            return {
                'trial_duration_days': np.nan,
                'trial_duration_years': np.nan,
                'planning_duration_days': np.nan,
                'expected_duration_days': np.nan,
                'duration_vs_expected': np.nan
            }
    
    def create_phase_specific_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create phase-specific outcome predictions"""
        result_df = df.copy()
        
        # Phase-specific success rates (industry averages for calibration)
        phase_success_rates = {'PHASE1': 0.63, 'PHASE2': 0.31, 'PHASE3': 0.58}
        
        for phase in ['PHASE1', 'PHASE2', 'PHASE3']:
            # Binary outcome for each phase
            phase_mask = result_df['primary_phase'] == phase
            result_df[f'{phase.lower()}_outcome'] = np.where(
                phase_mask, 
                result_df['binary_outcome'], 
                -1  # Not applicable
            )
            
            # Success probability based on historical rates and trial characteristics
            base_prob = phase_success_rates.get(phase, 0.5)
            result_df[f'{phase.lower()}_success_prob'] = np.where(
                phase_mask,
                base_prob,  # We'll improve this with ML later
                np.nan
            )
        
        return result_df
    
    def label_outcomes(self, trials_df: pd.DataFrame) -> pd.DataFrame:
        """Apply outcome labeling to the entire dataframe"""
        self.logger.info(f"Creating outcome labels for {len(trials_df)} trials...")
        
        results = []
        
        for idx, trial in trials_df.iterrows():
            trial_copy = trial.copy()
            
            # Get outcome labels
            outcome_info = self.determine_trial_success(trial)
            for key, value in outcome_info.items():
                trial_copy[key] = value
            
            # Get duration metrics
            duration_info = self.calculate_trial_duration(trial)
            for key, value in duration_info.items():
                trial_copy[key] = value
            
            results.append(trial_copy)
        
        result_df = pd.DataFrame(results)
        
        # Create phase-specific outcomes
        result_df = self.create_phase_specific_outcomes(result_df)
        
        # Calculate and log statistics
        self.calculate_outcome_statistics(result_df)
        
        return result_df
    
    def calculate_outcome_statistics(self, df: pd.DataFrame):
        """Calculate and log outcome statistics"""
        total_trials = len(df)
        
        # Overall outcome distribution
        outcome_counts = df['outcome_label'].value_counts()
        self.logger.info("Outcome Distribution:")
        for outcome, count in outcome_counts.items():
            percentage = count / total_trials * 100
            self.logger.info(f"  {outcome}: {count} ({percentage:.1f}%)")
        
        # Binary outcome distribution
        binary_counts = df[df['binary_outcome'] != -1]['binary_outcome'].value_counts()
        if len(binary_counts) > 0:
            success_rate = binary_counts.get(1, 0) / binary_counts.sum() * 100
            self.logger.info(f"Overall Success Rate: {success_rate:.1f}%")
        
        # Phase-specific statistics
        for phase in ['PHASE1', 'PHASE2', 'PHASE3']:
            phase_trials = df[df['primary_phase'] == phase]
            if len(phase_trials) > 0:
                phase_success = len(phase_trials[phase_trials['binary_outcome'] == 1])
                phase_total = len(phase_trials[phase_trials['binary_outcome'] != -1])
                if phase_total > 0:
                    phase_rate = phase_success / phase_total * 100
                    self.logger.info(f"{phase} Success Rate: {phase_rate:.1f}% ({phase_success}/{phase_total})")
        
        self.outcome_stats = {
            'total_trials': total_trials,
            'outcome_distribution': outcome_counts.to_dict(),
            'binary_distribution': binary_counts.to_dict() if len(binary_counts) > 0 else {},
        }

# Test function
def test_outcome_labeler():
    # Create sample data
    sample_data = pd.DataFrame([
        {
            'nct_id': 'NCT001', 'overall_status': 'COMPLETED', 'has_results': True,
            'why_stopped': '', 'primary_phase': 'PHASE2',
            'start_date': '2020-01-01', 'completion_date': '2021-06-01'
        },
        {
            'nct_id': 'NCT002', 'overall_status': 'TERMINATED', 'has_results': False,
            'why_stopped': 'Safety concerns', 'primary_phase': 'PHASE1',
            'start_date': '2019-03-01', 'completion_date': '2019-08-01'
        },
        {
            'nct_id': 'NCT003', 'overall_status': 'TERMINATED', 'has_results': False,
            'why_stopped': 'Lack of efficacy', 'primary_phase': 'PHASE3',
            'start_date': '2018-01-01', 'completion_date': '2020-12-01'
        }
    ])
    
    labeler = OutcomeLabeler()
    labeled_data = labeler.label_outcomes(sample_data)
    
    print("Sample outcome labeling:")
    for idx, row in labeled_data.iterrows():
        print(f"NCT{idx+1}: {row['outcome_label']} (binary: {row['binary_outcome']})")
        print(f"  Duration: {row.get('trial_duration_days', 'N/A')} days")
        print(f"  Failure reasons: Safety={row.get('failure_reason_safety', False)}, "
              f"Efficacy={row.get('failure_reason_efficacy', False)}")
    
    return labeled_data

if __name__ == "__main__":
    test_outcome_labeler()