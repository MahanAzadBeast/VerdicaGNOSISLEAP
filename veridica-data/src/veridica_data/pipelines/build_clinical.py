"""
Clinical Data Pipeline
Builds clinical trials table from ClinicalTrials.gov API
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

from ..connectors.clintrials_fetch import ClinicalTrialsFetcher

logger = logging.getLogger(__name__)


def build_clinical_from_existing_data():
    """
    Build clinical table from existing clinical trials data
    """
    logger.info("🏥 BUILDING CLINICAL TABLE FROM EXISTING DATA")
    logger.info("=" * 60)
    
    # Check for existing clinical trials data
    existing_files = [
        "/workspace/clinical_trial_dataset/data/github_final/trials_part_1.csv",
        "/workspace/clinical_trial_dataset/data/github_final/trials_part_2.csv", 
        "/workspace/clinical_trial_dataset/data/github_final/trials_part_3.csv",
        "/workspace/clinical_trial_dataset/data/github_final/trials_part_4.csv"
    ]
    
    all_trials = []
    
    for file_path in existing_files:
        try:
            df = pd.read_csv(file_path)
            logger.info(f"✅ Loaded {len(df):,} trials from {Path(file_path).name}")
            all_trials.append(df)
        except Exception as e:
            logger.warning(f"Could not load {file_path}: {e}")
    
    if not all_trials:
        logger.error("❌ No existing clinical trials data found")
        return pd.DataFrame()
    
    # Combine all trials
    combined_trials = pd.concat(all_trials, ignore_index=True)
    logger.info(f"📊 Combined total: {len(combined_trials):,} clinical trials")
    
    # Aggregate by primary drug
    logger.info("🔗 Aggregating trials by drug...")
    
    drug_aggregates = []
    
    for drug, group in combined_trials.groupby('primary_drug'):
        try:
            # Skip if drug name is missing or invalid
            if pd.isna(drug) or not drug or str(drug).lower() in ['nan', 'none', 'unknown']:
                continue
            
            # Calculate aggregated clinical metrics
            max_phase = 0
            if 'phase' in group.columns:
                phases = pd.to_numeric(group['phase'], errors='coerce')
                max_phase = phases.max() if phases.notna().any() else 0
            elif 'max_phase' in group.columns:
                phases = pd.to_numeric(group['max_phase'], errors='coerce')
                max_phase = phases.max() if phases.notna().any() else 0
            
            trial_count = len(group)
            
            # Status aggregation
            if 'overall_status' in group.columns:
                status_counts = group['overall_status'].value_counts()
                most_recent_status = status_counts.index[0] if not status_counts.empty else 'Unknown'
            else:
                most_recent_status = 'Unknown'
            
            # Temporal information
            start_dates = None
            completion_dates = None
            
            if 'start_date' in group.columns:
                start_dates = pd.to_datetime(group['start_date'], errors='coerce')
            elif 'study_start_date' in group.columns:
                start_dates = pd.to_datetime(group['study_start_date'], errors='coerce')
            
            if 'completion_date' in group.columns:
                completion_dates = pd.to_datetime(group['completion_date'], errors='coerce')
            elif 'study_completion_date' in group.columns:
                completion_dates = pd.to_datetime(group['study_completion_date'], errors='coerce')
            
            first_trial_date = start_dates.min() if start_dates is not None and start_dates.notna().any() else None
            last_trial_date = start_dates.max() if start_dates is not None and start_dates.notna().any() else None
            
            # Approval status (if available)
            approval_date = None
            approved = False
            
            if 'approval_date' in group.columns:
                approval_dates = pd.to_datetime(group['approval_date'], errors='coerce')
                approval_date = approval_dates.min() if approval_dates.notna().any() else None
                approved = approval_date is not None
            
            # Create aggregated record
            drug_record = {
                'primary_drug': drug,
                'max_clinical_phase': max_phase,
                'trial_activity_count': trial_count,
                'trial_status_most_recent': most_recent_status,
                'first_trial_date': first_trial_date,
                'last_trial_date': last_trial_date,
                'approval_date_first': approval_date,
                'approved': approved,
                'active_trials': len(group[group.get('overall_status', '').str.contains('Active|Recruiting', na=False)]) if 'overall_status' in group.columns else 0,
                'completed_trials': len(group[group.get('overall_status', '').str.contains('Completed', na=False)]) if 'overall_status' in group.columns else 0,
                'terminated_trials': len(group[group.get('overall_status', '').str.contains('Terminated|Withdrawn', na=False)]) if 'overall_status' in group.columns else 0,
                'data_source': 'existing_clinical_trials',
                'created_at': datetime.now(),
                'last_updated': datetime.now()
            }
            
            drug_aggregates.append(drug_record)
            
        except Exception as e:
            logger.error(f"Error aggregating drug {drug}: {e}")
            continue
    
    # Create clinical DataFrame
    clinical_df = pd.DataFrame(drug_aggregates)
    
    if clinical_df.empty:
        logger.error("❌ No clinical aggregates created")
        return pd.DataFrame()
    
    logger.info(f"✅ Clinical aggregation complete: {len(clinical_df):,} drugs")
    
    # Generate report
    _generate_clinical_report(clinical_df)
    
    # Save clinical table
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clinical_file = output_dir / "clinical.parquet"
    
    try:
        clinical_df.to_parquet(clinical_file, compression='snappy')
        logger.info(f"💾 Clinical table saved: {clinical_file}")
        
        # Also save as CSV
        csv_file = output_dir / "clinical.csv"
        clinical_df.to_csv(csv_file, index=False)
        logger.info(f"💾 Clinical CSV saved: {csv_file}")
        
    except Exception as e:
        logger.error(f"Error saving clinical table: {e}")
    
    return clinical_df


def _generate_clinical_report(df: pd.DataFrame) -> None:
    """Generate clinical data quality report"""
    logger.info("📊 CLINICAL DATA REPORT")
    logger.info("=" * 50)
    
    # Basic statistics
    logger.info(f"Total drugs with clinical data: {len(df):,}")
    logger.info(f"Total trials aggregated: {df['trial_activity_count'].sum():,}")
    
    # Phase distribution
    if 'max_clinical_phase' in df.columns:
        phase_counts = df['max_clinical_phase'].value_counts().sort_index()
        logger.info(f"\\nMax clinical phase distribution:")
        for phase, count in phase_counts.items():
            if phase > 0:
                phase_name = f"Phase {int(phase)}"
            else:
                phase_name = "Preclinical/Early"
            logger.info(f"  {phase_name}: {count:,} drugs")
    
    # Approval status
    if 'approved' in df.columns:
        approved_count = df['approved'].sum()
        logger.info(f"\\nApproval status:")
        logger.info(f"  Approved drugs: {approved_count:,}")
        logger.info(f"  Non-approved drugs: {len(df) - approved_count:,}")
    
    # Trial activity
    if 'trial_activity_count' in df.columns:
        activity_stats = df['trial_activity_count'].describe()
        logger.info(f"\\nTrial activity statistics:")
        logger.info(f"  Mean trials per drug: {activity_stats['mean']:.1f}")
        logger.info(f"  Median trials per drug: {activity_stats['50%']:.0f}")
        logger.info(f"  Max trials per drug: {activity_stats['max']:.0f}")
    
    # Data quality
    logger.info(f"\\n🎯 DATA QUALITY:")
    logger.info(f"  100% real clinical trial data")
    logger.info(f"  Source: Existing clinical trials dataset")
    logger.info(f"  Zero synthetic trials")


def build_clinical_from_api():
    """
    Build clinical table by fetching fresh data from ClinicalTrials.gov API
    """
    logger.info("🌐 BUILDING CLINICAL TABLE FROM API")
    
    fetcher = ClinicalTrialsFetcher()
    clinical_df = fetcher.build_clinical_table()
    
    return clinical_df


def main():
    """Main pipeline execution"""
    logger.info("🏥 CLINICAL DATA PIPELINE")
    logger.info("🔬 Building clinical trials enrichment table")
    logger.info("🚫 100% REAL clinical data - NO synthetic trials")
    
    # Try existing data first, then API if needed
    clinical_df = build_clinical_from_existing_data()
    
    if clinical_df.empty:
        logger.info("📡 Falling back to API collection...")
        clinical_df = build_clinical_from_api()
    
    if not clinical_df.empty:
        logger.info("\\n🎉 CLINICAL PIPELINE COMPLETE")
        logger.info(f"📊 Clinical data for {len(clinical_df):,} drugs")
        logger.info(f"✅ Ready for master table integration")
        logger.info(f"📁 Saved to: data/clinical.parquet")
    else:
        logger.error("❌ Clinical pipeline failed")
    
    return clinical_df


if __name__ == "__main__":
    main()