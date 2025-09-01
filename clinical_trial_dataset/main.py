"""
Clinical Trial Dataset Builder
Main orchestration script
"""

import logging
from pathlib import Path
from src.data_collectors.clinicaltrials_gov import ClinicalTrialsCollector
from src.data_collectors.pubchem_mapper import SMILESMapper
from src.processors.outcome_labeler import OutcomeLabeler
from src.processors.feature_engineer import FeatureEngineer
from src.processors.quality_control import QualityController
import config

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Clinical Trial Dataset Creation...")
    
    # Step 1: Collect clinical trials data
    collector = ClinicalTrialsCollector()
    raw_trials = collector.collect_all_trials(max_records=config.TARGET_RECORDS)
    logger.info(f"Collected {len(raw_trials)} clinical trials")
    
    # Step 2: Map drugs to SMILES
    mapper = SMILESMapper()
    trials_with_smiles = mapper.map_trials_to_smiles(raw_trials)
    logger.info(f"Mapped {len(trials_with_smiles[trials_with_smiles['smiles'].notna()])} trials to SMILES")
    
    # TODO: Step 3: Create outcome labels
    # TODO: Step 4: Engineer features  
    # TODO: Step 5: Quality control
    # TODO: Step 6: Save final dataset
    
    logger.info("Steps 1-2 completed!")

if __name__ == "__main__":
    main()