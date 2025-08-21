"""
Test script for Applicability Domain layer
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.append('/app/backend')

from applicability_domain import (
    ApplicabilityDomainLayer, 
    standardize_smiles,
    get_ad_layer,
    initialize_ad_layer
)
from ad_mock_data import generate_mock_training_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_standardize_smiles():
    """Test SMILES standardization function"""
    logger.info("Testing SMILES standardization...")
    
    test_cases = [
        "CCO",  # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",  # Imatinib
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CCO.[Na+]Cl-",  # Ethanol with salt
        "invalid_smiles",  # Invalid
        "",  # Empty
        None  # None
    ]
    
    for smiles in test_cases:
        result = standardize_smiles(smiles)
        logger.info(f"'{smiles}' -> '{result}'")

def test_ad_layer_initialization():
    """Test AD layer initialization with mock data"""
    logger.info("Testing AD layer initialization...")
    
    # Generate mock training data
    logger.info("Generating mock training data...")
    training_data = generate_mock_training_data(n_compounds=100, n_targets=5)
    
    # Initialize AD layer
    logger.info("Initializing AD layer...")
    ad_layer = ApplicabilityDomainLayer()
    ad_layer.initialize(training_data)
    
    logger.info("✅ AD layer initialized successfully")
    return ad_layer

def test_ad_scoring():
    """Test AD scoring functionality"""
    logger.info("Testing AD scoring...")
    
    # Initialize AD layer
    ad_layer = test_ad_layer_initialization()
    
    # Test compounds
    test_compounds = [
        ("CCO", "EGFR"),  # Simple compound
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "BRAF"),  # Aspirin
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CDK2"),  # Caffeine
        ("c1ccccc1", "PARP1"),  # Benzene
        ("invalid_smiles", "EGFR")  # Invalid SMILES
    ]
    
    for smiles, target in test_compounds:
        try:
            logger.info(f"\nTesting: {smiles} against {target}")
            
            # Score with AD
            result = ad_layer.score_with_ad(smiles, target, base_prediction=6.0)
            
            logger.info(f"  AD Score: {result.ad_score:.3f}")
            logger.info(f"  Confidence: {result.confidence_calibrated:.3f}")
            logger.info(f"  CI: {result.potency_ci}")
            logger.info(f"  Flags: {result.flags}")
            logger.info(f"  Tanimoto: {result.tanimoto_score:.3f}")
            logger.info(f"  Neighbors: {len(result.nearest_neighbors)}")
            
        except Exception as e:
            logger.error(f"Error testing {smiles}: {e}")

def test_integration_with_gnosis():
    """Test integration with existing Gnosis I predictor"""
    logger.info("Testing integration with Gnosis I...")
    
    try:
        # Try to import existing Gnosis predictor
        from gnosis_model1_predictor import get_gnosis_predictor
        
        predictor = get_gnosis_predictor()
        if predictor is None:
            logger.warning("Gnosis I predictor not available - skipping integration test")
            return
        
        # Get available targets
        available_targets = predictor.get_available_targets()
        logger.info(f"Available targets: {len(available_targets)}")
        
        if available_targets:
            test_target = available_targets[0]
            test_smiles = "CCO"
            
            logger.info(f"Testing integration with target: {test_target}")
            
            # Make prediction with Gnosis I
            gnosis_result = predictor.predict_with_confidence(
                smiles=test_smiles,
                targets=[test_target],
                assay_types=['IC50']
            )
            
            logger.info(f"Gnosis I result keys: {gnosis_result.keys()}")
            
            # Initialize AD layer with mock data for demonstration
            training_data = generate_mock_training_data(n_compounds=50, n_targets=3)
            ad_layer = initialize_ad_layer(training_data)
            
            # Score with AD layer
            if test_target in gnosis_result['predictions']:
                base_prediction = gnosis_result['predictions'][test_target].get('IC50', {}).get('pActivity', 6.0)
            else:
                base_prediction = 6.0
            
            ad_result = ad_layer.score_with_ad(test_smiles, test_target, base_prediction)
            
            logger.info(f"✅ Integration successful:")
            logger.info(f"  Gnosis prediction: {base_prediction}")
            logger.info(f"  AD score: {ad_result.ad_score:.3f}")
            logger.info(f"  Enhanced confidence: {ad_result.confidence_calibrated:.3f}")
            
    except Exception as e:
        logger.error(f"Integration test failed: {e}")

def main():
    """Run all tests"""
    logger.info("🧪 Starting AD Layer Tests")
    
    try:
        # Test 1: SMILES standardization
        test_standardize_smiles()
        
        # Test 2: AD layer initialization
        test_ad_layer_initialization()
        
        # Test 3: AD scoring
        test_ad_scoring()
        
        # Test 4: Integration with Gnosis I
        test_integration_with_gnosis()
        
        logger.info("✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()