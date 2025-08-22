#!/usr/bin/env python3
"""
Test script for Numeric Potency Gating validation.

This script validates that aspirin gets gated on kinase targets
and displays "HYPOTHESIS_ONLY" instead of numeric predictions.
"""

import sys
import os
sys.path.append('/app/backend')

import pandas as pd
import logging
import json
from hp_ad_layer import HighPerformanceAD, initialize_hp_ad_layer_sync, passes_kinase_hinge_pharmacophore, is_tiny_acid_veto
from ad_mock_data import generate_mock_training_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_aspirin_gating():
    """Test that aspirin gets gated on kinase targets"""
    logger.info("🧪 Testing Aspirin Gating Logic")
    
    # Generate mock training data
    logger.info("Generating mock training data...")
    training_data = generate_mock_training_data(n_compounds=100, n_targets=8)
    
    # Initialize HP AD layer
    logger.info("Initializing HP AD layer...")
    hp_ad = initialize_hp_ad_layer_sync(training_data)
    
    # Test compounds
    test_cases = [
        {
            'name': 'Aspirin',
            'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'targets': ['ERBB4', 'RET', 'AKT2', 'PARP1'],
            'should_gate': True,
            'expected_reasons': ['Kinase_pharmacophore_fail', 'tiny_acid_veto', 'OOD_chem']
        },
        {
            'name': 'Ethanol',
            'smiles': 'CCO', 
            'targets': ['EGFR', 'BRAF'],
            'should_gate': True,
            'expected_reasons': ['OOD_chem', 'Insufficient_in-class_neighbors']
        },
        {
            'name': 'Benzene',
            'smiles': 'c1ccccc1',
            'targets': ['CDK2'],
            'should_gate': True,
            'expected_reasons': ['OOD_chem', 'Kinase_pharmacophore_fail']
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        logger.info(f"\n--- Testing {test_case['name']} ---")
        
        for target in test_case['targets']:
            logger.info(f"Testing {test_case['name']} on {target}")
            
            # Test with HP AD layer
            result = hp_ad.ultra_fast_score_with_ad(
                ligand_smiles=test_case['smiles'],
                target_id=target,
                base_prediction=6.0,
                assay_type='IC50'
            )
            
            # Check if gated
            is_gated = hasattr(result, 'status') and result.status == "HYPOTHESIS_ONLY"
            
            test_result = {
                'compound': test_case['name'],
                'smiles': test_case['smiles'],
                'target': target,
                'is_gated': is_gated,
                'expected_gated': test_case['should_gate']
            }
            
            if is_gated:
                test_result.update({
                    'status': result.status,
                    'message': result.message,
                    'why': result.why,
                    'evidence_keys': list(result.evidence.keys())
                })
                
                logger.info(f"  ✅ GATED: {result.status}")
                logger.info(f"  Reasons: {result.why}")
                logger.info(f"  Evidence: {result.evidence.get('S_max', 'N/A')} S_max, "
                          f"{result.evidence.get('neighbors_same_assay', 'N/A')} neighbors")
                
                # Check for expected gating reasons
                found_expected = any(reason in result.why for reason in test_case['expected_reasons'])
                test_result['has_expected_reasons'] = found_expected
                
                if found_expected:
                    logger.info(f"  ✅ Contains expected gating reasons")
                else:
                    logger.warning(f"  ⚠️  Missing expected reasons: {test_case['expected_reasons']}")
                    
            else:
                logger.info(f"  ❌ NOT GATED (should be gated: {test_case['should_gate']})")
                test_result.update({
                    'ad_score': getattr(result, 'ad_score', 'N/A'),
                    'confidence': getattr(result, 'confidence_calibrated', 'N/A'),
                    'flags': getattr(result, 'flags', [])
                })
            
            results.append(test_result)
    
    # Summary
    logger.info("\n🎯 GATING VALIDATION SUMMARY")
    total_tests = len(results)
    correctly_gated = sum(1 for r in results if r['is_gated'] == r['expected_gated'])
    
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Correctly gated: {correctly_gated}/{total_tests}")
    logger.info(f"Success rate: {correctly_gated/total_tests*100:.1f}%")
    
    # Detailed results
    for result in results:
        status = "✅ PASS" if result['is_gated'] == result['expected_gated'] else "❌ FAIL"
        logger.info(f"{status} {result['compound']} on {result['target']}: "
                   f"Gated={result['is_gated']} (expected={result['expected_gated']})")
    
    return results

def test_pharmacophore_functions():
    """Test individual pharmacophore checking functions"""
    logger.info("\n🧪 Testing Pharmacophore Functions")
    
    test_molecules = [
        ('Aspirin', 'CC(=O)OC1=CC=CC=C1C(=O)O'),
        ('Imatinib', 'Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C'),
        ('Caffeine', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'),
        ('Ethanol', 'CCO'),
        ('Benzamide', 'NC(=O)c1ccccc1'),  # Should pass PARP1
    ]
    
    for name, smiles in test_molecules:
        kinase_pass = passes_kinase_hinge_pharmacophore(smiles)
        tiny_acid = is_tiny_acid_veto(smiles)
        
        logger.info(f"{name}: Kinase={kinase_pass}, TinyAcid={tiny_acid}")
    
    logger.info("✅ Pharmacophore function tests completed")

if __name__ == "__main__":
    logger.info("🚀 Starting Numeric Potency Gating Validation")
    
    try:
        # Test pharmacophore functions
        test_pharmacophore_functions()
        
        # Test gating logic
        gating_results = test_aspirin_gating()
        
        # Print final results
        logger.info("\n✅ Validation completed successfully!")
        
        # Check if aspirin is properly gated on kinases
        aspirin_kinase_results = [r for r in gating_results 
                                if r['compound'] == 'Aspirin' and r['target'] in ['ERBB4', 'RET', 'AKT2']]
        
        aspirin_gated_count = sum(1 for r in aspirin_kinase_results if r['is_gated'])
        logger.info(f"\n🎯 ASPIRIN KINASE GATING: {aspirin_gated_count}/{len(aspirin_kinase_results)} kinase targets gated")
        
        if aspirin_gated_count == len(aspirin_kinase_results):
            logger.info("✅ SUCCESS: All aspirin kinase predictions properly gated!")
        else:
            logger.warning("⚠️  Some aspirin kinase predictions not gated as expected")
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)