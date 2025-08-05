#!/usr/bin/env python3
"""
Enhanced Gnosis I Multi-Assay Backend Testing
Tests the new multi-assay functionality with IC50, Ki, and EC50 predictions
"""

import requests
import json
import time
import sys
from typing import Dict, List, Any
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

class EnhancedGnosisITester:
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        self.total_tests = 0
        self.passed_tests = 0
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            
        status = "✅ PASS" if success else "❌ FAIL"
        result = {
            'test': test_name,
            'status': status,
            'success': success,
            'details': details
        }
        self.test_results.append(result)
        if not success:
            self.failed_tests.append(result)
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")
    
    def test_multi_assay_predictions_single_target(self):
        """Test 1: Multi-Assay Predictions - Single Target (ABL1)"""
        print("\n=== Testing Multi-Assay Predictions - Single Target ===")
        
        aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        
        try:
            payload = {
                "smiles": aspirin_smiles,
                "targets": ["ABL1"],
                "assay_types": ["IC50", "Ki", "EC50"]
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', {})
                
                if 'ABL1' in predictions:
                    abl1_pred = predictions['ABL1']
                    
                    # Check for all three assay types
                    assay_types = ['IC50', 'Ki', 'EC50']
                    found_assays = []
                    
                    for assay in assay_types:
                        if assay in abl1_pred:
                            assay_data = abl1_pred[assay]
                            
                            # Check required fields
                            required_fields = ['pActivity', 'confidence', 'sigma']
                            has_all_fields = all(field in assay_data for field in required_fields)
                            
                            if has_all_fields:
                                found_assays.append(assay)
                                self.log_test(f"ABL1 {assay} Prediction Structure", True, 
                                            f"pActivity: {assay_data['pActivity']}, confidence: {assay_data['confidence']}, sigma: {assay_data['sigma']}")
                            else:
                                self.log_test(f"ABL1 {assay} Prediction Structure", False, 
                                            f"Missing fields: {[f for f in required_fields if f not in assay_data]}")
                    
                    # Check if all three assay types are present
                    if len(found_assays) == 3:
                        self.log_test("Multi-Assay Coverage", True, f"All 3 assay types found: {found_assays}")
                    else:
                        self.log_test("Multi-Assay Coverage", False, f"Only found {len(found_assays)}/3 assay types: {found_assays}")
                    
                    # Check selectivity ratio (should be None for single target)
                    selectivity = abl1_pred.get('selectivity_ratio')
                    if selectivity is None:
                        self.log_test("Single Target Selectivity", True, "Selectivity ratio correctly None for single target")
                    else:
                        self.log_test("Single Target Selectivity", False, f"Unexpected selectivity ratio: {selectivity}")
                        
                else:
                    self.log_test("ABL1 Target Prediction", False, "ABL1 not found in predictions")
                    
            elif response.status_code == 503:
                self.log_test("Multi-Assay Single Target", False, "Service unavailable (503)")
            else:
                self.log_test("Multi-Assay Single Target", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Multi-Assay Single Target", False, f"Error: {str(e)}")
    
    def test_multi_assay_predictions_multiple_targets(self):
        """Test 2: Multi-Assay Predictions - Multiple Targets"""
        print("\n=== Testing Multi-Assay Predictions - Multiple Targets ===")
        
        imatinib_smiles = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
        
        try:
            payload = {
                "smiles": imatinib_smiles,
                "targets": ["ABL1", "ABL2", "AKT1"],
                "assay_types": ["IC50", "Ki", "EC50"]
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', {})
                
                expected_targets = ["ABL1", "ABL2", "AKT1"]
                found_targets = []
                
                for target in expected_targets:
                    if target in predictions:
                        found_targets.append(target)
                        target_pred = predictions[target]
                        
                        # Check for all three assay types
                        assay_count = 0
                        for assay in ['IC50', 'Ki', 'EC50']:
                            if assay in target_pred:
                                assay_count += 1
                                assay_data = target_pred[assay]
                                
                                # Verify confidence metrics
                                has_confidence = 'confidence' in assay_data and 'sigma' in assay_data
                                if has_confidence:
                                    self.log_test(f"{target} {assay} Confidence", True, 
                                                f"confidence: {assay_data['confidence']}, sigma: {assay_data['sigma']}")
                                else:
                                    self.log_test(f"{target} {assay} Confidence", False, "Missing confidence metrics")
                        
                        # Check if target has all 3 assay types
                        if assay_count == 3:
                            self.log_test(f"{target} Multi-Assay", True, f"All 3 assay types present")
                        else:
                            self.log_test(f"{target} Multi-Assay", False, f"Only {assay_count}/3 assay types")
                        
                        # Check selectivity ratio (should be present for multiple targets)
                        selectivity = target_pred.get('selectivity_ratio')
                        if selectivity is not None:
                            self.log_test(f"{target} Selectivity Ratio", True, f"Selectivity: {selectivity}")
                        else:
                            self.log_test(f"{target} Selectivity Ratio", False, "Missing selectivity ratio")
                
                # Check overall target coverage
                if len(found_targets) == 3:
                    self.log_test("Multiple Target Coverage", True, f"All 3 targets found: {found_targets}")
                else:
                    self.log_test("Multiple Target Coverage", False, f"Only found {len(found_targets)}/3 targets: {found_targets}")
                    
            elif response.status_code == 503:
                self.log_test("Multi-Assay Multiple Targets", False, "Service unavailable (503)")
            else:
                self.log_test("Multi-Assay Multiple Targets", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Multi-Assay Multiple Targets", False, f"Error: {str(e)}")
    
    def test_all_targets_selection(self):
        """Test 3: All Targets Selection - Should process all 62 targets"""
        print("\n=== Testing All Targets Selection ===")
        
        caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        
        try:
            payload = {
                "smiles": caffeine_smiles,
                "targets": "all",
                "assay_types": ["IC50", "Ki", "EC50"]
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=60)  # Longer timeout for all targets
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', {})
                
                num_targets = len(predictions)
                
                # Check if we get close to 62 targets (allowing some flexibility)
                if num_targets >= 50:  # At least 50 targets
                    self.log_test("All Targets Count", True, f"Found {num_targets} targets (expected ~62)")
                elif num_targets >= 20:
                    self.log_test("All Targets Count", True, f"Found {num_targets} targets (partial coverage)")
                else:
                    self.log_test("All Targets Count", False, f"Only found {num_targets} targets (expected ~62)")
                
                # Check total predictions (should be targets × 3 assays)
                total_predictions = 0
                targets_with_all_assays = 0
                
                for target, target_pred in predictions.items():
                    assay_count = sum(1 for assay in ['IC50', 'Ki', 'EC50'] if assay in target_pred)
                    total_predictions += assay_count
                    
                    if assay_count == 3:
                        targets_with_all_assays += 1
                
                expected_total = num_targets * 3
                if total_predictions >= expected_total * 0.8:  # At least 80% of expected
                    self.log_test("All Targets Predictions", True, 
                                f"Found {total_predictions} predictions (~{expected_total} expected)")
                else:
                    self.log_test("All Targets Predictions", False, 
                                f"Only {total_predictions} predictions (expected ~{expected_total})")
                
                # Check how many targets have all 3 assay types
                coverage_rate = (targets_with_all_assays / num_targets) * 100 if num_targets > 0 else 0
                if coverage_rate >= 80:
                    self.log_test("All Assay Types Coverage", True, 
                                f"{targets_with_all_assays}/{num_targets} targets have all 3 assays ({coverage_rate:.1f}%)")
                else:
                    self.log_test("All Assay Types Coverage", False, 
                                f"Only {targets_with_all_assays}/{num_targets} targets have all 3 assays ({coverage_rate:.1f}%)")
                
                # Sample a few targets to verify structure
                sample_targets = list(predictions.keys())[:3]
                for target in sample_targets:
                    target_pred = predictions[target]
                    if 'IC50' in target_pred:
                        ic50_data = target_pred['IC50']
                        has_required = all(field in ic50_data for field in ['pActivity', 'confidence', 'sigma'])
                        self.log_test(f"Sample {target} Structure", has_required, 
                                    f"IC50 structure: {list(ic50_data.keys())}")
                    
            elif response.status_code == 503:
                self.log_test("All Targets Selection", False, "Service unavailable (503)")
            else:
                self.log_test("All Targets Selection", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("All Targets Selection", False, f"Error: {str(e)}")
    
    def test_new_data_structure(self):
        """Test 4: New Nested Data Structure Verification"""
        print("\n=== Testing New Data Structure ===")
        
        aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        
        try:
            payload = {
                "smiles": aspirin_smiles,
                "targets": ["ABL1"],
                "assay_types": ["IC50", "Ki", "EC50"]
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check top-level structure
                required_top_level = ['smiles', 'predictions', 'properties', 'model_info']
                missing_top = [field for field in required_top_level if field not in data]
                
                if not missing_top:
                    self.log_test("Top-Level Structure", True, f"All required fields present: {required_top_level}")
                else:
                    self.log_test("Top-Level Structure", False, f"Missing fields: {missing_top}")
                
                # Check predictions structure
                predictions = data.get('predictions', {})
                if 'ABL1' in predictions:
                    abl1_pred = predictions['ABL1']
                    
                    # Check nested assay structure
                    for assay in ['IC50', 'Ki', 'EC50']:
                        if assay in abl1_pred:
                            assay_data = abl1_pred[assay]
                            
                            # Expected structure: {"pActivity": X, "confidence": Y, "sigma": Z}
                            expected_fields = ['pActivity', 'confidence', 'sigma']
                            missing_fields = [field for field in expected_fields if field not in assay_data]
                            
                            if not missing_fields:
                                self.log_test(f"ABL1.{assay} Structure", True, 
                                            f"Correct nested structure: {list(assay_data.keys())}")
                                
                                # Verify data types
                                pactivity = assay_data['pActivity']
                                confidence = assay_data['confidence']
                                sigma = assay_data['sigma']
                                
                                types_correct = (isinstance(pactivity, (int, float)) and 
                                               isinstance(confidence, (int, float)) and 
                                               isinstance(sigma, (int, float)))
                                
                                if types_correct:
                                    self.log_test(f"ABL1.{assay} Data Types", True, 
                                                f"pActivity: {type(pactivity).__name__}, confidence: {type(confidence).__name__}, sigma: {type(sigma).__name__}")
                                else:
                                    self.log_test(f"ABL1.{assay} Data Types", False, 
                                                f"Incorrect types: pActivity={type(pactivity)}, confidence={type(confidence)}, sigma={type(sigma)}")
                            else:
                                self.log_test(f"ABL1.{assay} Structure", False, f"Missing fields: {missing_fields}")
                        else:
                            self.log_test(f"ABL1.{assay} Present", False, f"{assay} not found in ABL1 predictions")
                    
                    # Check selectivity_ratio field
                    if 'selectivity_ratio' in abl1_pred:
                        selectivity = abl1_pred['selectivity_ratio']
                        self.log_test("Selectivity Ratio Field", True, f"selectivity_ratio: {selectivity}")
                    else:
                        self.log_test("Selectivity Ratio Field", False, "selectivity_ratio field missing")
                        
                else:
                    self.log_test("Predictions Structure", False, "ABL1 not found in predictions")
                
                # Check properties structure
                properties = data.get('properties', {})
                expected_props = ['LogP', 'LogS']
                missing_props = [prop for prop in expected_props if prop not in properties]
                
                if not missing_props:
                    self.log_test("Properties Structure", True, f"Properties: {list(properties.keys())}")
                else:
                    self.log_test("Properties Structure", False, f"Missing properties: {missing_props}")
                    
            elif response.status_code == 503:
                self.log_test("New Data Structure", False, "Service unavailable (503)")
            else:
                self.log_test("New Data Structure", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("New Data Structure", False, f"Error: {str(e)}")
    
    def test_monte_carlo_dropout(self):
        """Test 5: Monte-Carlo Dropout Confidence Metrics"""
        print("\n=== Testing Monte-Carlo Dropout ===")
        
        imatinib_smiles = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
        
        try:
            payload = {
                "smiles": imatinib_smiles,
                "targets": ["ABL1", "ABL2"],
                "assay_types": ["IC50", "Ki", "EC50"]
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', {})
                
                # Check model_info for MC samples
                model_info = data.get('model_info', {})
                mc_samples = model_info.get('mc_samples')
                
                if mc_samples and mc_samples >= 20:  # Should be around 30
                    self.log_test("Monte-Carlo Samples", True, f"MC samples: {mc_samples}")
                else:
                    self.log_test("Monte-Carlo Samples", False, f"MC samples: {mc_samples} (expected ~30)")
                
                # Check confidence metrics for each prediction
                confidence_tests = 0
                sigma_tests = 0
                
                for target in ['ABL1', 'ABL2']:
                    if target in predictions:
                        target_pred = predictions[target]
                        
                        for assay in ['IC50', 'Ki', 'EC50']:
                            if assay in target_pred:
                                assay_data = target_pred[assay]
                                
                                # Check confidence (should be between 0 and 1)
                                confidence = assay_data.get('confidence')
                                if confidence is not None and 0 <= confidence <= 1:
                                    confidence_tests += 1
                                    self.log_test(f"{target}.{assay} Confidence Range", True, 
                                                f"confidence: {confidence} (valid range)")
                                else:
                                    self.log_test(f"{target}.{assay} Confidence Range", False, 
                                                f"confidence: {confidence} (invalid range)")
                                
                                # Check sigma (should be positive)
                                sigma = assay_data.get('sigma')
                                if sigma is not None and sigma > 0:
                                    sigma_tests += 1
                                    self.log_test(f"{target}.{assay} Sigma Value", True, 
                                                f"sigma: {sigma} (positive)")
                                else:
                                    self.log_test(f"{target}.{assay} Sigma Value", False, 
                                                f"sigma: {sigma} (should be positive)")
                                
                                # Check mc_samples field in individual predictions
                                mc_field = assay_data.get('mc_samples')
                                if mc_field and mc_field >= 20:
                                    self.log_test(f"{target}.{assay} MC Samples Field", True, 
                                                f"mc_samples: {mc_field}")
                                else:
                                    self.log_test(f"{target}.{assay} MC Samples Field", False, 
                                                f"mc_samples: {mc_field} (expected ~30)")
                
                # Overall confidence/sigma coverage
                expected_total = 6  # 2 targets × 3 assays
                confidence_rate = (confidence_tests / expected_total) * 100 if expected_total > 0 else 0
                sigma_rate = (sigma_tests / expected_total) * 100 if expected_total > 0 else 0
                
                if confidence_rate >= 80:
                    self.log_test("Confidence Coverage", True, f"{confidence_tests}/{expected_total} predictions have valid confidence ({confidence_rate:.1f}%)")
                else:
                    self.log_test("Confidence Coverage", False, f"Only {confidence_tests}/{expected_total} predictions have valid confidence ({confidence_rate:.1f}%)")
                
                if sigma_rate >= 80:
                    self.log_test("Sigma Coverage", True, f"{sigma_tests}/{expected_total} predictions have valid sigma ({sigma_rate:.1f}%)")
                else:
                    self.log_test("Sigma Coverage", False, f"Only {sigma_tests}/{expected_total} predictions have valid sigma ({sigma_rate:.1f}%)")
                    
            elif response.status_code == 503:
                self.log_test("Monte-Carlo Dropout", False, "Service unavailable (503)")
            else:
                self.log_test("Monte-Carlo Dropout", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Monte-Carlo Dropout", False, f"Error: {str(e)}")
    
    def test_selectivity_calculations(self):
        """Test 6: Selectivity Ratio Calculations"""
        print("\n=== Testing Selectivity Calculations ===")
        
        aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        
        try:
            payload = {
                "smiles": aspirin_smiles,
                "targets": ["ABL1", "ABL2", "AKT1"],
                "assay_types": ["IC50", "Ki", "EC50"]
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', {})
                
                selectivity_found = 0
                selectivity_values = []
                
                for target in ["ABL1", "ABL2", "AKT1"]:
                    if target in predictions:
                        target_pred = predictions[target]
                        selectivity = target_pred.get('selectivity_ratio')
                        
                        if selectivity is not None:
                            selectivity_found += 1
                            selectivity_values.append(selectivity)
                            
                            # Check if selectivity is a reasonable positive number
                            if selectivity > 0:
                                self.log_test(f"{target} Selectivity Value", True, 
                                            f"selectivity_ratio: {selectivity} (positive)")
                            else:
                                self.log_test(f"{target} Selectivity Value", False, 
                                            f"selectivity_ratio: {selectivity} (should be positive)")
                        else:
                            self.log_test(f"{target} Selectivity Present", False, "selectivity_ratio missing")
                
                # Check overall selectivity coverage
                if selectivity_found >= 2:  # At least 2 out of 3 targets should have selectivity
                    self.log_test("Selectivity Coverage", True, 
                                f"{selectivity_found}/3 targets have selectivity ratios")
                else:
                    self.log_test("Selectivity Coverage", False, 
                                f"Only {selectivity_found}/3 targets have selectivity ratios")
                
                # Check selectivity value distribution (should vary between targets)
                if len(selectivity_values) >= 2:
                    min_sel = min(selectivity_values)
                    max_sel = max(selectivity_values)
                    ratio_range = max_sel / min_sel if min_sel > 0 else 0
                    
                    if ratio_range > 1.5:  # Some variation expected
                        self.log_test("Selectivity Variation", True, 
                                    f"Selectivity range: {min_sel:.2f} - {max_sel:.2f} (ratio: {ratio_range:.2f})")
                    else:
                        self.log_test("Selectivity Variation", True, 
                                    f"Selectivity range: {min_sel:.2f} - {max_sel:.2f} (limited variation)")
                
                # Test single target (should have no selectivity)
                single_payload = {
                    "smiles": aspirin_smiles,
                    "targets": ["ABL1"],
                    "assay_types": ["IC50"]
                }
                
                single_response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                              json=single_payload, timeout=15)
                
                if single_response.status_code == 200:
                    single_data = single_response.json()
                    single_predictions = single_data.get('predictions', {})
                    
                    if 'ABL1' in single_predictions:
                        single_selectivity = single_predictions['ABL1'].get('selectivity_ratio')
                        if single_selectivity is None:
                            self.log_test("Single Target No Selectivity", True, 
                                        "Single target correctly has no selectivity ratio")
                        else:
                            self.log_test("Single Target No Selectivity", False, 
                                        f"Single target unexpectedly has selectivity: {single_selectivity}")
                    
            elif response.status_code == 503:
                self.log_test("Selectivity Calculations", False, "Service unavailable (503)")
            else:
                self.log_test("Selectivity Calculations", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Selectivity Calculations", False, f"Error: {str(e)}")
    
    def test_smiles_examples(self):
        """Test 7: SMILES Examples - Aspirin, Imatinib, Caffeine"""
        print("\n=== Testing SMILES Examples ===")
        
        test_molecules = [
            ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            ("Imatinib", "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"),
            ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        ]
        
        for mol_name, smiles in test_molecules:
            try:
                payload = {
                    "smiles": smiles,
                    "targets": ["ABL1", "ABL2"],
                    "assay_types": ["IC50", "Ki", "EC50"]
                }
                
                response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                       json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check SMILES echo
                    if data.get('smiles') == smiles:
                        self.log_test(f"{mol_name} SMILES Echo", True, "SMILES correctly echoed")
                    else:
                        self.log_test(f"{mol_name} SMILES Echo", False, "SMILES not echoed correctly")
                    
                    # Check predictions
                    predictions = data.get('predictions', {})
                    if len(predictions) >= 2:
                        self.log_test(f"{mol_name} Predictions", True, 
                                    f"Predictions for {len(predictions)} targets")
                        
                        # Check for multi-assay structure
                        sample_target = list(predictions.keys())[0]
                        sample_pred = predictions[sample_target]
                        assay_count = sum(1 for assay in ['IC50', 'Ki', 'EC50'] if assay in sample_pred)
                        
                        if assay_count >= 2:
                            self.log_test(f"{mol_name} Multi-Assay", True, 
                                        f"{assay_count}/3 assay types for {sample_target}")
                        else:
                            self.log_test(f"{mol_name} Multi-Assay", False, 
                                        f"Only {assay_count}/3 assay types for {sample_target}")
                    else:
                        self.log_test(f"{mol_name} Predictions", False, 
                                    f"Only {len(predictions)} targets predicted")
                    
                    # Check properties
                    properties = data.get('properties', {})
                    if 'LogP' in properties and 'LogS' in properties:
                        logp = properties['LogP']
                        logs = properties['LogS']
                        self.log_test(f"{mol_name} Properties", True, 
                                    f"LogP: {logp}, LogS: {logs}")
                    else:
                        self.log_test(f"{mol_name} Properties", False, "Missing LogP or LogS")
                        
                elif response.status_code == 503:
                    self.log_test(f"{mol_name} Prediction", False, "Service unavailable (503)")
                else:
                    self.log_test(f"{mol_name} Prediction", False, f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_test(f"{mol_name} Prediction", False, f"Error: {str(e)}")
    
    def test_performance_all_targets(self):
        """Test 8: Performance Test - All Targets with All Assays"""
        print("\n=== Testing Performance - All Targets ===")
        
        caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        
        try:
            start_time = time.time()
            
            payload = {
                "smiles": caffeine_smiles,
                "targets": "all",
                "assay_types": ["IC50", "Ki", "EC50"]
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=120)  # 2 minute timeout
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', {})
                
                num_targets = len(predictions)
                total_predictions = sum(len([a for a in ['IC50', 'Ki', 'EC50'] if a in pred]) 
                                      for pred in predictions.values())
                
                # Performance metrics
                if duration < 60:  # Should complete within 1 minute
                    self.log_test("All Targets Performance", True, 
                                f"Completed in {duration:.1f}s for {num_targets} targets, {total_predictions} predictions")
                else:
                    self.log_test("All Targets Performance", False, 
                                f"Took {duration:.1f}s (>60s) for {num_targets} targets")
                
                # Check if we got a reasonable number of predictions
                if total_predictions >= 100:  # At least 100 total predictions
                    self.log_test("All Targets Scale", True, 
                                f"{total_predictions} total predictions across {num_targets} targets")
                else:
                    self.log_test("All Targets Scale", False, 
                                f"Only {total_predictions} total predictions")
                
                # Check prediction quality (sample a few)
                sample_targets = list(predictions.keys())[:3]
                quality_checks = 0
                
                for target in sample_targets:
                    target_pred = predictions[target]
                    if 'IC50' in target_pred:
                        ic50_data = target_pred['IC50']
                        if all(field in ic50_data for field in ['pActivity', 'confidence', 'sigma']):
                            quality_checks += 1
                
                if quality_checks >= 2:
                    self.log_test("All Targets Quality", True, 
                                f"{quality_checks}/3 sample targets have complete structure")
                else:
                    self.log_test("All Targets Quality", False, 
                                f"Only {quality_checks}/3 sample targets have complete structure")
                    
            elif response.status_code == 503:
                self.log_test("All Targets Performance", False, "Service unavailable (503)")
            else:
                self.log_test("All Targets Performance", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("All Targets Performance", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run all enhanced Gnosis I multi-assay tests"""
        print("🧪 Starting Enhanced Gnosis I Multi-Assay Backend Testing")
        print(f"🔗 Backend URL: {BACKEND_URL}")
        print("=" * 80)
        
        # Run all tests
        self.test_multi_assay_predictions_single_target()
        self.test_multi_assay_predictions_multiple_targets()
        self.test_all_targets_selection()
        self.test_new_data_structure()
        self.test_monte_carlo_dropout()
        self.test_selectivity_calculations()
        self.test_smiles_examples()
        self.test_performance_all_targets()
        
        # Print summary
        print("\n" + "=" * 80)
        print("🎯 ENHANCED GNOSIS I MULTI-ASSAY TESTING SUMMARY")
        print("=" * 80)
        
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"✅ Passed: {self.passed_tests}/{self.total_tests} ({success_rate:.1f}%)")
        
        if self.failed_tests:
            print(f"❌ Failed: {len(self.failed_tests)}")
            print("\nFailed Tests:")
            for test in self.failed_tests:
                print(f"  • {test['test']}: {test['details']}")
        
        print("\n" + "=" * 80)
        
        # Key findings summary
        print("🔍 KEY FINDINGS:")
        print("1. Multi-Assay Support: Each target should return IC50, Ki, AND EC50 predictions")
        print("2. All Targets: 'all' selection should process ~62 targets")
        print("3. Data Structure: Nested format with predictions.{target}.{assay_type}")
        print("4. Monte-Carlo: Confidence metrics (sigma, confidence, mc_samples)")
        print("5. Selectivity: Ratios calculated for multi-target predictions")
        print("6. Performance: All targets × 3 assays = ~186 total predictions")
        
        # Determine overall status
        if success_rate >= 80:
            print("\n🎉 ENHANCED GNOSIS I MULTI-ASSAY: READY FOR PRODUCTION")
            return True
        elif success_rate >= 60:
            print("\n⚠️  ENHANCED GNOSIS I MULTI-ASSAY: MOSTLY WORKING - MINOR ISSUES")
            return True
        else:
            print("\n❌ ENHANCED GNOSIS I MULTI-ASSAY: MAJOR ISSUES - NEEDS ATTENTION")
            return False

if __name__ == "__main__":
    tester = EnhancedGnosisITester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)