#!/usr/bin/env python3
"""
Gnosis I Backend Testing - Focus on Binding_IC50 and Functional_IC50 Assay Types
Tests the updated Gnosis I backend system to verify new assay type support
"""

import requests
import json
import time
import sys
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment - use localhost for testing
BACKEND_URL = 'http://localhost:8001'  # Force localhost for backend testing
API_BASE = f"{BACKEND_URL}/api"

class GnosisIAssayTester:
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
    
    def test_gnosis_info_endpoint(self):
        """Test 1: Gnosis I Info Endpoint - GET /api/gnosis-i/info"""
        print("\n=== Testing Gnosis I Info Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/gnosis-i/info", timeout=10)
            
            if response.status_code == 200:
                info_data = response.json()
                
                # Check if model is available
                if info_data.get('available'):
                    self.log_test("Gnosis I Info Available", True, f"Model: {info_data.get('model_name')}")
                    
                    # Check R² score
                    r2_score = info_data.get('r2_score', 0.0)
                    if r2_score > 0.5:
                        self.log_test("Gnosis I R² Score", True, f"R² = {r2_score}")
                    else:
                        self.log_test("Gnosis I R² Score", False, f"R² = {r2_score} (too low)")
                    
                    # Check capabilities include IC50 and Ki prediction
                    capabilities = info_data.get('capabilities', [])
                    has_ic50 = any('IC50' in cap for cap in capabilities)
                    has_ki = any('Ki' in cap for cap in capabilities)
                    
                    self.log_test("IC50 Capability Listed", has_ic50, f"IC50 in capabilities: {has_ic50}")
                    self.log_test("Ki Capability Listed", has_ki, f"Ki in capabilities: {has_ki}")
                    
                else:
                    self.log_test("Gnosis I Info Available", False, f"Model not available: {info_data.get('message')}")
                    
            elif response.status_code == 503:
                self.log_test("Gnosis I Info Available", False, "Service unavailable (503)")
            else:
                self.log_test("Gnosis I Info Available", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Gnosis I Info Available", False, f"Error: {str(e)}")
    
    def test_gnosis_targets_endpoint(self):
        """Test 2: Gnosis I Targets Endpoint - GET /api/gnosis-i/targets"""
        print("\n=== Testing Gnosis I Targets Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/gnosis-i/targets", timeout=10)
            
            if response.status_code == 200:
                targets_data = response.json()
                
                # Check available targets
                available_targets = targets_data.get('available_targets', [])
                if available_targets:
                    self.log_test("Gnosis I Available Targets", True, 
                                f"Found {len(available_targets)} targets")
                else:
                    self.log_test("Gnosis I Available Targets", False, "No available targets found")
                
                # Check categorized targets
                categorized = targets_data.get('categorized_targets', {})
                oncoproteins = categorized.get('oncoproteins', [])
                
                if oncoproteins:
                    self.log_test("Gnosis I Oncoproteins", True, 
                                f"Found {len(oncoproteins)} oncoproteins")
                else:
                    self.log_test("Gnosis I Oncoproteins", False, "No oncoproteins found")
                
                # Check total targets
                total_targets = targets_data.get('total_targets', 0)
                self.log_test("Gnosis I Total Targets", total_targets > 0, f"Total targets: {total_targets}")
                    
            elif response.status_code == 503:
                self.log_test("Gnosis I Targets Endpoint", False, "Service unavailable (503)")
            else:
                self.log_test("Gnosis I Targets Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Gnosis I Targets Endpoint", False, f"Error: {str(e)}")
    
    def test_aspirin_binding_functional_ic50(self):
        """Test 3: Aspirin - Verify Binding_IC50 and Functional_IC50 Predictions"""
        print("\n=== Testing Aspirin - Binding_IC50 and Functional_IC50 ===")
        
        aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        
        try:
            payload = {
                "smiles": aspirin_smiles,
                "targets": ["EGFR", "BRAF"],  # Test with specific targets
                "assay_types": ["IC50", "Ki", "EC50"]  # Request all types
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                prediction_data = response.json()
                
                # Check SMILES echo
                if prediction_data.get('smiles') == aspirin_smiles:
                    self.log_test("Aspirin SMILES Echo", True, "SMILES correctly echoed")
                else:
                    self.log_test("Aspirin SMILES Echo", False, "SMILES not echoed correctly")
                
                # Check predictions structure
                predictions = prediction_data.get('predictions', {})
                if predictions:
                    self.log_test("Aspirin Predictions Structure", True, f"Predictions for {len(predictions)} targets")
                    
                    # Check for Binding_IC50 and Functional_IC50 assay types
                    binding_ic50_found = False
                    functional_ic50_found = False
                    ki_not_trained_found = False
                    
                    for target, target_preds in predictions.items():
                        print(f"   Target {target} predictions: {list(target_preds.keys())}")
                        
                        # Check for Binding_IC50
                        if 'Binding_IC50' in target_preds:
                            binding_ic50_found = True
                            binding_pred = target_preds['Binding_IC50']
                            
                            # Verify structure
                            required_fields = ['pActivity', 'activity_nM', 'confidence', 'sigma', 'assay_type']
                            has_all_fields = all(field in binding_pred for field in required_fields)
                            
                            if has_all_fields:
                                self.log_test(f"Aspirin {target} Binding_IC50 Structure", True, 
                                            f"pActivity={binding_pred.get('pActivity')}, confidence={binding_pred.get('confidence')}")
                            else:
                                missing = [f for f in required_fields if f not in binding_pred]
                                self.log_test(f"Aspirin {target} Binding_IC50 Structure", False, 
                                            f"Missing fields: {missing}")
                        
                        # Check for Functional_IC50
                        if 'Functional_IC50' in target_preds:
                            functional_ic50_found = True
                            functional_pred = target_preds['Functional_IC50']
                            
                            # Verify structure
                            required_fields = ['pActivity', 'activity_nM', 'confidence', 'sigma', 'assay_type']
                            has_all_fields = all(field in functional_pred for field in required_fields)
                            
                            if has_all_fields:
                                self.log_test(f"Aspirin {target} Functional_IC50 Structure", True, 
                                            f"pActivity={functional_pred.get('pActivity')}, confidence={functional_pred.get('confidence')}")
                            else:
                                missing = [f for f in required_fields if f not in functional_pred]
                                self.log_test(f"Aspirin {target} Functional_IC50 Structure", False, 
                                            f"Missing fields: {missing}")
                        
                        # Check for Ki "not_trained" handling
                        if 'Ki' in target_preds:
                            ki_pred = target_preds['Ki']
                            quality_flag = ki_pred.get('quality_flag')
                            confidence = ki_pred.get('confidence', 1.0)
                            
                            if quality_flag == 'not_trained' and confidence == 0.0:
                                ki_not_trained_found = True
                                self.log_test(f"Aspirin {target} Ki Not Trained Flag", True, 
                                            f"Ki correctly flagged as not_trained with 0% confidence")
                            else:
                                self.log_test(f"Aspirin {target} Ki Not Trained Flag", False, 
                                            f"Ki quality_flag={quality_flag}, confidence={confidence}")
                    
                    # Summary checks
                    self.log_test("Aspirin Binding_IC50 Found", binding_ic50_found, 
                                f"Binding_IC50 assay type present: {binding_ic50_found}")
                    self.log_test("Aspirin Functional_IC50 Found", functional_ic50_found, 
                                f"Functional_IC50 assay type present: {functional_ic50_found}")
                    self.log_test("Aspirin Ki Not Trained Handling", ki_not_trained_found, 
                                f"Ki properly flagged as not_trained: {ki_not_trained_found}")
                    
                else:
                    self.log_test("Aspirin Predictions Structure", False, "No predictions returned")
                    
            elif response.status_code == 400:
                self.log_test("Aspirin Prediction", False, f"Bad request (400): {response.text}")
            elif response.status_code == 503:
                self.log_test("Aspirin Prediction", False, "Service unavailable (503)")
            else:
                self.log_test("Aspirin Prediction", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Aspirin Prediction", False, f"Error: {str(e)}")
    
    def test_imatinib_binding_functional_ic50(self):
        """Test 4: Imatinib - Verify Binding_IC50 and Functional_IC50 Predictions"""
        print("\n=== Testing Imatinib - Binding_IC50 and Functional_IC50 ===")
        
        imatinib_smiles = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
        
        try:
            payload = {
                "smiles": imatinib_smiles,
                "targets": ["EGFR", "ABL1", "KIT"],  # Test with specific targets including ABL1 (imatinib's main target)
                "assay_types": ["IC50", "Ki", "EC50"]
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                prediction_data = response.json()
                
                # Check SMILES echo
                if prediction_data.get('smiles') == imatinib_smiles:
                    self.log_test("Imatinib SMILES Echo", True, "SMILES correctly echoed")
                else:
                    self.log_test("Imatinib SMILES Echo", False, "SMILES not echoed correctly")
                
                # Check predictions structure
                predictions = prediction_data.get('predictions', {})
                if predictions:
                    self.log_test("Imatinib Predictions Structure", True, f"Predictions for {len(predictions)} targets")
                    
                    # Check for proper assay type structure
                    binding_ic50_count = 0
                    functional_ic50_count = 0
                    ki_not_trained_count = 0
                    
                    for target, target_preds in predictions.items():
                        print(f"   Target {target} predictions: {list(target_preds.keys())}")
                        
                        # Check Binding_IC50
                        if 'Binding_IC50' in target_preds:
                            binding_ic50_count += 1
                            binding_pred = target_preds['Binding_IC50']
                            
                            # Verify reasonable values for imatinib (should be potent)
                            activity_nm = binding_pred.get('activity_nM', 0)
                            confidence = binding_pred.get('confidence', 0)
                            
                            if activity_nm > 0 and confidence > 0:
                                self.log_test(f"Imatinib {target} Binding_IC50 Values", True, 
                                            f"activity_nM={activity_nm}, confidence={confidence}")
                            else:
                                self.log_test(f"Imatinib {target} Binding_IC50 Values", False, 
                                            f"Invalid values: activity_nM={activity_nm}, confidence={confidence}")
                        
                        # Check Functional_IC50
                        if 'Functional_IC50' in target_preds:
                            functional_ic50_count += 1
                            functional_pred = target_preds['Functional_IC50']
                            
                            # Verify reasonable values
                            activity_nm = functional_pred.get('activity_nM', 0)
                            confidence = functional_pred.get('confidence', 0)
                            
                            if activity_nm > 0 and confidence > 0:
                                self.log_test(f"Imatinib {target} Functional_IC50 Values", True, 
                                            f"activity_nM={activity_nm}, confidence={confidence}")
                            else:
                                self.log_test(f"Imatinib {target} Functional_IC50 Values", False, 
                                            f"Invalid values: activity_nM={activity_nm}, confidence={confidence}")
                        
                        # Check Ki handling
                        if 'Ki' in target_preds:
                            ki_pred = target_preds['Ki']
                            quality_flag = ki_pred.get('quality_flag')
                            confidence = ki_pred.get('confidence', 1.0)
                            
                            if quality_flag == 'not_trained' and confidence == 0.0:
                                ki_not_trained_count += 1
                    
                    # Summary checks
                    self.log_test("Imatinib Binding_IC50 Count", binding_ic50_count > 0, 
                                f"Binding_IC50 predictions: {binding_ic50_count}")
                    self.log_test("Imatinib Functional_IC50 Count", functional_ic50_count > 0, 
                                f"Functional_IC50 predictions: {functional_ic50_count}")
                    self.log_test("Imatinib Ki Not Trained Count", ki_not_trained_count > 0, 
                                f"Ki not_trained flags: {ki_not_trained_count}")
                    
                else:
                    self.log_test("Imatinib Predictions Structure", False, "No predictions returned")
                    
            elif response.status_code == 400:
                self.log_test("Imatinib Prediction", False, f"Bad request (400): {response.text}")
            elif response.status_code == 503:
                self.log_test("Imatinib Prediction", False, "Service unavailable (503)")
            else:
                self.log_test("Imatinib Prediction", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Imatinib Prediction", False, f"Error: {str(e)}")
    
    def test_all_targets_prediction(self):
        """Test 5: All Targets Prediction - Verify Scale and Performance"""
        print("\n=== Testing All Targets Prediction ===")
        
        test_smiles = "CCO"  # Simple ethanol for quick test
        
        try:
            payload = {
                "smiles": test_smiles,
                "targets": "all",  # Request all targets
                "assay_types": ["IC50", "Ki", "EC50"]
            }
            
            start_time = time.time()
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=60)  # Longer timeout for all targets
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                prediction_data = response.json()
                
                # Check predictions structure
                predictions = prediction_data.get('predictions', {})
                num_targets = len(predictions)
                
                if num_targets > 0:
                    self.log_test("All Targets Prediction Count", True, 
                                f"Predictions for {num_targets} targets in {response_time:.1f}s")
                    
                    # Count assay types
                    binding_ic50_total = 0
                    functional_ic50_total = 0
                    ki_not_trained_total = 0
                    
                    for target, target_preds in predictions.items():
                        if 'Binding_IC50' in target_preds:
                            binding_ic50_total += 1
                        if 'Functional_IC50' in target_preds:
                            functional_ic50_total += 1
                        if 'Ki' in target_preds and target_preds['Ki'].get('quality_flag') == 'not_trained':
                            ki_not_trained_total += 1
                    
                    self.log_test("All Targets Binding_IC50 Coverage", binding_ic50_total == num_targets, 
                                f"Binding_IC50 for {binding_ic50_total}/{num_targets} targets")
                    self.log_test("All Targets Functional_IC50 Coverage", functional_ic50_total == num_targets, 
                                f"Functional_IC50 for {functional_ic50_total}/{num_targets} targets")
                    self.log_test("All Targets Ki Not Trained Coverage", ki_not_trained_total == num_targets, 
                                f"Ki not_trained for {ki_not_trained_total}/{num_targets} targets")
                    
                    # Check response time performance
                    if response_time < 30:
                        self.log_test("All Targets Response Time", True, f"Response time: {response_time:.1f}s (good)")
                    elif response_time < 60:
                        self.log_test("All Targets Response Time", True, f"Response time: {response_time:.1f}s (acceptable)")
                    else:
                        self.log_test("All Targets Response Time", False, f"Response time: {response_time:.1f}s (too slow)")
                    
                else:
                    self.log_test("All Targets Prediction Count", False, "No predictions returned")
                    
            elif response.status_code == 400:
                self.log_test("All Targets Prediction", False, f"Bad request (400): {response.text}")
            elif response.status_code == 503:
                self.log_test("All Targets Prediction", False, "Service unavailable (503)")
            else:
                self.log_test("All Targets Prediction", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("All Targets Prediction", False, f"Error: {str(e)}")
    
    def test_confidence_and_quality_flags(self):
        """Test 6: Confidence Scores and Quality Flags"""
        print("\n=== Testing Confidence Scores and Quality Flags ===")
        
        test_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
        
        try:
            payload = {
                "smiles": test_smiles,
                "targets": ["EGFR", "BRAF"],
                "assay_types": ["IC50", "Ki", "EC50"]
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                prediction_data = response.json()
                
                predictions = prediction_data.get('predictions', {})
                if predictions:
                    confidence_scores_valid = True
                    quality_flags_valid = True
                    
                    for target, target_preds in predictions.items():
                        # Check Binding_IC50 confidence
                        if 'Binding_IC50' in target_preds:
                            binding_pred = target_preds['Binding_IC50']
                            confidence = binding_pred.get('confidence', 0)
                            sigma = binding_pred.get('sigma', 0)
                            quality_flag = binding_pred.get('quality_flag', '')
                            
                            # Confidence should be between 0 and 1
                            if not (0 <= confidence <= 1):
                                confidence_scores_valid = False
                                print(f"   Invalid Binding_IC50 confidence for {target}: {confidence}")
                            
                            # Sigma should be positive
                            if sigma < 0:
                                confidence_scores_valid = False
                                print(f"   Invalid Binding_IC50 sigma for {target}: {sigma}")
                            
                            # Quality flag should be reasonable
                            if quality_flag not in ['good', 'uncertain', 'not_trained']:
                                quality_flags_valid = False
                                print(f"   Invalid Binding_IC50 quality_flag for {target}: {quality_flag}")
                        
                        # Check Functional_IC50 confidence
                        if 'Functional_IC50' in target_preds:
                            functional_pred = target_preds['Functional_IC50']
                            confidence = functional_pred.get('confidence', 0)
                            sigma = functional_pred.get('sigma', 0)
                            quality_flag = functional_pred.get('quality_flag', '')
                            
                            # Confidence should be between 0 and 1
                            if not (0 <= confidence <= 1):
                                confidence_scores_valid = False
                                print(f"   Invalid Functional_IC50 confidence for {target}: {confidence}")
                            
                            # Sigma should be positive
                            if sigma < 0:
                                confidence_scores_valid = False
                                print(f"   Invalid Functional_IC50 sigma for {target}: {sigma}")
                        
                        # Check Ki not_trained handling
                        if 'Ki' in target_preds:
                            ki_pred = target_preds['Ki']
                            confidence = ki_pred.get('confidence', 1.0)
                            quality_flag = ki_pred.get('quality_flag', '')
                            
                            # Ki should have 0 confidence and not_trained flag
                            if confidence != 0.0 or quality_flag != 'not_trained':
                                quality_flags_valid = False
                                print(f"   Invalid Ki handling for {target}: confidence={confidence}, flag={quality_flag}")
                    
                    self.log_test("Confidence Scores Valid", confidence_scores_valid, 
                                f"All confidence scores in valid range (0-1)")
                    self.log_test("Quality Flags Valid", quality_flags_valid, 
                                f"All quality flags properly set")
                    
                else:
                    self.log_test("Confidence and Quality Test", False, "No predictions returned")
                    
            elif response.status_code == 503:
                self.log_test("Confidence and Quality Test", False, "Service unavailable (503)")
            else:
                self.log_test("Confidence and Quality Test", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Confidence and Quality Test", False, f"Error: {str(e)}")
    
    def test_molecular_properties(self):
        """Test 7: Molecular Properties Calculation"""
        print("\n=== Testing Molecular Properties Calculation ===")
        
        test_molecules = [
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin"),
            ("Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C", "Imatinib"),
            ("CCO", "Ethanol")
        ]
        
        for smiles, name in test_molecules:
            try:
                payload = {
                    "smiles": smiles,
                    "targets": ["EGFR"],  # Single target for quick test
                    "assay_types": ["IC50"]
                }
                
                response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                       json=payload, timeout=15)
                
                if response.status_code == 200:
                    prediction_data = response.json()
                    
                    properties = prediction_data.get('properties', {})
                    logp = properties.get('LogP')
                    logs = properties.get('LogS')
                    
                    if logp is not None and logs is not None:
                        self.log_test(f"{name} Properties Calculation", True, 
                                    f"LogP={logp}, LogS={logs}")
                        
                        # Basic sanity checks
                        if -5 <= logp <= 10 and -10 <= logs <= 5:
                            self.log_test(f"{name} Properties Range", True, 
                                        f"Properties in reasonable range")
                        else:
                            self.log_test(f"{name} Properties Range", False, 
                                        f"Properties out of range: LogP={logp}, LogS={logs}")
                    else:
                        self.log_test(f"{name} Properties Calculation", False, 
                                    f"Missing properties: LogP={logp}, LogS={logs}")
                        
                elif response.status_code == 503:
                    self.log_test(f"{name} Properties Test", False, "Service unavailable (503)")
                else:
                    self.log_test(f"{name} Properties Test", False, f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_test(f"{name} Properties Test", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run all Gnosis I assay type tests"""
        print("🧪 Starting Gnosis I Binding_IC50 and Functional_IC50 Backend Testing")
        print(f"🔗 Backend URL: {BACKEND_URL}")
        print("=" * 80)
        
        # Run all tests
        self.test_gnosis_info_endpoint()
        self.test_gnosis_targets_endpoint()
        self.test_aspirin_binding_functional_ic50()
        self.test_imatinib_binding_functional_ic50()
        self.test_all_targets_prediction()
        self.test_confidence_and_quality_flags()
        self.test_molecular_properties()
        
        # Print summary
        print("\n" + "=" * 80)
        print("🎯 GNOSIS I ASSAY TYPE TESTING SUMMARY")
        print("=" * 80)
        
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"✅ Passed: {self.passed_tests}/{self.total_tests} ({success_rate:.1f}%)")
        
        if self.failed_tests:
            print(f"❌ Failed: {len(self.failed_tests)}")
            print("\nFailed Tests:")
            for test in self.failed_tests:
                print(f"  • {test['test']}: {test['details']}")
        
        print("\n" + "=" * 80)
        
        # Determine overall status
        if success_rate >= 80:
            print("🎉 GNOSIS I BINDING_IC50 & FUNCTIONAL_IC50: WORKING CORRECTLY")
            return True
        elif success_rate >= 60:
            print("⚠️  GNOSIS I ASSAY TYPES: MOSTLY WORKING - MINOR ISSUES")
            return True
        else:
            print("❌ GNOSIS I ASSAY TYPES: MAJOR ISSUES - NEEDS ATTENTION")
            return False

if __name__ == "__main__":
    tester = GnosisIAssayTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)