#!/usr/bin/env python3
"""
Comprehensive Backend Testing for Upgraded Predictive Chemistry Platform
Tests real ChEMBL integration, IC50 predictions, and target-specific models
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

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

class ChemistryPlatformTester:
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
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
    
    def test_health_endpoint_real_chembl(self):
        """Test the /api/health endpoint for real ChEMBL integration"""
        print("\n=== Testing Health Check with Real ChEMBL Models ===")
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields for ChEMBL integration
                required_fields = ['status', 'models_loaded', 'available_predictions', 'available_targets', 'real_chemprop_ready']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Health endpoint structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check ChEMBL-specific fields
                available_targets = data.get('available_targets', [])
                expected_targets = ['EGFR', 'BRAF', 'CDK2']
                real_chemprop_ready = data.get('real_chemprop_ready', False)
                
                self.log_test("Health endpoint response", True, f"Status: {data['status']}")
                self.log_test("Real ChEMBL ready", real_chemprop_ready, f"ChEMBL integration: {real_chemprop_ready}")
                self.log_test("Available targets", len(available_targets) > 0, f"Targets: {available_targets}")
                
                return True
            else:
                self.log_test("Health endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Health endpoint connectivity", False, f"Connection error: {str(e)}")
            return False
    
    def test_targets_endpoint(self):
        """Test the /api/targets endpoint for protein target information"""
        print("\n=== Testing Targets Endpoint ===")
        try:
            response = requests.get(f"{API_BASE}/targets", timeout=30)
            
            if response.status_code == 200:
                targets = response.json()
                
                if not isinstance(targets, list):
                    self.log_test("Targets endpoint format", False, "Response should be a list")
                    return False
                
                if len(targets) == 0:
                    self.log_test("Targets availability", False, "No targets available")
                    return False
                
                # Check target structure
                for target in targets:
                    required_fields = ['target', 'available', 'training_size']
                    missing_fields = [field for field in required_fields if field not in target]
                    
                    if missing_fields:
                        self.log_test(f"Target {target.get('target', 'unknown')} structure", False, 
                                    f"Missing fields: {missing_fields}")
                        return False
                
                target_names = [t['target'] for t in targets]
                self.log_test("Targets endpoint", True, f"Retrieved {len(targets)} targets: {target_names}")
                return True
                
            else:
                self.log_test("Targets endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Targets endpoint", False, f"Request error: {str(e)}")
            return False
    
    def test_real_ic50_predictions(self):
        """Test real IC50 predictions with ChEMBL data"""
        print("\n=== Testing Real IC50 Predictions ===")
        
        # Test with ethanol as specified in the review request
        test_smiles = "CCO"  # ethanol
        test_target = "EGFR"
        
        try:
            payload = {
                "smiles": test_smiles,
                "prediction_types": ["bioactivity_ic50"],
                "target": test_target
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=120)  # Longer timeout for real predictions
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' not in data or len(data['results']) == 0:
                    self.log_test("Real IC50 prediction structure", False, "No results returned")
                    return False
                
                result = data['results'][0]
                
                # Check for real ChEMBL prediction
                real_prediction = result.get('real_chemprop_prediction')
                if not real_prediction:
                    self.log_test("Real ChEMBL prediction", False, "No real_chemprop_prediction field")
                    return False
                
                # Check required fields in real prediction
                required_real_fields = ['pic50', 'ic50_nm', 'confidence', 'similarity']
                missing_real_fields = [field for field in required_real_fields if field not in real_prediction]
                
                if missing_real_fields:
                    self.log_test("Real prediction fields", False, f"Missing: {missing_real_fields}")
                    return False
                
                # Validate prediction values
                pic50 = real_prediction.get('pic50')
                ic50_nm = real_prediction.get('ic50_nm')
                confidence = real_prediction.get('confidence')
                similarity = real_prediction.get('similarity')
                
                # Check value ranges
                valid_pic50 = isinstance(pic50, (int, float)) and 3.0 <= pic50 <= 12.0
                valid_ic50 = isinstance(ic50_nm, (int, float)) and ic50_nm > 0
                valid_confidence = isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
                valid_similarity = isinstance(similarity, (int, float)) and 0.0 <= similarity <= 1.0
                
                self.log_test("Real IC50 prediction values", 
                            valid_pic50 and valid_ic50 and valid_confidence and valid_similarity,
                            f"pIC50: {pic50}, IC50: {ic50_nm} nM, Confidence: {confidence}, Similarity: {similarity}")
                
                # Check for model performance data
                model_performance = real_prediction.get('model_performance')
                if model_performance:
                    self.log_test("Model performance data", True, 
                                f"R²: {model_performance.get('test_r2')}, RMSE: {model_performance.get('test_rmse')}")
                else:
                    self.log_test("Model performance data", False, "No model performance metrics")
                
                return True
                
            else:
                self.log_test("Real IC50 prediction", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Real IC50 prediction", False, f"Request error: {str(e)}")
            return False
    
    def test_model_initialization(self):
        """Test the /api/initialize-target/BRAF endpoint"""
        print("\n=== Testing Model Initialization ===")
        
        target = "BRAF"
        
        try:
            response = requests.post(f"{API_BASE}/initialize-target/{target}", timeout=180)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'message' in data:
                    self.log_test("BRAF model initialization", True, data['message'])
                    return True
                else:
                    self.log_test("BRAF model initialization", False, "No message in response")
                    return False
                    
            else:
                self.log_test("BRAF model initialization", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("BRAF model initialization", False, f"Request error: {str(e)}")
            return False
    
    def test_multi_target_predictions(self):
        """Test predictions for different targets with same molecule"""
        print("\n=== Testing Multi-Target Predictions ===")
        
        test_smiles = "CCO"  # ethanol
        targets = ["EGFR", "BRAF"]
        
        all_passed = True
        
        for target in targets:
            try:
                payload = {
                    "smiles": test_smiles,
                    "prediction_types": ["bioactivity_ic50"],
                    "target": target
                }
                
                response = requests.post(f"{API_BASE}/predict", 
                                       json=payload, 
                                       headers={'Content-Type': 'application/json'},
                                       timeout=120)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'results' in data and len(data['results']) > 0:
                        result = data['results'][0]
                        real_prediction = result.get('real_chemprop_prediction')
                        
                        if real_prediction and 'pic50' in real_prediction:
                            self.log_test(f"Multi-target prediction - {target}", True, 
                                        f"pIC50: {real_prediction['pic50']}")
                        else:
                            self.log_test(f"Multi-target prediction - {target}", False, 
                                        "No real prediction data")
                            all_passed = False
                    else:
                        self.log_test(f"Multi-target prediction - {target}", False, "No results")
                        all_passed = False
                else:
                    self.log_test(f"Multi-target prediction - {target}", False, 
                                f"HTTP {response.status_code}")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"Multi-target prediction - {target}", False, f"Request error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_chembl_data_integration(self):
        """Test ChEMBL data integration verification"""
        print("\n=== Testing ChEMBL Data Integration ===")
        
        test_smiles = "CCO"  # ethanol
        
        try:
            payload = {
                "smiles": test_smiles,
                "prediction_types": ["bioactivity_ic50"],
                "target": "EGFR"
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check summary for real models usage
                summary = data.get('summary', {})
                real_models_used = summary.get('real_models_used', False)
                
                if not real_models_used:
                    self.log_test("ChEMBL data integration", False, "Real models not used in prediction")
                    return False
                
                # Check result for ChEMBL-specific data
                result = data['results'][0]
                real_prediction = result.get('real_chemprop_prediction')
                
                if not real_prediction:
                    self.log_test("ChEMBL data integration", False, "No real ChEMBL prediction")
                    return False
                
                # Check for training size (indicates real data)
                training_size = real_prediction.get('training_size', 0)
                if training_size < 50:
                    self.log_test("ChEMBL training data", False, f"Insufficient training data: {training_size}")
                    return False
                
                self.log_test("ChEMBL data integration", True, 
                            f"Real models used, training size: {training_size}")
                return True
                
            else:
                self.log_test("ChEMBL data integration", False, f"HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("ChEMBL data integration", False, f"Request error: {str(e)}")
            return False
    
    def test_error_handling_invalid_targets(self):
        """Test error handling with invalid targets and SMILES"""
        print("\n=== Testing Error Handling ===")
        
        all_passed = True
        
        # Test invalid target
        try:
            payload = {
                "smiles": "CCO",
                "prediction_types": ["bioactivity_ic50"],
                "target": "INVALID_TARGET"
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            # Should still work but may have error in real_chemprop_prediction
            if response.status_code == 200:
                data = response.json()
                result = data['results'][0]
                real_prediction = result.get('real_chemprop_prediction', {})
                
                if 'error' in real_prediction:
                    self.log_test("Invalid target handling", True, "Error properly handled for invalid target")
                else:
                    self.log_test("Invalid target handling", True, "Invalid target handled gracefully")
            else:
                self.log_test("Invalid target handling", False, f"HTTP {response.status_code}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Invalid target handling", False, f"Request error: {str(e)}")
            all_passed = False
        
        # Test invalid SMILES
        try:
            payload = {
                "smiles": "INVALID_SMILES",
                "prediction_types": ["bioactivity_ic50"],
                "target": "EGFR"
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code == 400:
                self.log_test("Invalid SMILES handling", True, "Invalid SMILES properly rejected")
            else:
                self.log_test("Invalid SMILES handling", False, f"Should return 400, got {response.status_code}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Invalid SMILES handling", False, f"Request error: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def test_response_validation_chembl(self):
        """Test response validation for ChEMBL integration"""
        print("\n=== Testing Response Validation for ChEMBL ===")
        
        test_smiles = "CCO"  # ethanol
        
        try:
            payload = {
                "smiles": test_smiles,
                "prediction_types": ["bioactivity_ic50"],
                "target": "EGFR"
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check top-level structure
                required_top_level = ['results', 'summary']
                missing_top = [field for field in required_top_level if field not in data]
                
                if missing_top:
                    self.log_test("Response validation - top level", False, f"Missing: {missing_top}")
                    return False
                
                # Check results structure for ChEMBL fields
                results = data.get('results', [])
                if len(results) > 0:
                    result = results[0]
                    
                    # Check for real_chemprop_prediction
                    real_prediction = result.get('real_chemprop_prediction')
                    if not real_prediction:
                        self.log_test("Response validation - real prediction", False, "Missing real_chemprop_prediction")
                        return False
                    
                    # Check real prediction structure
                    required_real_fields = ['pic50', 'ic50_nm', 'confidence', 'similarity']
                    missing_real = [field for field in required_real_fields if field not in real_prediction]
                    
                    if missing_real:
                        self.log_test("Response validation - real fields", False, f"Missing: {missing_real}")
                        return False
                    
                    # Check summary for ChEMBL-specific fields
                    summary = data.get('summary', {})
                    required_summary = ['molecule', 'target', 'total_predictions', 'real_models_used']
                    missing_summary = [field for field in required_summary if field not in summary]
                    
                    if missing_summary:
                        self.log_test("Response validation - summary", False, f"Missing: {missing_summary}")
                        return False
                    
                    self.log_test("Response validation for ChEMBL", True, "All required ChEMBL fields present")
                    return True
                else:
                    self.log_test("Response validation", False, "No results in response")
                    return False
            else:
                self.log_test("Response validation", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Response validation", False, f"Request error: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all tests and provide summary"""
        print(f"🧪 Starting Comprehensive Backend Testing")
        print(f"Backend URL: {API_BASE}")
        print("=" * 60)
        
        # Run all tests
        tests = [
            self.test_health_endpoint,
            self.test_smiles_validation,
            self.test_prediction_types,
            self.test_error_handling,
            self.test_model_integration,
            self.test_database_storage,
            self.test_response_format
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"❌ CRITICAL ERROR in {test.__name__}: {str(e)}")
                self.failed_tests.append({
                    'test': test.__name__,
                    'status': '❌ CRITICAL ERROR',
                    'success': False,
                    'details': str(e)
                })
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏁 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t['success']])
        failed_tests = len(self.failed_tests)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%")
        
        if self.failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"  • {test['test']}: {test['details']}")
        
        return passed_tests, failed_tests, self.test_results

if __name__ == "__main__":
    tester = ChemistryPlatformTester()
    passed, failed, results = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)