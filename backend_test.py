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
    
    def test_health_endpoint(self):
        """Test the /api/health endpoint"""
        print("\n=== Testing Health Check Endpoint ===")
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['status', 'models_loaded', 'available_predictions']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Health endpoint structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check if models are loaded
                models_loaded = data.get('models_loaded', [])
                expected_predictions = ['bioactivity_ic50', 'toxicity', 'logP', 'solubility']
                available_predictions = data.get('available_predictions', [])
                
                self.log_test("Health endpoint response", True, f"Status: {data['status']}")
                self.log_test("Models loaded check", len(models_loaded) > 0, f"Models: {models_loaded}")
                self.log_test("Available predictions", set(expected_predictions).issubset(set(available_predictions)), 
                            f"Available: {available_predictions}")
                
                return True
            else:
                self.log_test("Health endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Health endpoint connectivity", False, f"Connection error: {str(e)}")
            return False
    
    def test_smiles_validation(self):
        """Test SMILES validation with valid molecules"""
        print("\n=== Testing SMILES Validation ===")
        
        valid_molecules = [
            ("CCO", "ethanol"),
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine")
        ]
        
        all_passed = True
        
        for smiles, name in valid_molecules:
            try:
                payload = {
                    "smiles": smiles,
                    "prediction_types": ["logP"]
                }
                
                response = requests.post(f"{API_BASE}/predict", 
                                       json=payload, 
                                       headers={'Content-Type': 'application/json'},
                                       timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test(f"Valid SMILES - {name}", True, f"Successfully processed {smiles}")
                else:
                    self.log_test(f"Valid SMILES - {name}", False, f"HTTP {response.status_code}: {response.text}")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"Valid SMILES - {name}", False, f"Request error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_prediction_types(self):
        """Test all four priority prediction types"""
        print("\n=== Testing Prediction Types ===")
        
        test_smiles = "CCO"  # ethanol
        prediction_types = ["bioactivity_ic50", "toxicity", "logP", "solubility"]
        
        all_passed = True
        
        for pred_type in prediction_types:
            try:
                payload = {
                    "smiles": test_smiles,
                    "prediction_types": [pred_type]
                }
                
                response = requests.post(f"{API_BASE}/predict", 
                                       json=payload, 
                                       headers={'Content-Type': 'application/json'},
                                       timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check response structure
                    if 'results' in data and 'summary' in data:
                        results = data['results']
                        if len(results) > 0:
                            result = results[0]
                            
                            # Check if prediction was made
                            has_molbert = result.get('molbert_prediction') is not None
                            has_chemprop = result.get('chemprop_prediction') is not None
                            
                            self.log_test(f"Prediction type - {pred_type}", True, 
                                        f"MolBERT: {has_molbert}, Chemprop: {has_chemprop}")
                        else:
                            self.log_test(f"Prediction type - {pred_type}", False, "No results returned")
                            all_passed = False
                    else:
                        self.log_test(f"Prediction type - {pred_type}", False, "Invalid response structure")
                        all_passed = False
                else:
                    self.log_test(f"Prediction type - {pred_type}", False, f"HTTP {response.status_code}: {response.text}")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"Prediction type - {pred_type}", False, f"Request error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_error_handling(self):
        """Test error handling with invalid SMILES strings"""
        print("\n=== Testing Error Handling ===")
        
        invalid_smiles = [
            "INVALID_SMILES",
            "C[C@H](C)C(=O)O[C@H]1C[C@@H]2CC[C@H]1N2C(=O)C3=CC=CC=C3INVALID",
            "",
            "123456789"
        ]
        
        all_passed = True
        
        for invalid_smiles_str in invalid_smiles:
            try:
                payload = {
                    "smiles": invalid_smiles_str,
                    "prediction_types": ["logP"]
                }
                
                response = requests.post(f"{API_BASE}/predict", 
                                       json=payload, 
                                       headers={'Content-Type': 'application/json'},
                                       timeout=30)
                
                if response.status_code == 400:
                    self.log_test(f"Invalid SMILES error handling", True, f"Correctly rejected: {invalid_smiles_str}")
                else:
                    self.log_test(f"Invalid SMILES error handling", False, 
                                f"Should have returned 400, got {response.status_code}")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"Invalid SMILES error handling", False, f"Request error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_model_integration(self):
        """Test MolBERT and Chemprop model integration"""
        print("\n=== Testing Model Integration ===")
        
        test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin
        
        try:
            payload = {
                "smiles": test_smiles,
                "prediction_types": ["bioactivity_ic50", "toxicity", "logP", "solubility"]
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                molbert_predictions = 0
                chemprop_predictions = 0
                
                for result in results:
                    if result.get('molbert_prediction') is not None:
                        molbert_predictions += 1
                    if result.get('chemprop_prediction') is not None:
                        chemprop_predictions += 1
                
                self.log_test("MolBERT integration", molbert_predictions > 0, 
                            f"MolBERT predictions: {molbert_predictions}/{len(results)}")
                self.log_test("Chemprop integration", chemprop_predictions > 0, 
                            f"Chemprop predictions: {chemprop_predictions}/{len(results)}")
                
                return molbert_predictions > 0 and chemprop_predictions > 0
            else:
                self.log_test("Model integration", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Model integration", False, f"Request error: {str(e)}")
            return False
    
    def test_database_storage(self):
        """Test database storage functionality"""
        print("\n=== Testing Database Storage ===")
        
        test_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine
        
        try:
            # Make a prediction
            payload = {
                "smiles": test_smiles,
                "prediction_types": ["logP", "solubility"]
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                if len(results) > 0:
                    prediction_id = results[0].get('id')
                    
                    # Test history endpoint
                    history_response = requests.get(f"{API_BASE}/predictions/history", timeout=30)
                    
                    if history_response.status_code == 200:
                        history_data = history_response.json()
                        self.log_test("Database storage - history", True, f"Retrieved {len(history_data)} records")
                        
                        # Test specific prediction retrieval
                        if prediction_id:
                            pred_response = requests.get(f"{API_BASE}/predictions/{prediction_id}", timeout=30)
                            
                            if pred_response.status_code == 200:
                                self.log_test("Database storage - specific prediction", True, 
                                            f"Retrieved prediction {prediction_id}")
                                return True
                            else:
                                self.log_test("Database storage - specific prediction", False, 
                                            f"HTTP {pred_response.status_code}")
                                return False
                        else:
                            self.log_test("Database storage - prediction ID", False, "No prediction ID returned")
                            return False
                    else:
                        self.log_test("Database storage - history", False, f"HTTP {history_response.status_code}")
                        return False
                else:
                    self.log_test("Database storage", False, "No results to store")
                    return False
            else:
                self.log_test("Database storage", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Database storage", False, f"Request error: {str(e)}")
            return False
    
    def test_response_format(self):
        """Test response format validation"""
        print("\n=== Testing Response Format ===")
        
        test_smiles = "CCO"  # ethanol
        
        try:
            payload = {
                "smiles": test_smiles,
                "prediction_types": ["logP", "toxicity"]
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check top-level structure
                required_top_level = ['results', 'summary']
                missing_top = [field for field in required_top_level if field not in data]
                
                if missing_top:
                    self.log_test("Response format - top level", False, f"Missing: {missing_top}")
                    return False
                
                # Check results structure
                results = data.get('results', [])
                if len(results) > 0:
                    result = results[0]
                    required_result_fields = ['id', 'smiles', 'prediction_type', 'confidence', 'timestamp']
                    missing_result = [field for field in required_result_fields if field not in result]
                    
                    if missing_result:
                        self.log_test("Response format - result fields", False, f"Missing: {missing_result}")
                        return False
                    
                    # Check summary structure
                    summary = data.get('summary', {})
                    required_summary = ['molecule', 'total_predictions', 'molecular_properties', 'prediction_types']
                    missing_summary = [field for field in required_summary if field not in summary]
                    
                    if missing_summary:
                        self.log_test("Response format - summary fields", False, f"Missing: {missing_summary}")
                        return False
                    
                    self.log_test("Response format validation", True, "All required fields present")
                    return True
                else:
                    self.log_test("Response format", False, "No results in response")
                    return False
            else:
                self.log_test("Response format", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Response format", False, f"Request error: {str(e)}")
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