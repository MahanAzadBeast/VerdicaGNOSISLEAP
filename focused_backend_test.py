#!/usr/bin/env python3
"""
Focused Backend Testing for ChemBERTa and Chemprop Multi-Task Endpoints
Tests the specific endpoints mentioned in the review request
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

class FocusedBackendTester:
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
        """Test /api/health endpoint - should show all models loaded"""
        print("\n=== Testing Health Check Endpoint ===")
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check basic structure
                status = data.get('status')
                models_loaded = data.get('models_loaded', {})
                
                self.log_test("Health endpoint status", status == 'healthy', f"Status: {status}")
                
                # Check all models loaded
                expected_models = ['molbert', 'chemprop_simulation', 'real_ml_models', 'oncoprotein_chemberta']
                all_models_loaded = all(models_loaded.get(model, False) for model in expected_models)
                
                self.log_test("All models loaded", all_models_loaded, f"Models: {models_loaded}")
                
                return True
            else:
                self.log_test("Health endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Health endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_chemberta_endpoints(self):
        """Test ChemBERTa Multi-Task endpoints"""
        print("\n=== Testing ChemBERTa Multi-Task Endpoints ===")
        
        # Test /api/chemberta/status
        try:
            response = requests.get(f"{API_BASE}/chemberta/status", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                available = data.get('available', False)
                model_info = data.get('model_info', {})
                
                self.log_test("ChemBERTa status endpoint", available, 
                            f"Status: {status}, Available: {available}")
                
                if model_info:
                    trained_targets = model_info.get('trained_targets', [])
                    self.log_test("ChemBERTa model info", len(trained_targets) > 0, 
                                f"Trained targets: {len(trained_targets)}")
                
            else:
                self.log_test("ChemBERTa status endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("ChemBERTa status endpoint", False, f"Connection error: {str(e)}")
        
        # Test /api/chemberta/targets
        try:
            response = requests.get(f"{API_BASE}/chemberta/targets", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                targets = data.get('targets', {})
                total_targets = data.get('total_targets', 0)
                
                self.log_test("ChemBERTa targets endpoint", total_targets > 0, 
                            f"Total targets: {total_targets}")
                
                # Check for expected targets
                expected_targets = ['EGFR', 'BRAF', 'VEGFR2']
                found_targets = [t for t in expected_targets if t in targets]
                self.log_test("ChemBERTa expected targets", len(found_targets) > 0, 
                            f"Found targets: {found_targets}")
                
            else:
                self.log_test("ChemBERTa targets endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("ChemBERTa targets endpoint", False, f"Connection error: {str(e)}")
        
        # Test /api/chemberta/predict with aspirin
        try:
            payload = {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}  # aspirin
            
            response = requests.post(f"{API_BASE}/chemberta/predict", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                predictions = data.get('predictions', {})
                
                self.log_test("ChemBERTa predict endpoint", status == 'success', 
                            f"Status: {status}, Predictions count: {len(predictions)}")
                
                # Check for IC50 predictions
                if predictions:
                    target_count = len(predictions)
                    self.log_test("ChemBERTa IC50 predictions", target_count > 0, 
                                f"Predictions for {target_count} targets")
                
            else:
                self.log_test("ChemBERTa predict endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("ChemBERTa predict endpoint", False, f"Connection error: {str(e)}")
        
        # Test /api/chemberta/predict with imatinib
        try:
            payload = {"smiles": "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"}  # imatinib
            
            response = requests.post(f"{API_BASE}/chemberta/predict", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                predictions = data.get('predictions', {})
                
                self.log_test("ChemBERTa predict imatinib", status == 'success', 
                            f"Status: {status}, Predictions count: {len(predictions)}")
                
            else:
                self.log_test("ChemBERTa predict imatinib", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("ChemBERTa predict imatinib", False, f"Connection error: {str(e)}")
    
    def test_chemprop_multitask_endpoints(self):
        """Test Chemprop Multi-Task endpoints"""
        print("\n=== Testing Chemprop Multi-Task Endpoints ===")
        
        # Test /api/chemprop-multitask/status
        try:
            response = requests.get(f"{API_BASE}/chemprop-multitask/status", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                available = data.get('available', False)
                model_info = data.get('model_info', {})
                
                self.log_test("Chemprop multitask status endpoint", available, 
                            f"Status: {status}, Available: {available}")
                
                if model_info:
                    prediction_types = model_info.get('prediction_types', [])
                    self.log_test("Chemprop model info", len(prediction_types) > 0, 
                                f"Prediction types: {prediction_types}")
                
            else:
                self.log_test("Chemprop multitask status endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Chemprop multitask status endpoint", False, f"Connection error: {str(e)}")
        
        # Test /api/chemprop-multitask/properties
        try:
            response = requests.get(f"{API_BASE}/chemprop-multitask/properties", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                properties = data.get('properties', {})
                total_properties = data.get('total_properties', 0)
                
                self.log_test("Chemprop properties endpoint", total_properties > 0, 
                            f"Total properties: {total_properties}")
                
                # Check for expected properties
                expected_properties = ['bioactivity_ic50', 'toxicity', 'logP', 'solubility']
                found_properties = [p for p in expected_properties if p in properties]
                self.log_test("Chemprop expected properties", len(found_properties) == 4, 
                            f"Found properties: {found_properties}")
                
            else:
                self.log_test("Chemprop properties endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Chemprop properties endpoint", False, f"Connection error: {str(e)}")
        
        # Test /api/chemprop-multitask/predict with aspirin
        try:
            payload = {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
                "properties": ["bioactivity_ic50", "toxicity", "logP", "solubility"]
            }
            
            response = requests.post(f"{API_BASE}/chemprop-multitask/predict", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                predictions = data.get('predictions', {})
                
                self.log_test("Chemprop multitask predict endpoint", status == 'success', 
                            f"Status: {status}, Predictions count: {len(predictions)}")
                
                # Check for all property predictions
                if predictions:
                    expected_props = ['bioactivity_ic50', 'toxicity', 'logP', 'solubility']
                    found_props = [p for p in expected_props if p in predictions]
                    self.log_test("Chemprop all properties predicted", len(found_props) == 4, 
                                f"Predicted properties: {found_props}")
                
            else:
                self.log_test("Chemprop multitask predict endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Chemprop multitask predict endpoint", False, f"Connection error: {str(e)}")
        
        # Test /api/chemprop-multitask/predict with imatinib
        try:
            payload = {
                "smiles": "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",  # imatinib
                "properties": ["bioactivity_ic50", "toxicity", "logP", "solubility"]
            }
            
            response = requests.post(f"{API_BASE}/chemprop-multitask/predict", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                predictions = data.get('predictions', {})
                
                self.log_test("Chemprop multitask predict imatinib", status == 'success', 
                            f"Status: {status}, Predictions count: {len(predictions)}")
                
            else:
                self.log_test("Chemprop multitask predict imatinib", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Chemprop multitask predict imatinib", False, f"Connection error: {str(e)}")
    
    def test_main_predict_endpoint(self):
        """Test main /api/predict endpoint with various SMILES inputs"""
        print("\n=== Testing Main Prediction Endpoint ===")
        
        test_molecules = [
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
            ("Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C", "imatinib")
        ]
        
        for smiles, name in test_molecules:
            try:
                payload = {
                    "smiles": smiles,
                    "prediction_types": ["bioactivity_ic50", "toxicity", "logP", "solubility"],
                    "target": "EGFR"
                }
                
                response = requests.post(f"{API_BASE}/predict", 
                                       json=payload,
                                       headers={'Content-Type': 'application/json'},
                                       timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    summary = data.get('summary', {})
                    
                    self.log_test(f"Main predict endpoint - {name}", len(results) == 4, 
                                f"Results count: {len(results)}")
                    
                    # Check for enhanced models usage
                    enhanced_models_used = summary.get('enhanced_models_used', False)
                    self.log_test(f"Enhanced models used - {name}", enhanced_models_used, 
                                f"Enhanced models: {enhanced_models_used}")
                    
                    # Check for IC50 enhanced prediction
                    ic50_result = next((r for r in results if r.get('prediction_type') == 'bioactivity_ic50'), None)
                    if ic50_result:
                        has_enhanced = ic50_result.get('enhanced_chemprop_prediction') is not None
                        self.log_test(f"IC50 enhanced prediction - {name}", has_enhanced, 
                                    f"Enhanced prediction present: {has_enhanced}")
                    
                else:
                    self.log_test(f"Main predict endpoint - {name}", False, 
                                f"HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"Main predict endpoint - {name}", False, f"Connection error: {str(e)}")
    
    def test_database_endpoints(self):
        """Test database endpoints: /api/history"""
        print("\n=== Testing Database Endpoints ===")
        
        # Test /api/predictions/history
        try:
            response = requests.get(f"{API_BASE}/predictions/history", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list):
                    self.log_test("History endpoint", True, f"Retrieved {len(data)} history records")
                    
                    # Check if records have proper structure
                    if len(data) > 0:
                        first_record = data[0]
                        required_fields = ['id', 'smiles', 'prediction_type', 'timestamp']
                        has_required_fields = all(field in first_record for field in required_fields)
                        self.log_test("History record structure", has_required_fields, 
                                    f"Required fields present: {has_required_fields}")
                else:
                    self.log_test("History endpoint", False, "Response is not a list")
                    
            else:
                self.log_test("History endpoint", False, f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("History endpoint", False, f"Connection error: {str(e)}")
    
    def run_focused_tests(self):
        """Run focused tests for the review request"""
        print(f"🧪 Starting Focused Backend Testing")
        print(f"Backend URL: {API_BASE}")
        print("=" * 60)
        
        # Run focused tests
        tests = [
            self.test_health_endpoint,
            self.test_chemberta_endpoints,
            self.test_chemprop_multitask_endpoints,
            self.test_main_predict_endpoint,
            self.test_database_endpoints
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
        print("🏁 FOCUSED TEST SUMMARY")
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
    tester = FocusedBackendTester()
    passed, failed, results = tester.run_focused_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)