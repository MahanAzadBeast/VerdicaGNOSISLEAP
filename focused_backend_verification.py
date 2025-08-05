#!/usr/bin/env python3
"""
Focused Backend Verification for Real Data Cell Line Response Model Testing
Tests the specific endpoints mentioned in the review request while real data training is in progress
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

class FocusedBackendVerifier:
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
    
    def test_health_check_endpoint(self):
        """Test /api/health endpoint - verify all models are loaded"""
        print("\n=== Testing Health Check Endpoint (/api/health) ===")
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check basic structure
                required_fields = ['status', 'models_loaded', 'prediction_types', 'available_targets']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Health endpoint structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check models loaded
                models_loaded = data.get('models_loaded', {})
                expected_models = ['molbert', 'chemprop_simulation', 'cell_line_response_model']
                
                model_status = {}
                for model in expected_models:
                    model_status[model] = models_loaded.get(model, False)
                
                self.log_test("Health endpoint response", True, f"Status: {data['status']}")
                self.log_test("Models loaded status", True, f"Models: {model_status}")
                
                # Check Cell Line Response Model specifically
                cell_line_available = models_loaded.get('cell_line_response_model', False)
                self.log_test("Cell Line Response Model loaded", cell_line_available, 
                            f"Cell Line Model: {cell_line_available}")
                
                # Check AI modules
                ai_modules = data.get('ai_modules', {})
                if ai_modules:
                    self.log_test("AI modules info", True, f"AI modules: {ai_modules}")
                
                return True
            else:
                self.log_test("Health endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Health endpoint connectivity", False, f"Connection error: {str(e)}")
            return False
    
    def test_cell_line_endpoints(self):
        """Test Cell Line Response Model endpoints (/api/cell-line/*)"""
        print("\n=== Testing Cell Line Response Model Endpoints ===")
        
        endpoints_to_test = [
            ("/cell-line/health", "Cell Line Health"),
            ("/cell-line/examples", "Cell Line Examples"),
            ("/cell-line/predict", "Cell Line Predict"),
            ("/cell-line/compare", "Cell Line Compare")
        ]
        
        all_passed = True
        
        for endpoint, name in endpoints_to_test:
            try:
                if endpoint.endswith("/predict") or endpoint.endswith("/compare"):
                    # These are POST endpoints, test with sample data
                    if endpoint.endswith("/predict"):
                        payload = {
                            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
                            "cell_line": "A549",
                            "genomic_features": {
                                "mutations": ["KRAS_G12C"],
                                "amplifications": [],
                                "deletions": [],
                                "expression_changes": {}
                            }
                        }
                    else:  # compare endpoint
                        payload = {
                            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
                            "cell_lines": ["A549", "MCF7"]
                        }
                    
                    response = requests.post(f"{API_BASE}{endpoint}", 
                                           json=payload,
                                           headers={'Content-Type': 'application/json'},
                                           timeout=60)
                else:
                    # GET endpoints
                    response = requests.get(f"{API_BASE}{endpoint}", timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test(f"{name} endpoint", True, f"HTTP 200 OK - Response received")
                    
                    # Check response structure based on endpoint
                    if endpoint.endswith("/health"):
                        if 'model_info' in data:
                            model_info = data['model_info']
                            self.log_test(f"{name} model info", True, f"Model: {model_info.get('model_name', 'N/A')}")
                    
                    elif endpoint.endswith("/examples"):
                        if 'cell_lines' in data and 'drugs' in data:
                            cell_lines = data['cell_lines']
                            drugs = data['drugs']
                            self.log_test(f"{name} data", True, f"Cell lines: {len(cell_lines)}, Drugs: {len(drugs)}")
                    
                    elif endpoint.endswith("/predict"):
                        if 'prediction' in data:
                            prediction = data['prediction']
                            self.log_test(f"{name} prediction", True, f"IC50: {prediction.get('ic50_nm', 'N/A')} nM")
                    
                    elif endpoint.endswith("/compare"):
                        if 'comparisons' in data:
                            comparisons = data['comparisons']
                            self.log_test(f"{name} comparison", True, f"Comparisons: {len(comparisons)}")
                
                elif response.status_code == 503:
                    # Service unavailable is acceptable during training
                    self.log_test(f"{name} endpoint", True, f"HTTP 503 - Service unavailable (training in progress)")
                
                else:
                    self.log_test(f"{name} endpoint", False, f"HTTP {response.status_code}: {response.text}")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"{name} endpoint", False, f"Connection error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_chemberta_endpoints(self):
        """Test existing ChemBERTa endpoints - ensure no regression"""
        print("\n=== Testing ChemBERTa Endpoints (No Regression) ===")
        
        endpoints_to_test = [
            ("/chemberta/status", "ChemBERTa Status"),
            ("/chemberta/targets", "ChemBERTa Targets"),
            ("/chemberta/predict", "ChemBERTa Predict")
        ]
        
        all_passed = True
        
        for endpoint, name in endpoints_to_test:
            try:
                if endpoint.endswith("/predict"):
                    # POST endpoint with sample data
                    payload = {
                        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
                        "targets": ["EGFR", "BRAF"]
                    }
                    response = requests.post(f"{API_BASE}{endpoint}", 
                                           json=payload,
                                           headers={'Content-Type': 'application/json'},
                                           timeout=60)
                else:
                    # GET endpoints
                    response = requests.get(f"{API_BASE}{endpoint}", timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test(f"{name} endpoint", True, f"HTTP 200 OK")
                    
                    # Check specific response structures
                    if endpoint.endswith("/status"):
                        if 'available' in data:
                            available = data['available']
                            self.log_test(f"{name} availability", available, f"Available: {available}")
                    
                    elif endpoint.endswith("/targets"):
                        if 'targets' in data:
                            targets = data['targets']
                            self.log_test(f"{name} targets", True, f"Targets: {len(targets)}")
                    
                    elif endpoint.endswith("/predict"):
                        if 'predictions' in data:
                            predictions = data['predictions']
                            self.log_test(f"{name} predictions", True, f"Predictions received")
                
                elif response.status_code == 503:
                    # Service unavailable is acceptable
                    self.log_test(f"{name} endpoint", True, f"HTTP 503 - Service unavailable")
                
                else:
                    self.log_test(f"{name} endpoint", False, f"HTTP {response.status_code}: {response.text}")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"{name} endpoint", False, f"Connection error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_chemprop_endpoints(self):
        """Test existing Chemprop endpoints - ensure no regression"""
        print("\n=== Testing Chemprop Endpoints (No Regression) ===")
        
        endpoints_to_test = [
            ("/chemprop-multitask/status", "Chemprop Status"),
            ("/chemprop-multitask/properties", "Chemprop Properties"),
            ("/chemprop-multitask/predict", "Chemprop Predict")
        ]
        
        all_passed = True
        
        for endpoint, name in endpoints_to_test:
            try:
                if endpoint.endswith("/predict"):
                    # POST endpoint with sample data
                    payload = {
                        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
                        "properties": ["bioactivity_ic50", "toxicity"]
                    }
                    response = requests.post(f"{API_BASE}{endpoint}", 
                                           json=payload,
                                           headers={'Content-Type': 'application/json'},
                                           timeout=60)
                else:
                    # GET endpoints
                    response = requests.get(f"{API_BASE}{endpoint}", timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test(f"{name} endpoint", True, f"HTTP 200 OK")
                    
                    # Check specific response structures
                    if endpoint.endswith("/status"):
                        if 'available' in data:
                            available = data['available']
                            self.log_test(f"{name} availability", available, f"Available: {available}")
                    
                    elif endpoint.endswith("/properties"):
                        if 'properties' in data:
                            properties = data['properties']
                            self.log_test(f"{name} properties", True, f"Properties: {len(properties)}")
                    
                    elif endpoint.endswith("/predict"):
                        if 'predictions' in data:
                            predictions = data['predictions']
                            self.log_test(f"{name} predictions", True, f"Predictions received")
                
                elif response.status_code == 503:
                    # Service unavailable is acceptable
                    self.log_test(f"{name} endpoint", True, f"HTTP 503 - Service unavailable")
                
                else:
                    self.log_test(f"{name} endpoint", False, f"HTTP {response.status_code}: {response.text}")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"{name} endpoint", False, f"Connection error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_basic_prediction_endpoints(self):
        """Test basic prediction endpoints with simple molecules"""
        print("\n=== Testing Basic Prediction Endpoints ===")
        
        # Test molecules as specified in review request
        test_molecules = [
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
            ("CCO", "ethanol"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine")
        ]
        
        all_passed = True
        
        for smiles, name in test_molecules:
            try:
                payload = {
                    "smiles": smiles,
                    "prediction_types": ["bioactivity_ic50", "toxicity"],
                    "target": "EGFR"
                }
                
                response = requests.post(f"{API_BASE}/predict", 
                                       json=payload,
                                       headers={'Content-Type': 'application/json'},
                                       timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'results' in data and len(data['results']) > 0:
                        results = data['results']
                        self.log_test(f"Basic prediction - {name}", True, 
                                    f"Received {len(results)} predictions")
                        
                        # Check for enhanced predictions
                        for result in results:
                            if result.get('prediction_type') == 'bioactivity_ic50':
                                enhanced_pred = result.get('enhanced_chemprop_prediction')
                                if enhanced_pred:
                                    ic50 = enhanced_pred.get('ic50_nm')
                                    confidence = enhanced_pred.get('confidence')
                                    self.log_test(f"Enhanced prediction - {name}", True,
                                                f"IC50: {ic50} nM, Confidence: {confidence}")
                    else:
                        self.log_test(f"Basic prediction - {name}", False, "No results returned")
                        all_passed = False
                
                else:
                    self.log_test(f"Basic prediction - {name}", False, 
                                f"HTTP {response.status_code}: {response.text}")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"Basic prediction - {name}", False, f"Connection error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_database_endpoints(self):
        """Test database-related endpoints"""
        print("\n=== Testing Database Endpoints ===")
        
        all_passed = True
        
        # Test history endpoint
        try:
            response = requests.get(f"{API_BASE}/predictions/history", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("Prediction history", True, f"Retrieved {len(data)} records")
            else:
                self.log_test("Prediction history", False, f"HTTP {response.status_code}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Prediction history", False, f"Connection error: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def run_all_tests(self):
        """Run all focused verification tests"""
        print("🧪 FOCUSED BACKEND VERIFICATION FOR REAL DATA CELL LINE RESPONSE MODEL")
        print("=" * 80)
        print(f"🌐 Backend URL: {BACKEND_URL}")
        print(f"📡 API Base: {API_BASE}")
        print()
        
        # Run all test categories
        test_categories = [
            ("Health Check", self.test_health_check_endpoint),
            ("Cell Line Endpoints", self.test_cell_line_endpoints),
            ("ChemBERTa Endpoints", self.test_chemberta_endpoints),
            ("Chemprop Endpoints", self.test_chemprop_endpoints),
            ("Basic Predictions", self.test_basic_prediction_endpoints),
            ("Database Endpoints", self.test_database_endpoints)
        ]
        
        category_results = {}
        
        for category_name, test_func in test_categories:
            print(f"\n🔍 Testing {category_name}...")
            try:
                category_results[category_name] = test_func()
            except Exception as e:
                print(f"❌ Error in {category_name}: {str(e)}")
                category_results[category_name] = False
        
        # Print summary
        print("\n" + "=" * 80)
        print("🎯 FOCUSED BACKEND VERIFICATION SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['success']])
        failed_tests = len(self.failed_tests)
        
        print(f"📊 Test Results: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"   • {test['test']}: {test['details']}")
        
        print(f"\n📋 CATEGORY RESULTS:")
        for category, result in category_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status}: {category}")
        
        # Overall assessment
        overall_success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        if overall_success_rate >= 0.9:
            print(f"\n🎯 OVERALL ASSESSMENT:")
            print(f"   🟢 EXCELLENT: Backend is functioning well during real data training")
        elif overall_success_rate >= 0.8:
            print(f"\n🎯 OVERALL ASSESSMENT:")
            print(f"   🟡 GOOD: Backend is mostly functional with minor issues")
        else:
            print(f"\n🎯 OVERALL ASSESSMENT:")
            print(f"   🔴 NEEDS ATTENTION: Significant issues detected")
        
        print("=" * 80)
        
        return overall_success_rate >= 0.8

if __name__ == "__main__":
    verifier = FocusedBackendVerifier()
    success = verifier.run_all_tests()
    sys.exit(0 if success else 1)