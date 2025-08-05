#!/usr/bin/env python3
"""
Priority Backend Testing for Review Request
Tests the specific endpoints mentioned in the review request:
1. Health Check Endpoint (/api/health)
2. Cell Line Response Model endpoints (/api/cell-line/*)
3. ChemBERTa and Chemprop Models endpoints
4. Database connectivity
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

class PriorityBackendTester:
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
        """Test /api/health endpoint - Priority 1"""
        print("\n=== Testing Health Check Endpoint (/api/health) ===")
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check basic structure
                required_fields = ['status', 'models_loaded', 'ai_modules']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Health endpoint structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check status
                status = data.get('status')
                self.log_test("Health endpoint status", status == 'healthy', f"Status: {status}")
                
                # Check models loaded
                models_loaded = data.get('models_loaded', {})
                cell_line_model = models_loaded.get('cell_line_response_model', False)
                chemberta_available = models_loaded.get('oncoprotein_chemberta', False)
                
                self.log_test("Cell Line Response Model loaded", cell_line_model, f"Cell line model: {cell_line_model}")
                self.log_test("ChemBERTa model available", chemberta_available, f"ChemBERTa: {chemberta_available}")
                
                # Check AI modules
                ai_modules = data.get('ai_modules', {})
                chemberta_ai = ai_modules.get('chemberta_available', False)
                cell_line_ai = ai_modules.get('cell_line_model_available', False)
                
                self.log_test("AI Modules - ChemBERTa", chemberta_ai, f"ChemBERTa AI: {chemberta_ai}")
                self.log_test("AI Modules - Cell Line", cell_line_ai, f"Cell Line AI: {cell_line_ai}")
                
                return True
            else:
                self.log_test("Health endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Health endpoint connectivity", False, f"Connection error: {str(e)}")
            return False
    
    def test_cell_line_health_endpoint(self):
        """Test /api/cell-line/health endpoint - Priority 1"""
        print("\n=== Testing Cell Line Health Endpoint (/api/cell-line/health) ===")
        try:
            response = requests.get(f"{API_BASE}/cell-line/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['status', 'model_info', 'capabilities']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Cell Line health structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check status
                status = data.get('status')
                self.log_test("Cell Line health status", status == 'healthy', f"Status: {status}")
                
                # Check model info
                model_info = data.get('model_info', {})
                model_name = model_info.get('model_name', '')
                architecture = model_info.get('architecture', '')
                
                self.log_test("Cell Line model info", 'Cell_Line_Response_Model' in model_name, 
                            f"Model: {model_name}")
                self.log_test("Cell Line architecture", 'Multi_Modal' in architecture, 
                            f"Architecture: {architecture}")
                
                # Check capabilities
                capabilities = data.get('capabilities', [])
                expected_capabilities = ['multi_modal_prediction', 'genomic_integration', 'uncertainty_quantification']
                has_capabilities = all(cap in capabilities for cap in expected_capabilities)
                
                self.log_test("Cell Line capabilities", has_capabilities, 
                            f"Capabilities: {capabilities}")
                
                return True
            else:
                self.log_test("Cell Line health endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Cell Line health endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_cell_line_predict_endpoint(self):
        """Test /api/cell-line/predict endpoint - Priority 1"""
        print("\n=== Testing Cell Line Predict Endpoint (/api/cell-line/predict) ===")
        try:
            # Test with Erlotinib and A549 cell line (KRAS mutated)
            payload = {
                "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",  # Erlotinib
                "cell_line": "A549",
                "genomic_features": {
                    "mutations": ["KRAS_G12S"],
                    "amplifications": [],
                    "deletions": [],
                    "expression_changes": {}
                }
            }
            
            response = requests.post(f"{API_BASE}/cell-line/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['prediction', 'genomic_context', 'clinical_insights']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Cell Line predict structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check prediction
                prediction = data.get('prediction', {})
                ic50_nm = prediction.get('ic50_nm')
                confidence = prediction.get('confidence')
                
                self.log_test("Cell Line IC50 prediction", ic50_nm is not None and ic50_nm > 0, 
                            f"IC50: {ic50_nm} nM")
                self.log_test("Cell Line prediction confidence", confidence is not None and 0 <= confidence <= 1, 
                            f"Confidence: {confidence}")
                
                # Check genomic context
                genomic_context = data.get('genomic_context', {})
                detected_mutations = genomic_context.get('detected_mutations', [])
                
                self.log_test("Cell Line genomic context", len(detected_mutations) > 0, 
                            f"Detected mutations: {detected_mutations}")
                
                # Check clinical insights
                clinical_insights = data.get('clinical_insights', {})
                resistance_factors = clinical_insights.get('resistance_factors', [])
                
                self.log_test("Cell Line clinical insights", len(resistance_factors) >= 0, 
                            f"Resistance factors: {resistance_factors}")
                
                return True
            else:
                self.log_test("Cell Line predict endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Cell Line predict endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_cell_line_compare_endpoint(self):
        """Test /api/cell-line/compare endpoint - Priority 1"""
        print("\n=== Testing Cell Line Compare Endpoint (/api/cell-line/compare) ===")
        try:
            payload = {
                "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",  # Erlotinib
                "cell_lines": [
                    {
                        "name": "A549",
                        "genomic_features": {
                            "mutations": ["KRAS_G12S"],
                            "amplifications": [],
                            "deletions": [],
                            "expression_changes": {}
                        }
                    },
                    {
                        "name": "MCF7",
                        "genomic_features": {
                            "mutations": [],
                            "amplifications": ["EGFR"],
                            "deletions": [],
                            "expression_changes": {}
                        }
                    }
                ]
            }
            
            response = requests.post(f"{API_BASE}/cell-line/compare", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['comparisons', 'summary']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Cell Line compare structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check comparisons
                comparisons = data.get('comparisons', [])
                self.log_test("Cell Line compare results", len(comparisons) == 2, 
                            f"Comparisons: {len(comparisons)}")
                
                # Check summary
                summary = data.get('summary', {})
                fold_differences = summary.get('fold_differences', {})
                
                self.log_test("Cell Line compare summary", len(fold_differences) > 0, 
                            f"Fold differences: {fold_differences}")
                
                return True
            else:
                self.log_test("Cell Line compare endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Cell Line compare endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_cell_line_examples_endpoint(self):
        """Test /api/cell-line/examples endpoint - Priority 1"""
        print("\n=== Testing Cell Line Examples Endpoint (/api/cell-line/examples) ===")
        try:
            response = requests.get(f"{API_BASE}/cell-line/examples", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['example_drugs', 'example_cell_lines']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Cell Line examples structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check example drugs
                example_drugs = data.get('example_drugs', [])
                self.log_test("Cell Line example drugs", len(example_drugs) >= 2, 
                            f"Example drugs: {len(example_drugs)}")
                
                # Check example cell lines
                example_cell_lines = data.get('example_cell_lines', [])
                self.log_test("Cell Line example cell lines", len(example_cell_lines) >= 3, 
                            f"Example cell lines: {len(example_cell_lines)}")
                
                # Check structure of first drug
                if example_drugs:
                    first_drug = example_drugs[0]
                    drug_fields = ['name', 'smiles', 'description']
                    has_drug_fields = all(field in first_drug for field in drug_fields)
                    self.log_test("Cell Line drug structure", has_drug_fields, 
                                f"Drug fields: {list(first_drug.keys())}")
                
                # Check structure of first cell line
                if example_cell_lines:
                    first_cell_line = example_cell_lines[0]
                    cell_fields = ['name', 'cancer_type', 'genomic_features']
                    has_cell_fields = all(field in first_cell_line for field in cell_fields)
                    self.log_test("Cell Line cell structure", has_cell_fields, 
                                f"Cell fields: {list(first_cell_line.keys())}")
                
                return True
            else:
                self.log_test("Cell Line examples endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Cell Line examples endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_chemberta_endpoints(self):
        """Test ChemBERTa model endpoints - Priority 2"""
        print("\n=== Testing ChemBERTa Model Endpoints ===")
        
        # Test ChemBERTa status
        try:
            response = requests.get(f"{API_BASE}/chemberta/status", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                available = data.get('available', False)
                self.log_test("ChemBERTa status endpoint", True, f"Available: {available}")
                
                if available:
                    model_info = data.get('model_info', {})
                    trained_targets = model_info.get('trained_targets', 0)
                    self.log_test("ChemBERTa model info", trained_targets > 0, 
                                f"Trained targets: {trained_targets}")
            else:
                self.log_test("ChemBERTa status endpoint", False, f"HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("ChemBERTa status endpoint", False, f"Connection error: {str(e)}")
        
        # Test ChemBERTa predict
        try:
            payload = {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin
            }
            
            response = requests.post(f"{API_BASE}/chemberta/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', {})
                self.log_test("ChemBERTa predict endpoint", len(predictions) > 0, 
                            f"Predictions for {len(predictions)} targets")
            else:
                self.log_test("ChemBERTa predict endpoint", False, f"HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("ChemBERTa predict endpoint", False, f"Connection error: {str(e)}")
        
        # Test ChemBERTa targets
        try:
            response = requests.get(f"{API_BASE}/chemberta/targets", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                targets = data.get('targets', [])
                self.log_test("ChemBERTa targets endpoint", len(targets) > 0, 
                            f"Available targets: {len(targets)}")
            else:
                self.log_test("ChemBERTa targets endpoint", False, f"HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("ChemBERTa targets endpoint", False, f"Connection error: {str(e)}")
    
    def test_chemprop_endpoints(self):
        """Test Chemprop model endpoints - Priority 2"""
        print("\n=== Testing Chemprop Model Endpoints ===")
        
        # Test Chemprop Multi-Task status
        try:
            response = requests.get(f"{API_BASE}/chemprop-multitask/status", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                available = data.get('available', False)
                self.log_test("Chemprop Multi-Task status", True, f"Available: {available}")
                
                if available:
                    model_info = data.get('model_info', {})
                    prediction_types = model_info.get('prediction_types', [])
                    self.log_test("Chemprop model info", len(prediction_types) > 0, 
                                f"Prediction types: {len(prediction_types)}")
            else:
                self.log_test("Chemprop Multi-Task status", False, f"HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Chemprop Multi-Task status", False, f"Connection error: {str(e)}")
        
        # Test Chemprop predict
        try:
            payload = {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin
            }
            
            response = requests.post(f"{API_BASE}/chemprop-multitask/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', {})
                self.log_test("Chemprop Multi-Task predict", len(predictions) > 0, 
                            f"Predictions: {list(predictions.keys())}")
            else:
                self.log_test("Chemprop Multi-Task predict", False, f"HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Chemprop Multi-Task predict", False, f"Connection error: {str(e)}")
        
        # Test Chemprop properties
        try:
            response = requests.get(f"{API_BASE}/chemprop-multitask/properties", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                properties = data.get('properties', [])
                self.log_test("Chemprop properties endpoint", len(properties) > 0, 
                            f"Available properties: {len(properties)}")
            else:
                self.log_test("Chemprop properties endpoint", False, f"HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Chemprop properties endpoint", False, f"Connection error: {str(e)}")
    
    def test_database_connectivity(self):
        """Test database operations - Priority 3"""
        print("\n=== Testing Database Connectivity ===")
        
        # Test prediction history (MongoDB)
        try:
            response = requests.get(f"{API_BASE}/predictions/history?limit=5", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("Database history retrieval", isinstance(data, list), 
                            f"Retrieved {len(data)} records")
            else:
                self.log_test("Database history retrieval", False, f"HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Database history retrieval", False, f"Connection error: {str(e)}")
        
        # Test making a prediction (which stores to database)
        try:
            payload = {
                "smiles": "CCO",  # ethanol
                "prediction_types": ["bioactivity_ic50"],
                "target": "EGFR"
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                if results:
                    prediction_id = results[0].get('id')
                    self.log_test("Database prediction storage", prediction_id is not None, 
                                f"Stored prediction with ID: {prediction_id}")
                    
                    # Test retrieving specific prediction
                    if prediction_id:
                        try:
                            response = requests.get(f"{API_BASE}/predictions/{prediction_id}", timeout=30)
                            if response.status_code == 200:
                                self.log_test("Database specific retrieval", True, "Retrieved specific prediction")
                            else:
                                self.log_test("Database specific retrieval", False, f"HTTP {response.status_code}")
                        except:
                            self.log_test("Database specific retrieval", False, "Connection error")
                else:
                    self.log_test("Database prediction storage", False, "No results returned")
            else:
                self.log_test("Database prediction storage", False, f"HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Database prediction storage", False, f"Connection error: {str(e)}")
    
    def run_all_tests(self):
        """Run all priority tests"""
        print("🧪 PRIORITY BACKEND TESTING")
        print("=" * 80)
        print(f"🌐 Backend URL: {BACKEND_URL}")
        print(f"📡 API Base: {API_BASE}")
        print()
        
        # Priority 1: Critical endpoints
        self.test_health_check_endpoint()
        self.test_cell_line_health_endpoint()
        self.test_cell_line_predict_endpoint()
        self.test_cell_line_compare_endpoint()
        self.test_cell_line_examples_endpoint()
        
        # Priority 2: AI model endpoints
        self.test_chemberta_endpoints()
        self.test_chemprop_endpoints()
        
        # Priority 3: Database connectivity
        self.test_database_connectivity()
        
        # Summary
        print("\n" + "=" * 80)
        print("🎯 PRIORITY BACKEND TESTING SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = len(self.failed_tests)
        
        print(f"📊 Test Results: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"   • {test['test']}: {test['details']}")
        
        print(f"\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            print(f"   {result['status']}: {result['test']}")
            if result['details']:
                print(f"      Details: {result['details']}")
        
        # Overall assessment
        if passed_tests / total_tests >= 0.9:
            print(f"\n🎯 OVERALL ASSESSMENT:")
            print(f"   🟢 EXCELLENT: Backend services are operational during background processes")
        elif passed_tests / total_tests >= 0.7:
            print(f"\n🎯 OVERALL ASSESSMENT:")
            print(f"   🟡 GOOD: Most backend services operational with minor issues")
        else:
            print(f"\n🎯 OVERALL ASSESSMENT:")
            print(f"   🔴 NEEDS ATTENTION: Significant backend issues detected")
        
        print("=" * 80)

if __name__ == "__main__":
    tester = PriorityBackendTester()
    tester.run_all_tests()