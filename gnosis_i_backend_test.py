#!/usr/bin/env python3
"""
Gnosis I (Model 1) Ligand Activity Predictor Backend Integration Testing
Tests all Gnosis I endpoints, model loading, predictions, and database integration
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

class GnosisIBackendTester:
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
    
    def test_backend_service_status(self):
        """Test 1: Backend Service Status - verify backend running correctly"""
        print("\n=== Testing Backend Service Status ===")
        
        try:
            response = requests.get(f"{API_BASE}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Check if backend is healthy
                if health_data.get('status') == 'healthy':
                    self.log_test("Backend Service Health", True, f"Backend running at {BACKEND_URL}")
                else:
                    self.log_test("Backend Service Health", False, f"Backend status: {health_data.get('status')}")
                
                # Check if models are loaded
                models_loaded = health_data.get('models_loaded', {})
                if models_loaded:
                    self.log_test("Models Loading Status", True, f"Models loaded: {list(models_loaded.keys())}")
                else:
                    self.log_test("Models Loading Status", False, "No models loaded information")
                    
            else:
                self.log_test("Backend Service Health", False, f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Backend Service Health", False, f"Connection error: {str(e)}")
    
    def test_gnosis_model_loading(self):
        """Test 2: Gnosis I Model Loading - check if model file exists and loads"""
        print("\n=== Testing Gnosis I Model Loading ===")
        
        # Check if model file exists
        model_path = Path("/app/backend/models/gnosis_model1_best.pt")
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024 * 1024)  # MB
            self.log_test("Gnosis I Model File Exists", True, f"Model file found: {file_size:.1f} MB")
        else:
            self.log_test("Gnosis I Model File Exists", False, "gnosis_model1_best.pt not found")
        
        # Test model loading via health endpoint
        try:
            response = requests.get(f"{API_BASE}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                
                # Check for Gnosis I availability indicators
                gnosis_available = False
                if 'gnosis_i_available' in health_data:
                    gnosis_available = health_data['gnosis_i_available']
                elif 'models_loaded' in health_data and 'gnosis_i' in health_data['models_loaded']:
                    gnosis_available = health_data['models_loaded']['gnosis_i']
                
                self.log_test("Gnosis I Model Loading", gnosis_available, 
                            f"Gnosis I available: {gnosis_available}")
            else:
                self.log_test("Gnosis I Model Loading", False, f"Health check failed: {response.status_code}")
                
        except Exception as e:
            self.log_test("Gnosis I Model Loading", False, f"Error checking model loading: {str(e)}")
    
    def test_health_check_endpoint(self):
        """Test 3: Health Check Endpoint - verify gnosis_i_available status"""
        print("\n=== Testing Health Check Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Check overall health
                if health_data.get('status') == 'healthy':
                    self.log_test("Health Endpoint Status", True, "Backend healthy")
                else:
                    self.log_test("Health Endpoint Status", False, f"Status: {health_data.get('status')}")
                
                # Check for Gnosis I specific information
                gnosis_info_found = False
                gnosis_details = []
                
                # Look for Gnosis I indicators in various places
                if 'gnosis_i_available' in health_data:
                    gnosis_info_found = True
                    gnosis_details.append(f"gnosis_i_available: {health_data['gnosis_i_available']}")
                
                if 'models_loaded' in health_data:
                    models = health_data['models_loaded']
                    for key in models:
                        if 'gnosis' in key.lower():
                            gnosis_info_found = True
                            gnosis_details.append(f"{key}: {models[key]}")
                
                if 'ai_modules' in health_data:
                    ai_modules = health_data['ai_modules']
                    for key in ai_modules:
                        if 'gnosis' in key.lower():
                            gnosis_info_found = True
                            gnosis_details.append(f"{key}: {ai_modules[key]}")
                
                self.log_test("Gnosis I Health Information", gnosis_info_found, 
                            f"Found: {'; '.join(gnosis_details) if gnosis_details else 'No Gnosis I info'}")
                
            else:
                self.log_test("Health Endpoint Status", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Health Endpoint Status", False, f"Error: {str(e)}")
    
    def test_gnosis_info_endpoint(self):
        """Test 4: Gnosis I Info Endpoint - GET /api/gnosis-i/info"""
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
                    expected_r2 = 0.6281
                    if abs(r2_score - expected_r2) < 0.1:  # Allow some tolerance
                        self.log_test("Gnosis I R² Score", True, f"R² = {r2_score} (expected ~{expected_r2})")
                    else:
                        self.log_test("Gnosis I R² Score", False, f"R² = {r2_score}, expected ~{expected_r2}")
                    
                    # Check capabilities
                    capabilities = info_data.get('capabilities', [])
                    expected_capabilities = ['IC50 prediction', 'Ki prediction', 'EC50 prediction', 'LogP calculation', 'LogS calculation']
                    
                    capabilities_found = sum(1 for cap in expected_capabilities if cap in capabilities)
                    self.log_test("Gnosis I Capabilities", capabilities_found >= 3, 
                                f"Found {capabilities_found}/{len(expected_capabilities)} capabilities: {capabilities}")
                    
                    # Check version and description
                    version = info_data.get('version', 'unknown')
                    description = info_data.get('description', '')
                    self.log_test("Gnosis I Model Info", True, f"Version: {version}, Description: {description[:50]}...")
                    
                else:
                    self.log_test("Gnosis I Info Available", False, f"Model not available: {info_data.get('message')}")
                    
            elif response.status_code == 503:
                self.log_test("Gnosis I Info Available", False, "Service unavailable (503)")
            else:
                self.log_test("Gnosis I Info Available", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Gnosis I Info Available", False, f"Error: {str(e)}")
    
    def test_gnosis_targets_endpoint(self):
        """Test 5: Gnosis I Targets Endpoint - GET /api/gnosis-i/targets"""
        print("\n=== Testing Gnosis I Targets Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/gnosis-i/targets", timeout=10)
            
            if response.status_code == 200:
                targets_data = response.json()
                
                # Check available targets
                available_targets = targets_data.get('available_targets', [])
                if available_targets:
                    self.log_test("Gnosis I Available Targets", True, 
                                f"Found {len(available_targets)} targets: {available_targets[:5]}...")
                else:
                    self.log_test("Gnosis I Available Targets", False, "No available targets found")
                
                # Check categorized targets
                categorized = targets_data.get('categorized_targets', {})
                oncoproteins = categorized.get('oncoproteins', [])
                tumor_suppressors = categorized.get('tumor_suppressors', [])
                
                if oncoproteins:
                    self.log_test("Gnosis I Oncoproteins", True, 
                                f"Found {len(oncoproteins)} oncoproteins: {oncoproteins[:3]}...")
                else:
                    self.log_test("Gnosis I Oncoproteins", False, "No oncoproteins found")
                
                if tumor_suppressors:
                    self.log_test("Gnosis I Tumor Suppressors", True, 
                                f"Found {len(tumor_suppressors)} tumor suppressors: {tumor_suppressors[:3]}...")
                else:
                    self.log_test("Gnosis I Tumor Suppressors", False, "No tumor suppressors found")
                
                # Check total targets
                total_targets = targets_data.get('total_targets', 0)
                self.log_test("Gnosis I Total Targets", total_targets > 0, f"Total targets: {total_targets}")
                
                # Check model info
                model_info = targets_data.get('model_info', {})
                if model_info:
                    self.log_test("Gnosis I Targets Model Info", True, 
                                f"Model: {model_info.get('name')}, R²: {model_info.get('r2_score')}")
                else:
                    self.log_test("Gnosis I Targets Model Info", False, "No model info in targets response")
                    
            elif response.status_code == 503:
                self.log_test("Gnosis I Targets Endpoint", False, "Service unavailable (503)")
            else:
                self.log_test("Gnosis I Targets Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Gnosis I Targets Endpoint", False, f"Error: {str(e)}")
    
    def test_gnosis_prediction_aspirin(self):
        """Test 6: Gnosis I Prediction with Aspirin"""
        print("\n=== Testing Gnosis I Prediction - Aspirin ===")
        
        aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        
        try:
            payload = {
                "smiles": aspirin_smiles,
                "targets": "all",
                "assay_types": "IC50"
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                prediction_data = response.json()
                
                # Check SMILES echo
                if prediction_data.get('smiles') == aspirin_smiles:
                    self.log_test("Aspirin SMILES Echo", True, "SMILES correctly echoed")
                else:
                    self.log_test("Aspirin SMILES Echo", False, f"Expected {aspirin_smiles}, got {prediction_data.get('smiles')}")
                
                # Check molecular properties
                properties = prediction_data.get('properties', {})
                logp = properties.get('LogP')
                logs = properties.get('LogS')
                
                if logp is not None:
                    self.log_test("Aspirin LogP Calculation", True, f"LogP = {logp}")
                else:
                    self.log_test("Aspirin LogP Calculation", False, "LogP not calculated")
                
                if logs is not None:
                    self.log_test("Aspirin LogS Calculation", True, f"LogS = {logs}")
                else:
                    self.log_test("Aspirin LogS Calculation", False, "LogS not calculated")
                
                # Check predictions
                predictions = prediction_data.get('predictions', {})
                if predictions:
                    num_predictions = len(predictions)
                    self.log_test("Aspirin Target Predictions", True, 
                                f"Predictions for {num_predictions} targets")
                    
                    # Check a few specific predictions
                    sample_targets = list(predictions.keys())[:3]
                    for target in sample_targets:
                        pred = predictions[target]
                        pactivity = pred.get('pActivity')
                        activity_nm = pred.get('activity_nM')
                        assay_type = pred.get('assay_type')
                        
                        if pactivity is not None and activity_nm is not None:
                            self.log_test(f"Aspirin {target} Prediction", True, 
                                        f"{assay_type}: pActivity={pactivity}, {activity_nm} nM")
                        else:
                            self.log_test(f"Aspirin {target} Prediction", False, 
                                        f"Missing prediction values for {target}")
                else:
                    self.log_test("Aspirin Target Predictions", False, "No predictions returned")
                
                # Check model info
                model_info = prediction_data.get('model_info', {})
                if model_info.get('name') == 'Gnosis I':
                    self.log_test("Aspirin Model Info", True, f"Model: {model_info.get('name')}, R²: {model_info.get('r2_score')}")
                else:
                    self.log_test("Aspirin Model Info", False, f"Unexpected model info: {model_info}")
                    
            elif response.status_code == 400:
                self.log_test("Aspirin Prediction", False, f"Bad request (400): {response.text}")
            elif response.status_code == 503:
                self.log_test("Aspirin Prediction", False, "Service unavailable (503)")
            else:
                self.log_test("Aspirin Prediction", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Aspirin Prediction", False, f"Error: {str(e)}")
    
    def test_gnosis_prediction_imatinib(self):
        """Test 7: Gnosis I Prediction with Imatinib"""
        print("\n=== Testing Gnosis I Prediction - Imatinib ===")
        
        imatinib_smiles = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
        
        try:
            payload = {
                "smiles": imatinib_smiles,
                "targets": ["EGFR", "BRAF", "CDK2"],  # Test specific targets
                "assay_types": ["IC50", "Ki", "EC50"]  # Test different assay types
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
                
                # Check molecular properties
                properties = prediction_data.get('properties', {})
                logp = properties.get('LogP')
                logs = properties.get('LogS')
                
                # Imatinib should have higher LogP than aspirin (more lipophilic)
                if logp is not None and logp > 2.0:
                    self.log_test("Imatinib LogP Calculation", True, f"LogP = {logp} (lipophilic)")
                else:
                    self.log_test("Imatinib LogP Calculation", False, f"LogP = {logp} (expected > 2.0)")
                
                if logs is not None:
                    self.log_test("Imatinib LogS Calculation", True, f"LogS = {logs}")
                else:
                    self.log_test("Imatinib LogS Calculation", False, "LogS not calculated")
                
                # Check predictions for specific targets
                predictions = prediction_data.get('predictions', {})
                expected_targets = ["EGFR", "BRAF", "CDK2"]
                
                for target in expected_targets:
                    if target in predictions:
                        pred = predictions[target]
                        pactivity = pred.get('pActivity')
                        activity_nm = pred.get('activity_nM')
                        assay_type = pred.get('assay_type')
                        
                        if pactivity is not None and activity_nm is not None:
                            self.log_test(f"Imatinib {target} Prediction", True, 
                                        f"{assay_type}: pActivity={pactivity}, {activity_nm} nM")
                        else:
                            self.log_test(f"Imatinib {target} Prediction", False, 
                                        f"Missing prediction values for {target}")
                    else:
                        self.log_test(f"Imatinib {target} Prediction", False, f"No prediction for {target}")
                
                # Check different assay types
                assay_types_found = set()
                for pred in predictions.values():
                    assay_types_found.add(pred.get('assay_type'))
                
                expected_assays = {"IC50", "Ki", "EC50"}
                assays_match = len(assay_types_found.intersection(expected_assays)) > 0
                self.log_test("Imatinib Assay Types", assays_match, 
                            f"Found assay types: {assay_types_found}")
                    
            elif response.status_code == 400:
                self.log_test("Imatinib Prediction", False, f"Bad request (400): {response.text}")
            elif response.status_code == 503:
                self.log_test("Imatinib Prediction", False, "Service unavailable (503)")
            else:
                self.log_test("Imatinib Prediction", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Imatinib Prediction", False, f"Error: {str(e)}")
    
    def test_gnosis_error_handling(self):
        """Test 8: Gnosis I Error Handling with Invalid SMILES"""
        print("\n=== Testing Gnosis I Error Handling ===")
        
        invalid_smiles_tests = [
            ("", "Empty SMILES"),
            ("INVALID_SMILES", "Invalid SMILES string"),
            ("C[C@H](C)C(=O)O[INVALID]", "Malformed SMILES"),
            ("   ", "Whitespace only")
        ]
        
        for invalid_smiles, test_name in invalid_smiles_tests:
            try:
                payload = {
                    "smiles": invalid_smiles,
                    "targets": "all",
                    "assay_types": "IC50"
                }
                
                response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                       json=payload, timeout=10)
                
                if response.status_code == 400:
                    self.log_test(f"Error Handling - {test_name}", True, 
                                f"Correctly rejected with 400: {response.json().get('detail', 'No detail')}")
                elif response.status_code == 503:
                    self.log_test(f"Error Handling - {test_name}", False, 
                                "Service unavailable (503) - model not loaded")
                else:
                    self.log_test(f"Error Handling - {test_name}", False, 
                                f"Unexpected response {response.status_code}: {response.text}")
                    
            except Exception as e:
                self.log_test(f"Error Handling - {test_name}", False, f"Error: {str(e)}")
    
    def test_database_integration(self):
        """Test 9: Database Integration - verify predictions are stored"""
        print("\n=== Testing Database Integration ===")
        
        # First make a prediction to ensure something is stored
        test_smiles = "CCO"  # Simple ethanol
        
        try:
            payload = {
                "smiles": test_smiles,
                "targets": ["EGFR"],
                "assay_types": "IC50"
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=15)
            
            if response.status_code == 200:
                self.log_test("Database Test Prediction", True, "Test prediction made successfully")
                
                # Wait a moment for database write
                time.sleep(1)
                
                # Try to check if there's a history endpoint for Gnosis I predictions
                try:
                    history_response = requests.get(f"{API_BASE}/predictions/history", timeout=10)
                    if history_response.status_code == 200:
                        history_data = history_response.json()
                        if isinstance(history_data, list) and len(history_data) > 0:
                            self.log_test("Database History Retrieval", True, 
                                        f"Found {len(history_data)} prediction records")
                        else:
                            self.log_test("Database History Retrieval", False, "No prediction history found")
                    else:
                        self.log_test("Database History Retrieval", False, 
                                    f"History endpoint failed: {history_response.status_code}")
                except:
                    # Try alternative endpoint for Gnosis I specific history
                    try:
                        gnosis_history_response = requests.get(f"{API_BASE}/gnosis-i/history", timeout=10)
                        if gnosis_history_response.status_code == 200:
                            self.log_test("Gnosis I Database Storage", True, "Gnosis I history endpoint accessible")
                        else:
                            self.log_test("Gnosis I Database Storage", False, 
                                        f"Gnosis I history endpoint: {gnosis_history_response.status_code}")
                    except:
                        self.log_test("Gnosis I Database Storage", False, "No database history endpoints accessible")
                        
            elif response.status_code == 503:
                self.log_test("Database Test Prediction", False, "Service unavailable for database test")
            else:
                self.log_test("Database Test Prediction", False, f"Test prediction failed: {response.status_code}")
                
        except Exception as e:
            self.log_test("Database Test Prediction", False, f"Error: {str(e)}")
    
    def test_model_performance_indicators(self):
        """Test 10: Model Performance Indicators"""
        print("\n=== Testing Model Performance Indicators ===")
        
        try:
            # Check info endpoint for performance metrics
            response = requests.get(f"{API_BASE}/gnosis-i/info", timeout=10)
            
            if response.status_code == 200:
                info_data = response.json()
                
                if info_data.get('available'):
                    # Check R² score
                    r2_score = info_data.get('r2_score', 0.0)
                    expected_r2 = 0.6281
                    
                    if r2_score > 0.5:  # Reasonable performance threshold
                        self.log_test("Model Performance R²", True, f"R² = {r2_score} (good performance)")
                    elif r2_score > 0.0:
                        self.log_test("Model Performance R²", True, f"R² = {r2_score} (moderate performance)")
                    else:
                        self.log_test("Model Performance R²", False, f"R² = {r2_score} (poor performance)")
                    
                    # Check if R² is close to expected value
                    if abs(r2_score - expected_r2) < 0.1:
                        self.log_test("Expected R² Score", True, f"R² = {r2_score} ≈ {expected_r2}")
                    else:
                        self.log_test("Expected R² Score", False, f"R² = {r2_score}, expected ~{expected_r2}")
                    
                    # Check number of targets
                    num_targets = info_data.get('num_targets', 0)
                    if num_targets > 0:
                        self.log_test("Model Target Coverage", True, f"Covers {num_targets} targets")
                    else:
                        self.log_test("Model Target Coverage", False, "No target information")
                        
                else:
                    self.log_test("Model Performance Check", False, "Model not available for performance check")
                    
            else:
                self.log_test("Model Performance Check", False, f"Info endpoint failed: {response.status_code}")
                
        except Exception as e:
            self.log_test("Model Performance Check", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run all Gnosis I backend tests"""
        print("🧪 Starting Gnosis I (Model 1) Ligand Activity Predictor Backend Testing")
        print(f"🔗 Backend URL: {BACKEND_URL}")
        print("=" * 80)
        
        # Run all tests
        self.test_backend_service_status()
        self.test_gnosis_model_loading()
        self.test_health_check_endpoint()
        self.test_gnosis_info_endpoint()
        self.test_gnosis_targets_endpoint()
        self.test_gnosis_prediction_aspirin()
        self.test_gnosis_prediction_imatinib()
        self.test_gnosis_error_handling()
        self.test_database_integration()
        self.test_model_performance_indicators()
        
        # Print summary
        print("\n" + "=" * 80)
        print("🎯 GNOSIS I BACKEND TESTING SUMMARY")
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
            print("🎉 GNOSIS I INTEGRATION: READY FOR PRODUCTION")
            return True
        elif success_rate >= 60:
            print("⚠️  GNOSIS I INTEGRATION: MOSTLY WORKING - MINOR ISSUES")
            return True
        else:
            print("❌ GNOSIS I INTEGRATION: MAJOR ISSUES - NEEDS ATTENTION")
            return False

if __name__ == "__main__":
    tester = GnosisIBackendTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)