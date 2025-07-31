#!/usr/bin/env python3
"""
AI Modules Integration Testing
Tests the new comprehensive AI Modules integration with ChemBERTa, Chemprop Multi-Task, and Enhanced RDKit
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

class AIModulesIntegrationTester:
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
    
    def test_health_endpoint_ai_modules(self):
        """Test /api/health endpoint shows all systems healthy for AI Modules"""
        print("\n=== Testing AI Modules Health Check ===")
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check basic health
                status = data.get('status')
                self.log_test("Health endpoint status", status == 'healthy', f"Status: {status}")
                
                # Check models loaded for AI Modules
                models_loaded = data.get('models_loaded', {})
                expected_models = ['molbert', 'chemprop_simulation', 'real_ml_models']
                
                all_models_loaded = all(models_loaded.get(model, False) for model in expected_models)
                self.log_test("All AI models loaded", all_models_loaded, f"Models: {models_loaded}")
                
                # Check enhanced predictions
                enhanced_predictions = data.get('enhanced_predictions', False)
                self.log_test("Enhanced predictions available", enhanced_predictions, f"Enhanced: {enhanced_predictions}")
                
                # Check prediction types for Ligand Activity Predictor
                prediction_types = data.get('prediction_types', [])
                expected_types = ['bioactivity_ic50', 'toxicity', 'logP', 'solubility']
                has_all_types = all(ptype in prediction_types for ptype in expected_types)
                self.log_test("All prediction types available", has_all_types, f"Types: {prediction_types}")
                
                # Check available targets
                available_targets = data.get('available_targets', [])
                expected_targets = ['EGFR', 'BRAF', 'CDK2', 'PARP1', 'BCL2', 'VEGFR2']
                has_all_targets = all(target in available_targets for target in expected_targets)
                self.log_test("All targets available", has_all_targets, f"Targets: {available_targets}")
                
                return True
            else:
                self.log_test("Health endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Health endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_chemberta_endpoints(self):
        """Test ChemBERTa Multi-Task Model Integration endpoints"""
        print("\n=== Testing ChemBERTa Multi-Task Integration ===")
        
        all_passed = True
        
        # Test ChemBERTa status endpoint
        try:
            response = requests.get(f"{API_BASE}/chemberta/status", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("ChemBERTa status endpoint", True, f"Status: {data.get('status', 'unknown')}")
            elif response.status_code == 404:
                self.log_test("ChemBERTa status endpoint", False, "ChemBERTa endpoints not implemented (404)")
                all_passed = False
            else:
                self.log_test("ChemBERTa status endpoint", False, f"HTTP {response.status_code}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("ChemBERTa status endpoint", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test ChemBERTa targets endpoint
        try:
            response = requests.get(f"{API_BASE}/chemberta/targets", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                targets = data.get('targets', [])
                expected_oncoproteins = ['EGFR', 'HER2', 'VEGFR2', 'BRAF', 'MET', 'CDK4', 'CDK6', 'ALK', 'MDM2', 'PI3KCA']
                
                has_oncoproteins = any(target in str(targets) for target in expected_oncoproteins)
                self.log_test("ChemBERTa targets endpoint", has_oncoproteins, f"Targets include oncoproteins: {targets}")
            elif response.status_code == 404:
                self.log_test("ChemBERTa targets endpoint", False, "ChemBERTa endpoints not implemented (404)")
                all_passed = False
            else:
                self.log_test("ChemBERTa targets endpoint", False, f"HTTP {response.status_code}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("ChemBERTa targets endpoint", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test ChemBERTa predict endpoint with test molecules
        test_molecules = [
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
            ("CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", "imatinib")
        ]
        
        for smiles, name in test_molecules:
            try:
                payload = {"smiles": smiles}
                response = requests.post(f"{API_BASE}/chemberta/predict", 
                                       json=payload,
                                       headers={'Content-Type': 'application/json'},
                                       timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for IC50 predictions for 10 oncoproteins
                    predictions = data.get('predictions', {})
                    expected_oncoproteins = ['EGFR', 'HER2', 'VEGFR2', 'BRAF', 'MET', 'CDK4', 'CDK6', 'ALK', 'MDM2', 'PI3KCA']
                    
                    has_oncoprotein_predictions = any(onco in predictions for onco in expected_oncoproteins)
                    self.log_test(f"ChemBERTa predict - {name}", has_oncoprotein_predictions, 
                                f"Oncoprotein predictions: {list(predictions.keys())}")
                elif response.status_code == 404:
                    self.log_test(f"ChemBERTa predict - {name}", False, "ChemBERTa endpoints not implemented (404)")
                    all_passed = False
                else:
                    self.log_test(f"ChemBERTa predict - {name}", False, f"HTTP {response.status_code}")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"ChemBERTa predict - {name}", False, f"Connection error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_chemprop_multitask_endpoints(self):
        """Test Chemprop Multi-Task Model Integration endpoints"""
        print("\n=== Testing Chemprop Multi-Task Integration ===")
        
        all_passed = True
        
        # Test Chemprop Multi-Task status endpoint
        try:
            response = requests.get(f"{API_BASE}/chemprop-multitask/status", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("Chemprop Multi-Task status endpoint", True, f"Status: {data.get('status', 'unknown')}")
            elif response.status_code == 404:
                self.log_test("Chemprop Multi-Task status endpoint", False, "Chemprop Multi-Task endpoints not implemented (404)")
                all_passed = False
            else:
                self.log_test("Chemprop Multi-Task status endpoint", False, f"HTTP {response.status_code}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Chemprop Multi-Task status endpoint", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test Chemprop Multi-Task properties endpoint
        try:
            response = requests.get(f"{API_BASE}/chemprop-multitask/properties", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                properties = data.get('properties', [])
                expected_properties = ['bioactivity_ic50', 'toxicity', 'logP', 'solubility']
                
                has_all_properties = all(prop in str(properties) for prop in expected_properties)
                self.log_test("Chemprop Multi-Task properties endpoint", has_all_properties, 
                            f"Properties: {properties}")
            elif response.status_code == 404:
                self.log_test("Chemprop Multi-Task properties endpoint", False, "Chemprop Multi-Task endpoints not implemented (404)")
                all_passed = False
            else:
                self.log_test("Chemprop Multi-Task properties endpoint", False, f"HTTP {response.status_code}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Chemprop Multi-Task properties endpoint", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test Chemprop Multi-Task predict endpoint
        test_molecules = [
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
            ("CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", "imatinib")
        ]
        
        for smiles, name in test_molecules:
            try:
                payload = {
                    "smiles": smiles,
                    "properties": ["bioactivity_ic50", "toxicity", "logP", "solubility"]
                }
                response = requests.post(f"{API_BASE}/chemprop-multitask/predict", 
                                       json=payload,
                                       headers={'Content-Type': 'application/json'},
                                       timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for predictions with confidence scores
                    predictions = data.get('predictions', {})
                    expected_properties = ['bioactivity_ic50', 'toxicity', 'logP', 'solubility']
                    
                    has_all_predictions = all(prop in predictions for prop in expected_properties)
                    has_confidence_scores = all('confidence' in predictions.get(prop, {}) for prop in expected_properties if prop in predictions)
                    
                    self.log_test(f"Chemprop Multi-Task predict - {name}", 
                                has_all_predictions and has_confidence_scores,
                                f"Properties: {list(predictions.keys())}, Has confidence: {has_confidence_scores}")
                elif response.status_code == 404:
                    self.log_test(f"Chemprop Multi-Task predict - {name}", False, "Chemprop Multi-Task endpoints not implemented (404)")
                    all_passed = False
                else:
                    self.log_test(f"Chemprop Multi-Task predict - {name}", False, f"HTTP {response.status_code}")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"Chemprop Multi-Task predict - {name}", False, f"Connection error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_enhanced_rdkit_integration(self):
        """Test Enhanced RDKit Integration with unified prediction system"""
        print("\n=== Testing Enhanced RDKit Integration ===")
        
        # Test unified prediction system with all prediction types
        test_molecules = [
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
            ("CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", "imatinib")
        ]
        
        all_passed = True
        
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
                    
                    # Check that all prediction types are working
                    results = data.get('results', [])
                    prediction_types = [r.get('prediction_type') for r in results]
                    expected_types = ["bioactivity_ic50", "toxicity", "logP", "solubility"]
                    
                    has_all_types = all(ptype in prediction_types for ptype in expected_types)
                    
                    # Check for enhanced predictions
                    has_enhanced = any(r.get('enhanced_chemprop_prediction') for r in results)
                    
                    # Check for molecular properties
                    summary = data.get('summary', {})
                    has_molecular_props = 'molecular_properties' in summary
                    
                    self.log_test(f"Enhanced RDKit unified prediction - {name}", 
                                has_all_types and has_enhanced and has_molecular_props,
                                f"Types: {len(prediction_types)}/4, Enhanced: {has_enhanced}, MolProps: {has_molecular_props}")
                else:
                    self.log_test(f"Enhanced RDKit unified prediction - {name}", False, f"HTTP {response.status_code}")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"Enhanced RDKit unified prediction - {name}", False, f"Connection error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_ligand_activity_predictor_integration(self):
        """Test the main Ligand Activity Predictor Module integration"""
        print("\n=== Testing Ligand Activity Predictor Module Integration ===")
        
        # Test that all three AI models are accessible through the unified system
        test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin
        
        try:
            payload = {
                "smiles": test_smiles,
                "prediction_types": ["bioactivity_ic50", "toxicity", "logP", "solubility"],
                "target": "EGFR"
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for all three model types in results
                results = data['results']
                
                # Check for ChemBERTa predictions (MolBERT)
                has_chemberta = any(r.get('molbert_prediction') is not None for r in results)
                
                # Check for Chemprop predictions
                has_chemprop = any(r.get('chemprop_prediction') is not None for r in results)
                
                # Check for Enhanced RDKit predictions
                has_enhanced_rdkit = any(r.get('enhanced_chemprop_prediction') is not None for r in results)
                
                # Check summary shows enhanced models used
                summary = data.get('summary', {})
                enhanced_models_used = summary.get('enhanced_models_used', False)
                
                self.log_test("ChemBERTa model accessible", has_chemberta, f"ChemBERTa predictions: {has_chemberta}")
                self.log_test("Chemprop model accessible", has_chemprop, f"Chemprop predictions: {has_chemprop}")
                self.log_test("Enhanced RDKit accessible", has_enhanced_rdkit, f"Enhanced RDKit predictions: {has_enhanced_rdkit}")
                self.log_test("Enhanced models integration", enhanced_models_used, f"Enhanced models used: {enhanced_models_used}")
                
                # Test comprehensive property prediction
                prediction_types = [r.get('prediction_type') for r in results]
                expected_properties = ['bioactivity_ic50', 'toxicity', 'logP', 'solubility']
                has_all_properties = all(prop in prediction_types for prop in expected_properties)
                
                self.log_test("Comprehensive property prediction", has_all_properties, 
                            f"Properties: {prediction_types}")
                
                return has_chemberta and has_chemprop and has_enhanced_rdkit and enhanced_models_used
            else:
                self.log_test("Ligand Activity Predictor integration", False, f"HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Ligand Activity Predictor integration", False, f"Connection error: {str(e)}")
            return False
    
    def test_oncoprotein_predictions(self):
        """Test IC50 predictions for oncoproteins as specified in review"""
        print("\n=== Testing Oncoprotein IC50 Predictions ===")
        
        # Test with both test molecules
        test_molecules = [
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
            ("CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", "imatinib")
        ]
        
        # Expected oncoproteins from review request
        expected_oncoproteins = ['EGFR', 'HER2', 'VEGFR2', 'BRAF', 'MET', 'CDK4', 'CDK6', 'ALK', 'MDM2', 'PI3KCA']
        
        all_passed = True
        
        for smiles, name in test_molecules:
            # Test with available targets first
            available_targets = ['EGFR', 'BRAF', 'CDK2', 'PARP1', 'BCL2', 'VEGFR2']
            
            for target in available_targets:
                try:
                    payload = {
                        "smiles": smiles,
                        "prediction_types": ["bioactivity_ic50"],
                        "target": target
                    }
                    
                    response = requests.post(f"{API_BASE}/predict", 
                                           json=payload,
                                           headers={'Content-Type': 'application/json'},
                                           timeout=60)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'results' in data and len(data['results']) > 0:
                            result = data['results'][0]
                            enhanced_prediction = result.get('enhanced_chemprop_prediction')
                            
                            if enhanced_prediction:
                                ic50_nm = enhanced_prediction.get('ic50_nm')
                                confidence = enhanced_prediction.get('confidence')
                                
                                has_valid_ic50 = isinstance(ic50_nm, (int, float)) and ic50_nm > 0
                                has_valid_confidence = isinstance(confidence, (int, float)) and confidence > 0
                                
                                self.log_test(f"Oncoprotein IC50 - {name}/{target}", 
                                            has_valid_ic50 and has_valid_confidence,
                                            f"IC50: {ic50_nm} nM, Confidence: {confidence}")
                            else:
                                self.log_test(f"Oncoprotein IC50 - {name}/{target}", False, "No enhanced prediction")
                                all_passed = False
                        else:
                            self.log_test(f"Oncoprotein IC50 - {name}/{target}", False, "No results")
                            all_passed = False
                    else:
                        self.log_test(f"Oncoprotein IC50 - {name}/{target}", False, f"HTTP {response.status_code}")
                        all_passed = False
                        
                except requests.exceptions.RequestException as e:
                    self.log_test(f"Oncoprotein IC50 - {name}/{target}", False, f"Connection error: {str(e)}")
                    all_passed = False
        
        return all_passed
    
    def run_all_tests(self):
        """Run all AI Modules integration tests"""
        print(f"🧪 Starting AI Modules Integration Testing")
        print(f"Backend URL: {API_BASE}")
        print("=" * 60)
        
        # Run all AI Modules tests
        tests = [
            self.test_health_endpoint_ai_modules,
            self.test_chemberta_endpoints,
            self.test_chemprop_multitask_endpoints,
            self.test_enhanced_rdkit_integration,
            self.test_ligand_activity_predictor_integration,
            self.test_oncoprotein_predictions
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
        print("🏁 AI MODULES INTEGRATION TEST SUMMARY")
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
    tester = AIModulesIntegrationTester()
    passed, failed, results = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)