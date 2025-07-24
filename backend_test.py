#!/usr/bin/env python3
"""
Enhanced Predictive Chemistry Platform Backend Testing
Tests target-specific IC50 predictions and enhanced model validation
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

class EnhancedChemistryPlatformTester:
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
    
    def test_health_endpoint_enhanced(self):
        """Test the /api/health endpoint for enhanced predictions"""
        print("\n=== Testing Health Check with Enhanced Predictions ===")
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields for enhanced predictions
                required_fields = ['status', 'models_loaded', 'available_predictions', 'available_targets', 'enhanced_predictions']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Health endpoint structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check enhanced predictions availability
                enhanced_predictions = data.get('enhanced_predictions', False)
                available_targets = data.get('available_targets', [])
                expected_targets = ['EGFR', 'BRAF', 'CDK2', 'PARP1', 'BCL2', 'VEGFR2']
                
                self.log_test("Health endpoint response", True, f"Status: {data['status']}")
                self.log_test("Enhanced predictions available", enhanced_predictions, f"Enhanced predictions: {enhanced_predictions}")
                self.log_test("Available targets", len(available_targets) >= 3, f"Targets: {available_targets}")
                
                # Check prediction types
                available_predictions = data.get('available_predictions', [])
                expected_predictions = ['bioactivity_ic50', 'toxicity', 'logP', 'solubility']
                has_all_predictions = all(pred in available_predictions for pred in expected_predictions)
                self.log_test("All prediction types available", has_all_predictions, f"Predictions: {available_predictions}")
                
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
                    required_fields = ['target', 'available', 'description', 'model_type']
                    missing_fields = [field for field in required_fields if field not in target]
                    
                    if missing_fields:
                        self.log_test(f"Target {target.get('target', 'unknown')} structure", False, 
                                    f"Missing fields: {missing_fields}")
                        return False
                    
                    # Check model type
                    if target.get('model_type') != 'Enhanced RDKit-based':
                        self.log_test(f"Target {target.get('target')} model type", False, 
                                    f"Expected 'Enhanced RDKit-based', got '{target.get('model_type')}'")
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
    
    def test_enhanced_ic50_predictions(self):
        """Test enhanced IC50 predictions with aspirin and BRAF target"""
        print("\n=== Testing Enhanced IC50 Predictions ===")
        
        # Test with aspirin as specified in the review request
        test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin
        test_target = "BRAF"
        
        try:
            payload = {
                "smiles": test_smiles,
                "prediction_types": ["bioactivity_ic50"],
                "target": test_target
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' not in data or len(data['results']) == 0:
                    self.log_test("Enhanced IC50 prediction structure", False, "No results returned")
                    return False
                
                result = data['results'][0]
                
                # Check for enhanced ChemProp prediction
                enhanced_prediction = result.get('enhanced_chemprop_prediction')
                if not enhanced_prediction:
                    self.log_test("Enhanced ChemProp prediction", False, "No enhanced_chemprop_prediction field")
                    return False
                
                # Check required fields in enhanced prediction
                required_enhanced_fields = ['pic50', 'ic50_nm', 'confidence', 'similarity', 'target_specific', 'model_type']
                missing_enhanced_fields = [field for field in required_enhanced_fields if field not in enhanced_prediction]
                
                if missing_enhanced_fields:
                    self.log_test("Enhanced prediction fields", False, f"Missing: {missing_enhanced_fields}")
                    return False
                
                # Validate prediction values
                pic50 = enhanced_prediction.get('pic50')
                ic50_nm = enhanced_prediction.get('ic50_nm')
                confidence = enhanced_prediction.get('confidence')
                similarity = enhanced_prediction.get('similarity')
                target_specific = enhanced_prediction.get('target_specific')
                model_type = enhanced_prediction.get('model_type')
                
                # Check value ranges
                valid_pic50 = isinstance(pic50, (int, float)) and 4.0 <= pic50 <= 10.0
                valid_ic50 = isinstance(ic50_nm, (int, float)) and ic50_nm > 0
                valid_confidence = isinstance(confidence, (int, float)) and 0.4 <= confidence <= 0.95
                valid_similarity = isinstance(similarity, (int, float)) and 0.0 <= similarity <= 1.0
                
                self.log_test("Enhanced IC50 prediction values", 
                            valid_pic50 and valid_ic50 and valid_confidence and valid_similarity,
                            f"pIC50: {pic50}, IC50: {ic50_nm} nM, Confidence: {confidence}, Similarity: {similarity}")
                
                # Check target-specific and model type
                self.log_test("Target-specific prediction", target_specific == True, f"Target specific: {target_specific}")
                self.log_test("Enhanced model type", model_type == "Enhanced RDKit-based", f"Model type: {model_type}")
                
                # Check molecular properties
                molecular_properties = enhanced_prediction.get('molecular_properties')
                if molecular_properties:
                    self.log_test("Molecular properties data", True, 
                                f"Properties: {list(molecular_properties.keys())}")
                else:
                    self.log_test("Molecular properties data", False, "No molecular properties")
                
                return True
                
            else:
                self.log_test("Enhanced IC50 prediction", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Enhanced IC50 prediction", False, f"Request error: {str(e)}")
            return False
    
    def test_multi_target_comparison(self):
        """Test predictions for different targets with same molecule (EGFR vs BRAF)"""
        print("\n=== Testing Multi-Target Comparison ===")
        
        test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin
        targets = ["EGFR", "BRAF"]
        predictions = {}
        
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
                                       timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'results' in data and len(data['results']) > 0:
                        result = data['results'][0]
                        enhanced_prediction = result.get('enhanced_chemprop_prediction')
                        
                        if enhanced_prediction and 'pic50' in enhanced_prediction:
                            predictions[target] = enhanced_prediction['pic50']
                            self.log_test(f"Multi-target prediction - {target}", True, 
                                        f"pIC50: {enhanced_prediction['pic50']}")
                        else:
                            self.log_test(f"Multi-target prediction - {target}", False, 
                                        "No enhanced prediction data")
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
        
        # Check if different targets give different predictions
        if len(predictions) == 2:
            egfr_pic50 = predictions.get('EGFR')
            braf_pic50 = predictions.get('BRAF')
            
            if egfr_pic50 and braf_pic50:
                different_predictions = abs(egfr_pic50 - braf_pic50) > 0.1
                self.log_test("Target-specific logic verification", different_predictions, 
                            f"EGFR: {egfr_pic50}, BRAF: {braf_pic50}, Difference: {abs(egfr_pic50 - braf_pic50):.3f}")
            else:
                self.log_test("Target-specific logic verification", False, "Missing prediction values")
                all_passed = False
        
        return all_passed
    
    def test_all_prediction_types(self):
        """Test all 4 prediction types working together"""
        print("\n=== Testing All Prediction Types ===")
        
        test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin
        prediction_types = ["bioactivity_ic50", "toxicity", "logP", "solubility"]
        
        try:
            payload = {
                "smiles": test_smiles,
                "prediction_types": prediction_types,
                "target": "BRAF"
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' not in data or len(data['results']) != 4:
                    self.log_test("All prediction types", False, f"Expected 4 results, got {len(data.get('results', []))}")
                    return False
                
                results = data['results']
                prediction_success = {}
                
                for result in results:
                    pred_type = result.get('prediction_type')
                    
                    # Check basic fields
                    has_basic_fields = all(field in result for field in ['smiles', 'prediction_type', 'confidence'])
                    
                    # Check model predictions
                    has_molbert = result.get('molbert_prediction') is not None
                    has_chemprop = result.get('chemprop_prediction') is not None
                    
                    # For IC50, check enhanced prediction
                    if pred_type == "bioactivity_ic50":
                        has_enhanced = result.get('enhanced_chemprop_prediction') is not None
                        prediction_success[pred_type] = has_basic_fields and has_molbert and has_chemprop and has_enhanced
                    else:
                        prediction_success[pred_type] = has_basic_fields and has_molbert and has_chemprop
                    
                    self.log_test(f"Prediction type - {pred_type}", prediction_success[pred_type], 
                                f"MolBERT: {has_molbert}, ChemProp: {has_chemprop}")
                
                # Check summary
                summary = data.get('summary', {})
                enhanced_models_used = summary.get('enhanced_models_used', False)
                self.log_test("Enhanced models used in summary", enhanced_models_used, f"Enhanced models: {enhanced_models_used}")
                
                all_types_working = all(prediction_success.values())
                self.log_test("All prediction types working", all_types_working, 
                            f"Success: {sum(prediction_success.values())}/4")
                
                return all_types_working
                
            else:
                self.log_test("All prediction types", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("All prediction types", False, f"Request error: {str(e)}")
            return False
    
    def test_confidence_and_similarity_ranges(self):
        """Test that confidence scores are reasonable (0.4-0.95) and similarity is calculated"""
        print("\n=== Testing Confidence and Similarity Ranges ===")
        
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
                    "prediction_types": ["bioactivity_ic50"],
                    "target": "EGFR"
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
                            confidence = enhanced_prediction.get('confidence')
                            similarity = enhanced_prediction.get('similarity')
                            
                            # Check confidence range (0.4-0.95)
                            valid_confidence = isinstance(confidence, (int, float)) and 0.4 <= confidence <= 0.95
                            
                            # Check similarity range (0.0-1.0)
                            valid_similarity = isinstance(similarity, (int, float)) and 0.0 <= similarity <= 1.0
                            
                            self.log_test(f"Confidence range - {name}", valid_confidence, 
                                        f"Confidence: {confidence} (valid: {valid_confidence})")
                            self.log_test(f"Similarity calculation - {name}", valid_similarity, 
                                        f"Similarity: {similarity} (valid: {valid_similarity})")
                            
                            if not (valid_confidence and valid_similarity):
                                all_passed = False
                        else:
                            self.log_test(f"Enhanced prediction - {name}", False, "No enhanced prediction")
                            all_passed = False
                    else:
                        self.log_test(f"Prediction result - {name}", False, "No results")
                        all_passed = False
                else:
                    self.log_test(f"Prediction request - {name}", False, f"HTTP {response.status_code}")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"Prediction request - {name}", False, f"Request error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_error_handling(self):
        """Test error handling with invalid SMILES"""
        print("\n=== Testing Error Handling ===")
        
        all_passed = True
        
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
        
        # Test empty SMILES
        try:
            payload = {
                "smiles": "",
                "prediction_types": ["bioactivity_ic50"],
                "target": "EGFR"
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code == 400:
                self.log_test("Empty SMILES handling", True, "Empty SMILES properly rejected")
            else:
                self.log_test("Empty SMILES handling", False, f"Should return 400, got {response.status_code}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Empty SMILES handling", False, f"Request error: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def test_real_ml_model_status(self):
        """Test /api/health endpoint for real ML model status"""
        print("\n=== Testing Real ML Model Status ===")
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for real ML model fields
                models_loaded = data.get('models_loaded', {})
                real_ml_models = models_loaded.get('real_ml_models', False)
                real_ml_targets = data.get('real_ml_targets', {})
                model_type = data.get('model_type', 'heuristic')
                
                self.log_test("Real ML models status field", 'real_ml_models' in models_loaded, 
                            f"real_ml_models present: {'real_ml_models' in models_loaded}")
                
                self.log_test("Real ML targets status", 'real_ml_targets' in data,
                            f"real_ml_targets: {real_ml_targets}")
                
                # Check model type
                expected_model_type = "real_ml" if real_ml_models else "heuristic"
                self.log_test("Model type indication", model_type == expected_model_type,
                            f"Model type: {model_type} (expected: {expected_model_type})")
                
                # Check specific targets
                common_targets = ["EGFR", "BRAF", "CDK2"]
                for target in common_targets:
                    target_status = real_ml_targets.get(target, False)
                    self.log_test(f"Real ML model for {target}", target_status,
                                f"{target} real model loaded: {target_status}")
                
                return True
            else:
                self.log_test("Real ML model status check", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Real ML model status check", False, f"Connection error: {str(e)}")
            return False
    
    def test_chembl_data_integration(self):
        """Test ChEMBL data integration by checking if models can be initialized"""
        print("\n=== Testing ChEMBL Data Integration ===")
        
        # This test checks if the system can handle ChEMBL data by making predictions
        # and checking for real ML model responses vs heuristic fallbacks
        
        test_molecules = [
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),
            ("CCO", "ethanol")
        ]
        
        targets = ["EGFR", "BRAF", "CDK2"]
        
        all_passed = True
        real_model_responses = 0
        heuristic_responses = 0
        
        for smiles, name in test_molecules:
            for target in targets:
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
                                # Check if this looks like a real ML model response
                                has_performance_metrics = 'model_performance' in enhanced_prediction
                                has_training_size = 'training_size' in enhanced_prediction
                                
                                if has_performance_metrics and has_training_size:
                                    real_model_responses += 1
                                    self.log_test(f"Real ML response - {name}/{target}", True,
                                                f"Performance metrics and training size present")
                                else:
                                    heuristic_responses += 1
                                    self.log_test(f"Heuristic response - {name}/{target}", True,
                                                f"Using heuristic model (no performance metrics)")
                            else:
                                self.log_test(f"Prediction response - {name}/{target}", False,
                                            "No enhanced prediction data")
                                all_passed = False
                        else:
                            self.log_test(f"Prediction response - {name}/{target}", False, "No results")
                            all_passed = False
                    else:
                        self.log_test(f"Prediction request - {name}/{target}", False, 
                                    f"HTTP {response.status_code}")
                        all_passed = False
                        
                except requests.exceptions.RequestException as e:
                    self.log_test(f"Prediction request - {name}/{target}", False, f"Request error: {str(e)}")
                    all_passed = False
        
        # Summary of model usage
        total_requests = len(test_molecules) * len(targets)
        self.log_test("ChEMBL data integration summary", all_passed,
                    f"Real ML responses: {real_model_responses}, Heuristic responses: {heuristic_responses}, Total: {total_requests}")
        
        return all_passed
    
    def test_real_vs_heuristic_comparison(self):
        """Test performance comparison between real ML models and heuristic models"""
        print("\n=== Testing Real vs Heuristic Model Comparison ===")
        
        # Test with aspirin on EGFR to see if we get different types of responses
        test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin
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
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and len(data['results']) > 0:
                    result = data['results'][0]
                    enhanced_prediction = result.get('enhanced_chemprop_prediction')
                    
                    if enhanced_prediction:
                        # Check what type of model was used
                        has_performance = 'model_performance' in enhanced_prediction
                        has_training_size = 'training_size' in enhanced_prediction
                        model_type = enhanced_prediction.get('model_type', 'unknown')
                        
                        if has_performance and has_training_size:
                            # Real ML model response
                            performance = enhanced_prediction['model_performance']
                            training_size = enhanced_prediction['training_size']
                            
                            self.log_test("Real ML model detection", True,
                                        f"Real ML model used with {training_size} training samples")
                            self.log_test("Model performance metrics", True,
                                        f"Test R²: {performance.get('test_r2', 'N/A')}, RMSE: {performance.get('test_rmse', 'N/A')}")
                            
                            # Check if confidence is based on model performance
                            confidence = enhanced_prediction.get('confidence', 0)
                            similarity = enhanced_prediction.get('similarity', 0)
                            
                            self.log_test("Real ML confidence calculation", confidence > 0.3,
                                        f"Confidence: {confidence}, Similarity: {similarity}")
                            
                        else:
                            # Heuristic model response
                            self.log_test("Heuristic model fallback", True,
                                        f"Using heuristic model (model_type: {model_type})")
                            
                            # Check heuristic model characteristics
                            pic50 = enhanced_prediction.get('pic50')
                            confidence = enhanced_prediction.get('confidence')
                            target_specific = enhanced_prediction.get('target_specific', False)
                            
                            self.log_test("Heuristic model characteristics", 
                                        pic50 is not None and confidence is not None and target_specific,
                                        f"pIC50: {pic50}, confidence: {confidence}, target_specific: {target_specific}")
                        
                        return True
                    else:
                        self.log_test("Model comparison", False, "No enhanced prediction data")
                        return False
                else:
                    self.log_test("Model comparison", False, "No results returned")
                    return False
            else:
                self.log_test("Model comparison", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Model comparison", False, f"Request error: {str(e)}")
            return False
    
    def test_error_handling_and_fallback(self):
        """Test error handling scenarios and fallback to heuristic models"""
        print("\n=== Testing Error Handling and Fallback ===")
        
        all_passed = True
        
        # Test 1: Invalid SMILES should be rejected before reaching models
        try:
            payload = {
                "smiles": "INVALID_SMILES_STRING",
                "prediction_types": ["bioactivity_ic50"],
                "target": "EGFR"
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code == 400:
                self.log_test("Invalid SMILES rejection", True, "Invalid SMILES properly rejected with 400")
            else:
                self.log_test("Invalid SMILES rejection", False, f"Expected 400, got {response.status_code}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Invalid SMILES rejection", False, f"Request error: {str(e)}")
            all_passed = False
        
        # Test 2: Valid SMILES with unsupported target should still work (fallback)
        try:
            payload = {
                "smiles": "CCO",  # ethanol
                "prediction_types": ["bioactivity_ic50"],
                "target": "UNSUPPORTED_TARGET"
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    result = data['results'][0]
                    enhanced_prediction = result.get('enhanced_chemprop_prediction')
                    
                    if enhanced_prediction:
                        # Should fallback to heuristic model
                        has_performance = 'model_performance' in enhanced_prediction
                        self.log_test("Unsupported target fallback", not has_performance,
                                    f"Fallback to heuristic for unsupported target (no performance metrics)")
                    else:
                        self.log_test("Unsupported target fallback", False, "No prediction returned")
                        all_passed = False
                else:
                    self.log_test("Unsupported target fallback", False, "No results")
                    all_passed = False
            else:
                self.log_test("Unsupported target fallback", False, f"HTTP {response.status_code}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Unsupported target fallback", False, f"Request error: {str(e)}")
            all_passed = False
        
        # Test 3: Empty prediction types should be handled
        try:
            payload = {
                "smiles": "CCO",
                "prediction_types": [],
                "target": "EGFR"
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            # Should return empty results or handle gracefully
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                self.log_test("Empty prediction types handling", len(results) == 0,
                            f"Empty prediction types handled gracefully (results: {len(results)})")
            else:
                self.log_test("Empty prediction types handling", True, 
                            f"Rejected empty prediction types with {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Empty prediction types handling", False, f"Request error: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def test_model_information_reporting(self):
        """Test if health endpoint correctly reports model information"""
        print("\n=== Testing Model Information Reporting ===")
        
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check comprehensive model information
                models_loaded = data.get('models_loaded', {})
                real_ml_targets = data.get('real_ml_targets', {})
                model_type = data.get('model_type', 'unknown')
                available_targets = data.get('available_targets', [])
                
                # Test model loading status reporting
                has_molbert = models_loaded.get('molbert', False)
                has_chemprop_sim = models_loaded.get('chemprop_simulation', False)
                has_real_ml = models_loaded.get('real_ml_models', False)
                
                self.log_test("Model loading status complete", 
                            all(key in models_loaded for key in ['molbert', 'chemprop_simulation', 'real_ml_models']),
                            f"Models loaded: {models_loaded}")
                
                # Test target-specific model reporting
                expected_targets = ["EGFR", "BRAF", "CDK2"]
                targets_reported = all(target in real_ml_targets for target in expected_targets)
                
                self.log_test("Target-specific model reporting", targets_reported,
                            f"Real ML targets: {real_ml_targets}")
                
                # Test model type consistency
                real_models_available = any(real_ml_targets.values()) if real_ml_targets else False
                expected_type = "real_ml" if real_models_available else "heuristic"
                
                self.log_test("Model type consistency", model_type == expected_type,
                            f"Model type: {model_type}, Real models available: {real_models_available}")
                
                # Test available targets list
                self.log_test("Available targets reporting", len(available_targets) >= 3,
                            f"Available targets: {available_targets}")
                
                return True
            else:
                self.log_test("Model information reporting", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Model information reporting", False, f"Connection error: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all tests and provide summary"""
        print(f"🧪 Starting Enhanced Chemistry Platform Backend Testing with Real ML Models")
        print(f"Backend URL: {API_BASE}")
        print("=" * 60)
        
        # Run all tests including new real ML model tests
        tests = [
            self.test_health_endpoint_enhanced,
            self.test_real_ml_model_status,
            self.test_targets_endpoint,
            self.test_enhanced_ic50_predictions,
            self.test_multi_target_comparison,
            self.test_all_prediction_types,
            self.test_confidence_and_similarity_ranges,
            self.test_chembl_data_integration,
            self.test_real_vs_heuristic_comparison,
            self.test_error_handling_and_fallback,
            self.test_model_information_reporting,
            self.test_error_handling
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
    tester = EnhancedChemistryPlatformTester()
    passed, failed, results = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)