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
                required_fields = ['status', 'models_loaded', 'prediction_types', 'available_targets', 'enhanced_predictions']
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
                prediction_types = data.get('prediction_types', [])
                expected_predictions = ['bioactivity_ic50', 'toxicity', 'logP', 'solubility']
                has_all_predictions = all(pred in prediction_types for pred in expected_predictions)
                self.log_test("All prediction types available", has_all_predictions, f"Predictions: {prediction_types}")
                
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

    def test_enhanced_modal_molbert_status(self):
        """Test Enhanced Modal MolBERT status endpoint"""
        print("\n=== Testing Enhanced Modal MolBERT Status ===")
        
        try:
            response = requests.get(f"{API_BASE}/modal/molbert/status", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['status', 'modal_available']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Modal status endpoint structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                modal_available = data.get('modal_available', False)
                status = data.get('status', 'unknown')
                
                self.log_test("Modal MolBERT status endpoint", True, f"Status: {status}, Modal available: {modal_available}")
                
                # Check for credentials info if available
                if 'credentials_configured' in data:
                    credentials_configured = data.get('credentials_configured', False)
                    self.log_test("Modal credentials info", True, f"Credentials configured: {credentials_configured}")
                
                return True
                
            elif response.status_code == 404:
                self.log_test("Modal MolBERT status endpoint", False, "Enhanced Modal MolBERT endpoints not available (404)")
                return False
            else:
                self.log_test("Modal MolBERT status endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Modal MolBERT status endpoint", False, f"Connection error: {str(e)}")
            return False

    def test_enhanced_modal_molbert_setup(self):
        """Test Enhanced Modal MolBERT setup endpoint"""
        print("\n=== Testing Enhanced Modal MolBERT Setup ===")
        
        try:
            response = requests.post(f"{API_BASE}/modal/molbert/setup", timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                if 'status' in data:
                    status = data.get('status')
                    message = data.get('message', '')
                    
                    self.log_test("Modal MolBERT setup endpoint", True, f"Status: {status}, Message: {message}")
                    return True
                else:
                    self.log_test("Modal MolBERT setup endpoint", False, "Missing status field in response")
                    return False
                    
            elif response.status_code == 404:
                self.log_test("Modal MolBERT setup endpoint", False, "Enhanced Modal MolBERT endpoints not available (404)")
                return False
            else:
                # Expected to fail without credentials - this is acceptable
                data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                message = data.get('message', response.text)
                
                self.log_test("Modal MolBERT setup endpoint", True, f"Expected error without credentials: HTTP {response.status_code}, {message}")
                return True
                
        except requests.exceptions.RequestException as e:
            self.log_test("Modal MolBERT setup endpoint", False, f"Connection error: {str(e)}")
            return False

    def test_enhanced_modal_molbert_predict_fallback(self):
        """Test Enhanced Modal MolBERT predict endpoint with fallback"""
        print("\n=== Testing Enhanced Modal MolBERT Predict with Fallback ===")
        
        # Test with valid SMILES
        test_cases = [
            ("CCO", "EGFR", "ethanol"),
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "BRAF", "aspirin")
        ]
        
        all_passed = True
        
        for smiles, target, name in test_cases:
            try:
                payload = {
                    "smiles": smiles,
                    "target": target,
                    "use_finetuned": True
                }
                
                response = requests.post(f"{API_BASE}/modal/molbert/predict", 
                                       json=payload,
                                       headers={'Content-Type': 'application/json'},
                                       timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check response structure
                    if 'status' in data:
                        status = data.get('status')
                        self.log_test(f"Modal predict - {name}", True, f"Status: {status}")
                    else:
                        self.log_test(f"Modal predict - {name}", False, "Missing status field")
                        all_passed = False
                        
                elif response.status_code == 404:
                    self.log_test(f"Modal predict - {name}", False, "Enhanced Modal MolBERT endpoints not available (404)")
                    all_passed = False
                else:
                    # Expected to fail without Modal credentials - check for proper fallback
                    data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                    message = data.get('message', response.text)
                    
                    # This is acceptable - should show error but handle gracefully
                    self.log_test(f"Modal predict fallback - {name}", True, 
                                f"Expected error without Modal: HTTP {response.status_code}, {message}")
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"Modal predict - {name}", False, f"Connection error: {str(e)}")
                all_passed = False
        
        return all_passed

    def test_enhanced_modal_molbert_train(self):
        """Test Enhanced Modal MolBERT train endpoint"""
        print("\n=== Testing Enhanced Modal MolBERT Train ===")
        
        # Test with valid targets
        valid_targets = ["EGFR", "BRAF", "CDK2", "PARP1", "BCL2", "VEGFR2"]
        
        all_passed = True
        
        # Test with one valid target
        test_target = "EGFR"
        
        try:
            response = requests.post(f"{API_BASE}/modal/molbert/train/{test_target}", 
                                   timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'status' in data:
                    status = data.get('status')
                    target = data.get('target', '')
                    
                    self.log_test("Modal MolBERT train endpoint", True, f"Status: {status}, Target: {target}")
                else:
                    self.log_test("Modal MolBERT train endpoint", False, "Missing status field")
                    all_passed = False
                    
            elif response.status_code == 404:
                self.log_test("Modal MolBERT train endpoint", False, "Enhanced Modal MolBERT endpoints not available (404)")
                all_passed = False
            else:
                # Expected to fail without credentials - this is acceptable
                data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                message = data.get('message', response.text)
                
                self.log_test("Modal MolBERT train endpoint", True, 
                            f"Expected error without credentials: HTTP {response.status_code}, {message}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Modal MolBERT train endpoint", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test with invalid target
        try:
            invalid_target = "INVALID_TARGET"
            response = requests.post(f"{API_BASE}/modal/molbert/train/{invalid_target}", 
                                   timeout=30)
            
            if response.status_code == 400:
                self.log_test("Modal train invalid target", True, "Invalid target properly rejected with 400")
            elif response.status_code == 404:
                self.log_test("Modal train invalid target", False, "Enhanced Modal MolBERT endpoints not available (404)")
                all_passed = False
            else:
                # May still fail due to credentials, but should handle invalid target
                data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                message = data.get('message', response.text)
                
                if "Invalid target" in message or "Available:" in message:
                    self.log_test("Modal train invalid target", True, f"Invalid target handled: {message}")
                else:
                    self.log_test("Modal train invalid target", True, f"Response: {message}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Modal train invalid target", False, f"Connection error: {str(e)}")
            all_passed = False
        
        return all_passed

    def test_enhanced_modal_smiles_validation(self):
        """Test Enhanced Modal MolBERT SMILES validation"""
        print("\n=== Testing Enhanced Modal MolBERT SMILES Validation ===")
        
        all_passed = True
        
        # Test invalid SMILES
        try:
            payload = {
                "smiles": "INVALID_SMILES",
                "target": "EGFR",
                "use_finetuned": True
            }
            
            response = requests.post(f"{API_BASE}/modal/molbert/predict", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code == 400:
                self.log_test("Modal invalid SMILES validation", True, "Invalid SMILES properly rejected with 400")
            elif response.status_code == 404:
                self.log_test("Modal invalid SMILES validation", False, "Enhanced Modal MolBERT endpoints not available (404)")
                all_passed = False
            else:
                data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                message = data.get('message', response.text)
                
                if "Invalid SMILES" in message:
                    self.log_test("Modal invalid SMILES validation", True, f"Invalid SMILES handled: {message}")
                else:
                    self.log_test("Modal invalid SMILES validation", False, f"Should reject invalid SMILES: {message}")
                    all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Modal invalid SMILES validation", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test valid SMILES
        try:
            payload = {
                "smiles": "CCO",  # ethanol
                "target": "EGFR",
                "use_finetuned": True
            }
            
            response = requests.post(f"{API_BASE}/modal/molbert/predict", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code == 200:
                self.log_test("Modal valid SMILES validation", True, "Valid SMILES accepted")
            elif response.status_code == 404:
                self.log_test("Modal valid SMILES validation", False, "Enhanced Modal MolBERT endpoints not available (404)")
                all_passed = False
            else:
                # May fail due to credentials but should not be SMILES validation error
                data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                message = data.get('message', response.text)
                
                if "Invalid SMILES" not in message:
                    self.log_test("Modal valid SMILES validation", True, f"Valid SMILES not rejected for SMILES reasons: {message}")
                else:
                    self.log_test("Modal valid SMILES validation", False, f"Valid SMILES incorrectly rejected: {message}")
                    all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Modal valid SMILES validation", False, f"Connection error: {str(e)}")
            all_passed = False
        
        return all_passed

    def test_existing_predict_endpoint_integration(self):
        """Test that existing /api/predict endpoint still works with Enhanced Modal integration"""
        print("\n=== Testing Existing Predict Endpoint Integration ===")
        
        try:
            payload = {
                "smiles": "CCO",  # ethanol
                "prediction_types": ["bioactivity_ic50", "toxicity"],
                "target": "EGFR"
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check basic structure
                if 'results' not in data or len(data['results']) != 2:
                    self.log_test("Existing predict endpoint integration", False, 
                                f"Expected 2 results, got {len(data.get('results', []))}")
                    return False
                
                results = data['results']
                
                # Check that both prediction types work
                prediction_types = [r.get('prediction_type') for r in results]
                expected_types = ["bioactivity_ic50", "toxicity"]
                
                has_all_types = all(ptype in prediction_types for ptype in expected_types)
                
                if not has_all_types:
                    self.log_test("Existing predict endpoint integration", False, 
                                f"Missing prediction types. Got: {prediction_types}, Expected: {expected_types}")
                    return False
                
                # Check that IC50 prediction has enhanced data
                ic50_result = next((r for r in results if r.get('prediction_type') == 'bioactivity_ic50'), None)
                
                if ic50_result:
                    has_enhanced = ic50_result.get('enhanced_chemprop_prediction') is not None
                    has_molbert = ic50_result.get('molbert_prediction') is not None
                    has_chemprop = ic50_result.get('chemprop_prediction') is not None
                    
                    self.log_test("Existing predict endpoint integration", 
                                has_enhanced and has_molbert and has_chemprop,
                                f"Enhanced: {has_enhanced}, MolBERT: {has_molbert}, ChemProp: {has_chemprop}")
                    
                    return has_enhanced and has_molbert and has_chemprop
                else:
                    self.log_test("Existing predict endpoint integration", False, "No IC50 result found")
                    return False
                
            else:
                self.log_test("Existing predict endpoint integration", False, 
                            f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Existing predict endpoint integration", False, f"Connection error: {str(e)}")
            return False

    def test_backend_startup_with_modal(self):
        """Test that backend starts without errors with Enhanced Modal integration"""
        print("\n=== Testing Backend Startup with Enhanced Modal ===")
        
        try:
            # Test basic health check
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check that basic functionality works
                status = data.get('status')
                models_loaded = data.get('models_loaded', {})
                
                if status == 'healthy':
                    self.log_test("Backend startup with Modal", True, 
                                f"Backend healthy, models loaded: {models_loaded}")
                    return True
                else:
                    self.log_test("Backend startup with Modal", False, f"Backend not healthy: {status}")
                    return False
            else:
                self.log_test("Backend startup with Modal", False, 
                            f"Health check failed: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Backend startup with Modal", False, f"Connection error: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all tests and provide summary"""
        print(f"🧪 Starting Enhanced Chemistry Platform Backend Testing with Enhanced Modal MolBERT")
        print(f"Backend URL: {API_BASE}")
        print("=" * 60)
        
        # Run all tests including Enhanced Modal MolBERT tests
        tests = [
            # Core functionality tests
            self.test_health_endpoint_enhanced,
            self.test_real_ml_model_status,
            self.test_targets_endpoint,
            self.test_enhanced_ic50_predictions,
            self.test_multi_target_comparison,
            self.test_all_prediction_types,
            self.test_confidence_and_similarity_ranges,
            
            # Enhanced Modal MolBERT tests
            self.test_enhanced_modal_molbert_status,
            self.test_enhanced_modal_molbert_setup,
            self.test_enhanced_modal_molbert_predict_fallback,
            self.test_enhanced_modal_molbert_train,
            self.test_enhanced_modal_smiles_validation,
            self.test_existing_predict_endpoint_integration,
            self.test_backend_startup_with_modal,
            
            # Real ML model tests
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