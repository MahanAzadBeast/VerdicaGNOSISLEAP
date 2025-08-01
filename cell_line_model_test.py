#!/usr/bin/env python3
"""
Cell Line Response Model Integration Testing
Comprehensive testing of the Cell Line Response Model backend integration
Focus on multi-modal predictions, genomics-informed logic, and clinical insights
"""

import requests
import json
import time
import sys
from typing import Dict, List, Any
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from pathlib import Path

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment - use localhost for testing
BACKEND_URL = 'http://localhost:8001'
API_BASE = f"{BACKEND_URL}/api"

class CellLineModelTester:
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
    
    def test_cell_line_health_endpoint(self):
        """Test /api/cell-line/health endpoint functionality"""
        print("\n=== Testing Cell Line Health Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/cell-line/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['status', 'model_type', 'architecture', 'model_status', 'capabilities', 'supported_features']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Cell Line Health Endpoint Structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Validate specific values
                expected_model_type = "Cell_Line_Response_Model"
                expected_architecture = "Multi_Modal_Molecular_Genomic"
                
                model_type_correct = data.get('model_type') == expected_model_type
                architecture_correct = data.get('architecture') == expected_architecture
                
                self.log_test("Cell Line Model Type", model_type_correct, f"Model type: {data.get('model_type')}")
                self.log_test("Cell Line Architecture", architecture_correct, f"Architecture: {data.get('architecture')}")
                
                # Check capabilities
                capabilities = data.get('capabilities', {})
                expected_capabilities = ['multi_modal_prediction', 'genomic_integration', 'uncertainty_quantification', 'cancer_type_specific']
                
                capabilities_present = all(cap in capabilities for cap in expected_capabilities)
                self.log_test("Cell Line Capabilities", capabilities_present, f"Capabilities: {list(capabilities.keys())}")
                
                # Check supported features
                supported_features = data.get('supported_features', {})
                expected_features = ['molecular', 'genomic', 'fusion']
                
                features_present = all(feat in supported_features for feat in expected_features)
                self.log_test("Cell Line Supported Features", features_present, f"Features: {list(supported_features.keys())}")
                
                return True
                
            else:
                self.log_test("Cell Line Health Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Cell Line Health Endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_cell_line_examples_endpoint(self):
        """Test /api/cell-line/examples endpoint for sample data"""
        print("\n=== Testing Cell Line Examples Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/cell-line/examples", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check structure
                if 'cell_lines' not in data or 'example_drugs' not in data:
                    self.log_test("Cell Line Examples Structure", False, "Missing cell_lines or example_drugs")
                    return False
                
                cell_lines = data['cell_lines']
                example_drugs = data['example_drugs']
                
                # Validate cell lines
                if len(cell_lines) < 3:
                    self.log_test("Cell Line Examples Count", False, f"Expected at least 3 cell lines, got {len(cell_lines)}")
                    return False
                
                # Check cell line structure
                for cell_line in cell_lines:
                    required_fields = ['cell_line_name', 'cancer_type', 'genomic_features']
                    missing_fields = [field for field in required_fields if field not in cell_line]
                    
                    if missing_fields:
                        self.log_test(f"Cell Line {cell_line.get('cell_line_name', 'unknown')} Structure", False, 
                                    f"Missing fields: {missing_fields}")
                        return False
                    
                    # Check genomic features structure
                    genomic_features = cell_line['genomic_features']
                    required_genomic = ['mutations', 'cnvs', 'expression']
                    missing_genomic = [field for field in required_genomic if field not in genomic_features]
                    
                    if missing_genomic:
                        self.log_test(f"Cell Line {cell_line['cell_line_name']} Genomic Features", False,
                                    f"Missing genomic fields: {missing_genomic}")
                        return False
                
                # Check example drugs
                if len(example_drugs) < 2:
                    self.log_test("Example Drugs Count", False, f"Expected at least 2 drugs, got {len(example_drugs)}")
                    return False
                
                # Validate drug structure
                for drug in example_drugs:
                    required_drug_fields = ['name', 'smiles', 'target']
                    missing_drug_fields = [field for field in required_drug_fields if field not in drug]
                    
                    if missing_drug_fields:
                        self.log_test(f"Drug {drug.get('name', 'unknown')} Structure", False,
                                    f"Missing fields: {missing_drug_fields}")
                        return False
                
                # Check for specific expected cell lines and drugs
                cell_line_names = [cl['cell_line_name'] for cl in cell_lines]
                drug_names = [drug['name'] for drug in example_drugs]
                
                expected_cell_lines = ['A549', 'MCF7', 'HCT116']
                expected_drugs = ['Erlotinib', 'Trametinib']
                
                cell_lines_present = all(cl in cell_line_names for cl in expected_cell_lines)
                drugs_present = all(drug in drug_names for drug in expected_drugs)
                
                self.log_test("Expected Cell Lines Present", cell_lines_present, f"Cell lines: {cell_line_names}")
                self.log_test("Expected Drugs Present", drugs_present, f"Drugs: {drug_names}")
                
                return True
                
            else:
                self.log_test("Cell Line Examples Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Cell Line Examples Endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_cell_line_predict_endpoint(self):
        """Test /api/cell-line/predict endpoint with multi-modal predictions"""
        print("\n=== Testing Cell Line Predict Endpoint ===")
        
        # Test case: Erlotinib + A549 (KRAS mutated lung cancer) → Expected resistance
        test_payload = {
            "smiles": "Cc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCOC",  # Erlotinib
            "drug_name": "Erlotinib",
            "cell_line": {
                "cell_line_name": "A549",
                "cancer_type": "LUNG",
                "genomic_features": {
                    "mutations": {"TP53": 1, "KRAS": 1, "EGFR": 0, "BRAF": 0},
                    "cnvs": {"MYC": 1, "CDKN2A": -1, "PTEN": 0},
                    "expression": {"EGFR": -0.5, "KRAS": 1.2, "TP53": -1.8}
                }
            }
        }
        
        try:
            response = requests.post(f"{API_BASE}/cell-line/predict", 
                                   json=test_payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                required_fields = ['drug_name', 'cell_line_name', 'cancer_type', 'predicted_ic50_nm', 
                                 'predicted_pic50', 'uncertainty', 'confidence', 'sensitivity_class', 
                                 'genomic_context', 'prediction_timestamp']
                
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Cell Line Predict Response Structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Validate prediction values
                ic50_nm = data.get('predicted_ic50_nm')
                pic50 = data.get('predicted_pic50')
                uncertainty = data.get('uncertainty')
                confidence = data.get('confidence')
                sensitivity_class = data.get('sensitivity_class')
                
                # Check value ranges
                valid_ic50 = isinstance(ic50_nm, (int, float)) and ic50_nm > 0
                valid_pic50 = isinstance(pic50, (int, float)) and 4.0 <= pic50 <= 10.0
                valid_uncertainty = isinstance(uncertainty, (int, float)) and 0.0 <= uncertainty <= 1.0
                valid_confidence = isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
                valid_sensitivity = sensitivity_class in ['SENSITIVE', 'MODERATE', 'RESISTANT']
                
                self.log_test("Cell Line Prediction Values Valid", 
                            valid_ic50 and valid_pic50 and valid_uncertainty and valid_confidence and valid_sensitivity,
                            f"IC50: {ic50_nm} nM, pIC50: {pic50}, Uncertainty: {uncertainty}, Confidence: {confidence}, Class: {sensitivity_class}")
                
                # Check genomic context
                genomic_context = data.get('genomic_context', {})
                expected_context_fields = ['key_mutations', 'amplifications', 'deletions', 'high_expression', 'low_expression']
                
                context_complete = all(field in genomic_context for field in expected_context_fields)
                self.log_test("Genomic Context Complete", context_complete, f"Context fields: {list(genomic_context.keys())}")
                
                # Check if KRAS mutation is detected in context
                key_mutations = genomic_context.get('key_mutations', [])
                kras_detected = 'KRAS' in key_mutations
                self.log_test("KRAS Mutation Detected", kras_detected, f"Key mutations: {key_mutations}")
                
                # For Erlotinib + KRAS mutated A549, expect resistance (high IC50)
                expected_resistance = ic50_nm > 1000  # Should be resistant
                self.log_test("Expected Erlotinib Resistance in KRAS Mutated A549", expected_resistance,
                            f"IC50: {ic50_nm} nM (expected > 1000 nM for resistance)")
                
                return True
                
            else:
                self.log_test("Cell Line Predict Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Cell Line Predict Endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_cell_line_trametinib_sensitivity(self):
        """Test Trametinib + KRAS mutated cells → Expected sensitivity"""
        print("\n=== Testing Trametinib Sensitivity in KRAS Mutated Cells ===")
        
        # Test case: Trametinib + HCT116 (KRAS mutated colon cancer) → Expected sensitivity
        test_payload = {
            "smiles": "CC(C)[C@H](C(=O)N1CC2=C(C1=O)SC(=N2)C3=CC(=C(C=C3)F)F)N4C(=O)C5=C(C4=O)C=CC=C5I",  # Trametinib
            "drug_name": "Trametinib",
            "cell_line": {
                "cell_line_name": "HCT116",
                "cancer_type": "COLON",
                "genomic_features": {
                    "mutations": {"TP53": 0, "KRAS": 1, "PIK3CA": 1, "BRAF": 0},
                    "cnvs": {"MYC": 1, "PTEN": -1, "CDKN2A": 0},
                    "expression": {"EGFR": 1.5, "KRAS": 2.0, "TP53": 0.5}
                }
            }
        }
        
        try:
            response = requests.post(f"{API_BASE}/cell-line/predict", 
                                   json=test_payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                ic50_nm = data.get('predicted_ic50_nm')
                sensitivity_class = data.get('sensitivity_class')
                genomic_context = data.get('genomic_context', {})
                
                # Check if KRAS mutation is detected
                key_mutations = genomic_context.get('key_mutations', [])
                kras_detected = 'KRAS' in key_mutations
                
                # For Trametinib + KRAS mutated cells, expect sensitivity (low IC50)
                expected_sensitivity = ic50_nm < 500  # Should be sensitive
                
                self.log_test("KRAS Mutation Detected for Trametinib", kras_detected, f"Key mutations: {key_mutations}")
                self.log_test("Expected Trametinib Sensitivity in KRAS Mutated HCT116", expected_sensitivity,
                            f"IC50: {ic50_nm} nM (expected < 500 nM for sensitivity), Class: {sensitivity_class}")
                
                return True
                
            else:
                self.log_test("Trametinib Sensitivity Test", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Trametinib Sensitivity Test", False, f"Connection error: {str(e)}")
            return False
    
    def test_cell_line_compare_endpoint(self):
        """Test /api/cell-line/compare endpoint for multiple cell lines"""
        print("\n=== Testing Cell Line Compare Endpoint ===")
        
        # Compare Erlotinib sensitivity across different cancer types
        test_payload = {
            "smiles": "Cc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCOC",  # Erlotinib
            "drug_name": "Erlotinib",
            "cell_lines": [
                {
                    "cell_line_name": "A549",
                    "cancer_type": "LUNG",
                    "genomic_features": {
                        "mutations": {"TP53": 1, "KRAS": 1, "EGFR": 0, "BRAF": 0},
                        "cnvs": {"MYC": 1, "CDKN2A": -1, "PTEN": 0},
                        "expression": {"EGFR": -0.5, "KRAS": 1.2, "TP53": -1.8}
                    }
                },
                {
                    "cell_line_name": "MCF7",
                    "cancer_type": "BREAST",
                    "genomic_features": {
                        "mutations": {"TP53": 0, "PIK3CA": 1, "KRAS": 0, "EGFR": 0},
                        "cnvs": {"MYC": 0, "CDKN2A": 0, "PTEN": 0},
                        "expression": {"EGFR": 0.3, "KRAS": -0.2, "TP53": 0.8}
                    }
                },
                {
                    "cell_line_name": "HCT116",
                    "cancer_type": "COLON",
                    "genomic_features": {
                        "mutations": {"TP53": 0, "KRAS": 1, "PIK3CA": 1, "BRAF": 0},
                        "cnvs": {"MYC": 1, "PTEN": -1, "CDKN2A": 0},
                        "expression": {"EGFR": 1.5, "KRAS": 2.0, "TP53": 0.5}
                    }
                }
            ]
        }
        
        try:
            response = requests.post(f"{API_BASE}/cell-line/compare", 
                                   json=test_payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                if 'predictions' not in data or 'summary' not in data:
                    self.log_test("Cell Line Compare Response Structure", False, "Missing predictions or summary")
                    return False
                
                predictions = data['predictions']
                summary = data['summary']
                
                # Check number of predictions
                if len(predictions) != 3:
                    self.log_test("Cell Line Compare Predictions Count", False, f"Expected 3 predictions, got {len(predictions)}")
                    return False
                
                # Check summary structure
                required_summary_fields = ['total_cell_lines', 'ic50_range', 'sensitivity_distribution', 'most_sensitive', 'most_resistant']
                missing_summary_fields = [field for field in required_summary_fields if field not in summary]
                
                if missing_summary_fields:
                    self.log_test("Cell Line Compare Summary Structure", False, f"Missing summary fields: {missing_summary_fields}")
                    return False
                
                # Validate IC50 range
                ic50_range = summary.get('ic50_range', {})
                if 'min_nm' not in ic50_range or 'max_nm' not in ic50_range or 'fold_difference' not in ic50_range:
                    self.log_test("IC50 Range Structure", False, "Missing IC50 range fields")
                    return False
                
                # Check fold difference calculation
                min_ic50 = ic50_range['min_nm']
                max_ic50 = ic50_range['max_nm']
                fold_diff = ic50_range['fold_difference']
                
                expected_fold_diff = max_ic50 / min_ic50 if min_ic50 > 0 else 0
                fold_diff_correct = abs(fold_diff - expected_fold_diff) < 0.1
                
                self.log_test("Fold Difference Calculation", fold_diff_correct, 
                            f"Min: {min_ic50} nM, Max: {max_ic50} nM, Fold diff: {fold_diff}")
                
                # Check sensitivity distribution
                sensitivity_dist = summary.get('sensitivity_distribution', {})
                total_classified = sensitivity_dist.get('sensitive', 0) + sensitivity_dist.get('moderate', 0) + sensitivity_dist.get('resistant', 0)
                
                self.log_test("Sensitivity Distribution", total_classified == 3, 
                            f"Distribution: {sensitivity_dist}")
                
                # Check most sensitive/resistant identification
                most_sensitive = summary.get('most_sensitive')
                most_resistant = summary.get('most_resistant')
                
                cell_line_names = [p['cell_line_name'] for p in predictions]
                sensitive_valid = most_sensitive in cell_line_names
                resistant_valid = most_resistant in cell_line_names
                
                self.log_test("Most Sensitive/Resistant Identification", sensitive_valid and resistant_valid,
                            f"Most sensitive: {most_sensitive}, Most resistant: {most_resistant}")
                
                # Check genomics-informed differences
                # A549 (KRAS mutated) should be more resistant to Erlotinib than MCF7 (KRAS wild-type)
                a549_prediction = next((p for p in predictions if p['cell_line_name'] == 'A549'), None)
                mcf7_prediction = next((p for p in predictions if p['cell_line_name'] == 'MCF7'), None)
                
                if a549_prediction and mcf7_prediction:
                    a549_ic50 = a549_prediction['predicted_ic50_nm']
                    mcf7_ic50 = mcf7_prediction['predicted_ic50_nm']
                    
                    genomics_informed = a549_ic50 > mcf7_ic50  # KRAS mutated should be more resistant
                    self.log_test("Genomics-Informed Prediction Logic", genomics_informed,
                                f"A549 (KRAS mut): {a549_ic50} nM, MCF7 (KRAS wt): {mcf7_ic50} nM")
                
                return True
                
            else:
                self.log_test("Cell Line Compare Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Cell Line Compare Endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_smiles_processing_and_validation(self):
        """Test SMILES processing and validation"""
        print("\n=== Testing SMILES Processing and Validation ===")
        
        # Test valid SMILES
        valid_payload = {
            "smiles": "Cc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCOC",  # Erlotinib
            "drug_name": "Erlotinib",
            "cell_line": {
                "cell_line_name": "A549",
                "cancer_type": "LUNG",
                "genomic_features": {
                    "mutations": {"TP53": 1, "KRAS": 1},
                    "cnvs": {"MYC": 1},
                    "expression": {"EGFR": -0.5}
                }
            }
        }
        
        try:
            response = requests.post(f"{API_BASE}/cell-line/predict", 
                                   json=valid_payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code == 200:
                self.log_test("Valid SMILES Processing", True, "Valid SMILES processed successfully")
            else:
                self.log_test("Valid SMILES Processing", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Valid SMILES Processing", False, f"Error: {str(e)}")
            return False
        
        # Test invalid SMILES
        invalid_payload = valid_payload.copy()
        invalid_payload["smiles"] = "INVALID_SMILES_STRING"
        
        try:
            response = requests.post(f"{API_BASE}/cell-line/predict", 
                                   json=invalid_payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code == 400:
                self.log_test("Invalid SMILES Rejection", True, "Invalid SMILES properly rejected")
            else:
                self.log_test("Invalid SMILES Rejection", False, f"Expected 400, got {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Invalid SMILES Rejection", False, f"Error: {str(e)}")
            return False
        
        # Test empty SMILES
        empty_payload = valid_payload.copy()
        empty_payload["smiles"] = ""
        
        try:
            response = requests.post(f"{API_BASE}/cell-line/predict", 
                                   json=empty_payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code == 400:
                self.log_test("Empty SMILES Rejection", True, "Empty SMILES properly rejected")
            else:
                self.log_test("Empty SMILES Rejection", False, f"Expected 400, got {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Empty SMILES Rejection", False, f"Error: {str(e)}")
            return False
        
        return True
    
    def test_uncertainty_quantification(self):
        """Test uncertainty quantification and confidence scoring"""
        print("\n=== Testing Uncertainty Quantification ===")
        
        # Test with different genomic complexity levels
        test_cases = [
            {
                "name": "Simple genomics (low mutations)",
                "genomic_features": {
                    "mutations": {"TP53": 0, "KRAS": 0},
                    "cnvs": {"MYC": 0},
                    "expression": {"EGFR": 0.0}
                }
            },
            {
                "name": "Complex genomics (multiple mutations)",
                "genomic_features": {
                    "mutations": {"TP53": 1, "KRAS": 1, "PIK3CA": 1, "BRAF": 1},
                    "cnvs": {"MYC": 1, "PTEN": -1},
                    "expression": {"EGFR": 2.0, "KRAS": 1.5}
                }
            }
        ]
        
        all_passed = True
        
        for test_case in test_cases:
            payload = {
                "smiles": "Cc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCOC",  # Erlotinib
                "drug_name": "Erlotinib",
                "cell_line": {
                    "cell_line_name": "TestCell",
                    "cancer_type": "LUNG",
                    "genomic_features": test_case["genomic_features"]
                }
            }
            
            try:
                response = requests.post(f"{API_BASE}/cell-line/predict", 
                                       json=payload,
                                       headers={'Content-Type': 'application/json'},
                                       timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    uncertainty = data.get('uncertainty')
                    confidence = data.get('confidence')
                    
                    # Check value ranges
                    valid_uncertainty = isinstance(uncertainty, (int, float)) and 0.0 <= uncertainty <= 1.0
                    valid_confidence = isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
                    
                    # Check inverse relationship (higher confidence = lower uncertainty)
                    inverse_relationship = (confidence + uncertainty) <= 1.2  # Allow some tolerance
                    
                    self.log_test(f"Uncertainty/Confidence - {test_case['name']}", 
                                valid_uncertainty and valid_confidence and inverse_relationship,
                                f"Uncertainty: {uncertainty}, Confidence: {confidence}")
                    
                    if not (valid_uncertainty and valid_confidence and inverse_relationship):
                        all_passed = False
                else:
                    self.log_test(f"Uncertainty test - {test_case['name']}", False, f"HTTP {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"Uncertainty test - {test_case['name']}", False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_backend_compatibility(self):
        """Test that existing endpoints still work after Cell Line integration"""
        print("\n=== Testing Backend Compatibility ===")
        
        all_passed = True
        
        # Test main health endpoint
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if Cell Line model is shown as available
                models_loaded = data.get('models_loaded', {})
                cell_line_available = models_loaded.get('cell_line_response_model', False)
                
                self.log_test("Main Health Endpoint Shows Cell Line Model", cell_line_available,
                            f"Cell line model available: {cell_line_available}")
                
                # Check AI modules section
                ai_modules = data.get('ai_modules', {})
                cell_line_ai_available = ai_modules.get('cell_line_model_available', False)
                
                self.log_test("AI Modules Shows Cell Line Model", cell_line_ai_available,
                            f"Cell line AI module available: {cell_line_ai_available}")
                
            else:
                self.log_test("Main Health Endpoint Compatibility", False, f"HTTP {response.status_code}")
                all_passed = False
                
        except Exception as e:
            self.log_test("Main Health Endpoint Compatibility", False, f"Error: {str(e)}")
            all_passed = False
        
        # Test existing predict endpoint still works
        try:
            payload = {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
                "prediction_types": ["bioactivity_ic50"],
                "target": "EGFR"
            }
            
            response = requests.post(f"{API_BASE}/predict", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and len(data['results']) > 0:
                    self.log_test("Existing Predict Endpoint Compatibility", True, "Existing predict endpoint still works")
                else:
                    self.log_test("Existing Predict Endpoint Compatibility", False, "No results returned")
                    all_passed = False
            else:
                self.log_test("Existing Predict Endpoint Compatibility", False, f"HTTP {response.status_code}")
                all_passed = False
                
        except Exception as e:
            self.log_test("Existing Predict Endpoint Compatibility", False, f"Error: {str(e)}")
            all_passed = False
        
        # Test targets endpoint still works
        try:
            response = requests.get(f"{API_BASE}/targets", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'targets' in data and len(data['targets']) > 0:
                    self.log_test("Targets Endpoint Compatibility", True, "Targets endpoint still works")
                else:
                    self.log_test("Targets Endpoint Compatibility", False, "No targets returned")
                    all_passed = False
            else:
                self.log_test("Targets Endpoint Compatibility", False, f"HTTP {response.status_code}")
                all_passed = False
                
        except Exception as e:
            self.log_test("Targets Endpoint Compatibility", False, f"Error: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def run_all_tests(self):
        """Run all Cell Line Response Model tests"""
        print("🧬 Starting Cell Line Response Model Integration Testing")
        print(f"🔗 Backend URL: {BACKEND_URL}")
        print("=" * 80)
        
        # Run all tests
        test_methods = [
            self.test_cell_line_health_endpoint,
            self.test_cell_line_examples_endpoint,
            self.test_cell_line_predict_endpoint,
            self.test_cell_line_trametinib_sensitivity,
            self.test_cell_line_compare_endpoint,
            self.test_smiles_processing_and_validation,
            self.test_uncertainty_quantification,
            self.test_backend_compatibility
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.log_test(f"Test execution error - {test_method.__name__}", False, f"Exception: {str(e)}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("🧬 CELL LINE RESPONSE MODEL TESTING SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['success']])
        failed_tests = len(self.failed_tests)
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS ({len(self.failed_tests)}):")
            for failed_test in self.failed_tests:
                print(f"   • {failed_test['test']}: {failed_test['details']}")
        
        print("\n" + "=" * 80)
        
        return passed_tests, failed_tests, total_tests

if __name__ == "__main__":
    tester = CellLineModelTester()
    passed, failed, total = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)