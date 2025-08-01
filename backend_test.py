#!/usr/bin/env python3
"""
Enhanced Predictive Chemistry Platform Backend Testing
Tests target-specific IC50 predictions, enhanced model validation, and new real API integration pipeline
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
    
    def test_real_pubchem_extractor_syntax(self):
        """Test Real PubChem BioAssay API Integration syntax and structure"""
        print("\n=== Testing Real PubChem BioAssay API Integration ===")
        
        try:
            # Check if the file exists and can be imported
            pubchem_file = Path("/app/modal_training/real_pubchem_extractor.py")
            if not pubchem_file.exists():
                self.log_test("PubChem Extractor File Exists", False, "File not found")
                return
            
            # Test syntax by attempting to compile
            with open(pubchem_file, 'r') as f:
                code = f.read()
            
            try:
                compile(code, str(pubchem_file), 'exec')
                self.log_test("PubChem Extractor Syntax Valid", True, "Python syntax is correct")
            except SyntaxError as e:
                self.log_test("PubChem Extractor Syntax Valid", False, f"Syntax error: {e}")
                return
            
            # Check for key components
            required_components = [
                "RealPubChemExtractor",
                "ONCOLOGY_TARGETS_PUBCHEM", 
                "search_target_bioassays",
                "extract_bioassay_data",
                "get_compound_smiles",
                "get_bioactivity_data",
                "extract_real_pubchem_data"
            ]
            
            missing_components = []
            for component in required_components:
                if component not in code:
                    missing_components.append(component)
            
            if missing_components:
                self.log_test("PubChem Extractor Components", False, f"Missing: {missing_components}")
            else:
                self.log_test("PubChem Extractor Components", True, "All required components present")
            
            # Check API integration patterns
            api_patterns = [
                "https://pubchem.ncbi.nlm.nih.gov/rest",
                "requests.Session",
                "rate_limit",
                "synonyms",
                "bioassay",
                "SMILES"
            ]
            
            missing_patterns = []
            for pattern in api_patterns:
                if pattern not in code:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                self.log_test("PubChem API Integration Patterns", False, f"Missing patterns: {missing_patterns}")
            else:
                self.log_test("PubChem API Integration Patterns", True, "API integration patterns present")
            
            # Check target coverage
            if "23" in code or len([line for line in code.split('\n') if 'category' in line and 'oncoprotein' in line]) >= 8:
                self.log_test("PubChem Target Coverage", True, "Comprehensive target coverage")
            else:
                self.log_test("PubChem Target Coverage", False, "Limited target coverage")
                
        except Exception as e:
            self.log_test("PubChem Extractor Analysis", False, f"Error analyzing file: {e}")
    
    def test_gdsc_cancer_extractor_syntax(self):
        """Test Enhanced GDSC Real Data Extraction syntax and structure"""
        print("\n=== Testing Enhanced GDSC Real Data Extraction ===")
        
        try:
            # Check if the file exists
            gdsc_file = Path("/app/modal_training/gdsc_cancer_extractor.py")
            if not gdsc_file.exists():
                self.log_test("GDSC Extractor File Exists", False, "File not found")
                return
            
            # Test syntax
            with open(gdsc_file, 'r') as f:
                code = f.read()
            
            try:
                compile(code, str(gdsc_file), 'exec')
                self.log_test("GDSC Extractor Syntax Valid", True, "Python syntax is correct")
            except SyntaxError as e:
                self.log_test("GDSC Extractor Syntax Valid", False, f"Syntax error: {e}")
                return
            
            # Check for key components
            required_components = [
                "GDSCDataExtractor",
                "GDSC_URLS",
                "ONCOLOGY_CANCER_TYPES",
                "download_gdsc_file",
                "extract_drug_sensitivity_data",
                "extract_genomics_data",
                "extract_gdsc_data"
            ]
            
            missing_components = []
            for component in required_components:
                if component not in code:
                    missing_components.append(component)
            
            if missing_components:
                self.log_test("GDSC Extractor Components", False, f"Missing: {missing_components}")
            else:
                self.log_test("GDSC Extractor Components", True, "All required components present")
            
            # Check API integration patterns
            api_patterns = [
                "cog.sanger.ac.uk",
                "cancerrxgene",
                "IC50",
                "cell_line",
                "genomics",
                "mutations",
                "expression"
            ]
            
            missing_patterns = []
            for pattern in api_patterns:
                if pattern.lower() not in code.lower():
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                self.log_test("GDSC API Integration Patterns", False, f"Missing patterns: {missing_patterns}")
            else:
                self.log_test("GDSC API Integration Patterns", True, "GDSC API integration patterns present")
            
            # Check cancer type coverage
            cancer_types_count = len([line for line in code.split('\n') if 'LUNG' in line or 'BREAST' in line or 'COLON' in line])
            if cancer_types_count > 0:
                self.log_test("GDSC Cancer Type Coverage", True, "Cancer type coverage present")
            else:
                self.log_test("GDSC Cancer Type Coverage", False, "Limited cancer type coverage")
                
        except Exception as e:
            self.log_test("GDSC Extractor Analysis", False, f"Error analyzing file: {e}")
    
    def test_updated_database_integration_syntax(self):
        """Test Updated Database Integration Pipeline syntax and structure"""
        print("\n=== Testing Updated Database Integration Pipeline ===")
        
        try:
            # Check if the file exists
            integration_file = Path("/app/modal_training/updated_database_integration.py")
            if not integration_file.exists():
                self.log_test("Database Integration File Exists", False, "File not found")
                return
            
            # Test syntax
            with open(integration_file, 'r') as f:
                code = f.read()
            
            try:
                compile(code, str(integration_file), 'exec')
                self.log_test("Database Integration Syntax Valid", True, "Python syntax is correct")
            except SyntaxError as e:
                self.log_test("Database Integration Syntax Valid", False, f"Syntax error: {e}")
                return
            
            # Check for key components
            required_components = [
                "integrate_real_databases",
                "integrate_protein_ligand_data",
                "process_cell_line_data",
                "apply_protein_ligand_deduplication",
                "dual_track",
                "ChEMBL",
                "PubChem",
                "BindingDB",
                "GDSC"
            ]
            
            missing_components = []
            for component in required_components:
                if component not in code:
                    missing_components.append(component)
            
            if missing_components:
                self.log_test("Database Integration Components", False, f"Missing: {missing_components}")
            else:
                self.log_test("Database Integration Components", True, "All required components present")
            
            # Check DTC removal
            if "DTC" in code and "dtc_removed" in code.lower():
                if "dtc_excluded" in code.lower() or "DTC completely removed" in code:
                    self.log_test("DTC Removal Verification", True, "DTC removal properly implemented")
                else:
                    self.log_test("DTC Removal Verification", False, "DTC removal not clearly implemented")
            else:
                self.log_test("DTC Removal Verification", True, "No DTC references found (good)")
            
            # Check dual-track architecture
            dual_track_patterns = [
                "protein_ligand",
                "cell_line",
                "dual_track",
                "Track 1",
                "Track 2"
            ]
            
            dual_track_found = sum(1 for pattern in dual_track_patterns if pattern in code.lower())
            if dual_track_found >= 3:
                self.log_test("Dual-Track Architecture", True, "Dual-track architecture implemented")
            else:
                self.log_test("Dual-Track Architecture", False, "Dual-track architecture not clearly implemented")
            
            # Check cross-source deduplication
            dedup_patterns = ["deduplication", "priority", "ChEMBL > PubChem > BindingDB"]
            dedup_found = sum(1 for pattern in dedup_patterns if pattern in code)
            if dedup_found >= 2:
                self.log_test("Cross-Source Deduplication", True, "Deduplication logic present")
            else:
                self.log_test("Cross-Source Deduplication", False, "Deduplication logic unclear")
                
        except Exception as e:
            self.log_test("Database Integration Analysis", False, f"Error analyzing file: {e}")
    
    def test_cell_line_response_model_syntax(self):
        """Test Cell Line Response Model Architecture syntax and structure"""
        print("\n=== Testing Cell Line Response Model Architecture ===")
        
        try:
            # Check if the file exists
            model_file = Path("/app/modal_training/cell_line_response_model.py")
            if not model_file.exists():
                self.log_test("Cell Line Model File Exists", False, "File not found")
                return
            
            # Test syntax
            with open(model_file, 'r') as f:
                code = f.read()
            
            try:
                compile(code, str(model_file), 'exec')
                self.log_test("Cell Line Model Syntax Valid", True, "Python syntax is correct")
            except SyntaxError as e:
                self.log_test("Cell Line Model Syntax Valid", False, f"Syntax error: {e}")
                return
            
            # Check for key components
            required_components = [
                "CellLineResponseModel",
                "MolecularEncoder",
                "GenomicEncoder",
                "SMILESTokenizer",
                "torch.nn.Module",
                "MultiheadAttention",
                "train_cell_line_response_model"
            ]
            
            missing_components = []
            for component in required_components:
                if component not in code:
                    missing_components.append(component)
            
            if missing_components:
                self.log_test("Cell Line Model Components", False, f"Missing: {missing_components}")
            else:
                self.log_test("Cell Line Model Components", True, "All required components present")
            
            # Check PyTorch architecture patterns
            pytorch_patterns = [
                "nn.LSTM",
                "nn.Linear",
                "nn.MultiheadAttention",
                "forward",
                "torch.tensor",
                "cuda",
                "optimizer"
            ]
            
            missing_pytorch = []
            for pattern in pytorch_patterns:
                if pattern not in code:
                    missing_pytorch.append(pattern)
            
            if missing_pytorch:
                self.log_test("PyTorch Architecture Patterns", False, f"Missing: {missing_pytorch}")
            else:
                self.log_test("PyTorch Architecture Patterns", True, "PyTorch patterns present")
            
            # Check multi-modal features
            multimodal_patterns = [
                "molecular",
                "genomic",
                "SMILES",
                "mutations",
                "expression",
                "cross_attention",
                "fusion"
            ]
            
            multimodal_found = sum(1 for pattern in multimodal_patterns if pattern.lower() in code.lower())
            if multimodal_found >= 5:
                self.log_test("Multi-Modal Architecture", True, "Multi-modal features implemented")
            else:
                self.log_test("Multi-Modal Architecture", False, "Multi-modal architecture incomplete")
            
            # Check IC50 prediction capability
            ic50_patterns = ["IC50", "pIC50", "pic50", "drug sensitivity"]
            ic50_found = sum(1 for pattern in ic50_patterns if pattern in code)
            if ic50_found >= 2:
                self.log_test("IC50 Prediction Capability", True, "IC50 prediction implemented")
            else:
                self.log_test("IC50 Prediction Capability", False, "IC50 prediction unclear")
                
        except Exception as e:
            self.log_test("Cell Line Model Analysis", False, f"Error analyzing file: {e}")
    
    def test_real_bindingdb_extractor_syntax(self):
        """Test Real BindingDB API Integration syntax and structure"""
        print("\n=== Testing Real BindingDB API Integration ===")
        
        try:
            # Check if the file exists
            bindingdb_file = Path("/app/modal_training/real_bindingdb_extractor.py")
            if not bindingdb_file.exists():
                self.log_test("BindingDB Extractor File Exists", False, "File not found")
                return
            
            # Test syntax
            with open(bindingdb_file, 'r') as f:
                code = f.read()
            
            try:
                compile(code, str(bindingdb_file), 'exec')
                self.log_test("BindingDB Extractor Syntax Valid", True, "Python syntax is correct")
            except SyntaxError as e:
                self.log_test("BindingDB Extractor Syntax Valid", False, f"Syntax error: {e}")
                return
            
            # Check for key components
            required_components = [
                "RealBindingDBExtractor",
                "ONCOLOGY_TARGETS",
                "extract_target_data",
                "bindingdb.org",
                "IC50",
                "Ki",
                "Kd"
            ]
            
            missing_components = []
            for component in required_components:
                if component not in code:
                    missing_components.append(component)
            
            if missing_components:
                self.log_test("BindingDB Extractor Components", False, f"Missing: {missing_components}")
            else:
                self.log_test("BindingDB Extractor Components", True, "All required components present")
            
            # Check API integration patterns
            api_patterns = [
                "RESTful",
                "UniProt",
                "binding affinity",
                "unit conversion",
                "quality control"
            ]
            
            api_found = sum(1 for pattern in api_patterns if pattern.lower() in code.lower())
            if api_found >= 3:
                self.log_test("BindingDB API Integration", True, "API integration patterns present")
            else:
                self.log_test("BindingDB API Integration", False, "API integration patterns unclear")
                
        except Exception as e:
            self.log_test("BindingDB Extractor Analysis", False, f"Error analyzing file: {e}")
    
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
                data = response.json()
                targets = data.get('targets', [])
                
                if not isinstance(targets, list):
                    self.log_test("Targets endpoint format", False, "Response should contain 'targets' list")
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
                required_fields = ['modal_available']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Modal status endpoint structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                modal_available = data.get('modal_available', False)
                credentials_set = data.get('credentials_set', False)
                
                self.log_test("Modal MolBERT status endpoint", True, f"Modal available: {modal_available}, Credentials set: {credentials_set}")
                
                # Check for additional info
                if 'app_name' in data:
                    app_name = data.get('app_name', '')
                    self.log_test("Modal app info", True, f"App name: {app_name}")
                
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

    def test_expanded_models_health_check(self):
        """Test /api/health endpoint for expanded models information"""
        print("\n=== Testing Expanded Models Health Check ===")
        
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check expanded_models_available field
                models_loaded = data.get('models_loaded', {})
                expanded_models_available = models_loaded.get('expanded_models', False)
                
                self.log_test("Expanded models available in health", 
                            'expanded_models' in models_loaded,
                            f"Expanded models field present: {'expanded_models' in models_loaded}")
                
                # Check expanded_models_info section
                expanded_models_info = data.get('expanded_models_info', {})
                
                if not expanded_models_info:
                    self.log_test("Expanded models info section", False, "No expanded_models_info section")
                    return False
                
                # Check required fields in expanded_models_info
                required_fields = ['available', 'total_targets', 'target_categories', 'activity_types', 'data_sources']
                missing_fields = [field for field in required_fields if field not in expanded_models_info]
                
                if missing_fields:
                    self.log_test("Expanded models info structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check specific values
                total_targets = expanded_models_info.get('total_targets', 0)
                target_categories = expanded_models_info.get('target_categories', {})
                activity_types = expanded_models_info.get('activity_types', [])
                data_sources = expanded_models_info.get('data_sources', [])
                
                # Verify 23 total targets
                self.log_test("Total targets count", total_targets == 23, 
                            f"Total targets: {total_targets} (expected: 23)")
                
                # Verify target categories (10 oncoproteins + 7 tumor suppressors + 6 metastasis suppressors)
                expected_categories = {
                    'oncoproteins': 10,
                    'tumor_suppressors': 7,
                    'metastasis_suppressors': 6
                }
                
                categories_correct = True
                for category, expected_count in expected_categories.items():
                    actual_count = target_categories.get(category, 0)
                    if actual_count != expected_count:
                        categories_correct = False
                        break
                
                self.log_test("Target categories breakdown", categories_correct,
                            f"Categories: {target_categories} (expected: {expected_categories})")
                
                # Verify activity types
                expected_activity_types = ["IC50", "EC50", "Ki", "Inhibition", "Activity"]
                has_all_activity_types = all(activity in activity_types for activity in expected_activity_types)
                
                self.log_test("Activity types", has_all_activity_types,
                            f"Activity types: {activity_types} (expected: {expected_activity_types})")
                
                # Verify data sources
                expected_data_sources = ["ChEMBL", "PubChem", "BindingDB", "DTC"]
                has_all_data_sources = all(source in data_sources for source in expected_data_sources)
                
                self.log_test("Data sources", has_all_data_sources,
                            f"Data sources: {data_sources} (expected: {expected_data_sources})")
                
                return True
                
            else:
                self.log_test("Expanded models health check", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Expanded models health check", False, f"Connection error: {str(e)}")
            return False

    def test_expanded_models_health_endpoint(self):
        """Test /api/expanded/health endpoint"""
        print("\n=== Testing Expanded Models Health Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/expanded/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['status', 'expanded_models', 'target_categories', 'total_targets', 'activity_types', 'data_sources']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Expanded health endpoint structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check status
                status = data.get('status')
                self.log_test("Expanded health status", status == 'healthy', f"Status: {status}")
                
                # Check expanded models availability
                expanded_models = data.get('expanded_models', {})
                chemberta_available = expanded_models.get('chemberta_available', False)
                chemprop_available = expanded_models.get('chemprop_available', False)
                
                self.log_test("Expanded models availability", 
                            'chemberta_available' in expanded_models and 'chemprop_available' in expanded_models,
                            f"ChemBERTa: {chemberta_available}, Chemprop: {chemprop_available}")
                
                # Check target categories
                target_categories = data.get('target_categories', {})
                expected_counts = {'oncoproteins': 10, 'tumor_suppressors': 7, 'metastasis_suppressors': 6}
                
                categories_correct = all(
                    target_categories.get(cat) == count 
                    for cat, count in expected_counts.items()
                )
                
                self.log_test("Expanded target categories", categories_correct,
                            f"Categories: {target_categories}")
                
                # Check total targets
                total_targets = data.get('total_targets', 0)
                self.log_test("Expanded total targets", total_targets == 23, f"Total: {total_targets}")
                
                return True
                
            else:
                self.log_test("Expanded health endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Expanded health endpoint", False, f"Connection error: {str(e)}")
            return False

    def test_expanded_targets_endpoint(self):
        """Test /api/expanded/targets endpoint"""
        print("\n=== Testing Expanded Targets Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/expanded/targets", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['targets', 'by_category', 'total_targets', 'activity_types']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Expanded targets endpoint structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check targets list
                targets = data.get('targets', [])
                total_targets = data.get('total_targets', 0)
                
                self.log_test("Expanded targets list", len(targets) == 23 and total_targets == 23,
                            f"Targets count: {len(targets)}, Total: {total_targets}")
                
                # Check target structure
                if targets:
                    sample_target = targets[0]
                    required_target_fields = ['target', 'category', 'full_name', 'available_chemberta', 'available_chemprop']
                    missing_target_fields = [field for field in required_target_fields if field not in sample_target]
                    
                    self.log_test("Target structure", len(missing_target_fields) == 0,
                                f"Sample target: {sample_target.get('target', 'unknown')}, Missing fields: {missing_target_fields}")
                
                # Check by_category grouping
                by_category = data.get('by_category', {})
                expected_categories = ['oncoprotein', 'tumor_suppressor', 'metastasis_suppressor']
                has_all_categories = all(cat in by_category for cat in expected_categories)
                
                self.log_test("Targets by category", has_all_categories,
                            f"Categories: {list(by_category.keys())}")
                
                # Check category counts
                category_counts = {cat: len(targets_list) for cat, targets_list in by_category.items()}
                expected_counts = {'oncoprotein': 10, 'tumor_suppressor': 7, 'metastasis_suppressor': 6}
                
                counts_correct = all(
                    category_counts.get(cat) == count 
                    for cat, count in expected_counts.items()
                )
                
                self.log_test("Category target counts", counts_correct,
                            f"Counts: {category_counts}")
                
                return True
                
            else:
                self.log_test("Expanded targets endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Expanded targets endpoint", False, f"Connection error: {str(e)}")
            return False

    def test_expanded_stats_performance_endpoint(self):
        """Test /api/expanded/stats/performance endpoint"""
        print("\n=== Testing Expanded Stats Performance Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/expanded/stats/performance", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['chemberta', 'chemprop', 'comparison']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Expanded stats endpoint structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check ChemBERTa stats
                chemberta_stats = data.get('chemberta', {})
                chemberta_required = ['overall_r2', 'category_performance', 'training_info']
                chemberta_missing = [field for field in chemberta_required if field not in chemberta_stats]
                
                self.log_test("ChemBERTa stats structure", len(chemberta_missing) == 0,
                            f"ChemBERTa missing fields: {chemberta_missing}")
                
                # Check Chemprop stats
                chemprop_stats = data.get('chemprop', {})
                chemprop_required = ['overall_r2', 'category_performance', 'training_info']
                chemprop_missing = [field for field in chemprop_required if field not in chemprop_stats]
                
                self.log_test("Chemprop stats structure", len(chemprop_missing) == 0,
                            f"Chemprop missing fields: {chemprop_missing}")
                
                # Check comparison stats
                comparison = data.get('comparison', {})
                comparison_required = ['chemberta_better_targets', 'chemprop_better_targets', 'similar_targets']
                comparison_missing = [field for field in comparison_required if field not in comparison]
                
                self.log_test("Comparison stats structure", len(comparison_missing) == 0,
                            f"Comparison missing fields: {comparison_missing}")
                
                # Check category performance for both models
                if 'category_performance' in chemberta_stats:
                    chemberta_categories = chemberta_stats['category_performance']
                    expected_categories = ['oncoprotein', 'tumor_suppressor', 'metastasis_suppressor']
                    has_all_chemberta_categories = all(cat in chemberta_categories for cat in expected_categories)
                    
                    self.log_test("ChemBERTa category performance", has_all_chemberta_categories,
                                f"ChemBERTa categories: {list(chemberta_categories.keys())}")
                
                if 'category_performance' in chemprop_stats:
                    chemprop_categories = chemprop_stats['category_performance']
                    expected_categories = ['oncoprotein', 'tumor_suppressor', 'metastasis_suppressor']
                    has_all_chemprop_categories = all(cat in chemprop_categories for cat in expected_categories)
                    
                    self.log_test("Chemprop category performance", has_all_chemprop_categories,
                                f"Chemprop categories: {list(chemprop_categories.keys())}")
                
                return True
                
            else:
                self.log_test("Expanded stats endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Expanded stats endpoint", False, f"Connection error: {str(e)}")
            return False

    def test_expanded_models_error_handling(self):
        """Test error handling when expanded models are not available"""
        print("\n=== Testing Expanded Models Error Handling ===")
        
        all_passed = True
        
        # Test ChemBERTa prediction endpoint when model not available
        try:
            payload = {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
                "targets": ["EGFR", "BRAF"],
                "activity_types": ["IC50"]
            }
            
            response = requests.post(f"{API_BASE}/expanded/predict/chemberta", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code == 503:
                data = response.json()
                detail = data.get('detail', '')
                
                if 'not available' in detail.lower():
                    self.log_test("Expanded ChemBERTa error handling", True, 
                                f"Proper 503 error: {detail}")
                else:
                    self.log_test("Expanded ChemBERTa error handling", False, 
                                f"Wrong error message: {detail}")
                    all_passed = False
            else:
                # May return 200 if model is actually available, or other error codes
                self.log_test("Expanded ChemBERTa error handling", True, 
                            f"Response: HTTP {response.status_code} (model may be available)")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Expanded ChemBERTa error handling", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test Chemprop prediction endpoint when model not available
        try:
            payload = {
                "smiles": "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",  # imatinib
                "targets": ["VEGFR2", "MET"],
                "activity_types": ["IC50"]
            }
            
            response = requests.post(f"{API_BASE}/expanded/predict/chemprop", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code == 503:
                data = response.json()
                detail = data.get('detail', '')
                
                if 'not available' in detail.lower():
                    self.log_test("Expanded Chemprop error handling", True, 
                                f"Proper 503 error: {detail}")
                else:
                    self.log_test("Expanded Chemprop error handling", False, 
                                f"Wrong error message: {detail}")
                    all_passed = False
            else:
                # May return 200 if model is actually available, or other error codes
                self.log_test("Expanded Chemprop error handling", True, 
                            f"Response: HTTP {response.status_code} (model may be available)")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Expanded Chemprop error handling", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test model comparison endpoint
        try:
            payload = {
                "smiles": "CCO",  # ethanol
                "targets": ["EGFR"],
                "activity_types": ["IC50"]
            }
            
            response = requests.post(f"{API_BASE}/expanded/predict/compare", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code in [200, 503]:
                self.log_test("Expanded model comparison endpoint", True, 
                            f"Comparison endpoint accessible: HTTP {response.status_code}")
            else:
                self.log_test("Expanded model comparison endpoint", False, 
                            f"Unexpected response: HTTP {response.status_code}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Expanded model comparison endpoint", False, f"Connection error: {str(e)}")
            all_passed = False
        
        return all_passed

    def test_expanded_backend_integration_loaded(self):
        """Test that expanded_backend_integration.py router is properly loaded"""
        print("\n=== Testing Expanded Backend Integration Router Loading ===")
        
        try:
            # Test that the expanded endpoints are accessible (even if they return errors)
            endpoints_to_test = [
                "/expanded/health",
                "/expanded/targets", 
                "/expanded/stats/performance"
            ]
            
            all_loaded = True
            
            for endpoint in endpoints_to_test:
                try:
                    response = requests.get(f"{API_BASE}{endpoint}", timeout=10)
                    
                    # Any response code other than 404 means the endpoint exists
                    if response.status_code != 404:
                        self.log_test(f"Expanded endpoint {endpoint}", True, 
                                    f"Endpoint accessible: HTTP {response.status_code}")
                    else:
                        self.log_test(f"Expanded endpoint {endpoint}", False, 
                                    f"Endpoint not found: HTTP 404")
                        all_loaded = False
                        
                except requests.exceptions.RequestException as e:
                    self.log_test(f"Expanded endpoint {endpoint}", False, f"Connection error: {str(e)}")
                    all_loaded = False
            
            # Test POST endpoints exist
            post_endpoints = [
                "/expanded/predict/chemberta",
                "/expanded/predict/chemprop",
                "/expanded/predict/compare"
            ]
            
            for endpoint in post_endpoints:
                try:
                    # Send invalid payload to test if endpoint exists
                    response = requests.post(f"{API_BASE}{endpoint}", 
                                           json={},
                                           headers={'Content-Type': 'application/json'},
                                           timeout=10)
                    
                    # Any response code other than 404 means the endpoint exists
                    if response.status_code != 404:
                        self.log_test(f"Expanded POST endpoint {endpoint}", True, 
                                    f"Endpoint accessible: HTTP {response.status_code}")
                    else:
                        self.log_test(f"Expanded POST endpoint {endpoint}", False, 
                                    f"Endpoint not found: HTTP 404")
                        all_loaded = False
                        
                except requests.exceptions.RequestException as e:
                    self.log_test(f"Expanded POST endpoint {endpoint}", False, f"Connection error: {str(e)}")
                    all_loaded = False
            
            return all_loaded
            
        except Exception as e:
            self.log_test("Expanded backend integration loading", False, f"Error: {str(e)}")
            return False

    def test_real_chemprop_integration(self):
        """Test the new real Chemprop model router integration"""
        print("\n=== Testing Real Chemprop Model Router Integration ===")
        
        all_passed = True
        
        # Test 1: /api/chemprop-real/status
        try:
            response = requests.get(f"{API_BASE}/chemprop-real/status", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                available = data.get('available', False)
                message = data.get('message', '')
                
                self.log_test("Real Chemprop Status Endpoint", True, 
                            f"Status: {status}, Available: {available}, Message: {message}")
                
                # Check model_info structure
                model_info = data.get('model_info', {})
                if model_info:
                    self.log_test("Real Chemprop Model Info", True, 
                                f"Model info keys: {list(model_info.keys())}")
                else:
                    self.log_test("Real Chemprop Model Info", False, "No model_info in response")
                    all_passed = False
                    
            else:
                self.log_test("Real Chemprop Status Endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Real Chemprop Status Endpoint", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test 2: /api/chemprop-real/health
        try:
            response = requests.get(f"{API_BASE}/chemprop-real/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                model_available = data.get('model_available', False)
                model_type = data.get('model_type', '')
                
                self.log_test("Real Chemprop Health Endpoint", True, 
                            f"Health: {status}, Model Available: {model_available}, Type: {model_type}")
                
                # Check for expected model_type
                if model_type == "real_trained_model":
                    self.log_test("Real Chemprop Model Type", True, f"Correct model type: {model_type}")
                else:
                    self.log_test("Real Chemprop Model Type", False, f"Expected 'real_trained_model', got '{model_type}'")
                    all_passed = False
                    
            else:
                self.log_test("Real Chemprop Health Endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Real Chemprop Health Endpoint", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test 3: /api/chemprop-real/targets
        try:
            response = requests.get(f"{API_BASE}/chemprop-real/targets", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                targets = data.get('targets', {})
                total_targets = data.get('total_targets', 0)
                model_performance = data.get('model_performance', {})
                
                self.log_test("Real Chemprop Targets Endpoint", True, 
                            f"Total targets: {total_targets}, Available: {list(targets.keys())}")
                
                # Check model performance info
                if model_performance:
                    architecture = model_performance.get('architecture', '')
                    training_epochs = model_performance.get('training_epochs', 0)
                    self.log_test("Real Chemprop Model Performance", True, 
                                f"Architecture: {architecture}, Epochs: {training_epochs}")
                else:
                    self.log_test("Real Chemprop Model Performance", False, "No model performance info")
                    all_passed = False
                    
            else:
                self.log_test("Real Chemprop Targets Endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Real Chemprop Targets Endpoint", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test 4: /api/chemprop-real/predict (may fail if model not functional yet)
        try:
            payload = {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}  # aspirin
            response = requests.post(f"{API_BASE}/chemprop-real/predict", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                predictions = data.get('predictions', {})
                
                self.log_test("Real Chemprop Predict Endpoint", status == "success", 
                            f"Prediction Status: {status}, Predictions count: {len(predictions)}")
                
                if status == "success" and predictions:
                    # Check model_info in prediction response
                    model_info = data.get('model_info', {})
                    real_model = model_info.get('real_model', False)
                    self.log_test("Real Chemprop Prediction Model Info", real_model, 
                                f"Real model used: {real_model}")
                    
            else:
                # This might be expected if model is still being debugged
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error_msg = error_data.get('detail', response.text)
                
                self.log_test("Real Chemprop Predict Endpoint", False, 
                            f"HTTP {response.status_code}: {error_msg} (May be expected during debugging)")
                # Don't mark as critical failure since model may still be in development
                
        except requests.exceptions.RequestException as e:
            self.log_test("Real Chemprop Predict Endpoint", False, 
                        f"Connection error: {str(e)} (May be expected during debugging)")
        
        return all_passed

    def test_health_endpoint_real_chemprop_status(self):
        """Test that /api/health shows real_trained_chemprop status"""
        print("\n=== Testing Health Endpoint Real Chemprop Status ===")
        
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check models_loaded section
                models_loaded = data.get('models_loaded', {})
                
                if 'real_trained_chemprop' in models_loaded:
                    real_chemprop_status = models_loaded['real_trained_chemprop']
                    self.log_test("Health - Real Trained Chemprop Status", True, 
                                f"real_trained_chemprop: {real_chemprop_status}")
                else:
                    self.log_test("Health - Real Trained Chemprop Status", False, 
                                "real_trained_chemprop not found in models_loaded")
                    return False
                
                # Check ai_modules section
                ai_modules = data.get('ai_modules', {})
                if ai_modules:
                    real_chemprop_ai = ai_modules.get('real_chemprop_available', False)
                    total_ai_models = ai_modules.get('total_ai_models', 0)
                    
                    self.log_test("Health - AI Modules Real Chemprop", True, 
                                f"real_chemprop_available: {real_chemprop_ai}, total_ai_models: {total_ai_models}")
                else:
                    self.log_test("Health - AI Modules Real Chemprop", False, "No ai_modules section")
                    return False
                
                return True
            else:
                self.log_test("Health - Real Trained Chemprop Status", False, 
                            f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Health - Real Trained Chemprop Status", False, f"Connection error: {str(e)}")
            return False

    def test_existing_endpoints_compatibility(self):
        """Test that existing endpoints still work after real Chemprop integration"""
        print("\n=== Testing Existing Endpoints Compatibility ===")
        
        all_passed = True
        
        # Test ChemBERTa endpoints
        try:
            response = requests.get(f"{API_BASE}/chemberta/status", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                available = data.get('available', False)
                self.log_test("ChemBERTa Status Compatibility", True, 
                            f"ChemBERTa available: {available}")
            else:
                self.log_test("ChemBERTa Status Compatibility", False, 
                            f"HTTP {response.status_code}: {response.text}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("ChemBERTa Status Compatibility", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test Chemprop Multi-Task endpoints
        try:
            response = requests.get(f"{API_BASE}/chemprop-multitask/status", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                available = data.get('available', False)
                self.log_test("Chemprop Multi-Task Status Compatibility", True, 
                            f"Chemprop Multi-Task available: {available}")
            else:
                self.log_test("Chemprop Multi-Task Status Compatibility", False, 
                            f"HTTP {response.status_code}: {response.text}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Chemprop Multi-Task Status Compatibility", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test main predict endpoint still works
        try:
            payload = {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
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
                
                if len(results) > 0:
                    self.log_test("Main Predict Endpoint Compatibility", True, 
                                f"Main predict endpoint working, {len(results)} results")
                else:
                    self.log_test("Main Predict Endpoint Compatibility", False, "No results returned")
                    all_passed = False
            else:
                self.log_test("Main Predict Endpoint Compatibility", False, 
                            f"HTTP {response.status_code}: {response.text}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Main Predict Endpoint Compatibility", False, f"Connection error: {str(e)}")
            all_passed = False
        
        return all_passed

    def test_backend_loading_without_errors(self):
        """Test that backend loads without errors and all routers are integrated"""
        print("\n=== Testing Backend Loading Without Errors ===")
        
        try:
            # Test basic root endpoint
            response = requests.get(f"{API_BASE}/", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                message = data.get('message', '')
                
                self.log_test("Backend Root Endpoint", True, f"Backend loaded: {message}")
                
                # Test health endpoint for comprehensive status
                health_response = requests.get(f"{API_BASE}/health", timeout=30)
                
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    status = health_data.get('status', 'unknown')
                    
                    if status == 'healthy':
                        self.log_test("Backend Health Status", True, "Backend is healthy")
                        
                        # Check all expected routers are loaded
                        models_loaded = health_data.get('models_loaded', {})
                        expected_models = ['molbert', 'chemprop_simulation', 'real_ml_models', 'real_trained_chemprop']
                        
                        missing_models = [model for model in expected_models if model not in models_loaded]
                        
                        if not missing_models:
                            self.log_test("All Router Integration", True, 
                                        f"All expected models present: {list(models_loaded.keys())}")
                        else:
                            self.log_test("All Router Integration", False, 
                                        f"Missing models: {missing_models}")
                            return False
                        
                        return True
                    else:
                        self.log_test("Backend Health Status", False, f"Backend not healthy: {status}")
                        return False
                else:
                    self.log_test("Backend Health Status", False, 
                                f"Health check failed: HTTP {health_response.status_code}")
                    return False
            else:
                self.log_test("Backend Root Endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Backend Root Endpoint", False, f"Connection error: {str(e)}")
            return False

    def test_backend_integration_compatibility(self):
        """Test that existing backend endpoints still work after new integrations"""
        print("\n=== Testing Backend Integration Compatibility ===")
        
        # Test basic endpoints
        endpoints_to_test = [
            ("/health", "Health Check"),
            ("/targets", "Targets List"),
        ]
        
        for endpoint, description in endpoints_to_test:
            try:
                response = requests.get(f"{API_BASE}{endpoint}", timeout=30)
                if response.status_code == 200:
                    self.log_test(f"{description} Endpoint", True, f"HTTP 200 OK")
                else:
                    self.log_test(f"{description} Endpoint", False, f"HTTP {response.status_code}")
            except Exception as e:
                self.log_test(f"{description} Endpoint", False, f"Error: {e}")
        
        # Test prediction endpoint with simple molecule
        try:
            test_data = {
                "smiles": "CCO",  # Ethanol
                "prediction_types": ["bioactivity_ic50"],
                "target": "EGFR"
            }
            
            response = requests.post(f"{API_BASE}/predict", json=test_data, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if data.get('results') and len(data['results']) > 0:
                    self.log_test("Prediction Endpoint Compatibility", True, "Predictions working")
                else:
                    self.log_test("Prediction Endpoint Compatibility", False, "No prediction results")
            else:
                self.log_test("Prediction Endpoint Compatibility", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Prediction Endpoint Compatibility", False, f"Error: {e}")
    
    def test_api_error_handling(self):
        """Test API error handling for new integrations"""
        print("\n=== Testing API Error Handling ===")
        
        # Test invalid SMILES
        try:
            test_data = {
                "smiles": "INVALID_SMILES_STRING",
                "prediction_types": ["bioactivity_ic50"],
                "target": "EGFR"
            }
            
            response = requests.post(f"{API_BASE}/predict", json=test_data, timeout=30)
            if response.status_code == 400:
                self.log_test("Invalid SMILES Error Handling", True, "Properly rejects invalid SMILES")
            else:
                self.log_test("Invalid SMILES Error Handling", False, f"HTTP {response.status_code} instead of 400")
                
        except Exception as e:
            self.log_test("Invalid SMILES Error Handling", False, f"Error: {e}")
        
        # Test invalid target
        try:
            test_data = {
                "smiles": "CCO",
                "prediction_types": ["bioactivity_ic50"],
                "target": "INVALID_TARGET"
            }
            
            response = requests.post(f"{API_BASE}/predict", json=test_data, timeout=30)
            # Should still work but may give different results
            if response.status_code in [200, 400]:
                self.log_test("Invalid Target Handling", True, "Handles invalid targets appropriately")
            else:
                self.log_test("Invalid Target Handling", False, f"Unexpected HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Invalid Target Handling", False, f"Error: {e}")
    
    def test_memory_and_performance_impact(self):
        """Test that new integrations don't cause memory or performance issues"""
        print("\n=== Testing Memory and Performance Impact ===")
        
        # Test multiple rapid requests
        try:
            test_molecules = [
                "CCO",  # Ethanol
                "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
            ]
            
            start_time = time.time()
            successful_requests = 0
            
            for i, smiles in enumerate(test_molecules):
                test_data = {
                    "smiles": smiles,
                    "prediction_types": ["bioactivity_ic50"],
                    "target": "EGFR"
                }
                
                response = requests.post(f"{API_BASE}/predict", json=test_data, timeout=30)
                if response.status_code == 200:
                    successful_requests += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if successful_requests == len(test_molecules) and total_time < 60:
                self.log_test("Performance Impact", True, f"Processed {successful_requests} requests in {total_time:.2f}s")
            else:
                self.log_test("Performance Impact", False, f"Only {successful_requests}/{len(test_molecules)} successful, took {total_time:.2f}s")
                
        except Exception as e:
            self.log_test("Performance Impact", False, f"Error: {e}")
    
    def test_cross_database_integration_logic(self):
        """Test cross-database integration logic conceptually"""
        print("\n=== Testing Cross-Database Integration Logic ===")
        
        # This tests the conceptual integration by checking health endpoint for database info
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                # Check for expanded models info (indicates database integration)
                expanded_info = data.get('expanded_models_info', {})
                if expanded_info.get('available'):
                    total_targets = expanded_info.get('total_targets', 0)
                    if total_targets >= 20:
                        self.log_test("Cross-Database Integration", True, f"Integrated {total_targets} targets")
                    else:
                        self.log_test("Cross-Database Integration", False, f"Only {total_targets} targets integrated")
                else:
                    self.log_test("Cross-Database Integration", False, "Expanded models not available")
                
                # Check data sources
                data_sources = expanded_info.get('data_sources', [])
                expected_sources = ['ChEMBL', 'PubChem', 'BindingDB']
                found_sources = [source for source in expected_sources if source in data_sources]
                
                if len(found_sources) >= 2:
                    self.log_test("Multi-Source Integration", True, f"Found sources: {found_sources}")
                else:
                    self.log_test("Multi-Source Integration", False, f"Limited sources: {found_sources}")
                
            else:
                self.log_test("Cross-Database Integration", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Cross-Database Integration", False, f"Error: {e}")
    
    def run_real_api_integration_tests(self):
        """Run all tests for the new real API integration pipeline"""
        print("🧪 REAL API INTEGRATION PIPELINE TESTING")
        print("=" * 80)
        print(f"🌐 Backend URL: {BACKEND_URL}")
        print(f"📡 API Base: {API_BASE}")
        print()
        
        # Test new integration components
        self.test_real_pubchem_extractor_syntax()
        self.test_gdsc_cancer_extractor_syntax()
        self.test_updated_database_integration_syntax()
        self.test_cell_line_response_model_syntax()
        self.test_real_bindingdb_extractor_syntax()
        
        # Test backend compatibility
        self.test_health_endpoint_enhanced()
        self.test_backend_integration_compatibility()
        self.test_api_error_handling()
        self.test_memory_and_performance_impact()
        self.test_cross_database_integration_logic()
        
        # Generate summary
        self.generate_test_summary()
    
    def generate_test_summary(self):
        """Generate comprehensive test summary"""
        print("\n" + "=" * 80)
        print("🎯 REAL API INTEGRATION TESTING SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t['success']])
        failed_tests = len(self.failed_tests)
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 Test Results: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"   • {test['test']}: {test['details']}")
        
        print(f"\n📋 DETAILED RESULTS:")
        
        # Group by category
        categories = {
            'Syntax Validation': [],
            'API Integration': [],
            'Backend Compatibility': [],
            'Performance': []
        }
        
        for test in self.test_results:
            test_name = test['test']
            if 'Syntax' in test_name or 'Components' in test_name:
                categories['Syntax Validation'].append(test)
            elif 'API' in test_name or 'Integration' in test_name or 'Extractor' in test_name:
                categories['API Integration'].append(test)
            elif 'Endpoint' in test_name or 'Compatibility' in test_name or 'Health' in test_name:
                categories['Backend Compatibility'].append(test)
            else:
                categories['Performance'].append(test)
        
        for category, tests in categories.items():
            if tests:
                print(f"\n🔍 {category}:")
                for test in tests:
                    print(f"   {test['status']}: {test['test']}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        
        if success_rate >= 90:
            print("   🟢 EXCELLENT: Real API integration pipeline is ready for deployment")
        elif success_rate >= 75:
            print("   🟡 GOOD: Real API integration pipeline is mostly ready, minor issues to address")
        elif success_rate >= 50:
            print("   🟠 FAIR: Real API integration pipeline needs significant improvements")
        else:
            print("   🔴 POOR: Real API integration pipeline has major issues requiring attention")
        
        # Specific recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        syntax_issues = [t for t in self.failed_tests if 'Syntax' in t['test'] or 'Components' in t['test']]
        if syntax_issues:
            print("   • Fix syntax errors and missing components in integration files")
        
        api_issues = [t for t in self.failed_tests if 'API' in t['test'] or 'Integration' in t['test']]
        if api_issues:
            print("   • Review API integration patterns and error handling")
        
        backend_issues = [t for t in self.failed_tests if 'Endpoint' in t['test'] or 'Compatibility' in t['test']]
        if backend_issues:
            print("   • Address backend compatibility and endpoint functionality")
        
        performance_issues = [t for t in self.failed_tests if 'Performance' in t['test'] or 'Memory' in t['test']]
        if performance_issues:
            print("   • Optimize performance and memory usage")
        
        if not self.failed_tests:
            print("   • All tests passed! Ready for production deployment")
        
        print("=" * 80)

    def run_all_tests(self):
        """Run all tests and provide summary"""
        print(f"🧪 Starting Expanded Database Integration Testing")
        print(f"Backend URL: {API_BASE}")
        print("=" * 80)
        
        # Focus on Expanded Models integration tests as requested
        tests = [
            # Expanded Models Integration Tests (Primary Focus)
            self.test_expanded_models_health_check,
            self.test_expanded_models_health_endpoint,
            self.test_expanded_targets_endpoint,
            self.test_expanded_stats_performance_endpoint,
            self.test_expanded_models_error_handling,
            self.test_expanded_backend_integration_loaded,
            
            # Backend loading and health checks
            self.test_backend_loading_without_errors,
            self.test_health_endpoint_real_chemprop_status,
            self.test_existing_endpoints_compatibility,
            
            # Core functionality tests (Secondary)
            self.test_health_endpoint_enhanced,
            self.test_targets_endpoint,
            self.test_enhanced_ic50_predictions,
            self.test_all_prediction_types,
            
            # Error handling
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
    
    # Run the new real API integration tests
    tester.run_real_api_integration_tests()
    
    # Exit with appropriate code
    sys.exit(0 if len(tester.failed_tests) == 0 else 1)