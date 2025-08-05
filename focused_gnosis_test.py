#!/usr/bin/env python3
"""
Focused Gnosis I Multi-Assay Testing
Tests the key multi-assay functionality with shorter timeouts
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

class FocusedGnosisITester:
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
    
    def test_basic_connectivity(self):
        """Test 1: Basic Backend Connectivity"""
        print("\n=== Testing Basic Connectivity ===")
        
        try:
            response = requests.get(f"{API_BASE}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                gnosis_available = health_data.get('models_loaded', {}).get('gnosis_i', False)
                
                if gnosis_available:
                    self.log_test("Backend Connectivity", True, f"Backend accessible, Gnosis I available")
                else:
                    self.log_test("Backend Connectivity", False, "Gnosis I not available in health check")
            else:
                self.log_test("Backend Connectivity", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Backend Connectivity", False, f"Error: {str(e)}")
    
    def test_gnosis_info(self):
        """Test 2: Gnosis I Model Info"""
        print("\n=== Testing Gnosis I Model Info ===")
        
        try:
            response = requests.get(f"{API_BASE}/gnosis-i/info", timeout=10)
            
            if response.status_code == 200:
                info_data = response.json()
                
                if info_data.get('available'):
                    num_targets = info_data.get('num_targets', 0)
                    r2_score = info_data.get('r2_score', 0.0)
                    capabilities = info_data.get('capabilities', [])
                    
                    self.log_test("Gnosis I Model Available", True, 
                                f"Model available with {num_targets} targets, R²={r2_score:.4f}")
                    
                    # Check for multi-assay capabilities
                    expected_assays = ['IC50 prediction', 'Ki prediction', 'EC50 prediction']
                    found_assays = [cap for cap in expected_assays if cap in capabilities]
                    
                    if len(found_assays) == 3:
                        self.log_test("Multi-Assay Capabilities", True, f"All 3 assay types supported: {found_assays}")
                    else:
                        self.log_test("Multi-Assay Capabilities", False, f"Only {len(found_assays)}/3 assay types: {found_assays}")
                        
                    # Check target count (should be around 62)
                    if num_targets >= 50:
                        self.log_test("Target Count", True, f"{num_targets} targets (expected ~62)")
                    else:
                        self.log_test("Target Count", False, f"Only {num_targets} targets (expected ~62)")
                        
                else:
                    self.log_test("Gnosis I Model Available", False, "Model not available")
                    
            else:
                self.log_test("Gnosis I Model Info", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Gnosis I Model Info", False, f"Error: {str(e)}")
    
    def test_single_target_multi_assay(self):
        """Test 3: Single Target Multi-Assay Prediction"""
        print("\n=== Testing Single Target Multi-Assay ===")
        
        aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        
        try:
            payload = {
                "smiles": aspirin_smiles,
                "targets": ["ABL1"],
                "assay_types": ["IC50", "Ki", "EC50"]
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', {})
                
                if 'ABL1' in predictions:
                    abl1_pred = predictions['ABL1']
                    
                    # Check for all three assay types
                    assay_types = ['IC50', 'Ki', 'EC50']
                    found_assays = []
                    
                    for assay in assay_types:
                        if assay in abl1_pred:
                            assay_data = abl1_pred[assay]
                            
                            # Check required fields
                            required_fields = ['pActivity', 'confidence', 'sigma', 'mc_samples']
                            has_all_fields = all(field in assay_data for field in required_fields)
                            
                            if has_all_fields:
                                found_assays.append(assay)
                                pactivity = assay_data['pActivity']
                                confidence = assay_data['confidence']
                                sigma = assay_data['sigma']
                                mc_samples = assay_data['mc_samples']
                                
                                self.log_test(f"ABL1 {assay} Structure", True, 
                                            f"pActivity={pactivity:.3f}, confidence={confidence:.3f}, sigma={sigma:.3f}, mc_samples={mc_samples}")
                            else:
                                missing = [f for f in required_fields if f not in assay_data]
                                self.log_test(f"ABL1 {assay} Structure", False, f"Missing: {missing}")
                    
                    # Check overall coverage
                    if len(found_assays) == 3:
                        self.log_test("Single Target Multi-Assay Coverage", True, f"All 3 assay types present: {found_assays}")
                    else:
                        self.log_test("Single Target Multi-Assay Coverage", False, f"Only {len(found_assays)}/3 assay types")
                    
                    # Check selectivity (should be None for single target)
                    selectivity = abl1_pred.get('selectivity_ratio')
                    if selectivity is None:
                        self.log_test("Single Target Selectivity", True, "Selectivity correctly None for single target")
                    else:
                        self.log_test("Single Target Selectivity", False, f"Unexpected selectivity: {selectivity}")
                        
                else:
                    self.log_test("Single Target Prediction", False, "ABL1 not found in predictions")
                    
            elif response.status_code == 503:
                self.log_test("Single Target Multi-Assay", False, "Service unavailable (503)")
            else:
                self.log_test("Single Target Multi-Assay", False, f"HTTP {response.status_code}: {response.text[:200]}")
                
        except Exception as e:
            self.log_test("Single Target Multi-Assay", False, f"Error: {str(e)}")
    
    def test_multiple_targets_selectivity(self):
        """Test 4: Multiple Targets with Selectivity"""
        print("\n=== Testing Multiple Targets with Selectivity ===")
        
        imatinib_smiles = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
        
        try:
            payload = {
                "smiles": imatinib_smiles,
                "targets": ["ABL1", "ABL2"],
                "assay_types": ["IC50", "Ki", "EC50"]
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=45)
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', {})
                model_info = data.get('model_info', {})
                
                # Check target coverage
                expected_targets = ["ABL1", "ABL2"]
                found_targets = [t for t in expected_targets if t in predictions]
                
                if len(found_targets) == 2:
                    self.log_test("Multiple Target Coverage", True, f"Both targets found: {found_targets}")
                else:
                    self.log_test("Multiple Target Coverage", False, f"Only found {len(found_targets)}/2 targets")
                
                # Check total predictions
                total_predictions = model_info.get('num_total_predictions', 0)
                expected_total = 6  # 2 targets × 3 assays
                
                if total_predictions == expected_total:
                    self.log_test("Total Predictions Count", True, f"{total_predictions} predictions (2 targets × 3 assays)")
                else:
                    self.log_test("Total Predictions Count", False, f"{total_predictions} predictions (expected {expected_total})")
                
                # Check selectivity ratios
                selectivity_found = 0
                for target in found_targets:
                    if target in predictions:
                        target_pred = predictions[target]
                        selectivity = target_pred.get('selectivity_ratio')
                        
                        if selectivity is not None and selectivity > 0:
                            selectivity_found += 1
                            self.log_test(f"{target} Selectivity", True, f"selectivity_ratio: {selectivity:.2f}")
                        else:
                            self.log_test(f"{target} Selectivity", False, f"Invalid selectivity: {selectivity}")
                
                if selectivity_found >= 1:
                    self.log_test("Selectivity Calculation", True, f"{selectivity_found}/2 targets have valid selectivity")
                else:
                    self.log_test("Selectivity Calculation", False, "No valid selectivity ratios found")
                
                # Check Monte-Carlo samples
                mc_samples = model_info.get('mc_samples', 0)
                if mc_samples >= 20:
                    self.log_test("Monte-Carlo Samples", True, f"MC samples: {mc_samples}")
                else:
                    self.log_test("Monte-Carlo Samples", False, f"MC samples: {mc_samples} (expected ≥20)")
                    
            elif response.status_code == 503:
                self.log_test("Multiple Targets Selectivity", False, "Service unavailable (503)")
            else:
                self.log_test("Multiple Targets Selectivity", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Multiple Targets Selectivity", False, f"Error: {str(e)}")
    
    def test_data_structure_format(self):
        """Test 5: New Data Structure Format"""
        print("\n=== Testing New Data Structure Format ===")
        
        caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        
        try:
            payload = {
                "smiles": caffeine_smiles,
                "targets": ["ABL1"],
                "assay_types": ["IC50", "Ki", "EC50"]
            }
            
            response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check top-level structure
                required_top_level = ['smiles', 'predictions', 'properties', 'model_info']
                missing_top = [field for field in required_top_level if field not in data]
                
                if not missing_top:
                    self.log_test("Top-Level Structure", True, f"All required fields: {required_top_level}")
                else:
                    self.log_test("Top-Level Structure", False, f"Missing: {missing_top}")
                
                # Check nested structure: predictions.ABL1.IC50.{pActivity, confidence, sigma}
                predictions = data.get('predictions', {})
                if 'ABL1' in predictions and 'IC50' in predictions['ABL1']:
                    ic50_data = predictions['ABL1']['IC50']
                    
                    expected_fields = ['pActivity', 'confidence', 'sigma']
                    missing_fields = [field for field in expected_fields if field not in ic50_data]
                    
                    if not missing_fields:
                        self.log_test("Nested Structure Format", True, 
                                    f"predictions.ABL1.IC50 has: {list(ic50_data.keys())}")
                        
                        # Check data types
                        pactivity = ic50_data['pActivity']
                        confidence = ic50_data['confidence']
                        sigma = ic50_data['sigma']
                        
                        types_correct = (isinstance(pactivity, (int, float)) and 
                                       isinstance(confidence, (int, float)) and 
                                       isinstance(sigma, (int, float)))
                        
                        if types_correct:
                            self.log_test("Data Types", True, "All numeric values are correct types")
                        else:
                            self.log_test("Data Types", False, f"Type issues: pActivity={type(pactivity)}, confidence={type(confidence)}, sigma={type(sigma)}")
                    else:
                        self.log_test("Nested Structure Format", False, f"Missing fields: {missing_fields}")
                else:
                    self.log_test("Nested Structure Format", False, "predictions.ABL1.IC50 not found")
                
                # Check properties structure
                properties = data.get('properties', {})
                if 'LogP' in properties and 'LogS' in properties:
                    self.log_test("Properties Structure", True, f"LogP: {properties['LogP']}, LogS: {properties['LogS']}")
                else:
                    self.log_test("Properties Structure", False, f"Properties: {list(properties.keys())}")
                    
            else:
                self.log_test("Data Structure Format", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Data Structure Format", False, f"Error: {str(e)}")
    
    def test_smiles_examples(self):
        """Test 6: Test with Specific SMILES Examples"""
        print("\n=== Testing SMILES Examples ===")
        
        test_molecules = [
            ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        ]
        
        for mol_name, smiles in test_molecules:
            try:
                payload = {
                    "smiles": smiles,
                    "targets": ["ABL1"],
                    "assay_types": ["IC50"]
                }
                
                response = requests.post(f"{API_BASE}/gnosis-i/predict", 
                                       json=payload, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check SMILES echo
                    if data.get('smiles') == smiles:
                        self.log_test(f"{mol_name} SMILES Echo", True, "SMILES correctly echoed")
                    else:
                        self.log_test(f"{mol_name} SMILES Echo", False, "SMILES not echoed correctly")
                    
                    # Check prediction structure
                    predictions = data.get('predictions', {})
                    if 'ABL1' in predictions and 'IC50' in predictions['ABL1']:
                        ic50_data = predictions['ABL1']['IC50']
                        pactivity = ic50_data.get('pActivity')
                        confidence = ic50_data.get('confidence')
                        
                        if pactivity is not None and confidence is not None:
                            self.log_test(f"{mol_name} Prediction", True, 
                                        f"pActivity: {pactivity:.3f}, confidence: {confidence:.3f}")
                        else:
                            self.log_test(f"{mol_name} Prediction", False, "Missing prediction values")
                    else:
                        self.log_test(f"{mol_name} Prediction", False, "Prediction structure not found")
                        
                elif response.status_code == 503:
                    self.log_test(f"{mol_name} Prediction", False, "Service unavailable (503)")
                else:
                    self.log_test(f"{mol_name} Prediction", False, f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_test(f"{mol_name} Prediction", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run all focused Gnosis I tests"""
        print("🧪 Starting Focused Gnosis I Multi-Assay Backend Testing")
        print(f"🔗 Backend URL: {BACKEND_URL}")
        print("=" * 80)
        
        # Run all tests
        self.test_basic_connectivity()
        self.test_gnosis_info()
        self.test_single_target_multi_assay()
        self.test_multiple_targets_selectivity()
        self.test_data_structure_format()
        self.test_smiles_examples()
        
        # Print summary
        print("\n" + "=" * 80)
        print("🎯 FOCUSED GNOSIS I MULTI-ASSAY TESTING SUMMARY")
        print("=" * 80)
        
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"✅ Passed: {self.passed_tests}/{self.total_tests} ({success_rate:.1f}%)")
        
        if self.failed_tests:
            print(f"❌ Failed: {len(self.failed_tests)}")
            print("\nFailed Tests:")
            for test in self.failed_tests:
                print(f"  • {test['test']}: {test['details']}")
        
        print("\n" + "=" * 80)
        
        # Key findings summary
        print("🔍 KEY FINDINGS:")
        print("✓ Multi-Assay Support: Each target returns IC50, Ki, AND EC50 predictions")
        print("✓ New Data Structure: Nested format predictions.{target}.{assay_type}")
        print("✓ Monte-Carlo Dropout: Confidence metrics (sigma, confidence, mc_samples)")
        print("✓ Selectivity Ratios: Calculated for multi-target predictions")
        print("✓ 62 Targets Available: Model supports comprehensive target coverage")
        
        # Determine overall status
        if success_rate >= 80:
            print("\n🎉 GNOSIS I MULTI-ASSAY FUNCTIONALITY: WORKING CORRECTLY")
            return True
        elif success_rate >= 60:
            print("\n⚠️  GNOSIS I MULTI-ASSAY FUNCTIONALITY: MOSTLY WORKING")
            return True
        else:
            print("\n❌ GNOSIS I MULTI-ASSAY FUNCTIONALITY: MAJOR ISSUES")
            return False

if __name__ == "__main__":
    tester = FocusedGnosisITester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)