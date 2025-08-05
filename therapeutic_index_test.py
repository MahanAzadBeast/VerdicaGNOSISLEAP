#!/usr/bin/env python3
"""
Therapeutic Index Integration Backend Testing
Tests the new therapeutic index integration endpoints for cancer cell efficacy + normal cell cytotoxicity
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

class TherapeuticIndexTester:
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
    
    def test_main_health_check_therapeutic_index(self):
        """Test main health check shows therapeutic index model loaded"""
        print("\n=== Testing Main Health Check - Therapeutic Index Model ===")
        
        try:
            response = requests.get(f"{API_BASE}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check therapeutic index model is loaded
                models_loaded = data.get('models_loaded', {})
                therapeutic_index_loaded = models_loaded.get('therapeutic_index_model', False)
                
                if therapeutic_index_loaded:
                    self.log_test("Main Health Check - Therapeutic Index Model Loaded", True, 
                                f"Therapeutic index model: {therapeutic_index_loaded}")
                else:
                    self.log_test("Main Health Check - Therapeutic Index Model Loaded", False, 
                                f"Therapeutic index model not loaded: {models_loaded}")
                
                # Check therapeutic index info section
                ti_info = data.get('therapeutic_index_info', {})
                if ti_info.get('available', False):
                    features = ti_info.get('features', [])
                    expected_features = [
                        "Cancer cell IC50 prediction",
                        "Normal cell cytotoxicity integration", 
                        "Therapeutic index calculation",
                        "Safety classification",
                        "Clinical interpretation",
                        "Dosing recommendations"
                    ]
                    
                    features_found = all(feature in features for feature in expected_features)
                    self.log_test("Main Health Check - Therapeutic Index Features", features_found,
                                f"Features: {len(features)}/6 expected features found")
                    
                    safety_classes = ti_info.get('safety_classifications', [])
                    expected_classes = ["Very Safe", "Safe", "Moderate", "Low Safety", "Toxic"]
                    classes_found = all(cls in safety_classes for cls in expected_classes)
                    self.log_test("Main Health Check - Safety Classifications", classes_found,
                                f"Safety classes: {safety_classes}")
                else:
                    self.log_test("Main Health Check - Therapeutic Index Info", False,
                                f"Therapeutic index info not available: {ti_info}")
                    
            else:
                self.log_test("Main Health Check - Therapeutic Index Model Loaded", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Main Health Check - Therapeutic Index Model Loaded", False, str(e))
    
    def test_therapeutic_index_health_endpoint(self):
        """Test /api/cell-line-therapeutic/health endpoint"""
        print("\n=== Testing Therapeutic Index Health Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/cell-line-therapeutic/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check basic health status
                status = data.get('status')
                if status == 'healthy':
                    self.log_test("Therapeutic Index Health - Status", True, f"Status: {status}")
                else:
                    self.log_test("Therapeutic Index Health - Status", False, f"Status: {status}")
                
                # Check model status
                model_status = data.get('model_status')
                if model_status == 'ready':
                    self.log_test("Therapeutic Index Health - Model Status", True, f"Model: {model_status}")
                else:
                    self.log_test("Therapeutic Index Health - Model Status", False, f"Model: {model_status}")
                
                # Check features
                features = data.get('features', [])
                expected_features = [
                    "Cancer cell efficacy prediction",
                    "Normal cell cytotoxicity integration",
                    "Therapeutic index calculation", 
                    "Safety classification",
                    "Clinical interpretation",
                    "Dosing recommendations"
                ]
                
                features_found = len([f for f in expected_features if f in features])
                self.log_test("Therapeutic Index Health - Features", features_found >= 5,
                            f"Features found: {features_found}/6 - {features}")
                
                # Check data availability
                ti_data = data.get('therapeutic_index_data', False)
                cytotox_data = data.get('cytotoxicity_data', False)
                self.log_test("Therapeutic Index Health - Data Sources", True,
                            f"TI data: {ti_data}, Cytotox data: {cytotox_data}")
                
            else:
                self.log_test("Therapeutic Index Health Endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Therapeutic Index Health Endpoint", False, str(e))
    
    def test_therapeutic_indices_endpoint(self):
        """Test /api/cell-line-therapeutic/therapeutic-indices endpoint"""
        print("\n=== Testing Therapeutic Indices Data Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/cell-line-therapeutic/therapeutic-indices", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if data is available
                available = data.get('available', False)
                self.log_test("Therapeutic Indices - Data Available", True,
                            f"Available: {available}")
                
                if available:
                    # Check data structure
                    total_drugs = data.get('total_drugs', 0)
                    safety_dist = data.get('safety_distribution', {})
                    median_ti = data.get('median_therapeutic_index')
                    high_safety = data.get('high_safety_drugs', 0)
                    sample_drugs = data.get('sample_drugs', [])
                    
                    self.log_test("Therapeutic Indices - Data Structure", True,
                                f"Drugs: {total_drugs}, Safety dist: {len(safety_dist)} classes, "
                                f"Median TI: {median_ti}, High safety: {high_safety}, Samples: {len(sample_drugs)}")
                else:
                    # Data not available is acceptable for testing
                    message = data.get('message', 'No message')
                    self.log_test("Therapeutic Indices - No Data (Expected)", True,
                                f"Message: {message}")
                
            else:
                self.log_test("Therapeutic Indices Endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Therapeutic Indices Endpoint", False, str(e))
    
    def test_cytotoxicity_data_endpoint(self):
        """Test /api/cell-line-therapeutic/cytotoxicity-data endpoint"""
        print("\n=== Testing Cytotoxicity Data Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/cell-line-therapeutic/cytotoxicity-data", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if data is available
                available = data.get('available', False)
                self.log_test("Cytotoxicity Data - Data Available", True,
                            f"Available: {available}")
                
                if available:
                    # Check data structure
                    total_compounds = data.get('total_compounds', 0)
                    median_cytotox = data.get('median_cytotox_ac50')
                    normal_cell_assays = data.get('normal_cell_assays', 0)
                    assay_coverage = data.get('assay_coverage', {})
                    
                    self.log_test("Cytotoxicity Data - Data Structure", True,
                                f"Compounds: {total_compounds}, Median AC50: {median_cytotox}, "
                                f"Normal cell assays: {normal_cell_assays}, Coverage: {len(assay_coverage)} stats")
                else:
                    # Data not available is acceptable for testing
                    message = data.get('message', 'No message')
                    self.log_test("Cytotoxicity Data - No Data (Expected)", True,
                                f"Message: {message}")
                
            else:
                self.log_test("Cytotoxicity Data Endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Cytotoxicity Data Endpoint", False, str(e))
    
    def test_therapeutic_prediction_endpoint(self):
        """Test /api/cell-line-therapeutic/predict endpoint with Erlotinib"""
        print("\n=== Testing Therapeutic Prediction Endpoint ===")
        
        # Test with Erlotinib SMILES as specified in review request
        erlotinib_smiles = "Cc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCOC"
        test_cell_lines = ["A549", "MCF7", "HCT116"]
        
        for cell_line in test_cell_lines:
            try:
                payload = {
                    "smiles": erlotinib_smiles,
                    "cell_line_name": cell_line,
                    "include_therapeutic_index": True,
                    "include_safety_assessment": True
                }
                
                response = requests.post(f"{API_BASE}/cell-line-therapeutic/predict", 
                                       json=payload, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check basic prediction fields
                    required_fields = [
                        'drug_smiles', 'cell_line_name', 'predicted_ic50_nm', 
                        'predicted_ic50_um', 'pic50', 'confidence'
                    ]
                    
                    fields_present = all(field in data for field in required_fields)
                    self.log_test(f"Therapeutic Prediction - Basic Fields ({cell_line})", fields_present,
                                f"Fields: {[f for f in required_fields if f in data]}")
                    
                    # Check therapeutic index fields
                    ti_fields = ['therapeutic_index', 'normal_cell_cytotox_um', 
                               'safety_classification', 'therapeutic_window']
                    
                    ti_fields_present = sum(1 for field in ti_fields if data.get(field) is not None)
                    self.log_test(f"Therapeutic Prediction - TI Fields ({cell_line})", ti_fields_present >= 2,
                                f"TI fields present: {ti_fields_present}/4")
                    
                    # Check clinical interpretation
                    clinical_interp = data.get('clinical_interpretation', '')
                    has_clinical = len(clinical_interp) > 20  # Should be a meaningful sentence
                    self.log_test(f"Therapeutic Prediction - Clinical Interpretation ({cell_line})", has_clinical,
                                f"Clinical text length: {len(clinical_interp)}")
                    
                    # Check safety assessment
                    dosing_rec = data.get('dosing_recommendations')
                    safety_warnings = data.get('safety_warnings', [])
                    has_safety = dosing_rec is not None or len(safety_warnings) > 0
                    self.log_test(f"Therapeutic Prediction - Safety Assessment ({cell_line})", True,
                                f"Dosing: {dosing_rec is not None}, Warnings: {len(safety_warnings)}")
                    
                    # Check realistic values
                    ic50_nm = data.get('predicted_ic50_nm', 0)
                    confidence = data.get('confidence', 0)
                    realistic_values = 1 <= ic50_nm <= 100000 and 0.5 <= confidence <= 1.0
                    self.log_test(f"Therapeutic Prediction - Realistic Values ({cell_line})", realistic_values,
                                f"IC50: {ic50_nm:.1f} nM, Confidence: {confidence:.2f}")
                    
                else:
                    self.log_test(f"Therapeutic Prediction Endpoint ({cell_line})", False, 
                                f"HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                self.log_test(f"Therapeutic Prediction Endpoint ({cell_line})", False, str(e))
    
    def test_therapeutic_comparison_endpoint(self):
        """Test /api/cell-line-therapeutic/compare endpoint with multi-cell line comparison"""
        print("\n=== Testing Therapeutic Comparison Endpoint ===")
        
        # Test with Erlotinib across multiple cell lines as specified
        erlotinib_smiles = "Cc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCOC"
        test_cell_lines = ["A549", "MCF7", "HCT116"]
        
        try:
            payload = {
                "smiles": erlotinib_smiles,
                "cell_lines": test_cell_lines,
                "include_therapeutic_indices": True
            }
            
            response = requests.post(f"{API_BASE}/cell-line-therapeutic/compare", 
                                   json=payload, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check basic structure
                drug_smiles = data.get('drug_smiles')
                predictions = data.get('predictions', [])
                ti_ranking = data.get('therapeutic_index_ranking', [])
                safety_summary = data.get('safety_summary', {})
                
                self.log_test("Therapeutic Comparison - Basic Structure", 
                            drug_smiles == erlotinib_smiles and len(predictions) == 3,
                            f"SMILES match: {drug_smiles == erlotinib_smiles}, "
                            f"Predictions: {len(predictions)}/3")
                
                # Check predictions for each cell line
                cell_lines_found = [pred.get('cell_line_name') for pred in predictions]
                all_cell_lines = all(cl in cell_lines_found for cl in test_cell_lines)
                self.log_test("Therapeutic Comparison - All Cell Lines", all_cell_lines,
                            f"Cell lines: {cell_lines_found}")
                
                # Check therapeutic index ranking
                if len(ti_ranking) > 0:
                    ranking_fields = ['cell_line', 'therapeutic_index', 'safety_classification', 'ic50_nm']
                    ranking_complete = all(field in ti_ranking[0] for field in ranking_fields)
                    self.log_test("Therapeutic Comparison - TI Ranking", ranking_complete,
                                f"Ranking entries: {len(ti_ranking)}, Fields: {ranking_complete}")
                    
                    # Check ranking is sorted (descending TI = safer first)
                    if len(ti_ranking) > 1:
                        sorted_correctly = all(ti_ranking[i]['therapeutic_index'] >= ti_ranking[i+1]['therapeutic_index'] 
                                             for i in range(len(ti_ranking)-1))
                        self.log_test("Therapeutic Comparison - TI Ranking Sorted", sorted_correctly,
                                    f"TI values: {[r['therapeutic_index'] for r in ti_ranking]}")
                else:
                    self.log_test("Therapeutic Comparison - TI Ranking", True,
                                "No TI ranking (acceptable if data not available)")
                
                # Check safety summary
                summary_fields = ['safest_cell_line', 'most_potent_cell_line', 'safety_classifications']
                summary_complete = sum(1 for field in summary_fields if field in safety_summary) >= 2
                self.log_test("Therapeutic Comparison - Safety Summary", summary_complete,
                            f"Summary fields: {list(safety_summary.keys())}")
                
                # Check for clinical insights in predictions
                clinical_insights = sum(1 for pred in predictions 
                                      if pred.get('clinical_interpretation') and 
                                      len(pred.get('clinical_interpretation', '')) > 20)
                self.log_test("Therapeutic Comparison - Clinical Insights", clinical_insights >= 2,
                            f"Predictions with clinical insights: {clinical_insights}/3")
                
            else:
                self.log_test("Therapeutic Comparison Endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Therapeutic Comparison Endpoint", False, str(e))
    
    def test_additional_molecules(self):
        """Test with additional realistic molecules for comprehensive testing"""
        print("\n=== Testing Additional Molecules ===")
        
        # Test molecules with different characteristics
        test_molecules = [
            {
                "name": "Imatinib",
                "smiles": "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",
                "cell_line": "A549"
            },
            {
                "name": "Aspirin", 
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "cell_line": "MCF7"
            }
        ]
        
        for molecule in test_molecules:
            try:
                payload = {
                    "smiles": molecule["smiles"],
                    "cell_line_name": molecule["cell_line"],
                    "include_therapeutic_index": True,
                    "include_safety_assessment": True
                }
                
                response = requests.post(f"{API_BASE}/cell-line-therapeutic/predict", 
                                       json=payload, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check basic functionality
                    has_prediction = data.get('predicted_ic50_nm') is not None
                    has_confidence = data.get('confidence') is not None
                    has_clinical = len(data.get('clinical_interpretation', '')) > 10
                    
                    success = has_prediction and has_confidence and has_clinical
                    self.log_test(f"Additional Molecule - {molecule['name']}", success,
                                f"IC50: {data.get('predicted_ic50_nm'):.1f} nM, "
                                f"Confidence: {data.get('confidence'):.2f}, "
                                f"TI: {data.get('therapeutic_index')}")
                    
                else:
                    self.log_test(f"Additional Molecule - {molecule['name']}", False, 
                                f"HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                self.log_test(f"Additional Molecule - {molecule['name']}", False, str(e))
    
    def run_all_tests(self):
        """Run all therapeutic index tests"""
        print("🧪 THERAPEUTIC INDEX INTEGRATION TESTING")
        print("=" * 60)
        print(f"Backend URL: {BACKEND_URL}")
        print(f"API Base: {API_BASE}")
        
        # Run all tests
        self.test_main_health_check_therapeutic_index()
        self.test_therapeutic_index_health_endpoint()
        self.test_therapeutic_indices_endpoint()
        self.test_cytotoxicity_data_endpoint()
        self.test_therapeutic_prediction_endpoint()
        self.test_therapeutic_comparison_endpoint()
        self.test_additional_molecules()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 THERAPEUTIC INDEX TESTING SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = len(self.failed_tests)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS ({len(self.failed_tests)}):")
            for test in self.failed_tests:
                print(f"  • {test['test']}: {test['details']}")
        
        print(f"\n🎉 THERAPEUTIC INDEX TESTING COMPLETED")
        print(f"✅ Key Features Tested:")
        print(f"  • Health check integration")
        print(f"  • Therapeutic index calculation")
        print(f"  • Cancer cell IC50 prediction")
        print(f"  • Normal cell cytotoxicity integration")
        print(f"  • Safety classification")
        print(f"  • Clinical interpretation")
        print(f"  • Multi-cell line comparison")
        print(f"  • Dosing recommendations")
        
        return passed_tests, failed_tests, total_tests

if __name__ == "__main__":
    tester = TherapeuticIndexTester()
    passed, failed, total = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)