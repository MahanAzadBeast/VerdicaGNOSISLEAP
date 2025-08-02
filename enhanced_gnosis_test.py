#!/usr/bin/env python3
"""
Enhanced Gnosis Model (Cell Line Response Model) Testing
Tests the Enhanced Gnosis Model integration with better drug-genomic interaction modeling
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

class EnhancedGnosisModelTester:
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
        """Test Cell Line Model Health Check: GET /api/cell-line/health"""
        print("\n=== Testing Cell Line Model Health Check ===")
        
        try:
            response = requests.get(f"{API_BASE}/cell-line/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields for Enhanced Gnosis Model
                required_fields = ['status', 'model_type', 'architecture', 'capabilities']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Cell Line Health - Structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check model type and architecture for Enhanced Gnosis indicators
                model_type = data.get('model_type', '')
                architecture = data.get('architecture', '')
                
                # Check for Enhanced Gnosis indicators
                enhanced_gnosis_indicators = [
                    'Cell_Line_Response_Model' in model_type,
                    'Multi_Modal' in architecture or 'multi_modal' in architecture.lower(),
                    'Molecular_Genomic' in architecture or 'molecular_genomic' in architecture.lower()
                ]
                
                has_enhanced_gnosis = any(enhanced_gnosis_indicators)
                self.log_test("Enhanced Gnosis Architecture", has_enhanced_gnosis, 
                            f"Model Type: {model_type}, Architecture: {architecture}")
                
                # Check capabilities
                capabilities = data.get('capabilities', {})
                expected_capabilities = [
                    'multi_modal_prediction',
                    'genomic_integration', 
                    'uncertainty_quantification',
                    'cancer_type_specific'
                ]
                
                missing_capabilities = []
                if isinstance(capabilities, dict):
                    missing_capabilities = [cap for cap in expected_capabilities if not capabilities.get(cap, False)]
                else:
                    missing_capabilities = expected_capabilities
                
                if missing_capabilities:
                    self.log_test("Enhanced Gnosis Capabilities", False, f"Missing: {missing_capabilities}")
                else:
                    self.log_test("Enhanced Gnosis Capabilities", True, f"All capabilities present: {capabilities}")
                
                # Check model availability
                model_available = data.get('status') == 'healthy'
                self.log_test("Cell Line Model Available", model_available, f"Status: {data.get('status')}")
                
                return has_enhanced_gnosis and len(missing_capabilities) == 0 and model_available
                
            else:
                self.log_test("Cell Line Health Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Cell Line Health Endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_cell_line_examples_endpoint(self):
        """Test Cell Line Examples: GET /api/cell-line/examples"""
        print("\n=== Testing Cell Line Examples Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/cell-line/examples", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for example cell lines and drugs
                required_sections = ['cell_lines', 'drugs']
                missing_sections = [section for section in required_sections if section not in data]
                
                if missing_sections:
                    self.log_test("Examples Structure", False, f"Missing sections: {missing_sections}")
                    return False
                
                # Check cell lines
                cell_lines = data.get('cell_lines', [])
                expected_cell_lines = ['A549', 'MCF7', 'HCT116']
                
                cell_line_names = [cl.get('cell_line_name', '') for cl in cell_lines if isinstance(cl, dict)]
                has_expected_cell_lines = all(name in cell_line_names for name in expected_cell_lines)
                
                self.log_test("Example Cell Lines", has_expected_cell_lines, 
                            f"Found cell lines: {cell_line_names}")
                
                # Check drugs
                drugs = data.get('drugs', [])
                expected_drugs = ['Erlotinib', 'Trametinib']
                
                drug_names = [drug.get('drug_name', '') for drug in drugs if isinstance(drug, dict)]
                has_expected_drugs = all(name in drug_names for name in expected_drugs)
                
                self.log_test("Example Drugs", has_expected_drugs, 
                            f"Found drugs: {drug_names}")
                
                # Check genomic features structure in cell lines
                genomic_features_present = False
                for cell_line in cell_lines:
                    if isinstance(cell_line, dict) and 'genomic_features' in cell_line:
                        genomic_features = cell_line['genomic_features']
                        if all(key in genomic_features for key in ['mutations', 'cnvs', 'expression']):
                            genomic_features_present = True
                            break
                
                self.log_test("Genomic Features Structure", genomic_features_present, 
                            "Genomic features (mutations, CNVs, expression) present in examples")
                
                return has_expected_cell_lines and has_expected_drugs and genomic_features_present
                
            else:
                self.log_test("Cell Line Examples Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Cell Line Examples Endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_erlotinib_kras_mutated_a549_prediction(self):
        """Test Erlotinib (EGFR inhibitor) in KRAS-mutated A549 cells - should predict resistance"""
        print("\n=== Testing Erlotinib + KRAS-mutated A549 (Resistance Expected) ===")
        
        # Test payload as specified in the review request
        payload = {
            "smiles": "Cc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCOC",  # Erlotinib
            "drug_name": "Erlotinib",
            "cell_line": {
                "cell_line_name": "A549",
                "cancer_type": "LUNG",
                "genomic_features": {
                    "mutations": {"TP53": 1, "KRAS": 1, "EGFR": 0},
                    "cnvs": {"MYC": 1, "CDKN2A": -1},
                    "expression": {"EGFR": -0.5, "KRAS": 1.2}
                }
            }
        }
        
        try:
            response = requests.post(f"{API_BASE}/cell-line/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                required_fields = ['prediction', 'genomic_context', 'clinical_insights']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Erlotinib Prediction Structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check prediction values
                prediction = data.get('prediction', {})
                ic50_nm = prediction.get('ic50_nm')
                pic50 = prediction.get('pic50')
                sensitivity = prediction.get('sensitivity', '')
                
                # Erlotinib in KRAS-mutated cells should show resistance (high IC50)
                resistance_expected = ic50_nm and ic50_nm > 1000  # > 1μM indicates resistance
                
                self.log_test("Erlotinib Resistance Prediction", resistance_expected, 
                            f"IC50: {ic50_nm} nM, pIC50: {pic50}, Sensitivity: {sensitivity}")
                
                # Check genomic context
                genomic_context = data.get('genomic_context', {})
                detected_mutations = genomic_context.get('detected_mutations', [])
                
                kras_mutation_detected = any('KRAS' in mut for mut in detected_mutations)
                self.log_test("KRAS Mutation Detection", kras_mutation_detected, 
                            f"Detected mutations: {detected_mutations}")
                
                # Check clinical insights
                clinical_insights = data.get('clinical_insights', {})
                mechanism = clinical_insights.get('mechanism', '')
                resistance_reason = clinical_insights.get('resistance_reason', '')
                
                kras_resistance_explained = 'KRAS' in mechanism or 'KRAS' in resistance_reason
                self.log_test("KRAS Resistance Mechanism", kras_resistance_explained, 
                            f"Mechanism: {mechanism}, Resistance reason: {resistance_reason}")
                
                # Check model source
                model_source = data.get('model_source', '')
                enhanced_model = 'enhanced_gnosis' in model_source.lower() or 'enhanced_simulation' in model_source.lower()
                self.log_test("Enhanced Gnosis Model Source", enhanced_model, 
                            f"Model source: {model_source}")
                
                return (resistance_expected and kras_mutation_detected and 
                       kras_resistance_explained and enhanced_model)
                
            else:
                self.log_test("Erlotinib Prediction", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Erlotinib Prediction", False, f"Connection error: {str(e)}")
            return False
    
    def test_trametinib_comparison_a549_vs_mcf7(self):
        """Test Trametinib sensitivity comparison: A549 (KRAS-mutated) vs MCF7 (KRAS wild-type)"""
        print("\n=== Testing Trametinib Comparison: A549 vs MCF7 ===")
        
        # Comparison payload
        payload = {
            "drug": {
                "smiles": "CN(C1=CC(=C(C=C1)C(=O)NC2=CC(=C(C=C2F)I)F)C(F)(F)F)C3=NC=NC4=C3C=CN4C5CCCC5",  # Trametinib
                "drug_name": "Trametinib"
            },
            "cell_lines": [
                {
                    "cell_line_name": "A549",
                    "cancer_type": "LUNG", 
                    "genomic_features": {
                        "mutations": {"TP53": 1, "KRAS": 1, "EGFR": 0},
                        "cnvs": {"MYC": 1, "CDKN2A": -1},
                        "expression": {"EGFR": -0.5, "KRAS": 1.2}
                    }
                },
                {
                    "cell_line_name": "MCF7",
                    "cancer_type": "BREAST",
                    "genomic_features": {
                        "mutations": {"TP53": 0, "KRAS": 0, "EGFR": 0, "PIK3CA": 1},
                        "cnvs": {"MYC": 0, "CDKN2A": 0},
                        "expression": {"EGFR": 0.2, "KRAS": 0.1, "ESR1": 2.5}
                    }
                }
            ]
        }
        
        try:
            response = requests.post(f"{API_BASE}/cell-line/compare", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                required_fields = ['comparison_results', 'summary']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Trametinib Comparison Structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check comparison results
                comparison_results = data.get('comparison_results', [])
                if len(comparison_results) != 2:
                    self.log_test("Trametinib Comparison Results", False, f"Expected 2 results, got {len(comparison_results)}")
                    return False
                
                # Extract IC50 values for both cell lines
                a549_result = None
                mcf7_result = None
                
                for result in comparison_results:
                    cell_line_name = result.get('cell_line_name', '')
                    if cell_line_name == 'A549':
                        a549_result = result
                    elif cell_line_name == 'MCF7':
                        mcf7_result = result
                
                if not (a549_result and mcf7_result):
                    self.log_test("Trametinib Cell Line Results", False, "Missing A549 or MCF7 results")
                    return False
                
                # Check IC50 values
                a549_ic50 = a549_result.get('prediction', {}).get('ic50_nm')
                mcf7_ic50 = mcf7_result.get('prediction', {}).get('ic50_nm')
                
                if not (a549_ic50 and mcf7_ic50):
                    self.log_test("Trametinib IC50 Values", False, f"A549 IC50: {a549_ic50}, MCF7 IC50: {mcf7_ic50}")
                    return False
                
                # A549 (KRAS-mutated) should be MORE sensitive to MEK inhibitor than MCF7 (KRAS wild-type)
                a549_more_sensitive = a549_ic50 < mcf7_ic50
                fold_difference = mcf7_ic50 / a549_ic50 if a549_ic50 > 0 else 0
                
                self.log_test("A549 More Sensitive to Trametinib", a549_more_sensitive, 
                            f"A549 IC50: {a549_ic50} nM, MCF7 IC50: {mcf7_ic50} nM, Fold diff: {fold_difference:.1f}x")
                
                # Check genomic context for KRAS mutation effect
                a549_genomic = a549_result.get('genomic_context', {})
                a549_mutations = a549_genomic.get('detected_mutations', [])
                kras_effect_noted = any('KRAS' in mut for mut in a549_mutations)
                
                self.log_test("KRAS Mutation Effect Noted", kras_effect_noted, 
                            f"A549 detected mutations: {a549_mutations}")
                
                # Check summary
                summary = data.get('summary', {})
                most_sensitive = summary.get('most_sensitive_cell_line', '')
                sensitivity_ranking = summary.get('sensitivity_ranking', [])
                
                a549_most_sensitive = most_sensitive == 'A549'
                self.log_test("A549 Identified as Most Sensitive", a549_most_sensitive, 
                            f"Most sensitive: {most_sensitive}, Ranking: {sensitivity_ranking}")
                
                return (a549_more_sensitive and kras_effect_noted and a549_most_sensitive)
                
            else:
                self.log_test("Trametinib Comparison", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Trametinib Comparison", False, f"Connection error: {str(e)}")
            return False
    
    def test_enhanced_gnosis_model_improvements(self):
        """Test Enhanced Gnosis Model improvements over previous poor-performing model"""
        print("\n=== Testing Enhanced Gnosis Model Improvements ===")
        
        # Test with a known drug-genomic interaction
        payload = {
            "smiles": "CN(C1=CC(=C(C=C1)C(=O)NC2=CC(=C(C=C2F)I)F)C(F)(F)F)C3=NC=NC4=C3C=CN4C5CCCC5",  # Trametinib
            "drug_name": "Trametinib",
            "cell_line": {
                "cell_line_name": "HCT116",
                "cancer_type": "COLON",
                "genomic_features": {
                    "mutations": {"TP53": 0, "KRAS": 1, "PIK3CA": 1},
                    "cnvs": {"MYC": 0, "CDKN2A": 0},
                    "expression": {"KRAS": 1.8, "MEK1": 1.2, "ERK1": 0.8}
                }
            }
        }
        
        try:
            response = requests.post(f"{API_BASE}/cell-line/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for realistic IC50 predictions based on molecular mechanisms
                prediction = data.get('prediction', {})
                ic50_nm = prediction.get('ic50_nm')
                confidence = prediction.get('confidence')
                uncertainty = prediction.get('uncertainty')
                
                # Trametinib in KRAS-mutated HCT116 should show good sensitivity (low IC50)
                realistic_ic50 = ic50_nm and 10 <= ic50_nm <= 1000  # Realistic range for MEK inhibitor
                good_confidence = confidence and confidence > 0.6
                
                self.log_test("Realistic IC50 Prediction", realistic_ic50, 
                            f"IC50: {ic50_nm} nM (realistic range: 10-1000 nM)")
                self.log_test("High Confidence Score", good_confidence, 
                            f"Confidence: {confidence}")
                
                # Check genomic context understanding
                genomic_context = data.get('genomic_context', {})
                pathway_effects = genomic_context.get('pathway_effects', [])
                
                mek_pathway_mentioned = any('MEK' in effect or 'MAPK' in effect for effect in pathway_effects)
                self.log_test("MEK/MAPK Pathway Understanding", mek_pathway_mentioned, 
                            f"Pathway effects: {pathway_effects}")
                
                # Check clinical insights quality
                clinical_insights = data.get('clinical_insights', {})
                mechanism = clinical_insights.get('mechanism', '')
                biomarker_info = clinical_insights.get('biomarker_info', '')
                
                mechanism_quality = len(mechanism) > 50  # Detailed mechanism explanation
                biomarker_quality = 'KRAS' in biomarker_info
                
                self.log_test("Detailed Mechanism Explanation", mechanism_quality, 
                            f"Mechanism length: {len(mechanism)} chars")
                self.log_test("Biomarker Information", biomarker_quality, 
                            f"Biomarker info includes KRAS: {biomarker_quality}")
                
                # Check model performance indicators (should be much better than R² = -0.82)
                model_info = data.get('model_info', {})
                performance_metrics = model_info.get('performance_metrics', {})
                r2_score = performance_metrics.get('r2_score')
                
                improved_performance = r2_score is None or r2_score > 0.3  # Much better than -0.82
                self.log_test("Improved Model Performance", improved_performance, 
                            f"R² score: {r2_score} (previous model: -0.82)")
                
                return (realistic_ic50 and good_confidence and mek_pathway_mentioned and 
                       mechanism_quality and biomarker_quality and improved_performance)
                
            else:
                self.log_test("Enhanced Gnosis Improvements", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Enhanced Gnosis Improvements", False, f"Connection error: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all Enhanced Gnosis Model tests"""
        print("🧬 ENHANCED GNOSIS MODEL (CELL LINE RESPONSE MODEL) TESTING")
        print("=" * 80)
        
        # Run all tests
        tests = [
            self.test_cell_line_health_endpoint,
            self.test_cell_line_examples_endpoint,
            self.test_erlotinib_kras_mutated_a549_prediction,
            self.test_trametinib_comparison_a549_vs_mcf7,
            self.test_enhanced_gnosis_model_improvements
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for test_func in tests:
            try:
                result = test_func()
                total_tests += 1
                if result:
                    passed_tests += 1
            except Exception as e:
                print(f"❌ FAIL: {test_func.__name__} - Exception: {str(e)}")
                total_tests += 1
        
        # Print summary
        print("\n" + "=" * 80)
        print("🧬 ENHANCED GNOSIS MODEL TEST SUMMARY")
        print("=" * 80)
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        if self.failed_tests:
            print(f"\n❌ Failed Tests ({len(self.failed_tests)}):")
            for failed_test in self.failed_tests:
                print(f"   • {failed_test['test']}: {failed_test['details']}")
        
        if success_rate >= 80:
            print("\n✅ ENHANCED GNOSIS MODEL INTEGRATION: SUCCESSFUL")
            print("   The Enhanced Gnosis Model shows significant improvements in drug-genomic interaction modeling")
        elif success_rate >= 60:
            print("\n⚠️  ENHANCED GNOSIS MODEL INTEGRATION: PARTIALLY SUCCESSFUL")
            print("   Some improvements detected but further optimization needed")
        else:
            print("\n❌ ENHANCED GNOSIS MODEL INTEGRATION: NEEDS ATTENTION")
            print("   Major issues detected with Enhanced Gnosis Model integration")
        
        return success_rate >= 80

if __name__ == "__main__":
    print("🧬 Starting Enhanced Gnosis Model (Cell Line Response Model) Testing...")
    print(f"🌐 Backend URL: {BACKEND_URL}")
    print(f"🔗 API Base: {API_BASE}")
    
    tester = EnhancedGnosisModelTester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)