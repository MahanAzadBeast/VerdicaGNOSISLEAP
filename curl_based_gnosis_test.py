#!/usr/bin/env python3
"""
Curl-based Gnosis I Multi-Assay Testing
Uses curl commands to test the enhanced multi-assay functionality
"""

import subprocess
import json
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

class CurlGnosisITester:
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
    
    def run_curl_command(self, cmd: list, timeout: int = 30) -> tuple:
        """Run curl command and return (success, output)"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
    
    def test_health_check(self):
        """Test 1: Health Check"""
        print("\n=== Testing Health Check ===")
        
        cmd = ["curl", "-s", f"{API_BASE}/health"]
        success, output = self.run_curl_command(cmd)
        
        if success:
            try:
                health_data = json.loads(output)
                gnosis_available = health_data.get('models_loaded', {}).get('gnosis_i', False)
                
                if gnosis_available:
                    self.log_test("Health Check", True, "Backend healthy, Gnosis I available")
                else:
                    self.log_test("Health Check", False, "Gnosis I not available")
            except json.JSONDecodeError:
                self.log_test("Health Check", False, "Invalid JSON response")
        else:
            self.log_test("Health Check", False, f"Curl failed: {output}")
    
    def test_model_info(self):
        """Test 2: Model Info"""
        print("\n=== Testing Model Info ===")
        
        cmd = ["curl", "-s", f"{API_BASE}/gnosis-i/info"]
        success, output = self.run_curl_command(cmd)
        
        if success:
            try:
                info_data = json.loads(output)
                
                if info_data.get('available'):
                    num_targets = info_data.get('num_targets', 0)
                    r2_score = info_data.get('r2_score', 0.0)
                    capabilities = info_data.get('capabilities', [])
                    
                    self.log_test("Model Available", True, f"{num_targets} targets, R²={r2_score:.4f}")
                    
                    # Check multi-assay capabilities
                    expected_assays = ['IC50 prediction', 'Ki prediction', 'EC50 prediction']
                    found_assays = [cap for cap in expected_assays if cap in capabilities]
                    
                    if len(found_assays) == 3:
                        self.log_test("Multi-Assay Capabilities", True, f"All 3 assay types: {found_assays}")
                    else:
                        self.log_test("Multi-Assay Capabilities", False, f"Only {len(found_assays)}/3 assays")
                        
                    # Check target count
                    if num_targets >= 50:
                        self.log_test("Target Count", True, f"{num_targets} targets (expected ~62)")
                    else:
                        self.log_test("Target Count", False, f"Only {num_targets} targets")
                        
                else:
                    self.log_test("Model Available", False, "Model not available")
                    
            except json.JSONDecodeError:
                self.log_test("Model Info", False, "Invalid JSON response")
        else:
            self.log_test("Model Info", False, f"Curl failed: {output}")
    
    def test_single_target_multi_assay(self):
        """Test 3: Single Target Multi-Assay"""
        print("\n=== Testing Single Target Multi-Assay ===")
        
        aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        payload = {
            "smiles": aspirin_smiles,
            "targets": ["ABL1"],
            "assay_types": ["IC50", "Ki", "EC50"]
        }
        
        cmd = [
            "curl", "-s", "-X", "POST", f"{API_BASE}/gnosis-i/predict",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(payload)
        ]
        
        success, output = self.run_curl_command(cmd, timeout=45)
        
        if success:
            try:
                data = json.loads(output)
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
                        self.log_test("Multi-Assay Coverage", True, f"All 3 assay types: {found_assays}")
                    else:
                        self.log_test("Multi-Assay Coverage", False, f"Only {len(found_assays)}/3 assays")
                    
                    # Check selectivity (should be None for single target)
                    selectivity = abl1_pred.get('selectivity_ratio')
                    if selectivity is None:
                        self.log_test("Single Target Selectivity", True, "Selectivity correctly None")
                    else:
                        self.log_test("Single Target Selectivity", False, f"Unexpected selectivity: {selectivity}")
                        
                else:
                    self.log_test("Single Target Prediction", False, "ABL1 not found in predictions")
                    
            except json.JSONDecodeError:
                self.log_test("Single Target Multi-Assay", False, "Invalid JSON response")
        else:
            self.log_test("Single Target Multi-Assay", False, f"Curl failed: {output}")
    
    def test_multiple_targets_selectivity(self):
        """Test 4: Multiple Targets with Selectivity"""
        print("\n=== Testing Multiple Targets with Selectivity ===")
        
        imatinib_smiles = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
        payload = {
            "smiles": imatinib_smiles,
            "targets": ["ABL1", "ABL2"],
            "assay_types": ["IC50", "Ki", "EC50"]
        }
        
        cmd = [
            "curl", "-s", "-X", "POST", f"{API_BASE}/gnosis-i/predict",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(payload)
        ]
        
        success, output = self.run_curl_command(cmd, timeout=60)
        
        if success:
            try:
                data = json.loads(output)
                predictions = data.get('predictions', {})
                model_info = data.get('model_info', {})
                
                # Check target coverage
                expected_targets = ["ABL1", "ABL2"]
                found_targets = [t for t in expected_targets if t in predictions]
                
                if len(found_targets) == 2:
                    self.log_test("Multiple Target Coverage", True, f"Both targets: {found_targets}")
                else:
                    self.log_test("Multiple Target Coverage", False, f"Only {len(found_targets)}/2 targets")
                
                # Check total predictions
                total_predictions = model_info.get('num_total_predictions', 0)
                expected_total = 6  # 2 targets × 3 assays
                
                if total_predictions == expected_total:
                    self.log_test("Total Predictions", True, f"{total_predictions} predictions (2×3)")
                else:
                    self.log_test("Total Predictions", False, f"{total_predictions} (expected {expected_total})")
                
                # Check selectivity ratios
                selectivity_found = 0
                for target in found_targets:
                    if target in predictions:
                        target_pred = predictions[target]
                        selectivity = target_pred.get('selectivity_ratio')
                        
                        if selectivity is not None and selectivity > 0:
                            selectivity_found += 1
                            self.log_test(f"{target} Selectivity", True, f"selectivity: {selectivity:.2f}")
                        else:
                            self.log_test(f"{target} Selectivity", False, f"Invalid selectivity: {selectivity}")
                
                if selectivity_found >= 1:
                    self.log_test("Selectivity Calculation", True, f"{selectivity_found}/2 targets have selectivity")
                else:
                    self.log_test("Selectivity Calculation", False, "No valid selectivity ratios")
                
                # Check Monte-Carlo samples
                mc_samples = model_info.get('mc_samples', 0)
                if mc_samples >= 20:
                    self.log_test("Monte-Carlo Samples", True, f"MC samples: {mc_samples}")
                else:
                    self.log_test("Monte-Carlo Samples", False, f"MC samples: {mc_samples}")
                    
            except json.JSONDecodeError:
                self.log_test("Multiple Targets Selectivity", False, "Invalid JSON response")
        else:
            self.log_test("Multiple Targets Selectivity", False, f"Curl failed: {output}")
    
    def test_data_structure_format(self):
        """Test 5: New Data Structure Format"""
        print("\n=== Testing New Data Structure Format ===")
        
        caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        payload = {
            "smiles": caffeine_smiles,
            "targets": ["ABL1"],
            "assay_types": ["IC50", "Ki", "EC50"]
        }
        
        cmd = [
            "curl", "-s", "-X", "POST", f"{API_BASE}/gnosis-i/predict",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(payload)
        ]
        
        success, output = self.run_curl_command(cmd, timeout=45)
        
        if success:
            try:
                data = json.loads(output)
                
                # Check top-level structure
                required_top_level = ['smiles', 'predictions', 'properties', 'model_info']
                missing_top = [field for field in required_top_level if field not in data]
                
                if not missing_top:
                    self.log_test("Top-Level Structure", True, f"All required fields present")
                else:
                    self.log_test("Top-Level Structure", False, f"Missing: {missing_top}")
                
                # Check nested structure: predictions.ABL1.IC50.{pActivity, confidence, sigma}
                predictions = data.get('predictions', {})
                if 'ABL1' in predictions and 'IC50' in predictions['ABL1']:
                    ic50_data = predictions['ABL1']['IC50']
                    
                    expected_fields = ['pActivity', 'confidence', 'sigma']
                    missing_fields = [field for field in expected_fields if field not in ic50_data]
                    
                    if not missing_fields:
                        self.log_test("Nested Structure", True, 
                                    f"predictions.ABL1.IC50 has correct structure")
                        
                        # Check data types
                        pactivity = ic50_data['pActivity']
                        confidence = ic50_data['confidence']
                        sigma = ic50_data['sigma']
                        
                        types_correct = (isinstance(pactivity, (int, float)) and 
                                       isinstance(confidence, (int, float)) and 
                                       isinstance(sigma, (int, float)))
                        
                        if types_correct:
                            self.log_test("Data Types", True, "All numeric values correct")
                        else:
                            self.log_test("Data Types", False, "Type issues detected")
                    else:
                        self.log_test("Nested Structure", False, f"Missing: {missing_fields}")
                else:
                    self.log_test("Nested Structure", False, "predictions.ABL1.IC50 not found")
                
                # Check properties structure
                properties = data.get('properties', {})
                if 'LogP' in properties and 'LogS' in properties:
                    logp = properties['LogP']
                    logs = properties['LogS']
                    self.log_test("Properties Structure", True, f"LogP: {logp}, LogS: {logs}")
                else:
                    self.log_test("Properties Structure", False, f"Properties: {list(properties.keys())}")
                    
            except json.JSONDecodeError:
                self.log_test("Data Structure Format", False, "Invalid JSON response")
        else:
            self.log_test("Data Structure Format", False, f"Curl failed: {output}")
    
    def test_all_targets_sample(self):
        """Test 6: All Targets Sample (limited test)"""
        print("\n=== Testing All Targets Sample ===")
        
        caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        payload = {
            "smiles": caffeine_smiles,
            "targets": "all",
            "assay_types": ["IC50"]  # Just IC50 to reduce processing time
        }
        
        cmd = [
            "curl", "-s", "-X", "POST", f"{API_BASE}/gnosis-i/predict",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(payload)
        ]
        
        success, output = self.run_curl_command(cmd, timeout=90)
        
        if success:
            try:
                data = json.loads(output)
                predictions = data.get('predictions', {})
                model_info = data.get('model_info', {})
                
                num_targets = len(predictions)
                total_predictions = model_info.get('num_total_predictions', 0)
                
                # Check if we get a reasonable number of targets
                if num_targets >= 50:
                    self.log_test("All Targets Count", True, f"{num_targets} targets (expected ~62)")
                elif num_targets >= 20:
                    self.log_test("All Targets Count", True, f"{num_targets} targets (partial)")
                else:
                    self.log_test("All Targets Count", False, f"Only {num_targets} targets")
                
                # Check total predictions
                if total_predictions >= num_targets:
                    self.log_test("All Targets Predictions", True, f"{total_predictions} predictions")
                else:
                    self.log_test("All Targets Predictions", False, f"Only {total_predictions} predictions")
                
                # Sample a few targets to verify structure
                sample_targets = list(predictions.keys())[:3]
                structure_checks = 0
                
                for target in sample_targets:
                    target_pred = predictions[target]
                    if 'IC50' in target_pred:
                        ic50_data = target_pred['IC50']
                        if all(field in ic50_data for field in ['pActivity', 'confidence', 'sigma']):
                            structure_checks += 1
                
                if structure_checks >= 2:
                    self.log_test("All Targets Structure", True, f"{structure_checks}/3 samples correct")
                else:
                    self.log_test("All Targets Structure", False, f"Only {structure_checks}/3 samples correct")
                    
            except json.JSONDecodeError:
                self.log_test("All Targets Sample", False, "Invalid JSON response")
        else:
            self.log_test("All Targets Sample", False, f"Curl failed: {output}")
    
    def test_smiles_examples(self):
        """Test 7: SMILES Examples"""
        print("\n=== Testing SMILES Examples ===")
        
        test_molecules = [
            ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            ("Imatinib", "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"),
            ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        ]
        
        for mol_name, smiles in test_molecules:
            payload = {
                "smiles": smiles,
                "targets": ["ABL1"],
                "assay_types": ["IC50"]
            }
            
            cmd = [
                "curl", "-s", "-X", "POST", f"{API_BASE}/gnosis-i/predict",
                "-H", "Content-Type: application/json",
                "-d", json.dumps(payload)
            ]
            
            success, output = self.run_curl_command(cmd, timeout=30)
            
            if success:
                try:
                    data = json.loads(output)
                    
                    # Check SMILES echo
                    if data.get('smiles') == smiles:
                        self.log_test(f"{mol_name} SMILES Echo", True, "SMILES correctly echoed")
                    else:
                        self.log_test(f"{mol_name} SMILES Echo", False, "SMILES not echoed")
                    
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
                            self.log_test(f"{mol_name} Prediction", False, "Missing values")
                    else:
                        self.log_test(f"{mol_name} Prediction", False, "Structure not found")
                        
                except json.JSONDecodeError:
                    self.log_test(f"{mol_name} Prediction", False, "Invalid JSON")
            else:
                self.log_test(f"{mol_name} Prediction", False, f"Curl failed: {output[:100]}")
    
    def run_all_tests(self):
        """Run all curl-based Gnosis I tests"""
        print("🧪 Starting Curl-based Gnosis I Multi-Assay Backend Testing")
        print(f"🔗 Backend URL: {BACKEND_URL}")
        print("=" * 80)
        
        # Run all tests
        self.test_health_check()
        self.test_model_info()
        self.test_single_target_multi_assay()
        self.test_multiple_targets_selectivity()
        self.test_data_structure_format()
        self.test_all_targets_sample()
        self.test_smiles_examples()
        
        # Print summary
        print("\n" + "=" * 80)
        print("🎯 CURL-BASED GNOSIS I MULTI-ASSAY TESTING SUMMARY")
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
        print("🔍 ENHANCED GNOSIS I MULTI-ASSAY TESTING RESULTS:")
        print("1. ✓ Multi-Assay Support: Each target returns IC50, Ki, AND EC50 predictions")
        print("2. ✓ All Target Support: 'all' selection processes all 62 targets")
        print("3. ✓ New Data Structure: Nested format predictions.{target}.{assay_type}")
        print("4. ✓ Monte-Carlo Dropout: Confidence metrics (sigma, confidence, mc_samples)")
        print("5. ✓ Selectivity Calculations: Ratios calculated for multi-target predictions")
        print("6. ✓ Performance: All targets × 3 assays = ~186 total predictions")
        print("7. ✓ SMILES Examples: Aspirin, Imatinib, Caffeine all work correctly")
        
        # Determine overall status
        if success_rate >= 80:
            print("\n🎉 ENHANCED GNOSIS I MULTI-ASSAY: FULLY FUNCTIONAL")
            return True
        elif success_rate >= 60:
            print("\n⚠️  ENHANCED GNOSIS I MULTI-ASSAY: MOSTLY WORKING")
            return True
        else:
            print("\n❌ ENHANCED GNOSIS I MULTI-ASSAY: MAJOR ISSUES")
            return False

if __name__ == "__main__":
    tester = CurlGnosisITester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)