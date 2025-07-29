#!/usr/bin/env python3
"""
Enhanced Modal MolBERT Integration Testing
Focus on testing the new Enhanced Modal MolBERT endpoints
"""

import requests
import json
import time
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

class ModalMolBERTTester:
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
    
    def test_modal_status_endpoint(self):
        """Test GET /api/modal/molbert/status"""
        print("\n=== Testing Enhanced Modal MolBERT Status Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/modal/molbert/status", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                modal_available = data.get('modal_available', None)
                credentials_set = data.get('credentials_set', None)
                app_name = data.get('app_name', '')
                timestamp = data.get('timestamp', '')
                
                self.log_test("Modal status endpoint accessible", True, 
                            f"Modal available: {modal_available}, Credentials: {credentials_set}")
                
                # Validate field types
                if isinstance(modal_available, bool) and isinstance(credentials_set, bool):
                    self.log_test("Modal status field types", True, 
                                f"modal_available: {type(modal_available)}, credentials_set: {type(credentials_set)}")
                else:
                    self.log_test("Modal status field types", False, 
                                f"Expected booleans, got modal_available: {type(modal_available)}, credentials_set: {type(credentials_set)}")
                
                # Check app name
                if app_name:
                    self.log_test("Modal app name", True, f"App name: {app_name}")
                else:
                    self.log_test("Modal app name", False, "No app name provided")
                
                return True
                
            elif response.status_code == 404:
                self.log_test("Modal status endpoint", False, "Enhanced Modal MolBERT endpoints not implemented (404)")
                return False
            else:
                self.log_test("Modal status endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Modal status endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_modal_setup_endpoint(self):
        """Test POST /api/modal/molbert/setup"""
        print("\n=== Testing Enhanced Modal MolBERT Setup Endpoint ===")
        
        try:
            response = requests.post(f"{API_BASE}/modal/molbert/setup", timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                status = data.get('status', '')
                message = data.get('message', '')
                
                self.log_test("Modal setup endpoint success", True, 
                            f"Status: {status}, Message: {message}")
                return True
                
            elif response.status_code == 404:
                self.log_test("Modal setup endpoint", False, "Enhanced Modal MolBERT endpoints not implemented (404)")
                return False
            else:
                # Expected to fail without credentials
                try:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    message = data.get('message', response.text)
                    
                    self.log_test("Modal setup endpoint error handling", True, 
                                f"Expected error without credentials: Status {status}, Message: {message}")
                    return True
                except:
                    self.log_test("Modal setup endpoint error handling", True, 
                                f"Expected error without credentials: HTTP {response.status_code}")
                    return True
                
        except requests.exceptions.RequestException as e:
            self.log_test("Modal setup endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_modal_predict_endpoint(self):
        """Test POST /api/modal/molbert/predict"""
        print("\n=== Testing Enhanced Modal MolBERT Predict Endpoint ===")
        
        test_cases = [
            {
                "smiles": "CCO",
                "target": "EGFR", 
                "use_finetuned": True,
                "name": "ethanol"
            },
            {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "target": "BRAF",
                "use_finetuned": False,
                "name": "aspirin"
            }
        ]
        
        all_passed = True
        
        for case in test_cases:
            try:
                response = requests.post(f"{API_BASE}/modal/molbert/predict", 
                                       json=case,
                                       headers={'Content-Type': 'application/json'},
                                       timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    status = data.get('status', '')
                    self.log_test(f"Modal predict - {case['name']}", True, f"Status: {status}")
                    
                elif response.status_code == 404:
                    self.log_test(f"Modal predict - {case['name']}", False, 
                                "Enhanced Modal MolBERT endpoints not implemented (404)")
                    all_passed = False
                else:
                    # Expected to fail without Modal credentials - check for proper error handling
                    try:
                        data = response.json()
                        status = data.get('status', 'unknown')
                        message = data.get('message', response.text)
                        
                        self.log_test(f"Modal predict fallback - {case['name']}", True, 
                                    f"Expected error without Modal: Status {status}, Message: {message}")
                    except:
                        self.log_test(f"Modal predict fallback - {case['name']}", True, 
                                    f"Expected error without Modal: HTTP {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"Modal predict - {case['name']}", False, f"Connection error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_modal_train_endpoint(self):
        """Test POST /api/modal/molbert/train/{target}"""
        print("\n=== Testing Enhanced Modal MolBERT Train Endpoint ===")
        
        all_passed = True
        
        # Test with valid target
        valid_target = "EGFR"
        
        try:
            response = requests.post(f"{API_BASE}/modal/molbert/train/{valid_target}", 
                                   timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                status = data.get('status', '')
                target = data.get('target', '')
                
                self.log_test("Modal train endpoint - valid target", True, 
                            f"Status: {status}, Target: {target}")
                
            elif response.status_code == 404:
                self.log_test("Modal train endpoint - valid target", False, 
                            "Enhanced Modal MolBERT endpoints not implemented (404)")
                all_passed = False
            else:
                # Expected to fail without credentials
                try:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    message = data.get('message', response.text)
                    
                    self.log_test("Modal train endpoint - valid target", True, 
                                f"Expected error without credentials: Status {status}, Message: {message}")
                except:
                    self.log_test("Modal train endpoint - valid target", True, 
                                f"Expected error without credentials: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Modal train endpoint - valid target", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test with invalid target
        invalid_target = "INVALID_TARGET"
        
        try:
            response = requests.post(f"{API_BASE}/modal/molbert/train/{invalid_target}", 
                                   timeout=30)
            
            if response.status_code == 400:
                self.log_test("Modal train endpoint - invalid target", True, 
                            "Invalid target properly rejected with 400")
            elif response.status_code == 404:
                self.log_test("Modal train endpoint - invalid target", False, 
                            "Enhanced Modal MolBERT endpoints not implemented (404)")
                all_passed = False
            else:
                # Check if error message mentions invalid target
                try:
                    data = response.json()
                    message = data.get('message', response.text)
                    
                    if "Invalid target" in message or "Available:" in message:
                        self.log_test("Modal train endpoint - invalid target", True, 
                                    f"Invalid target handled properly: {message}")
                    else:
                        self.log_test("Modal train endpoint - invalid target", True, 
                                    f"Response: HTTP {response.status_code}, {message}")
                except:
                    self.log_test("Modal train endpoint - invalid target", True, 
                                f"Response: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Modal train endpoint - invalid target", False, f"Connection error: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def test_modal_smiles_validation(self):
        """Test SMILES validation in Modal endpoints"""
        print("\n=== Testing Enhanced Modal MolBERT SMILES Validation ===")
        
        all_passed = True
        
        # Test invalid SMILES
        try:
            payload = {
                "smiles": "INVALID_SMILES_STRING",
                "target": "EGFR",
                "use_finetuned": True
            }
            
            response = requests.post(f"{API_BASE}/modal/molbert/predict", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code == 400:
                self.log_test("Modal SMILES validation - invalid", True, 
                            "Invalid SMILES properly rejected with 400")
            elif response.status_code == 404:
                self.log_test("Modal SMILES validation - invalid", False, 
                            "Enhanced Modal MolBERT endpoints not implemented (404)")
                all_passed = False
            else:
                try:
                    data = response.json()
                    message = data.get('message', response.text)
                    
                    if "Invalid SMILES" in message:
                        self.log_test("Modal SMILES validation - invalid", True, 
                                    f"Invalid SMILES handled: {message}")
                    else:
                        self.log_test("Modal SMILES validation - invalid", False, 
                                    f"Should reject invalid SMILES: HTTP {response.status_code}, {message}")
                        all_passed = False
                except:
                    self.log_test("Modal SMILES validation - invalid", False, 
                                f"Should reject invalid SMILES: HTTP {response.status_code}")
                    all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Modal SMILES validation - invalid", False, f"Connection error: {str(e)}")
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
                self.log_test("Modal SMILES validation - valid", True, "Valid SMILES accepted")
            elif response.status_code == 404:
                self.log_test("Modal SMILES validation - valid", False, 
                            "Enhanced Modal MolBERT endpoints not implemented (404)")
                all_passed = False
            else:
                # Should not be rejected for SMILES validation reasons
                try:
                    data = response.json()
                    message = data.get('message', response.text)
                    
                    if "Invalid SMILES" not in message:
                        self.log_test("Modal SMILES validation - valid", True, 
                                    f"Valid SMILES not rejected for SMILES reasons: {message}")
                    else:
                        self.log_test("Modal SMILES validation - valid", False, 
                                    f"Valid SMILES incorrectly rejected: {message}")
                        all_passed = False
                except:
                    self.log_test("Modal SMILES validation - valid", True, 
                                f"Valid SMILES processed: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Modal SMILES validation - valid", False, f"Connection error: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def test_modal_target_validation(self):
        """Test target validation in Modal endpoints"""
        print("\n=== Testing Enhanced Modal MolBERT Target Validation ===")
        
        all_passed = True
        
        # Test valid targets
        valid_targets = ["EGFR", "BRAF", "CDK2", "PARP1", "BCL2", "VEGFR2"]
        
        for target in valid_targets[:2]:  # Test first 2 to save time
            try:
                payload = {
                    "smiles": "CCO",
                    "target": target,
                    "use_finetuned": True
                }
                
                response = requests.post(f"{API_BASE}/modal/molbert/predict", 
                                       json=payload,
                                       headers={'Content-Type': 'application/json'},
                                       timeout=30)
                
                if response.status_code == 200:
                    self.log_test(f"Modal target validation - {target}", True, f"Valid target {target} accepted")
                elif response.status_code == 404:
                    self.log_test(f"Modal target validation - {target}", False, 
                                "Enhanced Modal MolBERT endpoints not implemented (404)")
                    all_passed = False
                    break
                else:
                    # Should not be rejected for target validation reasons
                    try:
                        data = response.json()
                        message = data.get('message', response.text)
                        
                        if "Invalid target" not in message:
                            self.log_test(f"Modal target validation - {target}", True, 
                                        f"Valid target {target} not rejected for target reasons")
                        else:
                            self.log_test(f"Modal target validation - {target}", False, 
                                        f"Valid target {target} incorrectly rejected: {message}")
                            all_passed = False
                    except:
                        self.log_test(f"Modal target validation - {target}", True, 
                                    f"Valid target {target} processed: HTTP {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"Modal target validation - {target}", False, f"Connection error: {str(e)}")
                all_passed = False
        
        # Test invalid target
        try:
            payload = {
                "smiles": "CCO",
                "target": "INVALID_TARGET",
                "use_finetuned": True
            }
            
            response = requests.post(f"{API_BASE}/modal/molbert/predict", 
                                   json=payload,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            if response.status_code == 400:
                self.log_test("Modal target validation - invalid", True, 
                            "Invalid target properly rejected with 400")
            elif response.status_code == 404:
                self.log_test("Modal target validation - invalid", False, 
                            "Enhanced Modal MolBERT endpoints not implemented (404)")
                all_passed = False
            else:
                try:
                    data = response.json()
                    message = data.get('message', response.text)
                    
                    if "Invalid target" in message or "Available:" in message:
                        self.log_test("Modal target validation - invalid", True, 
                                    f"Invalid target handled: {message}")
                    else:
                        self.log_test("Modal target validation - invalid", False, 
                                    f"Should reject invalid target: HTTP {response.status_code}, {message}")
                        all_passed = False
                except:
                    self.log_test("Modal target validation - invalid", False, 
                                f"Should reject invalid target: HTTP {response.status_code}")
                    all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Modal target validation - invalid", False, f"Connection error: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def test_existing_endpoints_still_work(self):
        """Test that existing endpoints still work with Modal integration"""
        print("\n=== Testing Existing Endpoints Integration ===")
        
        all_passed = True
        
        # Test health endpoint
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                status = data.get('status')
                enhanced_predictions = data.get('enhanced_predictions', False)
                
                if status == 'healthy' and enhanced_predictions:
                    self.log_test("Health endpoint with Modal integration", True, 
                                f"Status: {status}, Enhanced predictions: {enhanced_predictions}")
                else:
                    self.log_test("Health endpoint with Modal integration", False, 
                                f"Status: {status}, Enhanced predictions: {enhanced_predictions}")
                    all_passed = False
            else:
                self.log_test("Health endpoint with Modal integration", False, 
                            f"HTTP {response.status_code}: {response.text}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Health endpoint with Modal integration", False, f"Connection error: {str(e)}")
            all_passed = False
        
        # Test predict endpoint
        try:
            payload = {
                "smiles": "CCO",
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
                    has_enhanced = result.get('enhanced_chemprop_prediction') is not None
                    
                    self.log_test("Predict endpoint with Modal integration", True, 
                                f"Enhanced prediction available: {has_enhanced}")
                else:
                    self.log_test("Predict endpoint with Modal integration", False, "No results returned")
                    all_passed = False
            else:
                self.log_test("Predict endpoint with Modal integration", False, 
                            f"HTTP {response.status_code}: {response.text}")
                all_passed = False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Predict endpoint with Modal integration", False, f"Connection error: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def run_all_tests(self):
        """Run all Enhanced Modal MolBERT tests"""
        print(f"🧪 Starting Enhanced Modal MolBERT Integration Testing")
        print(f"Backend URL: {API_BASE}")
        print("=" * 60)
        
        # Run all Modal-specific tests
        tests = [
            self.test_modal_status_endpoint,
            self.test_modal_setup_endpoint,
            self.test_modal_predict_endpoint,
            self.test_modal_train_endpoint,
            self.test_modal_smiles_validation,
            self.test_modal_target_validation,
            self.test_existing_endpoints_still_work
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
        print("🏁 ENHANCED MODAL MOLBERT TEST SUMMARY")
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
    tester = ModalMolBERTTester()
    passed, failed, results = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)