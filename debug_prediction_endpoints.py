#!/usr/bin/env python3
"""
Debug Prediction API Endpoints for Model Comparison
Focus on ChemBERTa and Chemprop Real prediction endpoints as requested
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

class PredictionEndpointDebugger:
    def __init__(self):
        self.test_results = []
        self.aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin as specified
        
    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        result = {
            'test': test_name,
            'status': status,
            'success': success,
            'details': details
        }
        self.test_results.append(result)
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")
    
    def debug_chemberta_predict_endpoint(self):
        """Debug ChemBERTa prediction endpoint: POST /api/chemberta/predict"""
        print("\n=== Debugging ChemBERTa Prediction Endpoint ===")
        
        try:
            payload = {"smiles": self.aspirin_smiles}
            
            print(f"Testing: POST {API_BASE}/chemberta/predict")
            print(f"Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(f"{API_BASE}/chemberta/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            print(f"Response Status: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"Response JSON Structure:")
                    print(json.dumps(data, indent=2))
                    
                    # Check response structure
                    if 'status' in data:
                        status = data.get('status')
                        self.log_result("ChemBERTa Predict - Status Field", True, f"Status: {status}")
                        
                        if status == "success":
                            predictions = data.get('predictions', {})
                            self.log_result("ChemBERTa Predict - Success Response", True, 
                                          f"Predictions count: {len(predictions)}")
                            
                            # Check prediction structure
                            if predictions:
                                print("Prediction Structure Analysis:")
                                for target, pred_data in predictions.items():
                                    print(f"  Target: {target}")
                                    if isinstance(pred_data, dict):
                                        print(f"    Keys: {list(pred_data.keys())}")
                                        if 'ic50_nm' in pred_data:
                                            print(f"    IC50: {pred_data['ic50_nm']} nM")
                                        if 'confidence' in pred_data:
                                            print(f"    Confidence: {pred_data['confidence']}")
                                        if 'activity_class' in pred_data:
                                            print(f"    Activity: {pred_data['activity_class']}")
                                    else:
                                        print(f"    Value: {pred_data}")
                                        
                                self.log_result("ChemBERTa Predict - Prediction Data", True, 
                                              f"Valid prediction structure with {len(predictions)} targets")
                            else:
                                self.log_result("ChemBERTa Predict - Prediction Data", False, 
                                              "No predictions in successful response")
                        else:
                            self.log_result("ChemBERTa Predict - Success Response", False, 
                                          f"Status not success: {status}")
                            if 'message' in data:
                                print(f"Error message: {data['message']}")
                    else:
                        self.log_result("ChemBERTa Predict - Response Structure", False, 
                                      "No status field in response")
                        
                except json.JSONDecodeError as e:
                    print(f"JSON Decode Error: {e}")
                    print(f"Raw Response: {response.text}")
                    self.log_result("ChemBERTa Predict - JSON Response", False, 
                                  f"Invalid JSON: {str(e)}")
                    
            else:
                print(f"Error Response: {response.text}")
                self.log_result("ChemBERTa Predict - HTTP Status", False, 
                              f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request Exception: {e}")
            self.log_result("ChemBERTa Predict - Connection", False, f"Request error: {str(e)}")
    
    def debug_chemprop_real_predict_endpoint(self):
        """Debug Chemprop Real prediction endpoint: POST /api/chemprop-real/predict"""
        print("\n=== Debugging Chemprop Real Prediction Endpoint ===")
        
        try:
            payload = {"smiles": self.aspirin_smiles}
            
            print(f"Testing: POST {API_BASE}/chemprop-real/predict")
            print(f"Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(f"{API_BASE}/chemprop-real/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            print(f"Response Status: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"Response JSON Structure:")
                    print(json.dumps(data, indent=2))
                    
                    # Check response structure
                    if 'status' in data:
                        status = data.get('status')
                        self.log_result("Chemprop Real Predict - Status Field", True, f"Status: {status}")
                        
                        if status == "success":
                            predictions = data.get('predictions', {})
                            self.log_result("Chemprop Real Predict - Success Response", True, 
                                          f"Predictions count: {len(predictions)}")
                            
                            # Check prediction structure
                            if predictions:
                                print("Prediction Structure Analysis:")
                                for prop_type, pred_value in predictions.items():
                                    print(f"  Property: {prop_type}")
                                    print(f"    Value: {pred_value}")
                                    
                                self.log_result("Chemprop Real Predict - Prediction Data", True, 
                                              f"Valid prediction structure with {len(predictions)} properties")
                            else:
                                self.log_result("Chemprop Real Predict - Prediction Data", False, 
                                              "No predictions in successful response")
                        else:
                            self.log_result("Chemprop Real Predict - Success Response", False, 
                                          f"Status not success: {status}")
                            if 'message' in data:
                                print(f"Error message: {data['message']}")
                    else:
                        self.log_result("Chemprop Real Predict - Response Structure", False, 
                                      "No status field in response")
                        
                except json.JSONDecodeError as e:
                    print(f"JSON Decode Error: {e}")
                    print(f"Raw Response: {response.text}")
                    self.log_result("Chemprop Real Predict - JSON Response", False, 
                                  f"Invalid JSON: {str(e)}")
                    
            elif response.status_code == 503:
                # Service unavailable - expected if model not loaded
                try:
                    data = response.json()
                    print(f"Service Unavailable Response:")
                    print(json.dumps(data, indent=2))
                    self.log_result("Chemprop Real Predict - Service Unavailable", True, 
                                  f"Expected 503 when model unavailable: {data.get('message', 'No message')}")
                except:
                    print(f"503 Response: {response.text}")
                    self.log_result("Chemprop Real Predict - Service Unavailable", True, 
                                  f"Expected 503 when model unavailable")
            else:
                print(f"Error Response: {response.text}")
                self.log_result("Chemprop Real Predict - HTTP Status", False, 
                              f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request Exception: {e}")
            self.log_result("Chemprop Real Predict - Connection", False, f"Request error: {str(e)}")
    
    def check_endpoint_availability(self):
        """Check if the endpoints are available"""
        print("\n=== Checking Endpoint Availability ===")
        
        # Check ChemBERTa status
        try:
            response = requests.get(f"{API_BASE}/chemberta/status", timeout=30)
            print(f"ChemBERTa Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"ChemBERTa Status Response: {json.dumps(data, indent=2)}")
                available = data.get('available', False)
                self.log_result("ChemBERTa Status Endpoint", True, f"Available: {available}")
            else:
                print(f"ChemBERTa Status Error: {response.text}")
                self.log_result("ChemBERTa Status Endpoint", False, f"HTTP {response.status_code}")
        except Exception as e:
            print(f"ChemBERTa Status Exception: {e}")
            self.log_result("ChemBERTa Status Endpoint", False, f"Error: {str(e)}")
        
        # Check Chemprop Real status
        try:
            response = requests.get(f"{API_BASE}/chemprop-real/status", timeout=30)
            print(f"Chemprop Real Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Chemprop Real Status Response: {json.dumps(data, indent=2)}")
                available = data.get('available', False)
                self.log_result("Chemprop Real Status Endpoint", True, f"Available: {available}")
            else:
                print(f"Chemprop Real Status Error: {response.text}")
                self.log_result("Chemprop Real Status Endpoint", False, f"HTTP {response.status_code}")
        except Exception as e:
            print(f"Chemprop Real Status Exception: {e}")
            self.log_result("Chemprop Real Status Endpoint", False, f"Error: {str(e)}")
    
    def check_health_endpoint_ai_modules(self):
        """Check health endpoint for AI modules status"""
        print("\n=== Checking Health Endpoint AI Modules ===")
        
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            print(f"Health Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Health Response: {json.dumps(data, indent=2)}")
                
                # Check AI modules section
                ai_modules = data.get('ai_modules', {})
                if ai_modules:
                    chemberta_available = ai_modules.get('chemberta_available', False)
                    real_chemprop_available = ai_modules.get('real_chemprop_available', False)
                    
                    self.log_result("Health - ChemBERTa Available", chemberta_available, 
                                  f"ChemBERTa available: {chemberta_available}")
                    self.log_result("Health - Real Chemprop Available", real_chemprop_available, 
                                  f"Real Chemprop available: {real_chemprop_available}")
                else:
                    self.log_result("Health - AI Modules Section", False, "No ai_modules section in health response")
                
                # Check models_loaded section
                models_loaded = data.get('models_loaded', {})
                if models_loaded:
                    real_trained_chemprop = models_loaded.get('real_trained_chemprop', False)
                    self.log_result("Health - Real Trained Chemprop", real_trained_chemprop, 
                                  f"Real trained Chemprop: {real_trained_chemprop}")
                else:
                    self.log_result("Health - Models Loaded Section", False, "No models_loaded section")
                    
            else:
                print(f"Health Error: {response.text}")
                self.log_result("Health Endpoint", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Health Exception: {e}")
            self.log_result("Health Endpoint", False, f"Error: {str(e)}")
    
    def analyze_response_format_mismatch(self):
        """Analyze if response formats match what UI expects"""
        print("\n=== Analyzing Response Format for UI Compatibility ===")
        
        # Based on the review request, the UI is showing "N/A" values
        # This suggests the response structure might not match what the frontend expects
        
        print("Expected UI data structure analysis:")
        print("- UI likely expects specific field names")
        print("- UI might expect numeric values in specific formats")
        print("- UI might expect confidence scores in specific ranges")
        print("- UI might expect error handling in specific format")
        
        # Test both endpoints and analyze structure
        endpoints_to_test = [
            ("ChemBERTa", f"{API_BASE}/chemberta/predict"),
            ("Chemprop Real", f"{API_BASE}/chemprop-real/predict")
        ]
        
        for endpoint_name, endpoint_url in endpoints_to_test:
            print(f"\n--- Analyzing {endpoint_name} Response Format ---")
            
            try:
                payload = {"smiles": self.aspirin_smiles}
                response = requests.post(endpoint_url, 
                                       json=payload, 
                                       headers={'Content-Type': 'application/json'},
                                       timeout=60)
                
                print(f"Status Code: {response.status_code}")
                
                if response.status_code in [200, 503]:  # 503 is acceptable for unavailable models
                    try:
                        data = response.json()
                        
                        # Analyze response structure
                        print(f"Response Keys: {list(data.keys())}")
                        
                        if 'status' in data:
                            print(f"Status: {data['status']}")
                        
                        if 'predictions' in data:
                            predictions = data['predictions']
                            print(f"Predictions Type: {type(predictions)}")
                            print(f"Predictions Content: {predictions}")
                            
                            # Check if predictions have expected structure for UI
                            if isinstance(predictions, dict):
                                for key, value in predictions.items():
                                    print(f"  {key}: {value} (type: {type(value)})")
                                    
                                    # Check if values are numeric (not None or "N/A")
                                    if value is None:
                                        print(f"    WARNING: {key} is None - UI will show N/A")
                                    elif isinstance(value, str) and value.lower() in ['n/a', 'na', 'null']:
                                        print(f"    WARNING: {key} is string N/A - UI issue")
                                    elif isinstance(value, (int, float)):
                                        print(f"    OK: {key} is numeric")
                                    elif isinstance(value, dict):
                                        print(f"    COMPLEX: {key} is dict with keys: {list(value.keys())}")
                        
                        if 'message' in data:
                            print(f"Message: {data['message']}")
                        
                        if 'error' in data:
                            print(f"Error: {data['error']}")
                            
                        # Check for common UI-expected fields
                        ui_expected_fields = ['ic50', 'ic50_nm', 'confidence', 'activity_class', 'predictions']
                        for field in ui_expected_fields:
                            if field in data:
                                print(f"UI Expected Field '{field}': Present")
                            else:
                                print(f"UI Expected Field '{field}': Missing")
                        
                        self.log_result(f"{endpoint_name} Response Format Analysis", True, 
                                      f"Response analyzed - Status: {response.status_code}")
                        
                    except json.JSONDecodeError as e:
                        print(f"JSON Decode Error: {e}")
                        print(f"Raw Response: {response.text}")
                        self.log_result(f"{endpoint_name} Response Format Analysis", False, 
                                      f"Invalid JSON response")
                else:
                    print(f"Non-success status: {response.text}")
                    self.log_result(f"{endpoint_name} Response Format Analysis", False, 
                                  f"HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"Exception: {e}")
                self.log_result(f"{endpoint_name} Response Format Analysis", False, f"Error: {str(e)}")
    
    def run_debug_session(self):
        """Run complete debug session"""
        print("🔍 DEBUGGING PREDICTION API ENDPOINTS FOR MODEL COMPARISON")
        print("=" * 60)
        print(f"Backend URL: {BACKEND_URL}")
        print(f"API Base: {API_BASE}")
        print(f"Test SMILES: {self.aspirin_smiles} (aspirin)")
        print("=" * 60)
        
        # Run all debug tests
        self.check_health_endpoint_ai_modules()
        self.check_endpoint_availability()
        self.debug_chemberta_predict_endpoint()
        self.debug_chemprop_real_predict_endpoint()
        self.analyze_response_format_mismatch()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 DEBUG SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            print(f"{result['status']}: {result['test']}")
            if result['details']:
                print(f"   {result['details']}")
        
        print("\n🔍 KEY FINDINGS FOR UI N/A VALUES:")
        print("1. Check if prediction endpoints return None values")
        print("2. Check if response structure matches UI expectations")
        print("3. Check if models are actually available and loaded")
        print("4. Check if error responses are being handled properly")
        print("5. Verify field names match what UI is looking for")
        
        return self.test_results

if __name__ == "__main__":
    debugger = PredictionEndpointDebugger()
    results = debugger.run_debug_session()