#!/usr/bin/env python3
"""
Expanded Database Integration Backend Testing
Tests the fixed expanded router endpoints after router integration fix
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

class ExpandedDatabaseTester:
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
    
    def test_expanded_health_endpoint(self):
        """Test GET /api/expanded/health endpoint"""
        print("\n=== Testing Expanded Health Endpoint ===")
        
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
                
                # Check total targets (should be 23)
                total_targets = data.get('total_targets', 0)
                self.log_test("Total targets count", total_targets == 23, f"Total targets: {total_targets} (expected: 23)")
                
                # Check target categories (10+7+6)
                target_categories = data.get('target_categories', {})
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
                
                # Check activity types
                activity_types = data.get('activity_types', [])
                expected_activity_types = ["IC50", "EC50", "Ki", "Inhibition", "Activity"]
                has_all_activity_types = all(activity in activity_types for activity in expected_activity_types)
                
                self.log_test("Activity types", has_all_activity_types,
                            f"Activity types: {activity_types}")
                
                # Check data sources
                data_sources = data.get('data_sources', [])
                expected_data_sources = ["ChEMBL", "PubChem", "BindingDB", "DTC"]
                has_all_data_sources = all(source in data_sources for source in expected_data_sources)
                
                self.log_test("Data sources", has_all_data_sources,
                            f"Data sources: {data_sources}")
                
                return True
                
            else:
                self.log_test("Expanded health endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Expanded health endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_expanded_targets_endpoint(self):
        """Test GET /api/expanded/targets endpoint"""
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
                
                # Check total targets
                total_targets = data.get('total_targets', 0)
                self.log_test("Expanded targets count", total_targets == 23, f"Total targets: {total_targets}")
                
                # Check targets list
                targets = data.get('targets', [])
                if len(targets) != 23:
                    self.log_test("Expanded targets list length", False, f"Expected 23 targets, got {len(targets)}")
                    return False
                
                # Check target structure
                for target in targets[:3]:  # Check first 3 targets
                    required_target_fields = ['target', 'category', 'full_name', 'available_chemberta', 'available_chemprop']
                    missing_target_fields = [field for field in required_target_fields if field not in target]
                    
                    if missing_target_fields:
                        self.log_test(f"Target structure - {target.get('target', 'unknown')}", False, 
                                    f"Missing fields: {missing_target_fields}")
                        return False
                
                # Check categories
                by_category = data.get('by_category', {})
                expected_categories = ['oncoprotein', 'tumor_suppressor', 'metastasis_suppressor']
                has_all_categories = all(category in by_category for category in expected_categories)
                
                self.log_test("Target categories present", has_all_categories,
                            f"Categories: {list(by_category.keys())}")
                
                # Check category counts
                category_counts = {category: len(targets) for category, targets in by_category.items()}
                expected_counts = {'oncoprotein': 10, 'tumor_suppressor': 7, 'metastasis_suppressor': 6}
                
                counts_correct = True
                for category, expected_count in expected_counts.items():
                    actual_count = category_counts.get(category, 0)
                    if actual_count != expected_count:
                        counts_correct = False
                        break
                
                self.log_test("Category target counts", counts_correct,
                            f"Counts: {category_counts} (expected: {expected_counts})")
                
                return True
                
            else:
                self.log_test("Expanded targets endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Expanded targets endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_expanded_stats_performance_endpoint(self):
        """Test GET /api/expanded/stats/performance endpoint"""
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
                
                if chemberta_missing:
                    self.log_test("ChemBERTa stats structure", False, f"Missing fields: {chemberta_missing}")
                    return False
                
                # Check Chemprop stats
                chemprop_stats = data.get('chemprop', {})
                chemprop_required = ['overall_r2', 'category_performance', 'training_info']
                chemprop_missing = [field for field in chemprop_required if field not in chemprop_stats]
                
                if chemprop_missing:
                    self.log_test("Chemprop stats structure", False, f"Missing fields: {chemprop_missing}")
                    return False
                
                # Check comparison stats
                comparison = data.get('comparison', {})
                comparison_required = ['chemberta_better_targets', 'chemprop_better_targets', 'similar_targets']
                comparison_missing = [field for field in comparison_required if field not in comparison]
                
                if comparison_missing:
                    self.log_test("Comparison stats structure", False, f"Missing fields: {comparison_missing}")
                    return False
                
                # Check category performance for both models
                chemberta_categories = chemberta_stats.get('category_performance', {})
                chemprop_categories = chemprop_stats.get('category_performance', {})
                
                expected_categories = ['oncoprotein', 'tumor_suppressor', 'metastasis_suppressor']
                
                chemberta_has_categories = all(cat in chemberta_categories for cat in expected_categories)
                chemprop_has_categories = all(cat in chemprop_categories for cat in expected_categories)
                
                self.log_test("ChemBERTa category performance", chemberta_has_categories,
                            f"Categories: {list(chemberta_categories.keys())}")
                self.log_test("Chemprop category performance", chemprop_has_categories,
                            f"Categories: {list(chemprop_categories.keys())}")
                
                # Check R² values are reasonable
                chemberta_r2 = chemberta_stats.get('overall_r2', 0)
                chemprop_r2 = chemprop_stats.get('overall_r2', 0)
                
                r2_reasonable = 0.0 <= chemberta_r2 <= 1.0 and 0.0 <= chemprop_r2 <= 1.0
                self.log_test("R² values reasonable", r2_reasonable,
                            f"ChemBERTa R²: {chemberta_r2}, Chemprop R²: {chemprop_r2}")
                
                return True
                
            else:
                self.log_test("Expanded stats endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Expanded stats endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_main_health_expanded_integration(self):
        """Test that /api/health includes expanded_models_info"""
        print("\n=== Testing Main Health Expanded Integration ===")
        
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check expanded_models_info section
                expanded_models_info = data.get('expanded_models_info', {})
                
                if not expanded_models_info:
                    self.log_test("Expanded models info in health", False, "No expanded_models_info section")
                    return False
                
                # Check required fields in expanded_models_info
                required_fields = ['available', 'total_targets', 'target_categories', 'activity_types', 'data_sources']
                missing_fields = [field for field in required_fields if field not in expanded_models_info]
                
                if missing_fields:
                    self.log_test("Expanded models info structure", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check availability
                available = expanded_models_info.get('available', False)
                self.log_test("Expanded models available", available, f"Available: {available}")
                
                # Check total targets (should be 23)
                total_targets = expanded_models_info.get('total_targets', 0)
                self.log_test("Health expanded total targets", total_targets == 23, 
                            f"Total targets: {total_targets} (expected: 23)")
                
                # Check target categories (10+7+6)
                target_categories = expanded_models_info.get('target_categories', {})
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
                
                self.log_test("Health expanded categories", categories_correct,
                            f"Categories: {target_categories} (expected: {expected_categories})")
                
                # Check activity types
                activity_types = expanded_models_info.get('activity_types', [])
                expected_activity_types = ["IC50", "EC50", "Ki", "Inhibition", "Activity"]
                has_all_activity_types = all(activity in activity_types for activity in expected_activity_types)
                
                self.log_test("Health expanded activity types", has_all_activity_types,
                            f"Activity types: {activity_types}")
                
                # Check data sources
                data_sources = expanded_models_info.get('data_sources', [])
                expected_data_sources = ["ChEMBL", "PubChem", "BindingDB", "DTC"]
                has_all_data_sources = all(source in data_sources for source in expected_data_sources)
                
                self.log_test("Health expanded data sources", has_all_data_sources,
                            f"Data sources: {data_sources}")
                
                # Check models_loaded section for expanded_models
                models_loaded = data.get('models_loaded', {})
                expanded_models_loaded = models_loaded.get('expanded_models', False)
                
                self.log_test("Expanded models in models_loaded", 'expanded_models' in models_loaded,
                            f"Expanded models loaded: {expanded_models_loaded}")
                
                return True
                
            else:
                self.log_test("Main health expanded integration", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Main health expanded integration", False, f"Connection error: {str(e)}")
            return False
    
    def test_backend_service_status(self):
        """Test that backend loads without errors and all routers are properly included"""
        print("\n=== Testing Backend Service Status ===")
        
        try:
            # Test main health endpoint
            response = requests.get(f"{API_BASE}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check basic health
                status = data.get('status')
                if status != 'healthy':
                    self.log_test("Backend service health", False, f"Backend not healthy: {status}")
                    return False
                
                # Check that expanded models are included in models_loaded
                models_loaded = data.get('models_loaded', {})
                has_expanded = 'expanded_models' in models_loaded
                
                self.log_test("Backend expanded router integration", has_expanded,
                            f"Expanded models in models_loaded: {has_expanded}")
                
                # Check AI modules section
                ai_modules = data.get('ai_modules', {})
                expanded_available = ai_modules.get('expanded_models_available', False)
                
                self.log_test("Backend AI modules expanded", 'expanded_models_available' in ai_modules,
                            f"Expanded models available: {expanded_available}")
                
                return True
                
            else:
                self.log_test("Backend service status", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Backend service status", False, f"Connection error: {str(e)}")
            return False
    
    def test_existing_endpoints_still_work(self):
        """Test that fixing expanded router didn't break existing endpoints"""
        print("\n=== Testing Existing Endpoints Still Work ===")
        
        all_passed = True
        
        # Test main health endpoint
        try:
            response = requests.get(f"{API_BASE}/health", timeout=30)
            if response.status_code == 200:
                self.log_test("Main health endpoint", True, "Main health endpoint working")
            else:
                self.log_test("Main health endpoint", False, f"HTTP {response.status_code}")
                all_passed = False
        except Exception as e:
            self.log_test("Main health endpoint", False, f"Error: {str(e)}")
            all_passed = False
        
        # Test targets endpoint
        try:
            response = requests.get(f"{API_BASE}/targets", timeout=30)
            if response.status_code == 200:
                self.log_test("Main targets endpoint", True, "Main targets endpoint working")
            else:
                self.log_test("Main targets endpoint", False, f"HTTP {response.status_code}")
                all_passed = False
        except Exception as e:
            self.log_test("Main targets endpoint", False, f"Error: {str(e)}")
            all_passed = False
        
        # Test predict endpoint with aspirin
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
                if 'results' in data and len(data['results']) > 0:
                    self.log_test("Main predict endpoint", True, "Main predict endpoint working")
                else:
                    self.log_test("Main predict endpoint", False, "No results returned")
                    all_passed = False
            else:
                self.log_test("Main predict endpoint", False, f"HTTP {response.status_code}")
                all_passed = False
        except Exception as e:
            self.log_test("Main predict endpoint", False, f"Error: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def test_expanded_prediction_endpoints_availability(self):
        """Test that expanded prediction endpoints are accessible (even if models not available)"""
        print("\n=== Testing Expanded Prediction Endpoints Availability ===")
        
        all_passed = True
        
        # Test ChemBERTa prediction endpoint
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
            
            # Expect 503 (service unavailable) since models aren't deployed, not 404
            if response.status_code == 503:
                self.log_test("Expanded ChemBERTa predict endpoint", True, 
                            "Endpoint accessible (503 expected - model not deployed)")
            elif response.status_code == 404:
                self.log_test("Expanded ChemBERTa predict endpoint", False, 
                            "404 error - endpoint not found (router not integrated)")
                all_passed = False
            else:
                # Any other response is also acceptable (might be working)
                self.log_test("Expanded ChemBERTa predict endpoint", True, 
                            f"Endpoint accessible (HTTP {response.status_code})")
                
        except Exception as e:
            self.log_test("Expanded ChemBERTa predict endpoint", False, f"Error: {str(e)}")
            all_passed = False
        
        # Test Chemprop prediction endpoint
        try:
            payload = {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
                "targets": ["EGFR", "BRAF"],
                "activity_types": ["IC50"]
            }
            
            response = requests.post(f"{API_BASE}/expanded/predict/chemprop", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            # Expect 503 (service unavailable) since models aren't deployed, not 404
            if response.status_code == 503:
                self.log_test("Expanded Chemprop predict endpoint", True, 
                            "Endpoint accessible (503 expected - model not deployed)")
            elif response.status_code == 404:
                self.log_test("Expanded Chemprop predict endpoint", False, 
                            "404 error - endpoint not found (router not integrated)")
                all_passed = False
            else:
                # Any other response is also acceptable (might be working)
                self.log_test("Expanded Chemprop predict endpoint", True, 
                            f"Endpoint accessible (HTTP {response.status_code})")
                
        except Exception as e:
            self.log_test("Expanded Chemprop predict endpoint", False, f"Error: {str(e)}")
            all_passed = False
        
        # Test comparison endpoint
        try:
            payload = {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
                "targets": ["EGFR", "BRAF"],
                "activity_types": ["IC50"]
            }
            
            response = requests.post(f"{API_BASE}/expanded/predict/compare", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=30)
            
            # Any response other than 404 is acceptable
            if response.status_code == 404:
                self.log_test("Expanded compare endpoint", False, 
                            "404 error - endpoint not found (router not integrated)")
                all_passed = False
            else:
                self.log_test("Expanded compare endpoint", True, 
                            f"Endpoint accessible (HTTP {response.status_code})")
                
        except Exception as e:
            self.log_test("Expanded compare endpoint", False, f"Error: {str(e)}")
            all_passed = False
        
        return all_passed
    
    def run_all_tests(self):
        """Run all expanded database integration tests"""
        print("🧪 Starting Expanded Database Integration Tests")
        print(f"🌐 Testing against: {BACKEND_URL}")
        print("=" * 80)
        
        # Test the specific requirements from the review request
        test_methods = [
            self.test_expanded_health_endpoint,
            self.test_expanded_targets_endpoint, 
            self.test_expanded_stats_performance_endpoint,
            self.test_main_health_expanded_integration,
            self.test_backend_service_status,
            self.test_existing_endpoints_still_work,
            self.test_expanded_prediction_endpoints_availability
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.log_test(f"Test {test_method.__name__}", False, f"Test crashed: {str(e)}")
        
        # Summary
        print("\n" + "=" * 80)
        print("🏁 EXPANDED DATABASE INTEGRATION TEST SUMMARY")
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
    tester = ExpandedDatabaseTester()
    passed, failed, total = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)