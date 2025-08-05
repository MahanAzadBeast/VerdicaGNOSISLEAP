#!/usr/bin/env python3
"""
Enhanced Model 2 (Cytotoxicity Prediction) Testing
Focus on testing the improved Model 2 with enhanced training approach
"""

import requests
import json
import sys
import os

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://7088916a-6fc8-4483-97c1-34b013f600a8.preview.emergentagent.com')
API_BASE = f"{BACKEND_URL}/api"

class EnhancedModel2Tester:
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
    
    def log_test(self, test_name: str, success: bool, details: str):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        print(f"   Details: {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'status': status
        })
        
        if not success:
            self.failed_tests.append({
                'test': test_name,
                'details': details,
                'status': status
            })
    
    def test_enhanced_model2_info(self):
        """Test Enhanced Model 2 Info Endpoint - should show improved metrics"""
        print("\n=== Testing Enhanced Model 2 Info Endpoint ===")
        
        try:
            response = requests.get(f"{API_BASE}/model2/info", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for enhanced model indicators
                model_name = data.get('model_name', '')
                model_loaded = data.get('model_loaded', False)
                version = data.get('version', '')
                
                # Look for enhanced/improved indicators
                enhanced_indicators = [
                    'Enhanced' in model_name or 'Fixed' in model_name,
                    'Enhanced' in version or 'Fixed' in version,
                    model_loaded
                ]
                
                self.log_test("Enhanced Model 2 Info Endpoint", all(enhanced_indicators),
                            f"Model: {model_name}, Version: {version}, Loaded: {model_loaded}")
                
                # Check performance metrics
                performance = data.get('performance', {})
                if performance:
                    validation_r2 = performance.get('validation_r2', 'Unknown')
                    target_r2 = performance.get('target_r2', 'Unknown')
                    
                    # Enhanced model should show improved R² values
                    improved_metrics = validation_r2 != 'Unknown' and target_r2 != 'Unknown'
                    
                    self.log_test("Enhanced Model 2 Performance Metrics", improved_metrics,
                                f"Validation R²: {validation_r2}, Target R²: {target_r2}")
                
                # Check enhanced features
                features = data.get('features', {})
                if features:
                    molecular = features.get('molecular', '')
                    genomic = features.get('genomic', '')
                    
                    enhanced_features = 'ChemBERTa' in molecular or 'RDKit' in molecular
                    
                    self.log_test("Enhanced Model 2 Features", enhanced_features,
                                f"Molecular: {molecular}, Genomic: {genomic}")
                
                return True
                
            else:
                self.log_test("Enhanced Model 2 Info Endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Enhanced Model 2 Info Endpoint", False, f"Connection error: {str(e)}")
            return False
    
    def test_enhanced_model2_aspirin_prediction(self):
        """Test Enhanced Model 2 with Aspirin - should show improved quality/realism"""
        print("\n=== Testing Enhanced Model 2 Aspirin Predictions ===")
        
        aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        
        try:
            payload = {
                "smiles": aspirin_smiles,
                "cell_lines": ["A549", "MCF7", "HCT116"]
            }
            
            response = requests.post(f"{API_BASE}/model2/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                if 'predictions' not in data:
                    self.log_test("Enhanced Model 2 Aspirin Structure", False, "No predictions field")
                    return False
                
                predictions = data.get('predictions', {})
                realistic_predictions = 0
                
                for cell_line in ["A549", "MCF7", "HCT116"]:
                    if cell_line in predictions:
                        pred = predictions[cell_line]
                        
                        ic50_uM = pred.get('ic50_uM')
                        confidence = pred.get('confidence')
                        
                        # Enhanced model should have realistic values
                        valid_ic50 = isinstance(ic50_uM, (int, float)) and ic50_uM > 0
                        valid_confidence = isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
                        
                        # Enhanced model should have more realistic IC50 values
                        realistic_ic50 = 0.01 <= ic50_uM <= 100.0 if valid_ic50 else False
                        realistic_confidence = 0.3 <= confidence <= 0.95 if valid_confidence else False
                        
                        if realistic_ic50 and realistic_confidence:
                            realistic_predictions += 1
                        
                        self.log_test(f"Enhanced Model 2 {cell_line} Aspirin", 
                                    valid_ic50 and valid_confidence,
                                    f"IC50: {ic50_uM} μM, Confidence: {confidence}, Realistic: {realistic_ic50 and realistic_confidence}")
                
                # Check model info in response
                model_info = data.get('model_info', {})
                if model_info:
                    model_version = model_info.get('model_version', '')
                    validation_r2 = model_info.get('validation_r2', 'Unknown')
                    
                    enhanced_version = 'Enhanced' in model_version or 'Fixed' in model_version
                    
                    self.log_test("Enhanced Model 2 Aspirin Model Info", enhanced_version,
                                f"Version: {model_version}, R²: {validation_r2}")
                
                self.log_test("Enhanced Model 2 Aspirin Overall Quality", realistic_predictions >= 2,
                            f"Realistic predictions: {realistic_predictions}/3")
                
                return True
                
            else:
                self.log_test("Enhanced Model 2 Aspirin Prediction", False, 
                            f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Enhanced Model 2 Aspirin Prediction", False, f"Connection error: {str(e)}")
            return False
    
    def test_enhanced_model2_imatinib_prediction(self):
        """Test Enhanced Model 2 with Imatinib - should show improved quality/realism"""
        print("\n=== Testing Enhanced Model 2 Imatinib Predictions ===")
        
        imatinib_smiles = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
        
        try:
            payload = {
                "smiles": imatinib_smiles,
                "cell_lines": ["A549", "MCF7", "HCT116"]
            }
            
            response = requests.post(f"{API_BASE}/model2/predict", 
                                   json=payload, 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                predictions = data.get('predictions', {})
                realistic_predictions = 0
                
                for cell_line in ["A549", "MCF7", "HCT116"]:
                    if cell_line in predictions:
                        pred = predictions[cell_line]
                        
                        ic50_uM = pred.get('ic50_uM')
                        confidence = pred.get('confidence')
                        
                        valid_ic50 = isinstance(ic50_uM, (int, float)) and ic50_uM > 0
                        valid_confidence = isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
                        
                        # Enhanced model should have realistic values for imatinib
                        realistic_ic50 = 0.01 <= ic50_uM <= 100.0 if valid_ic50 else False
                        realistic_confidence = 0.3 <= confidence <= 0.95 if valid_confidence else False
                        
                        if realistic_ic50 and realistic_confidence:
                            realistic_predictions += 1
                        
                        self.log_test(f"Enhanced Model 2 {cell_line} Imatinib", 
                                    valid_ic50 and valid_confidence,
                                    f"IC50: {ic50_uM} μM, Confidence: {confidence}")
                
                # Check SMILES echo
                compound_info = data.get('compound_info', {})
                smiles_echo = compound_info.get('smiles', '')
                
                self.log_test("Enhanced Model 2 Imatinib SMILES Echo", smiles_echo == imatinib_smiles,
                            f"SMILES echoed correctly: {smiles_echo == imatinib_smiles}")
                
                self.log_test("Enhanced Model 2 Imatinib Overall Quality", realistic_predictions >= 2,
                            f"Realistic predictions: {realistic_predictions}/3")
                
                return True
                
            else:
                self.log_test("Enhanced Model 2 Imatinib Prediction", False, 
                            f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Enhanced Model 2 Imatinib Prediction", False, f"Connection error: {str(e)}")
            return False
    
    def test_enhanced_model2_confidence_scoring(self):
        """Test Enhanced Model 2 confidence scoring is realistic"""
        print("\n=== Testing Enhanced Model 2 Confidence Scoring ===")
        
        test_molecules = [
            ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
            ("Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C", "imatinib")
        ]
        
        confidence_scores = []
        all_passed = True
        
        for smiles, name in test_molecules:
            try:
                payload = {
                    "smiles": smiles,
                    "cell_lines": ["A549"]
                }
                
                response = requests.post(f"{API_BASE}/model2/predict", 
                                       json=payload, 
                                       headers={'Content-Type': 'application/json'},
                                       timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('predictions', {})
                    
                    if 'A549' in predictions:
                        pred = predictions['A549']
                        confidence = pred.get('confidence')
                        
                        if isinstance(confidence, (int, float)):
                            confidence_scores.append(confidence)
                            
                            # Enhanced model should have realistic confidence scores
                            realistic_confidence = 0.3 <= confidence <= 0.95
                            
                            self.log_test(f"Enhanced Model 2 {name} Confidence", realistic_confidence,
                                        f"Confidence: {confidence} (realistic: {realistic_confidence})")
                            
                            if not realistic_confidence:
                                all_passed = False
                        else:
                            self.log_test(f"Enhanced Model 2 {name} Confidence", False, 
                                        f"Invalid confidence: {confidence}")
                            all_passed = False
                    else:
                        self.log_test(f"Enhanced Model 2 {name} Confidence", False, "No A549 prediction")
                        all_passed = False
                else:
                    self.log_test(f"Enhanced Model 2 {name} Confidence", False, 
                                f"HTTP {response.status_code}")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.log_test(f"Enhanced Model 2 {name} Confidence", False, f"Connection error: {str(e)}")
                all_passed = False
        
        # Check confidence distribution
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            confidence_range = max(confidence_scores) - min(confidence_scores)
            
            # Enhanced model should have reasonable confidence distribution
            reasonable_avg = 0.4 <= avg_confidence <= 0.9
            reasonable_range = confidence_range <= 0.6
            
            self.log_test("Enhanced Model 2 Confidence Distribution", reasonable_avg and reasonable_range,
                        f"Avg: {avg_confidence:.3f}, Range: {confidence_range:.3f}")
        
        return all_passed
    
    def test_enhanced_model2_performance_comparison(self):
        """Test that Enhanced Model 2 shows improved performance indicators"""
        print("\n=== Testing Enhanced Model 2 Performance Comparison ===")
        
        try:
            # Test info endpoint for performance metrics
            response = requests.get(f"{API_BASE}/model2/info", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for performance improvements
                performance = data.get('performance', {})
                improvements = data.get('improvements', [])
                
                # Look for R² improvements
                validation_r2 = performance.get('validation_r2', 'Unknown')
                target_r2 = performance.get('target_r2', 'Unknown')
                
                # Enhanced model should show R² > 0.3 (as mentioned in review)
                improved_r2 = False
                if validation_r2 != 'Unknown':
                    try:
                        r2_value = float(validation_r2)
                        improved_r2 = r2_value > 0.3
                    except:
                        pass
                
                self.log_test("Enhanced Model 2 R² Improvement", improved_r2,
                            f"Validation R²: {validation_r2} (should be > 0.3)")
                
                # Check for improvement indicators
                has_improvements = len(improvements) > 0
                
                self.log_test("Enhanced Model 2 Improvements Listed", has_improvements,
                            f"Improvements: {len(improvements)} items")
                
                # Check model path indicator
                model_path = data.get('model_path', '')
                enhanced_path = 'enhanced' in model_path.lower()
                
                self.log_test("Enhanced Model 2 Path Indicator", enhanced_path,
                            f"Model path: {model_path}")
                
                return improved_r2 or has_improvements or enhanced_path
                
            else:
                self.log_test("Enhanced Model 2 Performance Comparison", False, 
                            f"HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Enhanced Model 2 Performance Comparison", False, f"Connection error: {str(e)}")
            return False
    
    def run_enhanced_model2_tests(self):
        """Run all Enhanced Model 2 tests"""
        print("🎯 ENHANCED MODEL 2 (CYTOTOXICITY PREDICTION) TESTING")
        print("=" * 80)
        print("Testing improved Model 2 with enhanced training approach:")
        print("- Enhanced Model Architecture with better feature processing")
        print("- Better Molecular Features (20 RDKit descriptors)")
        print("- Realistic Genomic Features (30 features)")
        print("- New Training: R² = 0.42 (Random Forest), R² = 0.33 (neural network)")
        print("- Enhanced Model Path: model2_enhanced_v1.pth")
        print("=" * 80)
        
        # Run Enhanced Model 2 tests
        tests = [
            ("Enhanced Model 2 Info", self.test_enhanced_model2_info),
            ("Enhanced Model 2 Aspirin Prediction", self.test_enhanced_model2_aspirin_prediction),
            ("Enhanced Model 2 Imatinib Prediction", self.test_enhanced_model2_imatinib_prediction),
            ("Enhanced Model 2 Confidence Scoring", self.test_enhanced_model2_confidence_scoring),
            ("Enhanced Model 2 Performance Comparison", self.test_enhanced_model2_performance_comparison),
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                self.log_test(test_name, False, f"Test execution error: {str(e)}")
        
        # Summary
        print("\n" + "=" * 80)
        print("🎯 ENHANCED MODEL 2 TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS ({len(self.failed_tests)}):")
            for failed_test in self.failed_tests:
                print(f"  - {failed_test['test']}: {failed_test['details']}")
        
        print("\n🔍 KEY ENHANCED MODEL 2 FINDINGS:")
        
        # Check for enhanced model indicators
        enhanced_indicators = []
        for result in self.test_results:
            if result['success'] and ('Enhanced' in result['details'] or 'improved' in result['details'].lower()):
                enhanced_indicators.append(result['test'])
        
        if enhanced_indicators:
            print("✅ Enhanced Model 2 improvements detected:")
            for indicator in enhanced_indicators:
                print(f"  - {indicator}")
        
        # Check for performance improvements
        performance_improvements = []
        for result in self.test_results:
            if 'R²' in result['details'] or 'performance' in result['details'].lower():
                performance_improvements.append(result['details'])
        
        if performance_improvements:
            print("📊 Performance metrics found:")
            for perf in performance_improvements:
                print(f"  - {perf}")
        
        print("\n" + "=" * 80)
        print("🎯 ENHANCED MODEL 2 TESTING COMPLETE")
        print("=" * 80)
        
        return success_rate >= 70

if __name__ == "__main__":
    tester = EnhancedModel2Tester()
    success = tester.run_enhanced_model2_tests()
    sys.exit(0 if success else 1)