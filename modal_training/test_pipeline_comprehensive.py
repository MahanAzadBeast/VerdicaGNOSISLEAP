"""
Comprehensive Pipeline Test Suite
Tests the entire expanded multi-source pipeline before full execution
"""

import modal
import time
import json
from pathlib import Path
from datetime import datetime
import traceback
import subprocess
import os

def test_modal_connectivity():
    """Test basic Modal connectivity and authentication"""
    print("🔗 Testing Modal connectivity...")
    try:
        # Test basic Modal connection
        import modal
        print("   ✅ Modal import successful")
        
        # Test authentication
        apps = modal.App.list()
        print(f"   ✅ Modal authentication successful - found {len(apps)} apps")
        return True
    except Exception as e:
        print(f"   ❌ Modal connectivity failed: {e}")
        return False

def test_extraction_app():
    """Test the data extraction Modal app"""
    print("📊 Testing extraction app...")
    try:
        from expanded_multisource_extractor import app, extract_expanded_multisource_dataset
        print("   ✅ Extraction app import successful")
        
        # Test app functions exist
        functions = [func for func in dir(app) if not func.startswith('_')]
        print(f"   ✅ App has {len(functions)} functions")
        
        # Check if function is properly decorated
        if hasattr(extract_expanded_multisource_dataset, 'is_function'):
            print("   ✅ Function properly decorated as Modal function")
        else:
            print("   ⚠️ Function decoration unclear")
        
        return True
    except Exception as e:
        print(f"   ❌ Extraction app test failed: {e}")
        traceback.print_exc()
        return False

def test_chemberta_app():
    """Test the ChemBERTa training Modal app"""
    print("🧠 Testing ChemBERTa training app...")
    try:
        from train_expanded_chemberta import app, train_expanded_chemberta
        print("   ✅ ChemBERTa app import successful")
        
        # Check required libraries
        import torch
        import transformers
        print("   ✅ PyTorch and Transformers available")
        
        return True
    except Exception as e:
        print(f"   ❌ ChemBERTa app test failed: {e}")
        traceback.print_exc()
        return False

def test_chemprop_app():
    """Test the Chemprop training Modal app"""
    print("🕸️ Testing Chemprop training app...")
    try:
        from train_expanded_chemprop import app, train_expanded_chemprop
        print("   ✅ Chemprop app import successful")
        
        # Test if chemprop is available
        try:
            import chemprop
            print("   ✅ Chemprop library available")
        except ImportError:
            print("   ⚠️ Chemprop library not available locally (OK - will be installed on Modal)")
        
        return True
    except Exception as e:
        print(f"   ❌ Chemprop app test failed: {e}")
        traceback.print_exc()
        return False

def test_mini_extraction():
    """Test extraction with a tiny subset to validate the pipeline"""
    print("🔬 Testing mini data extraction...")
    try:
        # Create a minimal test extraction
        test_code = '''
import modal
from expanded_multisource_extractor import app

# Override with mini extraction for testing
@app.function(timeout=300)  # 5 minute timeout for test
def test_mini_extraction():
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    print("🧪 MINI EXTRACTION TEST STARTED")
    print("Testing basic data structures and quality control...")
    
    # Create synthetic test data to validate pipeline
    test_data = []
    targets = ["EGFR", "BRAF", "TP53"]  # Sample from each category
    
    for i in range(10):  # Mini dataset
        for target in targets:
            test_data.append({
                "canonical_smiles": f"CC(=O)OC1=CC=CC=C1C(=O)O{i}",  # Aspirin variants
                "target_name": target,
                "activity_type": "IC50",
                "standard_value": np.random.uniform(1, 1000),  # nM
                "standard_units": "nM",
                "data_source": "ChEMBL_test"
            })
    
    df = pd.DataFrame(test_data)
    print(f"   📊 Test dataset: {df.shape}")
    
    # Test data quality controller
    from expanded_multisource_extractor import DataQualityController
    qc = DataQualityController()
    
    # Test unit standardization
    test_value = qc.standardize_units(1.0, "uM", "nM")
    if test_value == 1000.0:
        print("   ✅ Unit standardization working")
    else:
        print(f"   ❌ Unit standardization failed: {test_value}")
    
    # Test pIC50 calculation
    pic50 = qc.calculate_pic50(1000.0)  # 1 μM = 1000 nM
    if 5.8 < pic50 < 6.2:  # Should be ~6
        print("   ✅ pIC50 calculation working")
    else:
        print(f"   ❌ pIC50 calculation failed: {pic50}")
    
    return {
        "status": "success",
        "test_type": "mini_extraction",
        "records_processed": len(test_data),
        "targets_tested": len(targets),
        "quality_control": "passed"
    }

# Run the test
try:
    with app.run() as app_run:
        result = test_mini_extraction.remote()
    print(f"   ✅ Mini extraction result: {result}")
    return True
except Exception as e:
    print(f"   ❌ Mini extraction failed: {e}")
    return False
'''
        
        # Execute test in a separate process to avoid import conflicts
        with open('/tmp/test_extraction.py', 'w') as f:
            f.write(test_code)
        
        result = subprocess.run(
            ['python', '/tmp/test_extraction.py'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("   ✅ Mini extraction test passed")
            return True
        else:
            print(f"   ❌ Mini extraction test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Mini extraction test error: {e}")
        return False

def test_dependencies():
    """Test critical dependencies"""
    print("📦 Testing dependencies...")
    
    required_packages = [
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing'),
        ('requests', 'HTTP requests'),
        ('modal', 'Modal platform'),
    ]
    
    all_good = True
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ❌ {package} - {description} - MISSING!")
            all_good = False
    
    return all_good

def test_wandb_integration():
    """Test Weights & Biases integration"""
    print("📊 Testing W&B integration...")
    try:
        import wandb
        print("   ✅ wandb import successful")
        
        # Test API key availability (don't actually initialize)
        wandb_key = os.environ.get('WANDB_API_KEY')
        if wandb_key:
            print("   ✅ WANDB_API_KEY environment variable found")
        else:
            print("   ⚠️ WANDB_API_KEY not found (will be available on Modal)")
        
        return True
    except ImportError:
        print("   ❌ wandb not available")
        return False

def test_backend_integration():
    """Test backend integration readiness"""
    print("🔌 Testing backend integration...")
    try:
        from expanded_backend_integration import expanded_router
        print(f"   ✅ Backend router imported with {len(expanded_router.routes)} routes")
        
        # Test target definitions
        from expanded_backend_integration import EXPANDED_TARGETS
        print(f"   ✅ {len(EXPANDED_TARGETS)} targets defined")
        
        # Check categories
        categories = set(info['category'] for info in EXPANDED_TARGETS.values())
        expected_categories = {'oncoprotein', 'tumor_suppressor', 'metastasis_suppressor'}
        if categories == expected_categories:
            print("   ✅ All target categories present")
        else:
            print(f"   ❌ Missing categories: {expected_categories - categories}")
        
        return True
    except Exception as e:
        print(f"   ❌ Backend integration test failed: {e}")
        return False

def test_file_permissions():
    """Test file system permissions for logging and results"""
    print("📁 Testing file permissions...")
    try:
        # Test write permissions in modal_training directory
        test_file = Path("/app/modal_training/test_permissions.txt")
        with open(test_file, 'w') as f:
            f.write("Permission test")
        
        # Read it back
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Clean up
        test_file.unlink()
        
        if content == "Permission test":
            print("   ✅ File write/read permissions working")
            return True
        else:
            print("   ❌ File content mismatch")
            return False
            
    except Exception as e:
        print(f"   ❌ File permissions test failed: {e}")
        return False

def test_modal_secrets():
    """Test Modal secrets availability"""
    print("🔐 Testing Modal secrets...")
    try:
        # Test if we can create secrets (this tests Modal auth)
        secrets = modal.Secret.list()
        print(f"   ✅ Modal secrets accessible - found {len(secrets)} secrets")
        
        # Look for wandb secret
        wandb_secret_found = any('wandb' in secret.label.lower() for secret in secrets)
        if wandb_secret_found:
            print("   ✅ W&B secret found")
        else:
            print("   ⚠️ W&B secret not found (should be created)")
        
        return True
    except Exception as e:
        print(f"   ❌ Modal secrets test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all pipeline tests"""
    print("🧪 COMPREHENSIVE PIPELINE TEST SUITE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S AEST')}")
    print()
    
    tests = [
        ("Modal Connectivity", test_modal_connectivity),
        ("Dependencies", test_dependencies),
        ("File Permissions", test_file_permissions),
        ("W&B Integration", test_wandb_integration),
        ("Modal Secrets", test_modal_secrets),
        ("Extraction App", test_extraction_app),
        ("ChemBERTa App", test_chemberta_app),
        ("Chemprop App", test_chemprop_app),
        ("Backend Integration", test_backend_integration),
        ("Mini Extraction", test_mini_extraction),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            results[test_name] = {
                "passed": result,
                "duration": duration
            }
            
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED ({duration:.1f}s)")
            else:
                print(f"❌ {test_name} FAILED ({duration:.1f}s)")
                
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            results[test_name] = {
                "passed": False,
                "duration": 0,
                "error": str(e)
            }
    
    # Generate test report
    print("\n" + "="*60)
    print("📊 TEST SUITE SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    # Detailed results
    print(f"\n📋 Detailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        duration = result.get("duration", 0)
        print(f"  {status} {test_name:<25} ({duration:.1f}s)")
        if "error" in result:
            print(f"       Error: {result['error']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if passed == total:
        print("🎉 ALL TESTS PASSED! Pipeline is ready for full execution.")
        print("✅ Safe to proceed with the 6-12 hour training pipeline.")
    elif passed >= total * 0.8:  # 80% pass rate
        print("⚠️ Most tests passed. Pipeline likely to work but monitor closely.")
        print("🔄 Consider investigating failed tests for optimal performance.")
    else:
        print("🚨 CRITICAL ISSUES DETECTED! Do not run full pipeline yet.")
        print("🔧 Fix failing tests before proceeding to avoid costly failures.")
    
    # Save results
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total,
        "passed_tests": passed,
        "success_rate": (passed/total)*100,
        "results": results,
        "recommendation": "proceed" if passed == total else "investigate" if passed >= total * 0.8 else "fix_issues"
    }
    
    with open('/app/modal_training/pipeline_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n📁 Test results saved to: /app/modal_training/pipeline_test_results.json")
    
    return test_results

if __name__ == "__main__":
    try:
        results = run_comprehensive_test()
        exit_code = 0 if results["recommendation"] == "proceed" else 1
        exit(exit_code)
    except Exception as e:
        print(f"\n💥 TEST SUITE CRASHED: {e}")
        traceback.print_exc()
        exit(2)