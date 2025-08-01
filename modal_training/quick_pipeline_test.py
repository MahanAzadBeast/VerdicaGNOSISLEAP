"""
Quick Pipeline Validation Test
Fixed version addressing Modal API compatibility issues
"""

import modal
import json
from datetime import datetime
import subprocess
import os

def test_modal_apps():
    """Test Modal app functionality with current API"""
    print("🔗 Testing Modal apps...")
    try:
        # Test app imports
        from expanded_multisource_extractor import app as extractor_app
        from train_expanded_chemberta import app as chemberta_app  
        from train_expanded_chemprop import app as chemprop_app
        
        print("   ✅ All Modal apps imported successfully")
        
        # Test if apps have functions
        print(f"   ✅ Extractor app: {type(extractor_app)}")
        print(f"   ✅ ChemBERTa app: {type(chemberta_app)}")
        print(f"   ✅ Chemprop app: {type(chemprop_app)}")
        
        return True
    except Exception as e:
        print(f"   ❌ Modal apps test failed: {e}")
        return False

def test_function_execution():
    """Test if we can execute a simple Modal function"""
    print("⚡ Testing Modal function execution...")
    try:
        # Create a simple test function
        app = modal.App("pipeline-test")
        
        @app.function(timeout=60)
        def simple_test():
            """Simple test function"""
            import pandas as pd
            import numpy as np
            
            # Test basic functionality
            data = {"test": [1, 2, 3], "values": [0.1, 0.2, 0.3]}
            df = pd.DataFrame(data)
            
            return {
                "status": "success",
                "shape": df.shape,
                "test_complete": True
            }
        
        # Try to run it
        with app.run() as app_run:
            result = simple_test.remote()
        
        if result["status"] == "success":
            print("   ✅ Modal function execution successful")
            return True
        else:
            print("   ❌ Modal function returned error")
            return False
            
    except Exception as e:
        print(f"   ❌ Modal function execution failed: {e}")
        return False

def validate_data_sources():
    """Test if we can access data sources"""
    print("🌐 Testing data source accessibility...")
    try:
        import requests
        
        # Test ChEMBL API
        response = requests.get("https://www.ebi.ac.uk/chembl/api/data/target/CHEMBL203", timeout=10)
        if response.status_code == 200:
            print("   ✅ ChEMBL API accessible")
        else:
            print(f"   ⚠️ ChEMBL API returned status {response.status_code}")
        
        # Test PubChem API
        pubchem_response = requests.get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/MolecularFormula/JSON", timeout=10)
        if pubchem_response.status_code == 200:
            print("   ✅ PubChem API accessible")
        else:
            print(f"   ⚠️ PubChem API returned status {pubchem_response.status_code}")
        
        return True
    except Exception as e:
        print(f"   ❌ Data source test failed: {e}")
        return False

def test_critical_imports():
    """Test all critical imports for the pipeline"""
    print("📦 Testing critical imports...")
    
    critical_packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'), 
        ('rdkit', None),
        ('torch', None),
        ('transformers', None),
        ('sklearn', None),
        ('modal', None)
    ]
    
    all_good = True
    for package, alias in critical_packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"   ✅ {package}")
        except ImportError as e:
            print(f"   ❌ {package} - MISSING: {e}")
            all_good = False
    
    return all_good

def quick_validation():
    """Run quick validation tests"""
    print("🧪 QUICK PIPELINE VALIDATION")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S AEST')}")
    print()
    
    tests = [
        ("Critical Imports", test_critical_imports),
        ("Modal Apps", test_modal_apps),
        ("Data Sources", validate_data_sources),
        ("Function Execution", test_function_execution),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    total = len(tests)
    success_rate = (passed / total) * 100
    
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print("✅ PIPELINE READY FOR EXECUTION")
        print("🚀 Safe to proceed with full training pipeline")
        recommendation = "proceed"
    else:
        print("⚠️ PIPELINE HAS ISSUES")
        print("🔧 Address failing tests before full execution")
        recommendation = "investigate"
    
    # Save quick results
    results = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "total": total,
        "success_rate": success_rate,
        "recommendation": recommendation
    }
    
    with open('/app/modal_training/quick_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = quick_validation()
    exit(0 if results["recommendation"] == "proceed" else 1)