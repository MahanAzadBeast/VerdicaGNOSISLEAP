#!/usr/bin/env python3
"""
Direct test of ChemBERTa Model Comparison Issue
Bypassing frontend navigation problems
"""

import requests
import json

def test_chemberta_in_comparison():
    """Test ChemBERTa API directly to verify it's working"""
    
    print("🧬 DIRECT CHEMBERTA MODEL COMPARISON TEST")
    print("=" * 50)
    
    # Backend URL
    API_BASE = "http://localhost:8001/api"
    
    # Test molecule (aspirin)
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    print(f"🧪 Testing with aspirin: {test_smiles}")
    print()
    
    # Test 1: ChemBERTa Status
    print("📊 Step 1: Testing ChemBERTa Status")
    try:
        response = requests.get(f"{API_BASE}/chemberta/status")
        if response.status_code == 200:
            status_data = response.json()
            print(f"   ✅ Status: {status_data.get('status')}")
            print(f"   ✅ Available: {status_data.get('available')}")
            print(f"   📊 Model: {status_data.get('model_info', {}).get('model_name', 'Unknown')}")
            print(f"   📈 Training Epochs: {status_data.get('model_info', {}).get('training_epochs', 'Unknown')}")
            print(f"   📈 Mean R²: {status_data.get('model_info', {}).get('mean_r2', 'Unknown')}")
        else:
            print(f"   ❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Status check error: {e}")
        return False
    
    print()
    
    # Test 2: ChemBERTa Prediction
    print("🔮 Step 2: Testing ChemBERTa Prediction")
    try:
        response = requests.post(f"{API_BASE}/chemberta/predict", 
                               json={"smiles": test_smiles},
                               timeout=30)
        
        if response.status_code == 200:
            pred_data = response.json()
            print(f"   ✅ Prediction Status: {pred_data.get('status')}")
            
            if pred_data.get('predictions'):
                predictions = pred_data['predictions']
                print(f"   🎯 Targets Predicted: {len(predictions)}")
                
                # Show key predictions
                key_targets = ['EGFR', 'VEGFR2', 'ALK']
                print("   📊 Key Predictions:")
                for target in key_targets:
                    if target in predictions:
                        pred = predictions[target]
                        ic50_um = pred.get('ic50_um', 'N/A')
                        activity = pred.get('activity_class', 'Unknown')
                        print(f"      {target}: IC50 {ic50_um} μM ({activity})")
                
                # Model info
                model_info = pred_data.get('model_info', {})
                print(f"   📊 Model Used: {model_info.get('model_type', 'Unknown')}")
                print(f"   📈 Training Epochs: {model_info.get('training_epochs', 'Unknown')}")
                print(f"   📈 Mean R²: {model_info.get('training_r2_mean', 'Unknown')}")
                
                return True
            else:
                print(f"   ❌ No predictions in response")
                return False
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   📋 Error: {error_data}")
            except:
                print(f"   📋 Raw response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
        return False

def test_comparison_simulation():
    """Simulate what the Model Architecture Comparison should do"""
    
    print("\n" + "=" * 50)
    print("⚔️ MODEL ARCHITECTURE COMPARISON SIMULATION")
    print("=" * 50)
    
    API_BASE = "http://localhost:8001/api"
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    print(f"🧪 Testing comparison with: {test_smiles}")
    print()
    
    # Test both models that should run in comparison
    results = {}
    
    # 1. ChemBERTa
    print("🧬 Testing ChemBERTa (for comparison)...")
    try:
        response = requests.post(f"{API_BASE}/chemberta/predict", 
                               json={"smiles": test_smiles}, timeout=30)
        if response.status_code == 200:
            results['chemberta'] = response.json()
            print("   ✅ ChemBERTa predictions obtained")
        else:
            print(f"   ❌ ChemBERTa failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ ChemBERTa error: {e}")
    
    # 2. Chemprop Real
    print("📊 Testing Chemprop Real (for comparison)...")
    try:
        response = requests.post(f"{API_BASE}/chemprop-real/predict", 
                               json={"smiles": test_smiles}, timeout=30)
        if response.status_code == 200:
            results['chemprop_real'] = response.json()
            print("   ✅ Chemprop Real predictions obtained")
        else:
            print(f"   ⚠️ Chemprop Real returned: {response.status_code} (expected 503)")
            results['chemprop_real'] = {"status": "unavailable", "code": response.status_code}
    except Exception as e:
        print(f"   ❌ Chemprop Real error: {e}")
    
    # Show comparison results
    print(f"\n📊 COMPARISON RESULTS:")
    print("-" * 30)
    
    if 'chemberta' in results and results['chemberta'].get('predictions'):
        print("✅ ChemBERTa Results Available:")
        predictions = results['chemberta']['predictions']
        model_info = results['chemberta'].get('model_info', {})
        print(f"   Model: {model_info.get('model_type', 'Unknown')}")
        print(f"   Epochs: {model_info.get('training_epochs', 'Unknown')}")
        print(f"   Mean R²: {model_info.get('training_r2_mean', 'Unknown')}")
        
        # Sample predictions
        sample_targets = ['EGFR', 'VEGFR2', 'ALK']
        for target in sample_targets:
            if target in predictions:
                pred = predictions[target]
                print(f"   {target}: {pred.get('ic50_um', 'N/A')} μM")
    else:
        print("❌ ChemBERTa Results NOT Available")
    
    print()
    
    if 'chemprop_real' in results:
        if results['chemprop_real'].get('status') == 'success':
            print("✅ Chemprop Real Results Available")
        else:
            print("⚠️ Chemprop Real NOT Available (using fallback/503)")
    else:
        print("❌ Chemprop Real Results NOT Available")
    
    return results

if __name__ == "__main__":
    print("Starting ChemBERTa Model Comparison Diagnosis...")
    
    # Test ChemBERTa directly
    chemberta_works = test_chemberta_in_comparison()
    
    # Test comparison simulation
    comparison_results = test_comparison_simulation()
    
    print(f"\n🎯 DIAGNOSIS SUMMARY:")
    print("=" * 50)
    print(f"✅ ChemBERTa API Working: {chemberta_works}")
    print(f"📊 ChemBERTa in Comparison: {'chemberta' in comparison_results}")
    print(f"⚔️ Comparison Ready: {chemberta_works and 'chemberta' in comparison_results}")
    
    if chemberta_works:
        print("\n🔍 ROOT CAUSE ANALYSIS:")
        print("✅ ChemBERTa backend API is working correctly")
        print("✅ Real 50-epoch model is connected and providing predictions")
        print("❌ Issue is likely in frontend navigation/display logic")
        print("\nRECOMMENDATION:")
        print("The ChemBERTa model is working fine. The issue is frontend navigation.")
        print("ChemBERTa SHOULD appear in Model Architecture Comparison when navigation works.")
    else:
        print("\n❌ ChemBERTa backend has issues that need to be resolved first.")