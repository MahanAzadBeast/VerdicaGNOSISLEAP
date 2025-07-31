#!/usr/bin/env python3
"""
Quick comparison test of both real models
"""

import requests
import json

def test_both_models():
    """Test both ChemBERTa and Chemprop with Imatinib"""
    
    print("⚔️ REAL MODEL ARCHITECTURE COMPARISON")
    print("=" * 60)
    
    imatinib = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
    print(f"🧪 Testing: Imatinib - {imatinib}")
    print()
    
    # Test ChemBERTa (working)
    print("🧬 ChemBERTa (50-epoch Transformer):")
    try:
        response = requests.post("http://localhost:8001/api/chemberta/predict", 
                               json={"smiles": imatinib}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            model_info = data.get('model_info', {})
            print(f"   ✅ Model: {model_info.get('model_type', 'Unknown')}")
            print(f"   📊 Training: {model_info.get('training_epochs', 'Unknown')} epochs")
            print(f"   📈 Mean R²: {model_info.get('training_r2_mean', 'Unknown')}")
            
            # Show key predictions in μM
            key_targets = ['EGFR', 'BRAF', 'ALK', 'VEGFR2']
            predictions = data.get('predictions', {})
            print(f"   🎯 Predictions (IC50 in μM):")
            for target in key_targets:
                if target in predictions:
                    ic50_um = predictions[target].get('ic50_um', 0)
                    activity = predictions[target].get('activity_class', 'Unknown')
                    print(f"      {target}: {ic50_um:.3f} μM ({activity})")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test Real Chemprop (attempt)
    print("📊 Chemprop Real (50-epoch GNN):")
    try:
        response = requests.post("http://localhost:8001/api/chemprop-real/predict", 
                               json={"smiles": imatinib}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            model_info = data.get('model_info', {})
            print(f"   ✅ Model: {model_info.get('model_used', 'Unknown')}")
            print(f"   📊 Training: {model_info.get('training_epochs', 'Unknown')} epochs")
            
            # Show key predictions
            predictions = data.get('predictions', {})
            print(f"   🎯 Predictions (IC50 in μM):")
            for target in key_targets:
                if target in predictions:
                    ic50_nm = predictions[target].get('IC50_nM', 0)
                    ic50_um = ic50_nm / 1000
                    activity = predictions[target].get('activity', 'Unknown')
                    print(f"      {target}: {ic50_um:.3f} μM ({activity})")
        else:
            print(f"   ⚠️ Status: {response.status_code} - Model deployment in progress")
            print("   📋 Note: Chemprop CLI fixed, but Modal app may need a moment to start")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Direct test of the fixed Chemprop system
    print("🔧 Direct Chemprop Test (Fixed CLI):")
    try:
        import modal
        app = modal.App.lookup("chemprop-production-inference", create_if_missing=False)
        function = getattr(app, "predict_oncoprotein_activity")
        
        with app.run():
            result = function.remote(imatinib)
            
        if result.get('status') == 'success':
            print("   ✅ Direct Modal test successful!")
            predictions = result.get('predictions', {})
            print(f"   🎯 Direct predictions (IC50 in μM):")
            for target in key_targets:
                if target in predictions:
                    ic50_nm = predictions[target].get('IC50_nM', 0)
                    ic50_um = ic50_nm / 1000
                    activity = predictions[target].get('activity', 'Unknown')
                    print(f"      {target}: {ic50_um:.3f} μM ({activity})")
        else:
            print(f"   ❌ Direct test failed: {result.get('error')}")
            
    except Exception as e:
        print(f"   ❌ Direct test error: {e}")

if __name__ == "__main__":
    test_both_models()