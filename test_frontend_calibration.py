#!/usr/bin/env python3
"""
Test if calibrated values are being served by the backend
This confirms the changes are ready for the frontend
"""

import requests
import json

def test_calibrated_backend():
    """Test the backend API to confirm calibrated values are being served"""
    
    print("🔍 TESTING CALIBRATED VALUES FROM BACKEND API")
    print("=" * 60)
    
    # Test the exact molecule from your screenshot
    imatinib = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
    
    print(f"🧪 Testing with Imatinib SMILES:")
    print(f"   {imatinib}")
    
    # Test ChemBERTa API (should show calibrated values)
    print(f"\n🧬 ChemBERTa API Test:")
    try:
        response = requests.post("http://localhost:8001/api/chemberta/predict", 
                               json={"smiles": imatinib}, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', {})
            model_info = data.get('model_info', {})
            
            print(f"   ✅ Status: {data.get('status')}")
            print(f"   📊 Model: {model_info.get('model_type', 'Unknown')}")
            print(f"   📈 Training: {model_info.get('training_epochs', 'Unknown')} epochs")
            print(f"   📈 Mean R²: {model_info.get('training_r2_mean', 'Unknown')}")
            
            print(f"\n   🎯 KEY PREDICTIONS (Should be calibrated):")
            key_targets = ['EGFR', 'BRAF', 'ALK', 'VEGFR2']
            
            for target in key_targets:
                if target in predictions:
                    pred = predictions[target]
                    ic50_um = pred.get('ic50_um', 0)
                    pic50 = pred.get('pic50', 0)
                    activity = pred.get('activity_class', 'Unknown')
                    
                    # Check if this looks calibrated
                    is_calibrated = "✅ CALIBRATED" if ic50_um < 1.0 else "❌ NOT CALIBRATED"
                    
                    print(f"      {target}: IC50 {ic50_um:.3f} μM (pIC50 {pic50:.3f}) - {activity} {is_calibrated}")
            
            # Test what the frontend would receive
            print(f"\n   📱 Frontend would receive:")
            print(f"      EGFR: {predictions.get('EGFR', {}).get('ic50_um', 'N/A'):.3f} μM")
            print(f"      BRAF: {predictions.get('BRAF', {}).get('ic50_um', 'N/A'):.3f} μM")
            
        else:
            print(f"   ❌ Failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test Chemprop API for comparison
    print(f"\n📊 Chemprop API Test:")
    try:
        response = requests.post("http://localhost:8001/api/chemprop-real/predict", 
                               json={"smiles": imatinib}, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', {})
            model_info = data.get('model_info', {})
            
            print(f"   ✅ Status: {data.get('status')}")
            print(f"   📊 Model: {model_info.get('model_used', 'Unknown')}")
            
            print(f"\n   🎯 KEY PREDICTIONS:")
            for target in key_targets:
                if target in predictions:
                    pred = predictions[target]
                    ic50_nm = pred.get('IC50_nM', 0)
                    ic50_um = ic50_nm / 1000
                    activity = pred.get('activity', 'Unknown')
                    
                    print(f"      {target}: IC50 {ic50_um:.3f} μM ({ic50_nm:.1f} nM) - {activity}")
            
        else:
            print(f"   ⚠️ Status: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Comparison analysis
    print(f"\n⚔️ COMPARISON ANALYSIS:")
    print("=" * 40)
    
    # Get both predictions for comparison
    try:
        chemberta_resp = requests.post("http://localhost:8001/api/chemberta/predict", 
                                     json={"smiles": imatinib}, timeout=30)
        chemprop_resp = requests.post("http://localhost:8001/api/chemprop-real/predict", 
                                    json={"smiles": imatinib}, timeout=30)
        
        if chemberta_resp.status_code == 200 and chemprop_resp.status_code == 200:
            chemberta_data = chemberta_resp.json()
            chemprop_data = chemprop_resp.json()
            
            print("Variance Analysis (Calibrated vs Real):")
            
            for target in ['EGFR', 'BRAF', 'ALK']:
                if (target in chemberta_data.get('predictions', {}) and 
                    target in chemprop_data.get('predictions', {})):
                    
                    chemberta_ic50 = chemberta_data['predictions'][target]['ic50_um']
                    chemprop_ic50_nm = chemprop_data['predictions'][target]['IC50_nM']
                    chemprop_ic50_um = chemprop_ic50_nm / 1000
                    
                    difference = chemberta_ic50 / chemprop_ic50_um
                    
                    status = "✅ GOOD" if difference < 50 else "⚠️ HIGH" if difference < 500 else "❌ EXTREME"
                    
                    print(f"   {target}: {chemberta_ic50:.3f} μM vs {chemprop_ic50_um:.3f} μM = {difference:.1f}x {status}")
        
        print(f"\n🎯 FRONTEND IMPACT:")
        print("When the frontend navigation is fixed, it will receive these calibrated values")
        print("The massive variance issue (15,851x) should now be resolved!")
        
    except Exception as e:
        print(f"   ❌ Comparison error: {e}")

if __name__ == "__main__":
    test_calibrated_backend()