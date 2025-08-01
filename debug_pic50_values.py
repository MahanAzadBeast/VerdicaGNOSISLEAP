#!/usr/bin/env python3
"""
Debug pIC50 Prediction Values from Both Models
"""

import requests
import json
import math

def debug_pic50_predictions():
    """Get raw pIC50 predictions from both models to debug the issue"""
    
    print("🔍 DEBUGGING pIC50 PREDICTION VALUES")
    print("=" * 60)
    
    imatinib = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
    
    print(f"🧪 Testing: Imatinib")
    print(f"SMILES: {imatinib}")
    print()
    
    # Get ChemBERTa prediction
    print("🧬 ChemBERTa Raw Predictions:")
    try:
        response = requests.post("http://localhost:8001/api/chemberta/predict", 
                               json={"smiles": imatinib}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', {})
            
            # Show raw pIC50 values and conversion
            key_targets = ['EGFR', 'BRAF', 'ALK']
            for target in key_targets:
                if target in predictions:
                    pred = predictions[target]
                    pic50 = pred.get('pic50', 0)
                    ic50_um = pred.get('ic50_um', 0)
                    
                    # Verify the conversion
                    expected_ic50_um = 10 ** (6 - pic50)
                    
                    print(f"   {target}:")
                    print(f"      Raw pIC50: {pic50:.3f}")
                    print(f"      Reported IC50: {ic50_um:.3f} μM")
                    print(f"      Expected IC50: {expected_ic50_um:.3f} μM")
                    print(f"      Conversion formula: 10^(6 - {pic50:.3f}) = {expected_ic50_um:.3f}")
                    
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Get Chemprop prediction (we can't get raw pIC50, but we can reverse-engineer it)
    print("📊 Chemprop Reverse-Engineered pIC50:")
    try:
        response = requests.post("http://localhost:8001/api/chemprop-real/predict", 
                               json={"smiles": imatinib}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', {})
            
            key_targets = ['EGFR', 'BRAF', 'ALK']
            for target in key_targets:
                if target in predictions:
                    pred = predictions[target]
                    ic50_nm = pred.get('IC50_nM', 0)
                    
                    # Reverse-engineer the pIC50 that Chemprop predicted
                    # Formula used: ic50_nm = 10 ** (9 - pred_value)
                    # So: pred_value = 9 - log10(ic50_nm)
                    reverse_pic50 = 9 - math.log10(ic50_nm)
                    
                    print(f"   {target}:")
                    print(f"      Reported IC50: {ic50_nm:.3f} nM ({ic50_nm/1000:.3f} μM)")
                    print(f"      Reverse-engineered pIC50: {reverse_pic50:.3f}")
                    print(f"      Chemprop formula: 10^(9 - {reverse_pic50:.3f}) = {ic50_nm:.3f} nM")
                    
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print(f"\n🎯 ANALYSIS:")
    print("If both models were properly calibrated, they should predict similar pIC50 values")
    print("The huge difference suggests:")
    print("1. Different training data preprocessing")
    print("2. One model trained on different target value ranges")
    print("3. Unit conversion errors during training")
    
    print(f"\n🔧 EXPECTED VALUES FOR IMATINIB:")
    print("Known EGFR IC50 for Imatinib: ~20-100 nM (literature)")
    print("Expected pIC50: ~7.0-7.7")
    print("Chemprop EGFR: 7.29 nM (pIC50 ≈ 8.14) - TOO HIGH")
    print("ChemBERTa EGFR: 110,956 nM (pIC50 ≈ 3.96) - TOO LOW")

if __name__ == "__main__":
    debug_pic50_predictions()