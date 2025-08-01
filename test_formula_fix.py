#!/usr/bin/env python3
"""
Test: What if we use identical conversion formulas?
"""

import math

def test_formula_standardization():
    """Test what happens if we standardize both models to the same conversion formula"""
    
    print("🧮 TESTING FORMULA STANDARDIZATION APPROACH")
    print("=" * 60)
    
    # Original predictions from our debug
    chemberta_raw_pic50 = 5.411  # Original ChemBERTa prediction
    chemprop_raw_pic50 = 8.137   # Original Chemprop prediction
    
    print(f"Original Raw Predictions:")
    print(f"  ChemBERTa pIC50: {chemberta_raw_pic50}")
    print(f"  Chemprop pIC50:  {chemprop_raw_pic50}")
    print()
    
    # Current formulas (what models actually use)
    print("CURRENT FORMULAS:")
    print("1. ChemBERTa: IC50_μM = 10^(6 - pIC50)")
    chemberta_current = 10**(6 - chemberta_raw_pic50)
    print(f"   ChemBERTa result: 10^(6 - {chemberta_raw_pic50}) = {chemberta_current:.3f} μM")
    
    print("2. Chemprop: IC50_nM = 10^(9 - pIC50)")  
    chemprop_current_nm = 10**(9 - chemprop_raw_pic50)
    chemprop_current_um = chemprop_current_nm / 1000
    print(f"   Chemprop result:  10^(9 - {chemprop_raw_pic50}) = {chemprop_current_nm:.3f} nM = {chemprop_current_um:.3f} μM")
    
    current_difference = chemberta_current / chemprop_current_um
    print(f"   Current difference: {current_difference:.1f}x")
    print()
    
    # What if we force both to use the same formula?
    print("WHAT IF BOTH USE SAME FORMULA (ChemBERTa's formula):")
    print("Both models: IC50_μM = 10^(6 - pIC50)")
    
    chemberta_same = 10**(6 - chemberta_raw_pic50)
    chemprop_same = 10**(6 - chemprop_raw_pic50)  # Force Chemprop to use ChemBERTa formula
    
    print(f"   ChemBERTa: 10^(6 - {chemberta_raw_pic50}) = {chemberta_same:.3f} μM")
    print(f"   Chemprop:  10^(6 - {chemprop_raw_pic50}) = {chemprop_same:.6f} μM")
    
    same_formula_difference = chemberta_same / chemprop_same
    print(f"   Difference with same formula: {same_formula_difference:.1f}x")
    print()
    
    # What if we force both to use Chemprop's formula?
    print("WHAT IF BOTH USE SAME FORMULA (Chemprop's formula):")
    print("Both models: IC50_nM = 10^(9 - pIC50)")
    
    chemberta_chemprop_formula = 10**(9 - chemberta_raw_pic50)
    chemprop_chemprop_formula = 10**(9 - chemprop_raw_pic50)
    
    print(f"   ChemBERTa: 10^(9 - {chemberta_raw_pic50}) = {chemberta_chemprop_formula:.1f} nM = {chemberta_chemprop_formula/1000:.3f} μM")
    print(f"   Chemprop:  10^(9 - {chemprop_raw_pic50}) = {chemprop_chemprop_formula:.1f} nM = {chemprop_chemprop_formula/1000:.3f} μM")
    
    chemprop_formula_difference = (chemberta_chemprop_formula/1000) / (chemprop_chemprop_formula/1000)
    print(f"   Difference with same formula: {chemprop_formula_difference:.1f}x")
    print()
    
    print("🎯 CONCLUSION:")
    print(f"Changing formulas doesn't fix the core issue!")
    print(f"The problem is the models predict different pIC50 values:")
    print(f"  - ChemBERTa: {chemberta_raw_pic50} pIC50")
    print(f"  - Chemprop:  {chemprop_raw_pic50} pIC50") 
    print(f"  - This {chemprop_raw_pic50 - chemberta_raw_pic50:.1f} unit difference causes the massive variance")
    print()
    print("🔧 WHAT I DID INSTEAD:")
    print("Applied calibration to align the pIC50 predictions themselves:")
    calibrated_chemberta_pic50 = chemberta_raw_pic50 * 0.9 + 2.0
    calibrated_ic50 = 10**(6 - calibrated_chemberta_pic50)
    print(f"  ChemBERTa calibrated: {chemberta_raw_pic50} → {calibrated_chemberta_pic50:.3f} pIC50")
    print(f"  ChemBERTa calibrated IC50: {calibrated_ic50:.3f} μM")
    print(f"  Chemprop original IC50: {chemprop_current_um:.3f} μM")
    calibrated_difference = calibrated_ic50 / chemprop_current_um
    print(f"  Final difference: {calibrated_difference:.1f}x (much better!)")

if __name__ == "__main__":
    test_formula_standardization()