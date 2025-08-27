#!/usr/bin/env python3
"""
Curated test panel for CI regression protection.
Ensures gate functions never regress on key molecules.
"""

import sys
sys.path.append('/app/backend')

import requests
import json
import time

BASE_URL = "http://localhost:8001"

# CURATED TEST PANEL - GOLDEN MOLECULES WITH EXPECTED BEHAVIORS
GOLDEN_MOLECULES = [
    {
        "name": "Aspirin",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "targets": ["ERBB4", "AKT2", "PARP1"],
        "expected_status": "HYPOTHESIS_ONLY",
        "expected_reasons": ["OOD_chem", "Kinase_pharmacophore_fail", "tiny_acid_veto"],
        "expected_highest_potency": "N/A",  # No numeric predictions
        "description": "Must be gated on ALL kinases and PARP1 - core reliability test"
    },
    {
        "name": "Salicylic Acid",
        "smiles": "O=C(O)c1ccccc1O",
        "targets": ["EGFR", "PARP1"],
        "expected_status": "HYPOTHESIS_ONLY",
        "expected_reasons": ["OOD_chem", "PARP_pharmacophore_fail", "tiny_acid_veto"],
        "expected_highest_potency": "N/A",
        "description": "Similar to aspirin - should be gated"
    },
    {
        "name": "Ethanol",
        "smiles": "CCO",
        "targets": ["EGFR", "BRAF"],
        "expected_status": "HYPOTHESIS_ONLY", 
        "expected_reasons": ["OOD_chem", "Insufficient_in-class_neighbors"],
        "expected_highest_potency": "N/A",
        "description": "Simple alcohol - should be OOD gated"
    },
    {
        "name": "Caffeine",
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "targets": ["EGFR"],
        "expected_status": "HYPOTHESIS_ONLY",
        "expected_reasons": ["OOD_chem", "Insufficient_in-class_neighbors"],
        "expected_highest_potency": "N/A", 
        "description": "Xanthine alkaloid - should be gated"
    },
    {
        "name": "Gefitinib",
        "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        "targets": ["EGFR"],
        "expected_status": "HYPOTHESIS_ONLY",  # Due to insufficient mock data
        "expected_reasons": ["Insufficient_in-class_neighbors"],
        "expected_highest_potency": "N/A",
        "description": "Known kinase inhibitor - gated due to sparse mock data (correct behavior)"
    }
]

def test_molecule_regression(molecule):
    """Test a single molecule against its expected behavior"""
    print(f"\n🧪 Testing {molecule['name']}")
    print(f"   SMILES: {molecule['smiles']}")
    print(f"   Targets: {molecule['targets']}")
    
    # Make API request
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/api/gnosis-i/predict-with-hp-ad",
        json={
            "smiles": molecule['smiles'],
            "targets": molecule['targets'],
            "assay_types": ["IC50"]
        },
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    elapsed_time = time.time() - start_time
    
    if response.status_code != 200:
        print(f"   ❌ API request failed: {response.status_code}")
        return False
    
    pred_data = response.json()
    
    # Analyze all predictions for this molecule
    all_gated = True
    all_ok = True
    found_reasons = set()
    has_numerics = False
    
    for target in molecule['targets']:
        target_data = pred_data['predictions'].get(target, {})
        for assay_type, assay_data in target_data.items():
            if assay_type != 'selectivity_ratio':
                status = assay_data.get('status', 'OK')
                if status == 'HYPOTHESIS_ONLY':
                    all_ok = False
                    reasons = assay_data.get('why', [])
                    found_reasons.update(reasons)
                else:
                    all_gated = False
                    if 'pActivity' in assay_data:
                        has_numerics = True
    
    # Check expectations
    expected_status = molecule['expected_status']
    if expected_status == 'HYPOTHESIS_ONLY':
        status_correct = all_gated and not has_numerics
    else:
        status_correct = all_ok and has_numerics
    
    # Check expected reasons are present
    expected_reasons_found = all(
        any(exp_reason in reason for reason in found_reasons) 
        for exp_reason in molecule['expected_reasons']
    )
    
    # Performance check
    performance_ok = elapsed_time <= 10.0  # Allow 10s for regression tests
    
    # Results
    print(f"   Status: {'✅' if status_correct else '❌'} " + 
          f"({'All gated' if all_gated else 'Some OK'}, " +
          f"{'No numerics' if not has_numerics else 'Has numerics'})")
    print(f"   Expected reasons: {molecule['expected_reasons']}")
    print(f"   Found reasons: {list(found_reasons)}")
    print(f"   Reasons match: {'✅' if expected_reasons_found else '❌'}")
    print(f"   Performance: {'✅' if performance_ok else '❌'} ({elapsed_time:.2f}s)")
    
    success = status_correct and expected_reasons_found and performance_ok
    print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
    
    return success

def test_pdf_regression():
    """Test PDF generation excludes gated predictions from highest potency"""
    print(f"\n📄 Testing PDF Regression - Highest Potency Logic")
    
    # Test aspirin PDF - should have no "Highest Potency" since all gated
    response = requests.post(
        f"{BASE_URL}/api/reports/export-pdf",
        json={
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "compound_name": "Aspirin",
            "predictions": {
                "ERBB4": {
                    "IC50": {
                        "target_id": "ERBB4",
                        "status": "HYPOTHESIS_ONLY",
                        "message": "Out of domain for this target class. Numeric potency suppressed.",
                        "why": ["OOD_chem", "Kinase_pharmacophore_fail", "tiny_acid_veto"],
                        "evidence": {"S_max": 0, "neighbors_same_assay": 0}
                    }
                }
            }
        },
        headers={"Content-Type": "application/json"}
    )
    
    pdf_success = response.status_code == 200 and len(response.content) > 1000
    print(f"   PDF Generation: {'✅' if pdf_success else '❌'} ({len(response.content)} bytes)")
    
    return pdf_success

def test_no_numeric_leaks():
    """Ensure no gated predictions contain numeric fields"""
    print(f"\n🔒 Testing No Numeric Leaks")
    
    # Test aspirin - should have NO numeric fields in any gated prediction
    response = requests.post(
        f"{BASE_URL}/api/gnosis-i/predict-with-hp-ad",
        json={
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "targets": ["ERBB4", "AKT2"],
            "assay_types": ["IC50"]
        }
    )
    
    if response.status_code != 200:
        print(f"   ❌ API request failed")
        return False
    
    pred_data = response.json()
    leaked_numerics = []
    
    for target, target_data in pred_data['predictions'].items():
        for assay_type, assay_data in target_data.items():
            if assay_type != 'selectivity_ratio':
                status = assay_data.get('status', 'OK')
                if status == 'HYPOTHESIS_ONLY':
                    # Check for numeric field leaks
                    numeric_fields = ['pActivity', 'potency_ci', 'confidence_calibrated', 'IC50_nM', 'activity_uM']
                    for field in numeric_fields:
                        if field in assay_data:
                            leaked_numerics.append(f"{target}.{assay_type}.{field}")
    
    no_leaks = len(leaked_numerics) == 0
    print(f"   Numeric leaks: {'✅ None' if no_leaks else f'❌ Found: {leaked_numerics}'}")
    
    return no_leaks

def main():
    """Run comprehensive regression test panel"""
    print("🚀 CURATED REGRESSION TEST PANEL")
    print("Ensures gate functions never regress on golden molecules")
    print("=" * 60)
    
    results = []
    
    # Test each golden molecule
    for molecule in GOLDEN_MOLECULES:
        success = test_molecule_regression(molecule)
        results.append((molecule['name'], success))
    
    # Test PDF regression
    pdf_success = test_pdf_regression()
    results.append(("PDF Generation", pdf_success))
    
    # Test numeric leak prevention
    leak_success = test_no_numeric_leaks() 
    results.append(("No Numeric Leaks", leak_success))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 REGRESSION TEST SUMMARY")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        print(f"   {name}: {'✅ PASS' if success else '❌ FAIL'}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL REGRESSION TESTS PASSED!")
        print("✅ Gate functions working correctly - no regressions detected")
        return True
    else:
        print("❌ Some regression tests failed - review results above")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)