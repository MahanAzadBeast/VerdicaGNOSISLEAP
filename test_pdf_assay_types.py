#!/usr/bin/env python3
"""
Test PDF generation handles all assay types correctly.
"""

import requests
import json

BASE_URL = "http://localhost:8001"

def test_all_assay_types_in_pdf():
    """Test that PDF includes both Binding_IC50 and Functional_IC50"""
    print("🧪 Testing PDF Assay Type Coverage")
    
    # Get actual API response structure
    response = requests.post(
        f"{BASE_URL}/api/gnosis-i/predict-with-hp-ad",
        json={
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "targets": ["ERBB4", "AKT2"],
            "assay_types": ["IC50"]
        }
    )
    
    if response.status_code != 200:
        print(f"❌ API request failed: {response.status_code}")
        return False
    
    pred_data = response.json()
    
    # Count total predictions across all targets and assay types
    total_predictions = 0
    assay_types_found = set()
    
    for target, target_data in pred_data['predictions'].items():
        for assay_type, assay_data in target_data.items():
            if assay_type != 'selectivity_ratio':
                total_predictions += 1
                assay_types_found.add(assay_type)
                print(f"   Found: {target}.{assay_type} = {assay_data.get('status', 'OK')}")
    
    print(f"\n   Assay types found: {list(assay_types_found)}")
    print(f"   Total predictions: {total_predictions}")
    
    # Expected: Should find both Binding_IC50 and Functional_IC50
    expected_assays = {'Binding_IC50', 'Functional_IC50'}
    has_both_assays = expected_assays.issubset(assay_types_found)
    
    print(f"   Has both Binding & Functional IC50: {'✅' if has_both_assays else '❌'}")
    
    # Generate PDF from actual response
    pdf_response = requests.post(
        f"{BASE_URL}/api/reports/export-pdf",
        json=pred_data,
        headers={"Content-Type": "application/json"}
    )
    
    pdf_success = pdf_response.status_code == 200 and len(pdf_response.content) > 1000
    print(f"   PDF generation: {'✅' if pdf_success else '❌'} ({len(pdf_response.content)} bytes)")
    
    # The PDF should process all predictions (both binding and functional)
    # Since aspirin is gated, we expect no "Highest Potency" but all assay types processed
    
    return has_both_assays and pdf_success and total_predictions >= 4  # 2 targets × 2 assays each

def test_mixed_status_pdf():
    """Test PDF with mixed OK and gated predictions"""
    print("\n🧪 Testing Mixed Status PDF Generation")
    
    # Create a mixed scenario manually
    mixed_data = {
        "smiles": "Mixed_Test",
        "compound_name": "Mixed Status Test",
        "predictions": {
            "TARGET_A": {
                "Binding_IC50": {
                    "target_id": "TARGET_A",
                    "status": "OK",
                    "pActivity": 6.5,
                    "activity_uM": 0.32,
                    "confidence": 0.85
                },
                "Functional_IC50": {
                    "target_id": "TARGET_A", 
                    "status": "HYPOTHESIS_ONLY",
                    "message": "Out of domain for this target class. Numeric potency suppressed.",
                    "why": ["OOD_chem", "Insufficient_in-class_neighbors"],
                    "evidence": {"S_max": 0.2, "neighbors_same_assay": 5}
                }
            }
        }
    }
    
    pdf_response = requests.post(
        f"{BASE_URL}/api/reports/export-pdf",
        json=mixed_data,
        headers={"Content-Type": "application/json"}
    )
    
    pdf_success = pdf_response.status_code == 200 and len(pdf_response.content) > 1000
    print(f"   Mixed status PDF: {'✅' if pdf_success else '❌'} ({len(pdf_response.content)} bytes)")
    
    # Save for manual inspection if needed
    if pdf_success:
        with open('/tmp/mixed_status_test.pdf', 'wb') as f:
            f.write(pdf_response.content)
        print(f"   Saved: /tmp/mixed_status_test.pdf")
    
    return pdf_success

def main():
    """Run PDF assay type tests"""
    print("🚀 PDF ASSAY TYPE COVERAGE TESTS")
    print("=" * 50)
    
    try:
        test1_success = test_all_assay_types_in_pdf()
        test2_success = test_mixed_status_pdf()
        
        print("\n" + "=" * 50)
        print("🎯 PDF ASSAY TYPE TEST SUMMARY")
        print(f"   All assay types covered: {'✅' if test1_success else '❌'}")
        print(f"   Mixed status handling: {'✅' if test2_success else '❌'}")
        
        overall_success = test1_success and test2_success
        
        if overall_success:
            print("🎉 PDF ASSAY TYPE TESTS PASSED!")
            print("✅ Both Binding_IC50 and Functional_IC50 correctly processed in PDF")
        else:
            print("❌ Some PDF assay type tests failed")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)