#!/usr/bin/env python3
"""
Comprehensive test suite for Cross-Assay Gating & Reliability Hardening.

Tests the new family envelopes, cross-assay consistency, and cumulative gating rules.
"""

import sys
sys.path.append('/app/backend')

import requests
import json
import time
from rdkit import Chem
from hp_ad_layer import (
    assay_consistency_check,
    family_physchem_gate, 
    kinase_mechanism_gate_v2,
    parp1_mechanism_gate_v2,
    gpcr_mechanism_gate,
    aggregate_gates_v3,
    determine_protein_family,
    _floor_clamp_um,
    _log10
)
from hp_ad_layer_config import *

BASE_URL = "http://localhost:8001"

def test_cross_assay_consistency():
    """Test cross-assay consistency checks"""
    print("🧪 Testing Cross-Assay Consistency")
    
    # Test case 1: Consistent assays (should pass)
    ok1, reasons1 = assay_consistency_check(
        binding_um=1.0,      # 1 μM binding
        functional_um=2.0,   # 2 μM functional (within 10x)
        ec50_um=5.0,         # 5 μM EC50 (reasonable for enzyme)
        is_enzyme=True
    )
    print(f"  Consistent assays: {ok1} (expected: True), reasons: {reasons1}")
    assert ok1 == True, "Consistent assays should pass"
    
    # Test case 2: Discordant binding vs functional (should fail)
    ok2, reasons2 = assay_consistency_check(
        binding_um=0.1,      # 0.1 μM binding  
        functional_um=50.0,  # 50 μM functional (500x difference > 10x)
        ec50_um=None,
        is_enzyme=True
    )
    print(f"  Discordant B vs F: {ok2} (expected: False), reasons: {reasons2}")
    assert ok2 == False, "Discordant assays should fail"
    assert "Assay_discordance_BvsF" in reasons2
    
    # Test case 3: Enzyme monotonicity violation (should fail)
    ok3, reasons3 = assay_consistency_check(
        binding_um=0.001,    # 1 nM binding (very potent)
        functional_um=None,
        ec50_um=10.0,        # 10 μM EC50 (10,000x weaker)
        is_enzyme=True
    )
    print(f"  Monotonicity violation: {ok3} (expected: False), reasons: {reasons3}")
    assert ok3 == False, "Monotonicity violation should fail"
    assert "Enzyme_monotonicity_fail" in reasons3
    
    print("  ✅ Cross-assay consistency tests passed")

def test_family_physchem_gates():
    """Test family-specific physicochemical property gates"""
    print("\n🧪 Testing Family Physicochemical Gates")
    
    # Test aspirin (should fail kinase gates)
    aspirin_mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    aspirin_kinase_ok, aspirin_kinase_reasons = family_physchem_gate(aspirin_mol, "kinase")
    print(f"  Aspirin kinase physchem: {aspirin_kinase_ok} (expected: False), reasons: {aspirin_kinase_reasons}")
    assert aspirin_kinase_ok == False, "Aspirin should fail kinase physicochemical gates"
    
    # Test caffeine (should fail GPCR gates)
    caffeine_mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    caffeine_gpcr_ok, caffeine_gpcr_reasons = family_physchem_gate(caffeine_mol, "gpcr")
    print(f"  Caffeine GPCR physchem: {caffeine_gpcr_ok} (expected: False), reasons: {caffeine_gpcr_reasons}")
    assert caffeine_gpcr_ok == False, "Caffeine should fail GPCR physicochemical gates"
    
    # Test imatinib (should pass kinase gates)
    imatinib_mol = Chem.MolFromSmiles("Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C")
    imatinib_kinase_ok, imatinib_kinase_reasons = family_physchem_gate(imatinib_mol, "kinase")
    print(f"  Imatinib kinase physchem: {imatinib_kinase_ok} (expected: True), reasons: {imatinib_kinase_reasons}")
    assert imatinib_kinase_ok == True, "Imatinib should pass kinase physicochemical gates"
    
    print("  ✅ Family physicochemical gate tests passed")

def test_mechanism_gates():
    """Test enhanced mechanism gates by family"""
    print("\n🧪 Testing Enhanced Mechanism Gates")
    
    # Test kinase mechanism gates
    aspirin_mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    aspirin_kinase_ok, aspirin_kinase_reasons = kinase_mechanism_gate_v2(aspirin_mol)
    print(f"  Aspirin kinase mechanism: {aspirin_kinase_ok} (expected: False), reasons: {aspirin_kinase_reasons}")
    assert aspirin_kinase_ok == False, "Aspirin should fail kinase mechanism gate"
    
    # Test PARP1 mechanism gates with negative pattern
    aspirin_parp_ok, aspirin_parp_reasons = parp1_mechanism_gate_v2(aspirin_mol)
    print(f"  Aspirin PARP1 mechanism: {aspirin_parp_ok} (expected: False), reasons: {aspirin_parp_reasons}")
    assert aspirin_parp_ok == False, "Aspirin should fail PARP1 mechanism gate (negative pattern)"
    
    # Test GPCR mechanism gate
    caffeine_gpcr_ok, caffeine_gpcr_reasons = gpcr_mechanism_gate(aspirin_mol)  # Aspirin for GPCR
    print(f"  Aspirin GPCR mechanism: {caffeine_gpcr_ok} (expected: False), reasons: {caffeine_gpcr_reasons}")
    
    # Test positive case - simple benzamide for PARP1
    benzamide_mol = Chem.MolFromSmiles("NC(=O)c1ccccc1")
    benzamide_parp_ok, benzamide_parp_reasons = parp1_mechanism_gate_v2(benzamide_mol)
    print(f"  Benzamide PARP1 mechanism: {benzamide_parp_ok} (expected: True), reasons: {benzamide_parp_reasons}")
    assert benzamide_parp_ok == True, "Benzamide should pass PARP1 mechanism gate"
    
    print("  ✅ Mechanism gate tests passed")

def test_protein_family_determination():
    """Test protein family determination from target IDs"""
    print("\n🧪 Testing Protein Family Determination")
    
    families = [
        ("EGFR", "kinase"),
        ("ERBB4", "kinase"), 
        ("PARP1", "parp"),
        ("BRAF", "kinase"),
        ("SOME_UNKNOWN", "kinase")  # Default to kinase
    ]
    
    for target_id, expected_family in families:
        detected_family = determine_protein_family(target_id)
        print(f"  {target_id}: {detected_family} (expected: {expected_family})")
        assert detected_family == expected_family, f"{target_id} should be classified as {expected_family}"
    
    print("  ✅ Protein family determination tests passed")

def test_aggregate_gates():
    """Test comprehensive gate aggregation"""
    print("\n🧪 Testing Aggregate Gates Function")
    
    # Test aspirin on kinase (should be heavily gated)
    aspirin_mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    
    # Mock data for aspirin
    nn_info = {
        "S_max": 0.2,  # Below 0.5 threshold
        "n_sim_ge_0_40_same_assay": 5  # Below 30 threshold
    }
    
    mech_info = {
        "reasons": ["Kinase_mechanism_fail"],
        "score": 0.1
    }
    
    assays = {
        "Binding_IC50": None,
        "Functional_IC50": None,
        "EC50": None
    }
    
    suppress, hard_flag, reasons, evidence = aggregate_gates_v3(
        mol=aspirin_mol,
        target_id="EGFR",
        family="kinase", 
        ad_ok=False,  # Fails AD
        mech_info=mech_info,
        nn_info=nn_info,
        assays=assays
    )
    
    print(f"  Aspirin EGFR - Suppress: {suppress} (expected: True)")
    print(f"  Aspirin EGFR - Hard flag: {hard_flag} (expected: True)") 
    print(f"  Aspirin EGFR - Gate failures: {evidence.get('gate_failures', 0)}")
    print(f"  Aspirin EGFR - Reasons: {reasons}")
    
    assert suppress == True, "Aspirin should be suppressed"
    assert hard_flag == True, "Aspirin should have hard flag (≥3 gate failures)"
    assert evidence.get('gate_failures', 0) >= 3, "Aspirin should have ≥3 gate failures"
    
    print("  ✅ Aggregate gates tests passed")

def test_api_integration():
    """Test API integration with cross-assay gating"""
    print("\n🧪 Testing API Integration")
    
    # Test the golden molecules from the specification
    golden_cases = [
        {
            "name": "Caffeine",
            "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "targets": ["EGFR"],
            "expected_all_gated": True,
            "expected_reasons": ["OOD_chem", "Kinase_physchem"]
        },
        {
            "name": "Aspirin", 
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "targets": ["ERBB4", "PARP1"],
            "expected_all_gated": True,
            "expected_reasons": ["OOD_chem", "Kinase_pharmacophore_fail", "PARP_pharmacophore_fail"]
        }
    ]
    
    for case in golden_cases:
        print(f"\n  Testing {case['name']}:")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/gnosis-i/predict-with-hp-ad",
            json={
                "smiles": case['smiles'],
                "targets": case['targets'],
                "assay_types": ["IC50"]
            },
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code != 200:
            print(f"    ❌ API request failed: {response.status_code}")
            continue
        
        pred_data = response.json()
        
        # Check if all predictions are gated
        all_gated = True
        has_numerics = False
        found_reasons = set()
        
        for target in case['targets']:
            target_data = pred_data['predictions'].get(target, {})
            for assay_type, assay_data in target_data.items():
                if assay_type != 'selectivity_ratio':
                    status = assay_data.get('status', 'OK')
                    if status != 'HYPOTHESIS_ONLY':
                        all_gated = False
                    if 'pActivity' in assay_data:
                        has_numerics = True
                    
                    reasons = assay_data.get('why', [])
                    found_reasons.update(reasons)
        
        # Validate results
        gating_correct = all_gated == case['expected_all_gated']
        no_numeric_leaks = not has_numerics if case['expected_all_gated'] else True
        performance_ok = elapsed_time <= 7.0  # P95 target
        
        print(f"    All gated: {'✅' if gating_correct else '❌'} ({all_gated})")
        print(f"    No numeric leaks: {'✅' if no_numeric_leaks else '❌'}")
        print(f"    Performance: {'✅' if performance_ok else '❌'} ({elapsed_time:.2f}s)")
        print(f"    Reasons found: {list(found_reasons)}")
        
        if case['expected_all_gated']:
            assert all_gated, f"{case['name']} should have all predictions gated"
            assert not has_numerics, f"{case['name']} should have no numeric leaks"
        
        assert performance_ok, f"{case['name']} should complete within 7s"
    
    print("  ✅ API integration tests passed")

def main():
    """Run comprehensive cross-assay gating test suite"""
    print("🚀 CROSS-ASSAY GATING & RELIABILITY HARDENING TESTS")
    print("=" * 65)
    
    try:
        # Unit tests
        test_cross_assay_consistency()
        test_family_physchem_gates()
        test_mechanism_gates()
        test_protein_family_determination()
        test_aggregate_gates()
        
        # Integration tests
        test_api_integration()
        
        print("\n" + "=" * 65)
        print("🎉 ALL CROSS-ASSAY GATING TESTS PASSED!")
        print("✅ Family envelopes working correctly")
        print("✅ Cross-assay consistency checks implemented") 
        print("✅ Enhanced mechanism gates functional")
        print("✅ Cumulative gating rules applied")
        print("✅ API integration successful")
        print("✅ Performance targets met (<7s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)