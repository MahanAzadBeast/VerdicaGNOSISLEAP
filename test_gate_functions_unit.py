#!/usr/bin/env python3
"""
Unit tests for gate functions as specified in the patch plan.
"""

import sys
sys.path.append('/app/backend')

from rdkit import Chem
from hp_ad_layer import (
    passes_kinase_pharmacophore_v3, 
    passes_parp1_pharmacophore_v3,
    physchem_implausible_for_atp_pocket,
    apply_gates_v2,
    neighbor_sanity,
    NEIGHBOR_SMAX_MIN,
    NEIGHBOR_MIN_COUNT_040
)

def test_kinase_pharmacophore():
    """passes_kinase_pharmacophore returns False for aspirin, True for hinge-binder"""
    print("Testing kinase pharmacophore...")
    
    # Aspirin - should fail
    aspirin_mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    aspirin_result = passes_kinase_pharmacophore_v3(aspirin_mol)
    print(f"  Aspirin kinase pharmacophore: {aspirin_result} (expected: False)")
    assert aspirin_result == False, "Aspirin should fail kinase pharmacophore check"
    
    # Gefitinib - should pass (has quinazoline + halogenated aniline)
    gefitinib_mol = Chem.MolFromSmiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1")
    gefitinib_result = passes_kinase_pharmacophore_v3(gefitinib_mol)
    print(f"  Gefitinib kinase pharmacophore: {gefitinib_result} (expected: True)")
    assert gefitinib_result == True, "Gefitinib should pass kinase pharmacophore check"
    
    print("  ✅ Kinase pharmacophore tests passed")

def test_parp1_pharmacophore():
    """passes_parp1_pharmacophore False for aspirin; True for nicotinamide analog; False for salicylate"""
    print("Testing PARP1 pharmacophore...")
    
    # Aspirin - should fail (negative pattern)
    aspirin_mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    aspirin_result = passes_parp1_pharmacophore_v3(aspirin_mol)
    print(f"  Aspirin PARP1 pharmacophore: {aspirin_result} (expected: False)")
    assert aspirin_result == False, "Aspirin should fail PARP1 pharmacophore (negative pattern)"
    
    # Simple benzamide - should pass
    benzamide_mol = Chem.MolFromSmiles("NC(=O)c1ccccc1")
    benzamide_result = passes_parp1_pharmacophore_v3(benzamide_mol)
    print(f"  Benzamide PARP1 pharmacophore: {benzamide_result} (expected: True)")
    assert benzamide_result == True, "Benzamide should pass PARP1 pharmacophore check"
    
    # Salicylic acid - should fail (negative pattern)
    salicylic_mol = Chem.MolFromSmiles("O=C(O)c1ccccc1O")
    salicylic_result = passes_parp1_pharmacophore_v3(salicylic_mol)
    print(f"  Salicylic acid PARP1 pharmacophore: {salicylic_result} (expected: False)")
    assert salicylic_result == False, "Salicylic acid should fail PARP1 pharmacophore (negative pattern)"
    
    print("  ✅ PARP1 pharmacophore tests passed")

def test_physchem_implausible():
    """physchem_implausible_for_ATP_pocket True for aspirin; False for gefitinib"""
    print("Testing physicochemical implausibility...")
    
    # Aspirin on kinase - should be implausible
    aspirin_mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    aspirin_result = physchem_implausible_for_atp_pocket(aspirin_mol, "EGFR")
    print(f"  Aspirin ATP pocket implausible: {aspirin_result} (expected: True)")
    assert aspirin_result == True, "Aspirin should be implausible for ATP pocket"
    
    # Gefitinib on kinase - should be plausible
    gefitinib_mol = Chem.MolFromSmiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1")
    gefitinib_result = physchem_implausible_for_atp_pocket(gefitinib_mol, "EGFR")
    print(f"  Gefitinib ATP pocket implausible: {gefitinib_result} (expected: False)")
    assert gefitinib_result == False, "Gefitinib should be plausible for ATP pocket"
    
    # Non-kinase target - should be plausible
    aspirin_nonkinase = physchem_implausible_for_atp_pocket(aspirin_mol, "SOME_OTHER_TARGET")
    print(f"  Aspirin non-kinase implausible: {aspirin_nonkinase} (expected: False)")
    assert aspirin_nonkinase == False, "Non-kinase targets should not trigger ATP pocket check"
    
    print("  ✅ Physicochemical implausibility tests passed")

def test_apply_gates():
    """apply_gates returns gated=True for aspirin on ERBB4; returns gated=False for gefitinib on EGFR"""
    print("Testing apply_gates function...")
    
    # Mock neighbor info for aspirin (insufficient)
    aspirin_nn = {
        "S_max": 0.2,
        "n_sim_ge_0_40_same_assay": 5,
        "assay_mismatch_possible": False
    }
    
    # Aspirin on ERBB4 - should be gated
    aspirin_mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    aspirin_gated, aspirin_hard, aspirin_reasons, aspirin_ev = apply_gates_v2(
        mol=aspirin_mol,
        target_id="ERBB4", 
        ad_score=0.3,  # Below 0.5 threshold
        mech_score=0.1,  # Below 0.25 threshold
        nn_info=aspirin_nn,
        assay_match=True
    )
    
    print(f"  Aspirin ERBB4 gated: {aspirin_gated} (expected: True)")
    print(f"  Aspirin gate reasons: {aspirin_reasons}")
    print(f"  Aspirin gate failures: {aspirin_ev.get('gate_failures', 0)}")
    assert aspirin_gated == True, "Aspirin should be gated on ERBB4"
    assert len(aspirin_reasons) >= 2, "Aspirin should have multiple gate failures"
    
    # Mock neighbor info for gefitinib (sufficient)
    gefitinib_nn = {
        "S_max": 0.8,
        "n_sim_ge_0_40_same_assay": 50,
        "assay_mismatch_possible": False
    }
    
    # Gefitinib on EGFR - should pass (if we had good training data)
    gefitinib_mol = Chem.MolFromSmiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1")
    gefitinib_gated, gefitinib_hard, gefitinib_reasons, gefitinib_ev = apply_gates_v2(
        mol=gefitinib_mol,
        target_id="EGFR",
        ad_score=0.7,  # Above 0.5 threshold
        mech_score=0.8,  # Above 0.25 threshold
        nn_info=gefitinib_nn,
        assay_match=True
    )
    
    print(f"  Gefitinib EGFR gated: {gefitinib_gated} (expected: False)")
    print(f"  Gefitinib gate reasons: {gefitinib_reasons}")
    assert gefitinib_gated == False, "Gefitinib should not be gated on EGFR with good neighbors"
    
    print("  ✅ Apply gates tests passed")

def test_neighbor_sanity():
    """Test neighbor sanity with hardened thresholds"""
    print("Testing neighbor sanity...")
    
    # Insufficient neighbors
    bad_nn = {"S_max": 0.3, "n_sim_ge_0_40_same_assay": 10}
    bad_ok, bad_ev = neighbor_sanity(bad_nn)
    print(f"  Bad neighbors OK: {bad_ok} (expected: False)")
    assert bad_ok == False, "Should fail with insufficient neighbors"
    
    # Sufficient neighbors
    good_nn = {"S_max": 0.6, "n_sim_ge_0_40_same_assay": 50}
    good_ok, good_ev = neighbor_sanity(good_nn)
    print(f"  Good neighbors OK: {good_ok} (expected: True)")
    assert good_ok == True, "Should pass with sufficient neighbors"
    
    # Check thresholds
    print(f"  S_max threshold: {NEIGHBOR_SMAX_MIN} (expected: 0.50)")
    print(f"  Neighbor count threshold: {NEIGHBOR_MIN_COUNT_040} (expected: 30)")
    assert NEIGHBOR_SMAX_MIN == 0.50, "S_max threshold should be 0.50"
    assert NEIGHBOR_MIN_COUNT_040 == 30, "Neighbor count threshold should be 30"
    
    print("  ✅ Neighbor sanity tests passed")

def main():
    """Run all unit tests"""
    print("🧪 Running Gate Function Unit Tests")
    print("=" * 50)
    
    try:
        test_kinase_pharmacophore()
        test_parp1_pharmacophore()
        test_physchem_implausible()
        test_neighbor_sanity()
        test_apply_gates()
        
        print("=" * 50)
        print("🎉 ALL UNIT TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)