"""
Universal Family-Based Gating for All Targets
Applies biological plausibility rules even without target-specific training data
"""

import logging
import numpy as np
from typing import Dict, Any, List
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

logger = logging.getLogger(__name__)

class UniversalFamilyGating:
    """Universal gating based on target family and molecular properties"""
    
    @staticmethod
    def determine_target_family(target_id: str) -> str:
        """Determine target family for universal gating rules"""
        target_upper = target_id.upper()
        
        # Kinase family (most important for aspirin gating)
        if any(k in target_upper for k in ["CDK", "JAK", "ABL", "KIT", "FLT", "ALK", "EGFR", "BRAF", "ERBB", "SRC", "BTK", "TRK", "AURK", "PLK", "CHK", "WEE", "DYRK", "GSK", "MAPK", "PIK3", "AKT", "MTOR", "ATM", "ATR", "PARP"]):
            return "kinase"
        elif any(onco in target_upper for onco in ["MYC", "MYCN", "MYCL", "BCL2", "MCL1", "BCLXL", "MDM2", "HDM2"]):
            return "oncoprotein"
        elif any(ts in target_upper for ts in ["TP53", "P53", "RB1", "RB", "BRCA1", "BRCA2", "PTEN", "VHL", "APC"]):
            return "tumor_suppressor"
        else:
            return "other"
    
    @staticmethod
    def apply_family_gating(smiles: str, target_id: str, target_family: str, base_prediction: float, assay_type: str):
        """
        Apply universal family-based gating rules without requiring target-specific training data
        
        This implements the core Universal Gating logic for ALL targets
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return UniversalFamilyGating._create_gated_result(
                    target_id, ["Invalid_SMILES"], "Invalid SMILES structure"
                )
            
            reasons = []
            
            # Universal physicochemical checks by family
            mw = Descriptors.MolWt(mol)
            rings = mol.GetRingInfo().NumRings()
            logp = Crippen.MolLogP(mol)
            
            if target_family == "kinase":
                # Kinase family rules (universal for all kinases)
                
                # 1. Molecular weight check
                if mw < 300:  # Too small for kinase ATP pocket
                    reasons.append("Kinase_physchem_MW_low")
                
                # 2. Ring count check  
                if rings < 2:  # Insufficient complexity for kinase binding
                    reasons.append("Kinase_physchem_rings_low")
                
                # 3. Strong anion check (critical for aspirin)
                if UniversalFamilyGating._is_strongly_anionic(mol):
                    reasons.append("Kinase_physchem_strong_anion")
                
                # 4. Kinase pharmacophore check
                if not UniversalFamilyGating._passes_kinase_pharmacophore(mol):
                    reasons.append("Kinase_pharmacophore_fail")
                
                # 5. Tiny acid veto (specifically catches aspirin)
                if UniversalFamilyGating._is_tiny_acid(mol):
                    reasons.append("Tiny_acid_veto")
            
            elif target_family == "gpcr":
                # GPCR family rules
                if not (250 <= mw <= 600):
                    reasons.append("GPCR_physchem_MW_out")
                if logp < 1.5:
                    reasons.append("GPCR_physchem_logP_low")
            
            elif target_family == "parp":
                # PARP family rules  
                if not UniversalFamilyGating._passes_parp_pharmacophore(mol):
                    reasons.append("PARP_pharmacophore_fail")
            
            # Apply cumulative gating rules
            fail_count = len(reasons)
            
            if fail_count >= 2:  # ≥2 failures → gate prediction
                if fail_count >= 3:
                    reasons.append("Mechanistically_implausible")
                
                return UniversalFamilyGating._create_gated_result(
                    target_id, reasons, f"Biologically implausible for {target_family} family"
                )
            else:
                # Passes universal family gating
                return UniversalFamilyGating._create_passed_result(target_id, reasons)
                
        except Exception as e:
            logger.error(f"Universal family gating error: {e}")
            return UniversalFamilyGating._create_gated_result(
                target_id, ["Gating_calculation_error"], "Gating calculation failed"
            )
    
    @staticmethod
    def _is_strongly_anionic(mol) -> bool:
        """Check if compound is strongly anionic (like aspirin)"""
        try:
            # Check for carboxylic acid groups
            carboxylic_acid = mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)O"))
            sulfonic_acid = mol.HasSubstructMatch(Chem.MolFromSmarts("S(=O)(=O)O"))
            return carboxylic_acid or sulfonic_acid
        except:
            return False
    
    @staticmethod
    def _passes_kinase_pharmacophore(mol) -> bool:
        """Check basic kinase pharmacophore requirements"""
        try:
            # Basic kinase requirements: aromatic system + H-bond capability
            has_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
            has_nitrogen = any(atom.GetSymbol() == 'N' for atom in mol.GetAtoms())
            has_hba = Descriptors.NOCount(mol) > 0  # N or O atoms
            
            return has_aromatic and (has_nitrogen or has_hba)
        except:
            return False
    
    @staticmethod 
    def _passes_parp_pharmacophore(mol) -> bool:
        """Check PARP pharmacophore requirements"""
        try:
            # PARP requires amide or similar H-bond pattern
            has_amide = mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)N"))
            has_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
            return has_amide and has_aromatic
        except:
            return False
    
    @staticmethod
    def _is_tiny_acid(mol) -> bool:
        """Check for tiny acid compounds (like aspirin)"""
        try:
            mw = Descriptors.MolWt(mol)
            has_acid = mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)O"))
            return mw < 250 and has_acid
        except:
            return False
    
    @staticmethod
    def _create_gated_result(target_id: str, reasons: List[str], message: str):
        """Create a gated AD result"""
        class GatedADResult:
            def __init__(self):
                self.status = "HYPOTHESIS_ONLY"
                self.target_id = target_id
                self.message = message
                self.why = reasons
                self.evidence = {
                    "gating_type": "Universal_Family_Based",
                    "gate_failures": len(reasons)
                }
        return GatedADResult()
    
    @staticmethod
    def _create_passed_result(target_id: str, reasons: List[str]):
        """Create a passed AD result"""  
        class PassedADResult:
            def __init__(self):
                self.status = "OK"
                self.target_id = target_id
                self.message = "Passes universal family gating"
                self.why = reasons
                self.evidence = {
                    "gating_type": "Universal_Family_Based",
                    "gate_failures": len(reasons)
                }
        return PassedADResult()