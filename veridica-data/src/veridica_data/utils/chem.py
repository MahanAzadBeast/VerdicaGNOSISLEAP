"""
Chemical Structure Utilities
SMILES canonicalization and molecular descriptor computation using RDKit
"""

from typing import Dict, Optional, Tuple
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


def canonicalize(smiles: str) -> Tuple[Optional[str], Optional[str], Optional[str], Dict]:
    """
    Canonicalize SMILES and compute molecular descriptors
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Tuple of (canonical_smiles, inchi, inchikey, descriptors_dict)
    """
    if not RDKIT_AVAILABLE:
        logger.error("RDKit not available - cannot canonicalize SMILES")
        return None, None, None, {}
    
    if not smiles or not isinstance(smiles, str):
        return None, None, None, {}
    
    try:
        # Parse molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None, None, None, {}
        
        # Remove salts and neutralize
        mol = Chem.rdMolStandardize.StandardizeMol(mol)
        
        # Generate canonical SMILES
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        
        # Generate InChI and InChIKey
        inchi = Chem.MolToInchi(mol)
        inchikey = Chem.InchiToInchiKey(inchi)
        
        # Compute molecular descriptors
        descriptors = {
            "mol_molecular_weight": Descriptors.MolWt(mol),
            "mol_logp": Crippen.MolLogP(mol),
            "mol_tpsa": Descriptors.TPSA(mol),
            "mol_num_hbd": Descriptors.NumHDonors(mol),
            "mol_num_hba": Descriptors.NumHAcceptors(mol),
            "mol_num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "mol_num_heavy_atoms": Descriptors.HeavyAtomCount(mol),
            "mol_num_rings": rdMolDescriptors.CalcNumRings(mol),
            "mol_num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "mol_formal_charge": Chem.rdmolops.GetFormalCharge(mol),
            "mol_num_heteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
            "mol_fraction_csp3": rdMolDescriptors.CalcFractionCsp3(mol)
        }
        
        return canonical_smiles, inchi, inchikey, descriptors
        
    except Exception as e:
        logger.error(f"Error canonicalizing SMILES {smiles}: {e}")
        return None, None, None, {}


def validate_smiles(smiles: str) -> bool:
    """
    Validate SMILES string format and parseability
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not RDKIT_AVAILABLE:
        # Basic validation without RDKit
        return _basic_smiles_validation(smiles)
    
    if not smiles or not isinstance(smiles, str):
        return False
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def _basic_smiles_validation(smiles: str) -> bool:
    """
    Basic SMILES validation without RDKit
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if passes basic validation
    """
    if not smiles or not isinstance(smiles, str):
        return False
    
    # Length check
    if len(smiles) < 3 or len(smiles) > 1000:
        return False
    
    # Valid characters
    valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@+\\-[]()=#$:/.%\\\\')
    if not all(c in valid_chars for c in smiles):
        return False
    
    # Bracket balancing
    brackets = {'(': ')', '[': ']'}
    stack = []
    for char in smiles:
        if char in brackets:
            stack.append(char)
        elif char in brackets.values():
            if not stack:
                return False
            expected = brackets.get(stack.pop())
            if expected != char:
                return False
    
    return len(stack) == 0


def compute_molecular_descriptors(smiles: str) -> Dict:
    """
    Compute molecular descriptors for a SMILES string
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of molecular descriptors
    """
    if not RDKIT_AVAILABLE:
        logger.warning("RDKit not available - cannot compute descriptors")
        return {}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        # Standardize molecule
        mol = Chem.rdMolStandardize.StandardizeMol(mol)
        
        # Compute comprehensive descriptor set
        descriptors = {
            # Basic properties
            "mol_molecular_weight": Descriptors.MolWt(mol),
            "mol_logp": Crippen.MolLogP(mol),
            "mol_tpsa": Descriptors.TPSA(mol),
            
            # Hydrogen bonding
            "mol_num_hbd": Descriptors.NumHDonors(mol),
            "mol_num_hba": Descriptors.NumHAcceptors(mol),
            
            # Structural features
            "mol_num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "mol_num_heavy_atoms": Descriptors.HeavyAtomCount(mol),
            "mol_num_rings": rdMolDescriptors.CalcNumRings(mol),
            "mol_num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "mol_num_heteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
            
            # Electronic properties
            "mol_formal_charge": Chem.rdmolops.GetFormalCharge(mol),
            "mol_fraction_csp3": rdMolDescriptors.CalcFractionCsp3(mol),
            
            # Complexity measures
            "mol_bertz_complexity": rdMolDescriptors.BertzCT(mol),
            "mol_lipinski_violations": Descriptors.NumLipinskiHBD(mol) + Descriptors.NumLipinskiHBA(mol),
        }
        
        return descriptors
        
    except Exception as e:
        logger.error(f"Error computing descriptors for {smiles}: {e}")
        return {}


def neutralize_molecule(smiles: str) -> Optional[str]:
    """
    Neutralize charged molecules
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Neutralized SMILES or None if error
    """
    if not RDKIT_AVAILABLE:
        return smiles  # Return as-is if RDKit not available
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Neutralize
        neutralized = Chem.rdMolStandardize.Normalize(mol)
        return Chem.MolToSmiles(neutralized, canonical=True)
        
    except Exception as e:
        logger.error(f"Error neutralizing {smiles}: {e}")
        return None


def remove_salts(smiles: str) -> Optional[str]:
    """
    Remove salts from SMILES string
    
    Args:
        smiles: Input SMILES with potential salts
        
    Returns:
        SMILES with salts removed or None if error
    """
    if not RDKIT_AVAILABLE:
        return smiles
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Remove salts (keep largest fragment)
        desalted = Chem.rdMolStandardize.FragmentParent(mol)
        return Chem.MolToSmiles(desalted, canonical=True)
        
    except Exception as e:
        logger.error(f"Error removing salts from {smiles}: {e}")
        return None