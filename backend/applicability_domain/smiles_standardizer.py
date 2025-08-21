"""
SMILES Standardization Module
Implements robust SMILES standardization using RDKit
"""
from rdkit import Chem
from rdkit.Chem import SaltRemover, rdMolStandardize
import logging

logger = logging.getLogger(__name__)

class SMILESStandardizer:
    """Standardizes SMILES strings for consistent processing"""
    
    def __init__(self):
        # Initialize salt remover
        self.salt_remover = SaltRemover.SaltRemover()
        
        # Initialize standardizer components
        self.normalizer = rdMolStandardize.Normalizer()
        self.reionizer = rdMolStandardize.Reionizer()
        
    def standardize_smiles(self, smiles: str) -> str:
        """
        Standardize SMILES string using RDKit
        
        Args:
            smiles: Raw SMILES string
            
        Returns:
            Standardized canonical SMILES with stereochemistry, or None if invalid
        """
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Step 1: Remove salts - keep largest fragment
            mol = self.salt_remover.StripMol(mol)
            if mol is None:
                return None
            
            # Step 2: Normalize (standardize functional groups, charges, etc.)
            mol = self.normalizer.normalize(mol)
            if mol is None:
                return None
            
            # Step 3: Reionize (standardize protonation states)
            mol = self.reionizer.reionize(mol)
            if mol is None:
                return None
            
            # Step 4: Generate canonical SMILES with stereochemistry
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            
            return canonical_smiles
            
        except Exception as e:
            logger.warning(f"Failed to standardize SMILES '{smiles}': {e}")
            return None
    
    def validate_smiles(self, smiles: str) -> bool:
        """Check if SMILES string is valid"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

# Global standardizer instance
_standardizer = None

def get_standardizer() -> SMILESStandardizer:
    """Get global standardizer instance"""
    global _standardizer
    if _standardizer is None:
        _standardizer = SMILESStandardizer()
    return _standardizer

def standardize_smiles(smiles: str) -> str:
    """Convenience function for SMILES standardization"""
    return get_standardizer().standardize_smiles(smiles)