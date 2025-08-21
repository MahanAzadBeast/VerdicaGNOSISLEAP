"""
Molecular Fingerprint Generation
ECFP4 fingerprints for AD calculations
"""
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FingerprintGenerator:
    """Generates molecular fingerprints for AD calculations"""
    
    def __init__(self, radius: int = 2, n_bits: int = 2048):
        """
        Args:
            radius: ECFP radius (2 = ECFP4)
            n_bits: Number of bits in fingerprint
        """
        self.radius = radius
        self.n_bits = n_bits
    
    def ecfp4_bits(self, mol) -> Optional[np.ndarray]:
        """
        Generate ECFP4 fingerprint as bit vector
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Binary fingerprint as numpy array, or None if failed
        """
        try:
            # Generate ECFP fingerprint
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, 
                radius=self.radius, 
                nBits=self.n_bits
            )
            
            # Convert to numpy array
            fp_array = np.zeros((self.n_bits,), dtype=np.uint8)
            for i in range(self.n_bits):
                fp_array[i] = fp.GetBit(i)
            
            return fp_array
            
        except Exception as e:
            logger.error(f"Failed to generate fingerprint: {e}")
            return None
    
    def ecfp4_from_smiles(self, smiles: str) -> Optional[np.ndarray]:
        """
        Generate ECFP4 fingerprint from SMILES string
        
        Args:
            smiles: SMILES string
            
        Returns:
            Binary fingerprint as numpy array, or None if failed
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return self.ecfp4_bits(mol)
        except:
            return None
    
    def tanimoto_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """
        Calculate Tanimoto similarity between two fingerprints
        
        Args:
            fp1, fp2: Binary fingerprints as numpy arrays
            
        Returns:
            Tanimoto similarity (0-1)
        """
        try:
            intersection = np.sum(fp1 & fp2)
            union = np.sum(fp1 | fp2)
            
            if union == 0:
                return 0.0
            
            return float(intersection) / float(union)
            
        except Exception as e:
            logger.error(f"Failed to calculate Tanimoto similarity: {e}")
            return 0.0
    
    def bulk_tanimoto_similarities(self, query_fp: np.ndarray, database_fps: np.ndarray) -> np.ndarray:
        """
        Calculate Tanimoto similarities between query and database fingerprints
        
        Args:
            query_fp: Query fingerprint (1D array)
            database_fps: Database fingerprints (2D array, each row is a fingerprint)
            
        Returns:
            Array of similarities
        """
        try:
            # Vectorized Tanimoto calculation
            intersections = np.sum(database_fps & query_fp, axis=1)
            unions = np.sum(database_fps | query_fp, axis=1)
            
            # Avoid division by zero
            similarities = np.divide(intersections, unions, 
                                   out=np.zeros_like(intersections, dtype=float), 
                                   where=unions!=0)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Failed to calculate bulk similarities: {e}")
            return np.zeros(len(database_fps))

# Global fingerprint generator
_fp_generator = None

def get_fingerprint_generator() -> FingerprintGenerator:
    """Get global fingerprint generator instance"""
    global _fp_generator
    if _fp_generator is None:
        _fp_generator = FingerprintGenerator()
    return _fp_generator

def ecfp4_from_smiles(smiles: str) -> Optional[np.ndarray]:
    """Convenience function for ECFP4 generation"""
    return get_fingerprint_generator().ecfp4_from_smiles(smiles)

def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Convenience function for Tanimoto similarity"""
    return get_fingerprint_generator().tanimoto_similarity(fp1, fp2)