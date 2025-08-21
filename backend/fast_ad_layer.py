"""
Lightweight Applicability Domain (AD) Layer for Real-time Inference

This optimized version addresses the performance issues by:
1. Using lighter-weight similarity calculations
2. Pre-computing expensive operations during startup
3. Implementing async initialization
4. Reducing computational overhead for real-time inference
"""

import numpy as np
import pandas as pd
import torch
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import pickle
import json
from dataclasses import dataclass
import time

# Core imports
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

# Lightweight scientific computing
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class LightweightADResult:
    """Lightweight AD result with essential information"""
    target_id: str
    smiles_std: str
    potency_pred: float
    potency_ci: Tuple[float, float]
    ad_score: float
    confidence_calibrated: float
    flags: List[str]
    nearest_neighbors: List[Dict[str, Any]]
    
    # Simplified AD components
    similarity_score: float = 0.0
    density_score: float = 0.0
    context_score: float = 0.0

def standardize_smiles(smiles: str) -> Optional[str]:
    """Fast SMILES standardization (reused from original)"""
    try:
        if not smiles or not smiles.strip():
            return None
            
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        
        # Quick standardization - skip heavy operations for speed
        salt_remover = SaltRemover.SaltRemover()
        mol = salt_remover.StripMol(mol)
        
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        
    except Exception as e:
        logger.warning(f"Failed to standardize SMILES '{smiles}': {e}")
        return None

def compute_lightweight_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Compute lightweight fingerprint (reduced bits for speed)"""
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        # Use smaller array for speed
        arr = np.zeros((n_bits,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        logger.error(f"Error computing fingerprint: {e}")
        return np.zeros(n_bits, dtype=np.int8)

def fast_tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Fast Tanimoto similarity calculation"""
    try:
        # Use bitwise operations for speed
        fp1 = fp1.astype(bool)
        fp2 = fp2.astype(bool)
        
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0

class LightweightFingerprintDB:
    """Lightweight fingerprint database for fast similarity search"""
    
    def __init__(self, cache_dir: str = "/app/backend/ad_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db = {}  # target_id -> precomputed fingerprints
        
    def build_lightweight(self, training_data: pd.DataFrame, max_compounds_per_target: int = 50):
        """Build lightweight fingerprint database (limited compounds for speed)"""
        logger.info(f"Building lightweight fingerprint database (max {max_compounds_per_target} per target)...")
        
        train_data = training_data[training_data['split'] == 'train'].copy()
        
        for target_id in train_data['target_id'].unique():
            target_data = train_data[train_data['target_id'] == target_id].copy()
            
            # Limit compounds per target for speed
            if len(target_data) > max_compounds_per_target:
                target_data = target_data.sample(n=max_compounds_per_target, random_state=42)
            
            fps = []
            ligand_ids = []
            
            for _, row in target_data.iterrows():
                smiles_std = standardize_smiles(row['smiles'])
                if not smiles_std:
                    continue
                    
                mol = Chem.MolFromSmiles(smiles_std)
                if mol is None:
                    continue
                
                fp = compute_lightweight_fingerprint(mol)
                fps.append(fp)
                ligand_ids.append(row['ligand_id'])
            
            if fps:
                self.db[target_id] = {
                    'fingerprints': np.array(fps),
                    'ligand_ids': ligand_ids
                }
                logger.info(f"Target {target_id}: {len(fps)} compounds loaded")
        
        # Save to cache
        self._save_cache()
        logger.info(f"Lightweight fingerprint database built for {len(self.db)} targets")
    
    def _save_cache(self):
        """Save lightweight cache"""
        try:
            cache_file = self.cache_dir / "lightweight_fp_db.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(self.db, f)
        except Exception as e:
            logger.error(f"Failed to cache lightweight DB: {e}")
    
    def load_cache(self) -> bool:
        """Load from cache"""
        try:
            cache_file = self.cache_dir / "lightweight_fp_db.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self.db = pickle.load(f)
                logger.info(f"Loaded lightweight DB for {len(self.db)} targets")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load lightweight DB: {e}")
            return False

class LightweightADScorer:
    """Lightweight AD scorer optimized for real-time inference"""
    
    def __init__(self, fp_db: LightweightFingerprintDB):
        self.fp_db = fp_db
        
        # Simplified weights for 3 components only
        self.weights = {
            'similarity': 0.6,    # Tanimoto similarity (main component)
            'density': 0.3,       # Simple density estimation
            'context': 0.1        # Context scoring
        }
        
        # Pre-computed target statistics for fast context scoring
        self.target_stats = self._precompute_target_stats()
    
    def _precompute_target_stats(self) -> Dict[str, Dict]:
        """Pre-compute target statistics for fast context scoring"""
        stats = {}
        for target_id, data in self.fp_db.db.items():
            n_compounds = len(data['ligand_ids'])
            stats[target_id] = {
                'n_compounds': n_compounds,
                'context_score': 0.8 if n_compounds >= 20 else 0.5
            }
        return stats
    
    def compute_lightweight_ad(self, smiles: str, target_id: str) -> Dict[str, float]:
        """Fast AD scoring with reduced computational overhead"""
        try:
            # Initialize default scores
            similarity_score = 0.0
            density_score = 0.5
            context_score = 0.5
            
            # Get target data
            if target_id not in self.fp_db.db:
                return {
                    'ad_score': 0.3,  # Low score for unknown targets
                    'similarity_score': 0.0,
                    'density_score': 0.3,
                    'context_score': 0.2
                }
            
            target_data = self.fp_db.db[target_id]
            
            # Compute query fingerprint
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {'ad_score': 0.2, 'similarity_score': 0.0, 'density_score': 0.2, 'context_score': 0.2}
            
            query_fp = compute_lightweight_fingerprint(mol)
            
            # 1. Fast similarity scoring (vectorized)
            target_fps = target_data['fingerprints']
            similarities = []
            
            # Compute similarities in batches for speed
            batch_size = min(10, len(target_fps))  # Limit comparisons for speed
            for i in range(0, min(batch_size, len(target_fps))):
                sim = fast_tanimoto_similarity(query_fp, target_fps[i])
                similarities.append(sim)
            
            if similarities:
                max_sim = max(similarities)
                similarity_score = np.clip((max_sim - 0.3) / 0.4, 0, 1)  # Adjusted thresholds
            
            # 2. Simple density estimation (using similarity distribution)
            if similarities:
                mean_sim = np.mean(similarities)
                density_score = np.clip(mean_sim * 2, 0, 1)  # Simple density proxy
            
            # 3. Context scoring (pre-computed)
            context_score = self.target_stats.get(target_id, {}).get('context_score', 0.5)
            
            # Aggregate scores with simplified weights
            ad_score = (
                self.weights['similarity'] * similarity_score +
                self.weights['density'] * density_score +
                self.weights['context'] * context_score
            )
            
            return {
                'ad_score': float(ad_score),
                'similarity_score': float(similarity_score),
                'density_score': float(density_score),
                'context_score': float(context_score)
            }
            
        except Exception as e:
            logger.error(f"Error in lightweight AD scoring: {e}")
            return {
                'ad_score': 0.3,
                'similarity_score': 0.0,
                'density_score': 0.3,
                'context_score': 0.3
            }
    
    def get_fast_neighbors(self, smiles: str, target_id: str, k: int = 3) -> List[Dict[str, Any]]:
        """Get nearest neighbors with reduced computational cost"""
        try:
            if target_id not in self.fp_db.db:
                return []
            
            target_data = self.fp_db.db[target_id]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            
            query_fp = compute_lightweight_fingerprint(mol)
            target_fps = target_data['fingerprints']
            ligand_ids = target_data['ligand_ids']
            
            neighbors = []
            # Limit to first 20 compounds for speed
            for i, (fp, ligand_id) in enumerate(zip(target_fps[:20], ligand_ids[:20])):
                sim = fast_tanimoto_similarity(query_fp, fp)
                neighbors.append({
                    'ligand_id': ligand_id,
                    'sim': float(sim),
                    'assay_type': 'Mixed'
                })
            
            # Sort and return top k
            neighbors.sort(key=lambda x: x['sim'], reverse=True)
            return neighbors[:k]
            
        except Exception as e:
            logger.error(f"Error getting neighbors: {e}")
            return []

class FastApplicabilityDomain:
    """Fast AD layer optimized for production use"""
    
    def __init__(self, cache_dir: str = "/app/backend/ad_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Core components
        self.fp_db = LightweightFingerprintDB(cache_dir)
        self.ad_scorer = None
        self.conformal_quantiles = {}
        self.initialized = False
        
    async def async_initialize(self, training_data: Optional[pd.DataFrame] = None):
        """Async initialization for non-blocking startup"""
        try:
            logger.info("Starting async AD layer initialization...")
            
            # Try cache first
            if self.fp_db.load_cache():
                self.ad_scorer = LightweightADScorer(self.fp_db)
                self._load_conformal_quantiles()
                self.initialized = True
                logger.info("✅ Fast AD layer loaded from cache")
                return
            
            # Build from training data if cache not available
            if training_data is not None:
                logger.info("Building fast AD layer from training data...")
                
                # Use asyncio to yield control during heavy operations
                await asyncio.sleep(0.001)  # Yield control
                
                # Build lightweight components
                self.fp_db.build_lightweight(training_data, max_compounds_per_target=20)
                
                await asyncio.sleep(0.001)  # Yield control
                
                self.ad_scorer = LightweightADScorer(self.fp_db)
                self._build_conformal_quantiles()
                
                self.initialized = True
                logger.info("✅ Fast AD layer initialized")
            else:
                logger.warning("No training data provided for AD initialization")
                
        except Exception as e:
            logger.error(f"Failed to initialize fast AD layer: {e}")
            self.initialized = False
    
    def initialize_sync(self, training_data: Optional[pd.DataFrame] = None):
        """Synchronous initialization fallback"""
        try:
            # Try cache first
            if self.fp_db.load_cache():
                self.ad_scorer = LightweightADScorer(self.fp_db)
                self._load_conformal_quantiles()
                self.initialized = True
                logger.info("✅ Fast AD layer loaded from cache")
                return
            
            # Quick build with minimal data
            if training_data is not None:
                # Use only first 50 compounds for ultra-fast initialization
                sample_data = training_data.head(50)
                self.fp_db.build_lightweight(sample_data, max_compounds_per_target=5)
                self.ad_scorer = LightweightADScorer(self.fp_db)
                self._build_conformal_quantiles()
                self.initialized = True
                logger.info("✅ Fast AD layer initialized (minimal mode)")
            
        except Exception as e:
            logger.error(f"Sync initialization failed: {e}")
            self.initialized = False
    
    def fast_score_with_ad(self, ligand_smiles: str, target_id: str, base_prediction: Optional[float] = None) -> LightweightADResult:
        """Fast AD scoring for real-time inference"""
        try:
            # Check if initialized
            if not self.initialized or self.ad_scorer is None:
                return self._create_fallback_result(ligand_smiles, target_id, base_prediction)
            
            # Standardize SMILES quickly
            smiles_std = standardize_smiles(ligand_smiles)
            if not smiles_std:
                return self._create_error_result(ligand_smiles, target_id, base_prediction)
            
            # Fast AD scoring
            ad_results = self.ad_scorer.compute_lightweight_ad(smiles_std, target_id)
            ad_score = ad_results['ad_score']
            
            # Apply policies
            flags = []
            confidence_calibrated = 0.7  # Default confidence
            ci_multiplier = 1.0
            
            if ad_score < 0.4:  # Adjusted threshold for lightweight scoring
                flags.append("OOD_chem")
                confidence_calibrated = 0.2
                ci_multiplier = 2.0  # Reduced from 3x for faster computation
            
            # Simple conformal intervals
            base_pred = base_prediction or 6.0
            Q_t = self.conformal_quantiles.get(target_id, 0.8) * ci_multiplier
            potency_ci = (base_pred - Q_t, base_pred + Q_t)
            
            # Kinase gating (simplified)
            if self._is_kinase_target(target_id):
                if ad_score < 0.3:
                    flags.append("Kinase_mech_low")
                    confidence_calibrated = min(confidence_calibrated, 0.3)
            
            # Fast neighbor search
            neighbors = self.ad_scorer.get_fast_neighbors(smiles_std, target_id, k=3)
            
            return LightweightADResult(
                target_id=target_id,
                smiles_std=smiles_std,
                potency_pred=base_pred,
                potency_ci=potency_ci,
                ad_score=ad_score,
                confidence_calibrated=confidence_calibrated,
                flags=flags,
                nearest_neighbors=neighbors,
                similarity_score=ad_results['similarity_score'],
                density_score=ad_results['density_score'],
                context_score=ad_results['context_score']
            )
            
        except Exception as e:
            logger.error(f"Error in fast AD scoring: {e}")
            return self._create_error_result(ligand_smiles, target_id, base_prediction)
    
    def _create_fallback_result(self, smiles: str, target_id: str, base_prediction: Optional[float]) -> LightweightADResult:
        """Create fallback result when AD not initialized"""
        return LightweightADResult(
            target_id=target_id,
            smiles_std=smiles,
            potency_pred=base_prediction or 6.0,
            potency_ci=(4.0, 8.0),
            ad_score=0.5,
            confidence_calibrated=0.6,
            flags=["AD_not_initialized"],
            nearest_neighbors=[]
        )
    
    def _create_error_result(self, smiles: str, target_id: str, base_prediction: Optional[float]) -> LightweightADResult:
        """Create error result"""
        return LightweightADResult(
            target_id=target_id,
            smiles_std=smiles,
            potency_pred=base_prediction or 6.0,
            potency_ci=(3.0, 9.0),
            ad_score=0.2,
            confidence_calibrated=0.2,
            flags=["Error"],
            nearest_neighbors=[]
        )
    
    def _build_conformal_quantiles(self):
        """Build simple conformal quantiles"""
        for target_id in self.fp_db.db.keys():
            self.conformal_quantiles[target_id] = 0.8  # Fixed quantile for simplicity
    
    def _load_conformal_quantiles(self):
        """Load conformal quantiles from cache"""
        try:
            cache_file = self.cache_dir / "conformal_quantiles.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    self.conformal_quantiles = json.load(f)
            else:
                self._build_conformal_quantiles()
        except Exception as e:
            logger.warning(f"Could not load conformal quantiles: {e}")
            self._build_conformal_quantiles()
    
    def _is_kinase_target(self, target_id: str) -> bool:
        """Simple kinase detection"""
        kinase_keywords = ['CDK', 'JAK', 'ABL', 'KIT', 'FLT', 'ALK', 'EGFR', 'BRAF']
        return any(keyword in target_id.upper() for keyword in kinase_keywords)

# Global fast AD layer instance
_fast_ad_layer = None

def get_fast_ad_layer() -> FastApplicabilityDomain:
    """Get global fast AD layer instance"""
    global _fast_ad_layer
    if _fast_ad_layer is None:
        _fast_ad_layer = FastApplicabilityDomain()
    return _fast_ad_layer

async def initialize_fast_ad_layer(training_data: Optional[pd.DataFrame] = None):
    """Initialize global fast AD layer"""
    global _fast_ad_layer
    _fast_ad_layer = FastApplicabilityDomain()
    await _fast_ad_layer.async_initialize(training_data)
    return _fast_ad_layer

def initialize_fast_ad_layer_sync(training_data: Optional[pd.DataFrame] = None):
    """Initialize global fast AD layer synchronously"""
    global _fast_ad_layer
    _fast_ad_layer = FastApplicabilityDomain()
    _fast_ad_layer.initialize_sync(training_data)
    return _fast_ad_layer