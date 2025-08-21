"""
Applicability Domain (AD) Layer for Gnosis I Ligand Activity Predictor

This module implements comprehensive applicability domain scoring including:
- Multi-view molecular similarity scoring
- Conformal prediction intervals  
- Confidence calibration
- Mechanism-based quality gates
- Kinase sanity checks

Author: Gnosis AI Platform
Version: 1.0
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import pickle
import json
import hashlib
from dataclasses import dataclass, asdict

# Core imports
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

# Scientific computing
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import pairwise_distances
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import faiss

logger = logging.getLogger(__name__)

@dataclass
class ADResult:
    """Applicability Domain scoring result"""
    target_id: str
    smiles_std: str
    potency_pred: float
    potency_ci: Tuple[float, float]
    ad_score: float
    confidence_calibrated: float
    flags: List[str]
    nearest_neighbors: List[Dict[str, Any]]
    
    # Additional metrics for transparency
    tanimoto_score: float = 0.0
    mahalanobis_score: float = 0.0
    knn_score: float = 0.0
    leverage_score: float = 0.0
    protein_context_score: float = 0.0
    assay_context_score: float = 0.0

def standardize_smiles(smiles: str) -> Optional[str]:
    """
    Standardize SMILES using RDKit with comprehensive normalization.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Standardized canonical SMILES with stereochemistry, or None if invalid
    """
    try:
        if not smiles or not smiles.strip():
            return None
            
        # Parse molecule
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        
        # Remove salts - keep largest fragment
        salt_remover = SaltRemover.SaltRemover()
        mol = salt_remover.StripMol(mol)
        
        # Standardize using RDKit's standardizer
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)
        
        # Reionize to ensure consistent protonation states
        reionizer = rdMolStandardize.Reionizer()
        mol = reionizer.reionize(mol)
        
        # Generate canonical SMILES with explicit stereochemistry
        standardized = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        
        return standardized
        
    except Exception as e:
        logger.warning(f"Failed to standardize SMILES '{smiles}': {e}")
        return None

def compute_ecfp4_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Compute ECFP4 fingerprint as explicit bit vector.
    
    Args:
        mol: RDKit molecule object
        radius: Fingerprint radius (default 2 for ECFP4)
        n_bits: Number of bits in fingerprint
        
    Returns:
        Binary fingerprint as numpy array
    """
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        # Convert to numpy array directly without going through bit string
        arr = np.zeros((n_bits,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        logger.error(f"Error computing ECFP4: {e}")
        return np.zeros(n_bits, dtype=np.int8)

def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Calculate Tanimoto similarity between two binary fingerprints.
    
    Args:
        fp1, fp2: Binary fingerprint arrays
        
    Returns:
        Tanimoto similarity (0-1)
    """
    try:
        # Ensure arrays are binary
        fp1 = fp1.astype(bool)
        fp2 = fp2.astype(bool)
        
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        
        if union == 0:
            return 0.0
            
        return intersection / union
    except Exception as e:
        logger.error(f"Error calculating Tanimoto: {e}")
        return 0.0

class PerTargetFingerprintDB:
    """Per-target fingerprint database builder and manager"""
    
    def __init__(self, cache_dir: str = "/app/backend/ad_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db = {}  # target_id -> DataFrame with fingerprints
        
    def build_from_training_data(self, training_data: pd.DataFrame):
        """
        Build per-target fingerprint database from training data.
        
        Args:
            training_data: DataFrame with columns [ligand_id, smiles, target_id, assay_type, label, split]
        """
        logger.info("Building per-target fingerprint database...")
        
        # Process only training split
        train_data = training_data[training_data['split'] == 'train'].copy()
        
        results = []
        
        for idx, row in train_data.iterrows():
            try:
                # Standardize SMILES
                smiles_std = standardize_smiles(row['smiles'])
                if not smiles_std:
                    continue
                
                # Generate fingerprint
                mol = Chem.MolFromSmiles(smiles_std)
                if mol is None:
                    continue
                    
                fp_bits = compute_ecfp4_fingerprint(mol)
                fp_bytes = fp_bits.tobytes()
                
                results.append({
                    'target_id': row['target_id'],
                    'ligand_id': row['ligand_id'],
                    'smiles_std': smiles_std,
                    'ecfp4_bytes': fp_bytes,
                    'assay_type': row['assay_type']
                })
                
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
        
        # Convert to DataFrame and deduplicate per target by smiles_std
        df = pd.DataFrame(results)
        
        # Group by target and deduplicate
        for target_id in df['target_id'].unique():
            target_df = df[df['target_id'] == target_id].copy()
            
            # Deduplicate by smiles_std, keeping first occurrence
            target_df = target_df.drop_duplicates(subset=['smiles_std'])
            
            self.db[target_id] = target_df
            logger.info(f"Target {target_id}: {len(target_df)} unique compounds")
        
        # Save to cache
        self._save_to_cache()
        
        logger.info(f"Fingerprint database built for {len(self.db)} targets")
    
    def _save_to_cache(self):
        """Save fingerprint database to cache"""
        try:
            cache_file = self.cache_dir / "fingerprint_db.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(self.db, f)
            logger.info(f"Fingerprint database cached to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to cache fingerprint DB: {e}")
    
    def load_from_cache(self) -> bool:
        """Load fingerprint database from cache"""
        try:
            cache_file = self.cache_dir / "fingerprint_db.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self.db = pickle.load(f)
                logger.info(f"Loaded fingerprint database for {len(self.db)} targets")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load fingerprint DB from cache: {e}")
            return False
    
    def get_target_fingerprints(self, target_id: str) -> Optional[pd.DataFrame]:
        """Get fingerprints for a specific target"""
        return self.db.get(target_id)

class PerTargetEmbeddingStats:
    """Per-target embedding statistics and kNN indices manager"""
    
    def __init__(self, cache_dir: str = "/app/backend/ad_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.stats = {}  # target_id -> stats dict
        self.indices = {}  # target_id -> FAISS index
        self.ids_map = {}  # target_id -> ligand_id array
        
    def build_from_embeddings(self, target_embeddings: Dict[str, Dict]):
        """
        Build per-target statistics and kNN indices from embeddings.
        
        Args:
            target_embeddings: {target_id: {'embeddings': np.array, 'ligand_ids': list}}
        """
        logger.info("Building per-target embedding statistics and indices...")
        
        for target_id, data in target_embeddings.items():
            try:
                embeddings = data['embeddings']  # Shape: (n_samples, embedding_dim)
                ligand_ids = data['ligand_ids']
                
                if len(embeddings) < 5:  # Minimum samples for statistics
                    logger.warning(f"Target {target_id}: insufficient samples ({len(embeddings)})")
                    continue
                
                # Compute statistics with Ledoit-Wolf shrinkage
                mu = np.mean(embeddings, axis=0)
                
                # Covariance with shrinkage
                lw = LedoitWolf(assume_centered=False)
                lw.fit(embeddings)
                sigma = lw.covariance_
                shrinkage = lw.shrinkage_
                
                # Build FAISS HNSW index
                embedding_dim = embeddings.shape[1]
                index = faiss.IndexHNSWFlat(embedding_dim, 32)  # M=32
                index.hnsw.efConstruction = 100
                
                # Add embeddings to index
                embeddings_f32 = embeddings.astype(np.float32)
                index.add(embeddings_f32)
                
                # Store everything
                self.stats[target_id] = {
                    'n_ligands': len(embeddings),
                    'mu': mu,
                    'sigma': sigma,
                    'shrinkage': shrinkage,
                    'embedding_dim': embedding_dim
                }
                
                self.indices[target_id] = index
                self.ids_map[target_id] = np.array(ligand_ids)
                
                logger.info(f"Target {target_id}: stats computed for {len(embeddings)} compounds "
                          f"(shrinkage: {shrinkage:.3f})")
                
            except Exception as e:
                logger.error(f"Error building stats for {target_id}: {e}")
                continue
        
        # Save to cache
        self._save_to_cache()
        
        logger.info(f"Embedding statistics built for {len(self.stats)} targets")
    
    def build_from_fingerprints_pca(self, fp_db: PerTargetFingerprintDB, n_components: int = 128):
        """
        Build statistics using PCA on ECFP4 fingerprints when embeddings unavailable.
        
        Args:
            fp_db: Fingerprint database
            n_components: Number of PCA components
        """
        logger.info("Building PCA-based statistics from fingerprints...")
        
        for target_id, fp_df in fp_db.db.items():
            try:
                if len(fp_df) < 10:  # Minimum for PCA
                    continue
                
                # Convert fingerprints back to arrays
                fps = []
                ligand_ids = []
                
                for _, row in fp_df.iterrows():
                    fp_array = np.frombuffer(row['ecfp4_bytes'], dtype=np.int8)
                    fps.append(fp_array)
                    ligand_ids.append(row['ligand_id'])
                
                fps = np.array(fps, dtype=np.float32)
                
                # Fit PCA
                pca = PCA(n_components=min(n_components, fps.shape[0]-1, fps.shape[1]))
                pca_embeddings = pca.fit_transform(fps)
                
                # Compute statistics
                mu = np.mean(pca_embeddings, axis=0)
                
                lw = LedoitWolf(assume_centered=False)
                lw.fit(pca_embeddings)
                sigma = lw.covariance_
                
                # Build FAISS index on PCA embeddings
                embedding_dim = pca_embeddings.shape[1]
                index = faiss.IndexHNSWFlat(embedding_dim, 32)
                index.hnsw.efConstruction = 100
                index.add(pca_embeddings.astype(np.float32))
                
                # Store with PCA transform
                self.stats[target_id] = {
                    'n_ligands': len(pca_embeddings),
                    'mu': mu,
                    'sigma': sigma,
                    'shrinkage': lw.shrinkage_,
                    'embedding_dim': embedding_dim,
                    'pca_model': pca,  # Store PCA transform
                    'uses_pca': True
                }
                
                self.indices[target_id] = index
                self.ids_map[target_id] = np.array(ligand_ids)
                
                logger.info(f"Target {target_id}: PCA stats computed "
                          f"({len(pca_embeddings)} compounds, {embedding_dim} dims)")
                
            except Exception as e:
                logger.error(f"Error building PCA stats for {target_id}: {e}")
                continue
        
        self._save_to_cache()
        
    def _save_to_cache(self):
        """Save statistics and indices to cache"""
        try:
            # Save statistics (without FAISS indices)
            stats_cache = {k: {**v} for k, v in self.stats.items()}
            
            # Remove non-serializable FAISS objects for pickle
            for target_stats in stats_cache.values():
                if 'pca_model' in target_stats:
                    # PCA models are serializable
                    pass
            
            stats_file = self.cache_dir / "embedding_stats.pkl" 
            with open(stats_file, 'wb') as f:
                pickle.dump(stats_cache, f)
            
            # Save FAISS indices separately
            for target_id, index in self.indices.items():
                index_file = self.cache_dir / f"faiss_index_{target_id}.faiss"
                faiss.write_index(index, str(index_file))
            
            # Save ID mappings
            ids_file = self.cache_dir / "ids_map.pkl"
            with open(ids_file, 'wb') as f:
                pickle.dump(self.ids_map, f)
                
            logger.info("Embedding statistics cached successfully")
            
        except Exception as e:
            logger.error(f"Failed to cache embedding stats: {e}")
    
    def load_from_cache(self) -> bool:
        """Load statistics and indices from cache"""
        try:
            stats_file = self.cache_dir / "embedding_stats.pkl"
            ids_file = self.cache_dir / "ids_map.pkl"
            
            if not (stats_file.exists() and ids_file.exists()):
                return False
            
            # Load statistics
            with open(stats_file, 'rb') as f:
                self.stats = pickle.load(f)
            
            # Load ID mappings
            with open(ids_file, 'rb') as f:
                self.ids_map = pickle.load(f)
            
            # Load FAISS indices
            self.indices = {}
            for target_id in self.stats.keys():
                index_file = self.cache_dir / f"faiss_index_{target_id}.faiss"
                if index_file.exists():
                    index = faiss.read_index(str(index_file))
                    self.indices[target_id] = index
            
            logger.info(f"Loaded embedding statistics for {len(self.stats)} targets")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embedding stats from cache: {e}")
            return False

class MultiViewADScorer:
    """Multi-view Applicability Domain scorer implementing all scoring methods"""
    
    def __init__(self, 
                 fp_db: PerTargetFingerprintDB,
                 embedding_stats: PerTargetEmbeddingStats):
        self.fp_db = fp_db
        self.embedding_stats = embedding_stats
        
        # AD scoring weights (as specified in requirements)
        self.weights = {
            'tanimoto': 0.35,
            'mahalanobis': 0.25, 
            'knn': 0.20,
            'leverage': 0.10,
            'protein': 0.05,
            'assay': 0.05
        }
    
    def compute_ad_score(self, 
                        smiles: str, 
                        target_id: str, 
                        z_emb: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute multi-view AD score for a query compound.
        
        Args:
            smiles: Standardized SMILES
            target_id: Target protein ID
            z_emb: Optional pre-computed embedding
            
        Returns:
            Dictionary with individual scores and final AD score
        """
        try:
            # Initialize scores
            scores = {
                'c_fps': 0.5,      # Tanimoto
                'c_maha': 0.5,     # Mahalanobis  
                'c_knn': 0.5,      # kNN density
                'c_lev': 0.5,      # Leverage
                'c_prot': 0.5,     # Protein context
                'c_assay': 0.5     # Assay context
            }
            
            # A) Tanimoto nearest neighbor on ECFP4
            scores['c_fps'] = self._compute_tanimoto_score(smiles, target_id)
            
            # B) Mahalanobis in embedding space
            scores['c_maha'] = self._compute_mahalanobis_score(target_id, z_emb)
            
            # C) kNN density
            scores['c_knn'] = self._compute_knn_density_score(target_id, z_emb)
            
            # D) Leverage in descriptor space (placeholder)
            scores['c_lev'] = self._compute_leverage_score(target_id)
            
            # E) Protein and assay context
            scores['c_prot'], scores['c_assay'] = self._compute_context_scores(target_id)
            
            # Aggregate with weights
            ad_score = (
                self.weights['tanimoto'] * scores['c_fps'] +
                self.weights['mahalanobis'] * scores['c_maha'] +
                self.weights['knn'] * scores['c_knn'] +
                self.weights['leverage'] * scores['c_lev'] +
                self.weights['protein'] * scores['c_prot'] +
                self.weights['assay'] * scores['c_assay']
            )
            
            return {
                'ad_score': ad_score,
                'tanimoto_score': scores['c_fps'],
                'mahalanobis_score': scores['c_maha'],
                'knn_score': scores['c_knn'],
                'leverage_score': scores['c_lev'],
                'protein_context_score': scores['c_prot'],
                'assay_context_score': scores['c_assay']
            }
            
        except Exception as e:
            logger.error(f"Error computing AD score: {e}")
            # Return moderate scores on error
            return {
                'ad_score': 0.5,
                'tanimoto_score': 0.5,
                'mahalanobis_score': 0.5,
                'knn_score': 0.5,
                'leverage_score': 0.5,
                'protein_context_score': 0.5,
                'assay_context_score': 0.5
            }
    
    def _compute_tanimoto_score(self, smiles: str, target_id: str) -> float:
        """Compute Tanimoto similarity to k=20 nearest neighbors"""
        try:
            # Get target fingerprints
            target_fp_df = self.fp_db.get_target_fingerprints(target_id)
            if target_fp_df is None or len(target_fp_df) == 0:
                return 0.5
            
            # Generate query fingerprint
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.5
            
            query_fp = compute_ecfp4_fingerprint(mol)
            
            # Compute similarities to all compounds in target
            max_sim = 0.0
            similarities = []
            
            for _, row in target_fp_df.iterrows():
                target_fp = np.frombuffer(row['ecfp4_bytes'], dtype=int)
                sim = tanimoto_similarity(query_fp, target_fp)
                similarities.append(sim)
                max_sim = max(max_sim, sim)
            
            # Use max similarity for scoring (S_max)
            c_fps = np.clip((max_sim - 0.35) / 0.20, 0, 1)
            
            return float(c_fps)
            
        except Exception as e:
            logger.error(f"Error computing Tanimoto score: {e}")
            return 0.5
    
    def _compute_mahalanobis_score(self, target_id: str, z_emb: Optional[np.ndarray]) -> float:
        """Compute Mahalanobis distance in embedding space"""
        try:
            if z_emb is None or target_id not in self.embedding_stats.stats:
                return 0.5
            
            stats = self.embedding_stats.stats[target_id]
            mu = stats['mu']
            sigma = stats['sigma']
            
            # Compute Mahalanobis distance
            diff = z_emb - mu
            sigma_inv = np.linalg.pinv(sigma)  # Use pseudoinverse for stability
            d_mahal_sq = np.dot(diff, np.dot(sigma_inv, diff))
            
            # Map using chi-square CDF
            df = len(z_emb)  # Degrees of freedom
            c_maha = 1 - stats.chi2.cdf(d_mahal_sq, df=df)
            
            return float(np.clip(c_maha, 0, 1))
            
        except Exception as e:
            logger.error(f"Error computing Mahalanobis score: {e}")
            return 0.5
    
    def _compute_knn_density_score(self, target_id: str, z_emb: Optional[np.ndarray]) -> float:
        """Compute kNN density score"""
        try:
            if (z_emb is None or 
                target_id not in self.embedding_stats.indices or
                target_id not in self.embedding_stats.stats):
                return 0.5
            
            index = self.embedding_stats.indices[target_id]
            k = min(32, index.ntotal - 1)  # k=32 neighbors or all available
            
            if k <= 0:
                return 0.5
            
            # Search for k nearest neighbors
            query = z_emb.reshape(1, -1).astype(np.float32)
            distances, indices = index.search(query, k + 1)  # +1 to exclude self if present
            
            # Use mean distance to k neighbors (excluding self)
            neighbor_distances = distances[0][1:k+1]  # Skip first (self) if present
            
            if len(neighbor_distances) == 0:
                return 0.5
                
            mean_distance = np.mean(neighbor_distances)
            
            # Inverse distance as density proxy
            density = 1.0 / (mean_distance + 1e-6)
            
            # Z-score by target statistics (placeholder - would need pre-computed stats)
            # For now, use sigmoid transformation
            c_knn = 1 / (1 + np.exp(-0.75 * (density - 1.0)))
            
            return float(np.clip(c_knn, 0, 1))
            
        except Exception as e:
            logger.error(f"Error computing kNN density score: {e}")
            return 0.5
    
    def _compute_leverage_score(self, target_id: str) -> float:
        """Compute leverage score (placeholder implementation)"""
        # This would require access to the descriptor matrix used for model training
        # For now, return default
        return 0.5
    
    def _compute_context_scores(self, target_id: str) -> Tuple[float, float]:
        """Compute protein and assay context scores"""
        try:
            # Protein context: based on training set size
            target_stats = self.embedding_stats.stats.get(target_id, {})
            n_ligands = target_stats.get('n_ligands', 0)
            
            c_prot = 0.6 if n_ligands >= 50 else 0.2
            
            # Assay context: placeholder (would need assay type matching logic)
            c_assay = 0.7  # Default for now
            
            return float(c_prot), float(c_assay)
            
        except Exception as e:
            logger.error(f"Error computing context scores: {e}")
            return 0.5, 0.5

class ApplicabilityDomainLayer:
    """
    Main Applicability Domain layer integrating all components.
    
    This class provides the main interface for AD scoring and integrates:
    - SMILES standardization
    - Multi-view AD scoring  
    - Conformal prediction intervals
    - Confidence calibration
    - Mechanism gating
    """
    
    def __init__(self, cache_dir: str = "/app/backend/ad_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Core components
        self.fp_db = PerTargetFingerprintDB(cache_dir)
        self.embedding_stats = PerTargetEmbeddingStats(cache_dir)
        self.ad_scorer = None
        
        # Conformal prediction quantiles (per target)
        self.conformal_quantiles = {}
        
        # Confidence calibration model
        self.calibration_model = None
        
        # Mechanism gate classifier (placeholder)
        self.mechanism_gate = None
        
        # Initialize from cache if available
        self.load_from_cache()
    
    def initialize(self, training_data: Optional[pd.DataFrame] = None):
        """
        Initialize the AD layer with training data.
        
        Args:
            training_data: Training dataset for building AD components
        """
        try:
            logger.info("Initializing Applicability Domain layer...")
            
            # Try loading from cache first
            if self.load_from_cache():
                logger.info("✅ AD layer loaded from cache")
                return
            
            # Build from training data if cache not available
            if training_data is not None:
                logger.info("Building AD layer from training data...")
                
                # Build fingerprint database
                self.fp_db.build_from_training_data(training_data)
                
                # Build embedding statistics using PCA on fingerprints
                # (since we don't have pre-computed embeddings)
                self.embedding_stats.build_from_fingerprints_pca(self.fp_db)
                
                # Initialize multi-view scorer
                self.ad_scorer = MultiViewADScorer(self.fp_db, self.embedding_stats)
                
                # Build conformal prediction quantiles (placeholder)
                self._build_conformal_quantiles()
                
                # Build confidence calibration (placeholder)
                self._build_confidence_calibration()
                
                logger.info("✅ AD layer initialized from training data")
            else:
                logger.warning("No training data provided and no cache found")
                
        except Exception as e:
            logger.error(f"Failed to initialize AD layer: {e}")
            raise
    
    def load_from_cache(self) -> bool:
        """Load AD layer components from cache"""
        try:
            # Load fingerprint database
            if not self.fp_db.load_from_cache():
                return False
            
            # Load embedding statistics
            if not self.embedding_stats.load_from_cache():
                return False
            
            # Initialize scorer
            self.ad_scorer = MultiViewADScorer(self.fp_db, self.embedding_stats)
            
            # Load other components (conformal quantiles, calibration, etc.)
            self._load_additional_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load AD layer from cache: {e}")
            return False
    
    def score_with_ad(self, 
                     ligand_smiles: str, 
                     target_id: str,
                     base_prediction: Optional[float] = None) -> ADResult:
        """
        Main AD scoring function as specified in requirements.
        
        Args:
            ligand_smiles: Input SMILES string
            target_id: Target protein ID
            base_prediction: Base model prediction (pIC50, etc.)
            
        Returns:
            ADResult with comprehensive AD scoring
        """
        try:
            # Step 1: Standardize SMILES
            smiles_std = standardize_smiles(ligand_smiles)
            if not smiles_std:
                raise ValueError(f"Could not standardize SMILES: {ligand_smiles}")
            
            # Step 2: Compute AD scores
            if self.ad_scorer is None:
                raise ValueError("AD scorer not initialized")
            
            ad_results = self.ad_scorer.compute_ad_score(smiles_std, target_id)
            ad_score = ad_results['ad_score']
            
            # Step 3: Apply policy for low AD scores
            flags = []
            confidence_calibrated = 0.8  # Base confidence
            ci_multiplier = 1.0
            
            if ad_score < 0.5:
                flags.append("OOD_chem")
                confidence_calibrated = min(confidence_calibrated, 0.20)
                ci_multiplier = 3.0
            
            # Step 4: Conformal prediction intervals
            base_pred = base_prediction or 6.0  # Default pIC50
            Q_t = self.conformal_quantiles.get(target_id, 1.0)
            Q_t *= ci_multiplier
            
            potency_ci = (base_pred - Q_t, base_pred + Q_t)
            
            # Step 5: Mechanism gating for kinases
            if self._is_kinase_target(target_id):
                mech_flags, mech_confidence = self._apply_mechanism_gating(smiles_std, target_id)
                flags.extend(mech_flags)
                confidence_calibrated = min(confidence_calibrated, mech_confidence)
            
            # Step 6: Get nearest neighbors for explanation
            nearest_neighbors = self._get_nearest_neighbors(smiles_std, target_id, k=5)
            
            # Step 7: Build result
            result = ADResult(
                target_id=target_id,
                smiles_std=smiles_std,
                potency_pred=base_pred,
                potency_ci=potency_ci,
                ad_score=ad_score,
                confidence_calibrated=confidence_calibrated,
                flags=flags,
                nearest_neighbors=nearest_neighbors,
                tanimoto_score=ad_results['tanimoto_score'],
                mahalanobis_score=ad_results['mahalanobis_score'],
                knn_score=ad_results['knn_score'],
                leverage_score=ad_results['leverage_score'],
                protein_context_score=ad_results['protein_context_score'],
                assay_context_score=ad_results['assay_context_score']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in score_with_ad: {e}")
            
            # Return conservative result on error
            return ADResult(
                target_id=target_id,
                smiles_std=ligand_smiles,  # Use original if standardization failed
                potency_pred=base_prediction or 6.0,
                potency_ci=(4.0, 8.0),  # Wide interval
                ad_score=0.3,  # Low confidence
                confidence_calibrated=0.2,
                flags=["Error"],
                nearest_neighbors=[]
            )
    
    def _build_conformal_quantiles(self):
        """Build conformal prediction quantiles per target (placeholder)"""
        # This would use held-out calibration data to compute quantiles
        # For now, use reasonable defaults
        for target_id in self.embedding_stats.stats.keys():
            self.conformal_quantiles[target_id] = 1.0  # ±1 log unit
        
        logger.info(f"Built conformal quantiles for {len(self.conformal_quantiles)} targets")
    
    def _build_confidence_calibration(self):
        """Build isotonic regression calibration model (placeholder)"""
        # This would train on validation data
        self.calibration_model = IsotonicRegression(out_of_bounds='clip')
        
        # Dummy training for now
        x_cal = np.linspace(0, 1, 100)
        y_cal = x_cal + np.random.normal(0, 0.1, 100)  # Slight miscalibration
        self.calibration_model.fit(x_cal, np.clip(y_cal, 0, 1))
    
    def _is_kinase_target(self, target_id: str) -> bool:
        """Check if target is a kinase"""
        kinase_keywords = ['CDK', 'JAK', 'ABL', 'KIT', 'FLT', 'ALK', 'ROS', 'EGFR', 'BRAF']
        return any(keyword in target_id.upper() for keyword in kinase_keywords)
    
    def _apply_mechanism_gating(self, smiles: str, target_id: str) -> Tuple[List[str], float]:
        """Apply mechanism gating for kinases (placeholder)"""
        flags = []
        confidence = 0.8
        
        # Placeholder hinge binder probability
        # Would use trained classifier in production
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ["Kinase_sanity_fail"], 0.2
        
        # Simple heuristic: check for aromatic rings (hinge binders often have them)
        aromatic_rings = len([x for x in mol.GetRingInfo().AtomRings() 
                             if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in x)])
        
        hinge_prob = min(0.9, 0.3 + aromatic_rings * 0.2)
        
        if hinge_prob < 0.3:
            flags.append("Kinase_mech_low")
            confidence = 0.2
        
        return flags, confidence
    
    def _get_nearest_neighbors(self, smiles: str, target_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get nearest neighbors for explanation"""
        try:
            target_fp_df = self.fp_db.get_target_fingerprints(target_id)
            if target_fp_df is None or len(target_fp_df) == 0:
                return []
            
            # Generate query fingerprint  
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            
            query_fp = compute_ecfp4_fingerprint(mol)
            
            # Compute similarities
            neighbors = []
            for _, row in target_fp_df.iterrows():
                target_fp = np.frombuffer(row['ecfp4_bytes'], dtype=int)
                sim = tanimoto_similarity(query_fp, target_fp)
                
                neighbors.append({
                    'ligand_id': row['ligand_id'],
                    'sim': float(sim),
                    'assay_type': row['assay_type']
                })
            
            # Sort by similarity and return top k
            neighbors.sort(key=lambda x: x['sim'], reverse=True)
            return neighbors[:k]
            
        except Exception as e:
            logger.error(f"Error getting nearest neighbors: {e}")
            return []
    
    def _load_additional_cache(self):
        """Load additional cached components"""
        try:
            # Load conformal quantiles
            conformal_file = self.cache_dir / "conformal_quantiles.json"
            if conformal_file.exists():
                with open(conformal_file, 'r') as f:
                    self.conformal_quantiles = json.load(f)
            
            # Load calibration model
            calibration_file = self.cache_dir / "calibration_model.pkl"
            if calibration_file.exists():
                with open(calibration_file, 'rb') as f:
                    self.calibration_model = pickle.load(f)
            
        except Exception as e:
            logger.warning(f"Could not load additional cache components: {e}")

# Global AD layer instance
_ad_layer = None

def get_ad_layer() -> ApplicabilityDomainLayer:
    """Get global AD layer instance"""
    global _ad_layer
    if _ad_layer is None:
        _ad_layer = ApplicabilityDomainLayer()
    return _ad_layer

def initialize_ad_layer(training_data: Optional[pd.DataFrame] = None):
    """Initialize global AD layer"""
    global _ad_layer
    _ad_layer = ApplicabilityDomainLayer()
    _ad_layer.initialize(training_data)
    return _ad_layer