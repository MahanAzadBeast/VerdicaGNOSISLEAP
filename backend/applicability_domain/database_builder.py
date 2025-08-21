"""
Fingerprint Database Builder
Builds per-target fingerprint database from training data
"""
import pandas as pd
import numpy as np
import os
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict

from .smiles_standardizer import standardize_smiles
from .fingerprints import get_fingerprint_generator
from rdkit import Chem

logger = logging.getLogger(__name__)

class FingerprintDatabaseBuilder:
    """Builds fingerprint databases for AD calculations"""
    
    def __init__(self, output_dir: str = "/app/backend/ad_data"):
        """
        Args:
            output_dir: Directory to store AD data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / "fpdb").mkdir(exist_ok=True)
        (self.output_dir / "stats").mkdir(exist_ok=True)
        (self.output_dir / "indices").mkdir(exist_ok=True)
        (self.output_dir / "ids").mkdir(exist_ok=True)
        
        self.fp_gen = get_fingerprint_generator()
    
    def build_fpdb_from_training_data(self, training_data_path: str) -> Dict[str, int]:
        """
        Build fingerprint database from training CSV/JSON data
        
        Args:
            training_data_path: Path to training data file
            
        Returns:
            Dictionary with target counts
        """
        logger.info(f"Building fingerprint database from {training_data_path}")
        
        # Load training data
        if training_data_path.endswith('.csv'):
            df = pd.read_csv(training_data_path)
        elif training_data_path.endswith('.json'):
            df = pd.read_json(training_data_path)
        else:
            raise ValueError("Training data must be CSV or JSON format")
        
        # Expected columns: [ligand_id, smiles, target_id, assay_type, label, split]
        required_cols = ['ligand_id', 'smiles', 'target_id', 'assay_type', 'label', 'split']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter to training split only
        train_df = df[df['split'] == 'train'].copy()
        logger.info(f"Using {len(train_df)} training records")
        
        # Process records
        processed_records = []
        target_counts = defaultdict(int)
        standardization_failures = 0
        fingerprint_failures = 0
        
        for _, row in train_df.iterrows():
            # Standardize SMILES
            smiles_std = standardize_smiles(row['smiles'])
            if not smiles_std:
                standardization_failures += 1
                continue
            
            # Generate fingerprint
            mol = Chem.MolFromSmiles(smiles_std)
            if mol is None:
                fingerprint_failures += 1
                continue
                
            fp = self.fp_gen.ecfp4_bits(mol)
            if fp is None:
                fingerprint_failures += 1
                continue
            
            # Store record
            record = {
                'target_id': row['target_id'],
                'ligand_id': row['ligand_id'],
                'smiles_std': smiles_std,
                'ecfp4_bytes': fp.tobytes(),
                'assay_type': row['assay_type'],
                'label': row['label']
            }
            
            processed_records.append(record)
            target_counts[row['target_id']] += 1
        
        logger.info(f"Processed {len(processed_records)} records")
        logger.info(f"Standardization failures: {standardization_failures}")
        logger.info(f"Fingerprint failures: {fingerprint_failures}")
        
        # Convert to DataFrame and deduplicate per target
        fpdb_df = pd.DataFrame(processed_records)
        
        # Deduplicate by (target_id, smiles_std), keeping highest quality assay
        assay_priority = {'IC50': 3, 'Ki': 2, 'EC50': 1, 'Kd': 1}  # IC50 highest priority
        fpdb_df['assay_priority'] = fpdb_df['assay_type'].map(assay_priority).fillna(0)
        
        # Sort and deduplicate
        fpdb_df = fpdb_df.sort_values(['target_id', 'smiles_std', 'assay_priority'], ascending=[True, True, False])
        fpdb_df_dedup = fpdb_df.drop_duplicates(subset=['target_id', 'smiles_std'], keep='first')
        
        logger.info(f"After deduplication: {len(fpdb_df_dedup)} unique target-ligand pairs")
        
        # Save as parquet
        output_path = self.output_dir / "fpdb" / "per_target_fp_db.parquet"
        fpdb_df_dedup.drop('assay_priority', axis=1).to_parquet(output_path, index=False)
        
        logger.info(f"Fingerprint database saved to {output_path}")
        
        # Update target counts after deduplication
        final_target_counts = fpdb_df_dedup.groupby('target_id').size().to_dict()
        
        return final_target_counts
    
    def build_per_target_stats(self, target_counts: Dict[str, int]) -> Dict[str, Dict]:
        """
        Build per-target statistics and indices
        
        Args:
            target_counts: Dictionary of target counts
            
        Returns:
            Dictionary of per-target statistics
        """
        logger.info("Building per-target statistics and indices")
        
        # Load fingerprint database
        fpdb_path = self.output_dir / "fpdb" / "per_target_fp_db.parquet"
        fpdb_df = pd.read_parquet(fpdb_path)
        
        target_stats = {}
        
        for target_id, count in target_counts.items():
            logger.info(f"Processing target {target_id} with {count} ligands")
            
            # Get target data
            target_data = fpdb_df[fpdb_df['target_id'] == target_id].copy()
            
            if len(target_data) == 0:
                continue
            
            # Convert fingerprints back to arrays
            fingerprints = []
            ligand_ids = []
            
            for _, row in target_data.iterrows():
                fp_bytes = row['ecfp4_bytes']
                fp_array = np.frombuffer(fp_bytes, dtype=np.uint8)
                fingerprints.append(fp_array)
                ligand_ids.append(row['ligand_id'])
            
            fingerprints = np.array(fingerprints)
            
            # Build statistics
            stats = {
                'n_ligands': len(target_data),
                'fingerprint_dim': len(fingerprints[0]) if len(fingerprints) > 0 else 0,
                'has_embeddings': False,  # Will be updated if embeddings available
                'pca_params': None,
                'embedding_stats': None
            }
            
            # For now, we'll use PCA on ECFP4 since we don't have embeddings yet
            if len(fingerprints) >= 10:  # Need minimum samples for PCA
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                
                # Convert fingerprints to float for PCA
                fp_float = fingerprints.astype(np.float32)
                
                # Standardize
                scaler = StandardScaler()
                fp_scaled = scaler.fit_transform(fp_float)
                
                # Apply PCA to reduce to 128 dimensions
                n_components = min(128, len(fingerprints) - 1, fp_float.shape[1])
                pca = PCA(n_components=n_components)
                fp_pca = pca.fit_transform(fp_scaled)
                
                # Store PCA parameters
                stats['pca_params'] = {
                    'n_components': n_components,
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist(),
                    'pca_components': pca.components_.tolist(),
                    'pca_mean': pca.mean_.tolist()
                }
                
                # Build FAISS index on PCA vectors
                try:
                    import faiss
                    
                    # Create HNSW index
                    index = faiss.IndexHNSWFlat(n_components, 32)  # M=32
                    index.hnsw.efConstruction = 100
                    
                    # Add vectors
                    index.add(fp_pca.astype(np.float32))
                    
                    # Save index and IDs
                    index_path = self.output_dir / "indices" / f"{target_id}.faiss"
                    ids_path = self.output_dir / "ids" / f"{target_id}.npy"
                    
                    faiss.write_index(index, str(index_path))
                    np.save(ids_path, np.array(ligand_ids))
                    
                    logger.info(f"Built FAISS index for {target_id}: {n_components}D, {len(fingerprints)} vectors")
                    
                except ImportError:
                    logger.warning("FAISS not available, skipping index creation")
                except Exception as e:
                    logger.error(f"Failed to build FAISS index for {target_id}: {e}")
            
            # Save target statistics
            stats_path = self.output_dir / "stats" / f"{target_id}.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            target_stats[target_id] = stats
        
        logger.info(f"Built statistics for {len(target_stats)} targets")
        return target_stats
    
    def build_complete_database(self, training_data_path: str) -> Dict:
        """
        Build complete AD database from training data
        
        Args:
            training_data_path: Path to training data
            
        Returns:
            Summary of built database
        """
        logger.info("Starting complete AD database build")
        
        # Step 1: Build fingerprint database
        target_counts = self.build_fpdb_from_training_data(training_data_path)
        
        # Step 2: Build per-target statistics and indices  
        target_stats = self.build_per_target_stats(target_counts)
        
        # Step 3: Save build manifest
        manifest = {
            'build_timestamp': pd.Timestamp.now().isoformat(),
            'training_data_path': training_data_path,
            'total_targets': len(target_counts),
            'target_counts': target_counts,
            'fingerprint_params': {
                'radius': self.fp_gen.radius,
                'n_bits': self.fp_gen.n_bits
            },
            'output_directory': str(self.output_dir)
        }
        
        manifest_path = self.output_dir / "build_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"AD database build complete. Manifest saved to {manifest_path}")
        
        return manifest

def build_ad_database(training_data_path: str, output_dir: str = "/app/backend/ad_data") -> Dict:
    """
    Convenience function to build AD database
    
    Args:
        training_data_path: Path to training data file
        output_dir: Output directory for AD data
        
    Returns:
        Build manifest
    """
    builder = FingerprintDatabaseBuilder(output_dir)
    return builder.build_complete_database(training_data_path)