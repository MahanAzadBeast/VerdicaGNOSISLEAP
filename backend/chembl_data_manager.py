"""
ChEMBL Data Manager for Real IC50 Prediction
Handles downloading, processing, and caching ChEMBL bioactivity data
"""

import os
import pandas as pd
import sqlite3
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import asyncio
import aiofiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChEMBLDataManager:
    """Manages ChEMBL data download, processing, and caching"""
    
    def __init__(self, cache_dir: str = "/app/backend/data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # ChEMBL client setup
        self.activity = new_client.activity
        self.molecule = new_client.molecule
        self.target = new_client.target
        
        # Target mappings for common proteins
        self.target_mappings = {
            "EGFR": "CHEMBL203",
            "BRAF": "CHEMBL5145", 
            "CDK2": "CHEMBL301",
            "PARP1": "CHEMBL3105",
            "BCL2": "CHEMBL2842",
            "P53": "CHEMBL4722",
            "VEGFR2": "CHEMBL279"
        }
        
        # Cache file paths
        self.ic50_cache_file = self.cache_dir / "chembl_ic50_data.pkl"
        self.fingerprint_cache_file = self.cache_dir / "chembl_fingerprints.pkl"
        self.training_data_file = self.cache_dir / "training_data.csv"
        
    async def download_ic50_data(self, target_name: str = "EGFR", limit: int = 1000) -> pd.DataFrame:
        """Download IC50 data from ChEMBL for specific target"""
        logger.info(f"Downloading IC50 data for {target_name} (limit: {limit})")
        
        try:
            # Get target ChEMBL ID
            target_id = self.target_mappings.get(target_name)
            if not target_id:
                logger.warning(f"Target {target_name} not found in mappings, using EGFR")
                target_id = "CHEMBL203"
            
            # Query activities
            activities = self.activity.filter(
                target_chembl_id=target_id,
                standard_type="IC50",
                standard_units="nM",
                pchembl_value__isnull=False
            )[:limit]
            
            # Convert to DataFrame
            data = []
            for activity in activities:
                try:
                    # Get molecule SMILES
                    mol_id = activity.get('molecule_chembl_id')
                    if mol_id:
                        mol_data = self.molecule.get(mol_id)
                        smiles = mol_data.get('molecule_structures', {}).get('canonical_smiles')
                        
                        if smiles and self._is_valid_smiles(smiles):
                            ic50_value = activity.get('standard_value')
                            pchembl_value = activity.get('pchembl_value')
                            
                            if ic50_value and pchembl_value:
                                data.append({
                                    'smiles': smiles,
                                    'ic50_nm': float(ic50_value),
                                    'pic50': float(pchembl_value),
                                    'target': target_name,
                                    'chembl_id': mol_id,
                                    'assay_id': activity.get('assay_chembl_id')
                                })
                except Exception as e:
                    logger.warning(f"Error processing activity: {e}")
                    continue
                    
            df = pd.DataFrame(data)
            logger.info(f"Downloaded {len(df)} IC50 records for {target_name}")
            
            # Save to cache
            await self._save_cache(df, self.ic50_cache_file)
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading ChEMBL data: {e}")
            return pd.DataFrame()
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """Validate SMILES string using RDKit"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    async def _save_cache(self, data: pd.DataFrame, cache_file: Path):
        """Save data to cache file"""
        try:
            async with aiofiles.open(cache_file, 'wb') as f:
                await f.write(pickle.dumps(data))
            logger.info(f"Cached data to {cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    async def load_cached_data(self, cache_file: Path) -> Optional[pd.DataFrame]:
        """Load data from cache file"""
        try:
            if cache_file.exists():
                async with aiofiles.open(cache_file, 'rb') as f:
                    data = pickle.loads(await f.read())
                logger.info(f"Loaded cached data from {cache_file}")
                return data
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
        return None
    
    def calculate_molecular_fingerprints(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """Calculate Morgan fingerprints for SMILES list"""
        fingerprints = {}
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fp_array = np.zeros((1,))
                    DataStructs.ConvertToNumpyArray(fp, fp_array)
                    fingerprints[smiles] = fp_array
            except Exception as e:
                logger.warning(f"Error calculating fingerprint for {smiles}: {e}")
                
        return fingerprints
    
    def calculate_tanimoto_similarity(self, query_smiles: str, reference_smiles_list: List[str]) -> float:
        """Calculate maximum Tanimoto similarity to training set"""
        try:
            query_mol = Chem.MolFromSmiles(query_smiles)
            if not query_mol:
                return 0.0
                
            query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2)
            
            max_similarity = 0.0
            for ref_smiles in reference_smiles_list:
                ref_mol = Chem.MolFromSmiles(ref_smiles)
                if ref_mol:
                    ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2)
                    similarity = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
                    max_similarity = max(max_similarity, similarity)
                    
            return max_similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def prepare_training_data(self, target: str = "EGFR") -> Tuple[pd.DataFrame, List[str]]:
        """Prepare training data and reference SMILES for similarity calculations"""
        
        # Try to load cached data first
        cached_data = await self.load_cached_data(self.ic50_cache_file)
        
        if cached_data is not None and len(cached_data) > 100:
            logger.info("Using cached ChEMBL data")
            training_data = cached_data
        else:
            logger.info("Downloading fresh ChEMBL data")
            training_data = await self.download_ic50_data(target, limit=2000)
        
        # Clean and prepare data
        if len(training_data) > 0:
            # Remove duplicates and outliers
            training_data = training_data.drop_duplicates(subset=['smiles'])
            training_data = training_data[
                (training_data['pic50'] >= 3.0) & 
                (training_data['pic50'] <= 12.0)
            ]
            
            # Save processed training data
            training_data.to_csv(self.training_data_file, index=False)
            
            reference_smiles = training_data['smiles'].tolist()
            
            logger.info(f"Prepared {len(training_data)} training compounds for {target}")
            return training_data, reference_smiles
        else:
            logger.warning("No training data available")
            return pd.DataFrame(), []
    
    def get_available_targets(self) -> List[str]:
        """Get list of available protein targets"""
        return list(self.target_mappings.keys())

# Global instance
chembl_manager = ChEMBLDataManager()