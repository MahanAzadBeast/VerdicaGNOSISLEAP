"""
ChEMBL Data Manager for Real IC50 Prediction
Handles downloading, processing, and caching ChEMBL bioactivity data using REST API
"""

import os
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import asyncio
import aiofiles
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChEMBLDataManager:
    """Manages ChEMBL data download, processing, and caching using REST API"""
    
    def __init__(self, cache_dir: str = "/app/backend/data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # ChEMBL API base URL
        self.api_base = "https://www.ebi.ac.uk/chembl/api/data"
        
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
        """Download IC50 data from ChEMBL REST API for specific target"""
        logger.info(f"Downloading IC50 data for {target_name} (limit: {limit})")
        
        target_id = self.target_mappings.get(target_name, "CHEMBL203")
        logger.info(f"Using ChEMBL target ID: {target_id}")
        
        # Build API URL
        api_url = f"{self.api_base}/activity.json"
        params = {
            "target_chembl_id": target_id,
            "standard_type": "IC50",
            "limit": limit,
            "offset": 0
        }
        
        all_data = []
        
        try:
            while len(all_data) < limit:
                logger.info(f"Fetching batch with offset {params['offset']}")
                
                response = requests.get(api_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                activities = data.get('activities', [])
                
                if not activities:
                    logger.info("No more activities found")
                    break
                
                # Process activities
                for activity in activities:
                    try:
                        canonical_smiles = activity.get('canonical_smiles')
                        standard_value = activity.get('standard_value')
                        pchembl_value = activity.get('pchembl_value')
                        standard_units = activity.get('standard_units')
                        
                        # Validate data
                        if (canonical_smiles and 
                            standard_value and 
                            pchembl_value and
                            standard_units == 'nM' and
                            self._is_valid_smiles(canonical_smiles)):
                            
                            all_data.append({
                                'smiles': canonical_smiles,
                                'ic50_nm': float(standard_value),
                                'pic50': float(pchembl_value),
                                'target': target_name,
                                'chembl_id': activity.get('molecule_chembl_id'),
                                'assay_id': activity.get('assay_chembl_id'),
                                'assay_type': activity.get('assay_type')
                            })
                            
                    except Exception as e:
                        logger.warning(f"Error processing activity: {e}")
                        continue
                
                # Check if there are more pages
                page_meta = data.get('page_meta', {})
                if not page_meta.get('next'):
                    logger.info("No more pages available")
                    break
                
                # Update offset for next batch
                params['offset'] += params['limit']
                
                # Add small delay to be respectful to the API
                await asyncio.sleep(0.1)
            
            df = pd.DataFrame(all_data)
            logger.info(f"Downloaded {len(df)} IC50 records for {target_name}")
            
            # Save to cache
            if len(df) > 0:
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
            logger.info(f"Using cached ChEMBL data: {len(cached_data)} compounds")
            training_data = cached_data
        else:
            logger.info("Downloading fresh ChEMBL data...")
            training_data = await self.download_ic50_data(target, limit=2000)
        
        # Clean and prepare data
        if len(training_data) > 0:
            logger.info(f"Raw data: {len(training_data)} compounds")
            
            # Remove duplicates by SMILES
            training_data = training_data.drop_duplicates(subset=['smiles'])
            logger.info(f"After deduplication: {len(training_data)} compounds")
            
            # Remove outliers (keep reasonable IC50 range)
            training_data = training_data[
                (training_data['pic50'] >= 3.0) & 
                (training_data['pic50'] <= 12.0) &
                (training_data['ic50_nm'] > 0.001) &
                (training_data['ic50_nm'] < 1000000)  # 1 mM max
            ]
            logger.info(f"After outlier removal: {len(training_data)} compounds")
            
            # Validate SMILES
            valid_smiles = []
            for _, row in training_data.iterrows():
                if self._is_valid_smiles(row['smiles']):
                    valid_smiles.append(row)
            
            training_data = pd.DataFrame(valid_smiles)
            logger.info(f"After SMILES validation: {len(training_data)} compounds")
            
            # Save processed training data
            if len(training_data) > 0:
                training_data.to_csv(self.training_data_file, index=False)
                reference_smiles = training_data['smiles'].tolist()
                
                logger.info(f"✅ Prepared {len(training_data)} training compounds for {target}")
                logger.info(f"IC50 range: {training_data['ic50_nm'].min():.2f} - {training_data['ic50_nm'].max():.2f} nM")
                logger.info(f"pIC50 range: {training_data['pic50'].min():.2f} - {training_data['pic50'].max():.2f}")
                
                return training_data, reference_smiles
        
        logger.warning("No training data available")
        return pd.DataFrame(), []
    
    def get_available_targets(self) -> List[str]:
        """Get list of available protein targets"""
        return list(self.target_mappings.keys())

# Global instance
chembl_manager = ChEMBLDataManager()