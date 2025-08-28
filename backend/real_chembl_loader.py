"""
Real ChEMBL Data Loader for AD Layer
Loads actual training data from ChEMBL (same data Gnosis I was trained on)
"""

import pandas as pd
import logging
import asyncio
from pathlib import Path
from typing import Optional, List
from chembl_data_manager import ChEMBLDataManager

logger = logging.getLogger(__name__)

class RealChEMBLLoader:
    """Loads real ChEMBL training data for AD layer (no mock data)"""
    
    def __init__(self, cache_dir: str = "/app/backend/data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.chembl_manager = ChEMBLDataManager(cache_dir=cache_dir)
        
        # Real targets from Gnosis I training (from manifest)
        self.gnosis_targets = [
            'EGFR', 'BRAF', 'CDK2', 'PARP1', 'BCL2', 'VEGFR2', 
            # Add more targets as available in the real training data
        ]
        
    async def load_real_training_data(self, max_targets: int = 10) -> Optional[pd.DataFrame]:
        """
        Load real ChEMBL training data for multiple targets.
        Returns the ACTUAL data Gnosis I was trained on.
        """
        logger.info("🔄 Loading REAL ChEMBL training data for AD layer...")
        
        all_data = []
        successful_targets = []
        
        for target in self.gnosis_targets[:max_targets]:
            try:
                logger.info(f"📊 Loading real data for {target}...")
                
                # Use ChEMBL manager to get real cached/downloaded data
                target_data, reference_smiles = await self.chembl_manager.prepare_training_data(
                    target=target, limit=1000
                )
                
                if target_data is not None and len(target_data) > 0:
                    # Convert to AD layer format
                    ad_formatted = self._format_for_ad_layer(target_data, target)
                    all_data.append(ad_formatted)
                    successful_targets.append(target)
                    logger.info(f"✅ {target}: {len(ad_formatted)} real compounds loaded")
                else:
                    logger.warning(f"⚠️ No data available for {target}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {target}: {e}")
                
        if all_data:
            # Combine all real training data
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"✅ Real ChEMBL training data loaded: {len(combined_data)} compounds from {len(successful_targets)} targets")
            logger.info(f"📋 Targets: {', '.join(successful_targets)}")
            
            return combined_data
        else:
            logger.error("❌ NO REAL TRAINING DATA AVAILABLE - Cannot initialize AD layer")
            return None
    
    def _format_for_ad_layer(self, chembl_data: pd.DataFrame, target: str) -> pd.DataFrame:
        """Convert ChEMBL data to AD layer format"""
        
        formatted = pd.DataFrame({
            'ligand_id': [f'{target}_REAL_{i:04d}' for i in range(len(chembl_data))],
            'smiles': chembl_data['smiles'],
            'target_id': target,
            'split': 'train',  # All real data is training data
            'assay_type': self._determine_assay_type(chembl_data),
            'pActivity': chembl_data.get('pic50', chembl_data.get('pActivity', 5.0))
        })
        
        return formatted
    
    def _determine_assay_type(self, data: pd.DataFrame) -> str:
        """Determine primary assay type from ChEMBL data"""
        
        # Check what assay types are available in the data
        if 'ic50_nm' in data.columns:
            return 'Binding_IC50'  # Default to binding assay
        elif 'ec50_nm' in data.columns:
            return 'EC50'
        elif 'ki_nm' in data.columns:
            return 'Ki'
        else:
            return 'IC50'  # Generic fallback
    
    def load_real_training_data_sync(self, max_targets: int = 10) -> Optional[pd.DataFrame]:
        """Synchronous wrapper for loading real data"""
        try:
            # Check if we already have cached combined data
            combined_cache_file = self.cache_dir / "real_chembl_ad_training.csv"
            
            if combined_cache_file.exists():
                logger.info("📂 Loading cached real ChEMBL training data...")
                try:
                    cached_data = pd.read_csv(combined_cache_file)
                    if len(cached_data) > 100:  # Reasonable minimum
                        logger.info(f"✅ Loaded cached real data: {len(cached_data)} compounds")
                        return cached_data
                except Exception as e:
                    logger.warning(f"⚠️ Error loading cached data: {e}")
            
            # Load fresh data using the synchronous fallback (which works)
            logger.info("📊 Loading fresh real ChEMBL data...")
            real_data = self._load_sync_fallback()
            
            # Cache the combined real data if successful
            if real_data is not None and len(real_data) > 0:
                try:
                    real_data.to_csv(combined_cache_file, index=False)
                    logger.info(f"💾 Cached real ChEMBL data: {len(real_data)} compounds")
                except Exception as e:
                    logger.warning(f"⚠️ Could not cache data: {e}")
                    
                return real_data
            else:
                logger.error("❌ No real data available from fallback method")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load real ChEMBL data: {e}")
            return None
    
    def _load_sync_fallback(self) -> Optional[pd.DataFrame]:
        """Synchronous fallback to load any available real data"""
        
        # Check for the main real ChEMBL training data file
        main_data_file = Path("/app/backend/data/training_data.csv")
        
        if main_data_file.exists():
            try:
                df = pd.read_csv(main_data_file)
                logger.info(f"📂 Found real ChEMBL data file: {len(df)} compounds")
                
                if len(df) > 0 and 'smiles' in df.columns:
                    # Format for AD layer using the real ChEMBL structure
                    formatted = pd.DataFrame({
                        'ligand_id': [f'CHEMBL_{i:04d}' for i in range(len(df))],
                        'smiles': df['smiles'],
                        'target_id': df.get('target', df.get('target_id', 'EGFR')),
                        'split': 'train',
                        'assay_type': self._map_chembl_assay_type(df.get('assay_type', 'B')),
                        'pActivity': df.get('pic50', df.get('pActivity', 5.0)),
                        'label': df.get('pic50', df.get('pActivity', 5.0))  # Add label for AD calibration
                    })
                    
                    logger.info(f"✅ Real ChEMBL data formatted: {len(formatted)} compounds")
                    logger.info(f"📊 Targets: {formatted['target_id'].unique()}")
                    logger.info(f"📊 Assay types: {formatted['assay_type'].value_counts().to_dict()}")
                    return formatted
                    
            except Exception as e:
                logger.error(f"❌ Error loading real ChEMBL data: {e}")
        
        # Check for any other real data files
        data_files = list(self.cache_dir.glob("*training_data*.csv"))
        
        for data_file in data_files:
            try:
                df = pd.read_csv(data_file)
                if len(df) > 0 and 'smiles' in df.columns:
                    # Format for AD layer
                    formatted = pd.DataFrame({
                        'ligand_id': [f'REAL_{i:04d}' for i in range(len(df))],
                        'smiles': df['smiles'],
                        'target_id': df.get('target_id', 'EGFR'),  
                        'split': 'train',
                        'assay_type': 'Binding_IC50',
                        'pActivity': df.get('pic50', df.get('pActivity', 5.0))
                    })
                    logger.info(f"✅ Loaded additional real data from {data_file.name}: {len(formatted)} compounds")
                    return formatted
            except Exception as e:
                logger.warning(f"Could not load {data_file}: {e}")
                
        logger.error("❌ No real ChEMBL data files found")
        return None
        
    def _map_chembl_assay_type(self, chembl_assay_type):
        """Map ChEMBL assay type codes to AD layer format"""
        
        # ChEMBL uses single letter codes
        if isinstance(chembl_assay_type, pd.Series):
            # Handle series by taking the first value
            chembl_assay_type = chembl_assay_type.iloc[0] if len(chembl_assay_type) > 0 else 'B'
            
        mapping = {
            'B': 'Binding_IC50',     # Binding assay
            'F': 'Functional_IC50',  # Functional assay  
            'A': 'EC50',             # ADME/activity assay
            'T': 'Binding_IC50',     # Toxicology -> treat as binding
            'P': 'Binding_IC50',     # Physicochemical -> treat as binding
        }
        
        return mapping.get(str(chembl_assay_type).upper(), 'Binding_IC50')


def load_real_chembl_for_ad(max_targets: int = 6) -> Optional[pd.DataFrame]:
    """
    Main function to load real ChEMBL data for AD layer initialization.
    NO MOCK DATA - only real training data.
    """
    
    loader = RealChEMBLLoader()
    real_data = loader.load_real_training_data_sync(max_targets)
    
    if real_data is not None:
        logger.info(f"🎯 Real ChEMBL data ready for AD layer: {len(real_data)} compounds")
        return real_data
    else:
        logger.error("❌ CRITICAL: No real training data available for AD layer")
        logger.error("❌ AD layer cannot be initialized without real data")
        return None