"""
Real Chemprop Predictor using actual ChEMBL data
Implements training and prediction with real IC50 data
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
from rdkit import Chem
from chembl_data_manager import chembl_manager
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class RealChempropPredictor:
    """Real IC50 predictor using ChEMBL data and ML models"""
    
    def __init__(self, model_dir: str = "/app/backend/trained_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.models = {}
        self.training_data = {}
        self.reference_smiles = {}
        
        # Initialize with default target
        self.current_target = "EGFR"
    
    async def initialize_models(self, target: str = "EGFR"):
        """Initialize models for specific target"""
        logger.info(f"🎯 Initializing models for {target}")
        
        try:
            # Load or download training data
            training_data, reference_smiles = await chembl_manager.prepare_training_data(target)
            
            if len(training_data) == 0:
                logger.warning(f"❌ No training data available for {target}")
                return False
            
            self.training_data[target] = training_data
            self.reference_smiles[target] = reference_smiles
            
            # Try to load existing model
            model_file = self.model_dir / f"{target}_ic50_model.pkl"
            
            if model_file.exists():
                try:
                    self.models[target] = joblib.load(model_file)
                    logger.info(f"✅ Loaded cached model for {target}")
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Error loading cached model: {e}")
            
            # Train new model
            success = await self._train_model(target, training_data)
            return success
            
        except Exception as e:
            logger.error(f"❌ Error initializing models for {target}: {e}")
            return False
    
    async def _train_model(self, target: str, training_data: pd.DataFrame) -> bool:
        """Train Random Forest model on ChEMBL data"""
        logger.info(f"🔬 Training new model for {target} with {len(training_data)} compounds")
        
        try:
            # Calculate molecular descriptors
            X, y = self._prepare_features(training_data)
            
            if len(X) < 50:
                logger.warning(f"❌ Insufficient training data for {target}: {len(X)} samples")
                return False
            
            logger.info(f"📊 Features prepared: {X.shape}, targets: {y.shape}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            logger.info(f"📈 Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
            
            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            logger.info("🌳 Training Random Forest model...")
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            logger.info(f"📊 Model performance for {target}:")
            logger.info(f"  🎯 Train R²: {train_r2:.3f}")
            logger.info(f"  🎯 Test R²: {test_r2:.3f}")  
            logger.info(f"  📏 Test RMSE: {test_rmse:.3f}")
            
            # Save model
            self.models[target] = {
                'model': model,
                'performance': {
                    'train_r2': train_r2,
                    'test_r2': test_r2,  
                    'test_rmse': test_rmse
                },
                'training_size': len(training_data),
                'model_type': 'real_ml_rf'
            }
            
            model_file = self.model_dir / f"{target}_ic50_model.pkl"
            joblib.dump(self.models[target], model_file)
            
            logger.info(f"✅ Model saved for {target}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training model for {target}: {e}")
            return False
    
    def _prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare molecular features from SMILES"""
        from rdkit.Chem import Descriptors, Crippen, Lipinski
        
        features = []
        targets = []
        
        for _, row in data.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol:
                    # Calculate molecular descriptors (using only available descriptors)
                    desc = [
                        Descriptors.MolWt(mol),
                        Crippen.MolLogP(mol),
                        Descriptors.NumHDonors(mol),
                        Descriptors.NumHAcceptors(mol),
                        Descriptors.TPSA(mol),
                        Descriptors.NumRotatableBonds(mol),
                        Descriptors.NumAromaticRings(mol),
                        Descriptors.FpDensityMorgan1(mol),  # Alternative to FractionCsp3
                        Descriptors.qed(mol),
                        Descriptors.BertzCT(mol)
                    ]
                    
                    features.append(desc)
                    targets.append(row['pic50'])
                    
            except Exception as e:
                logger.warning(f"Error calculating descriptors for SMILES {row.get('smiles', 'unknown')}: {e}")
                continue
        
        logger.info(f"📊 Features extracted: {len(features)} samples, {len(desc) if features else 0} features each")
        return np.array(features), np.array(targets)
    
    async def predict_ic50(self, smiles: str, target: str = "EGFR") -> Dict:
        """Predict IC50 for a molecule using real trained model"""
        
        # Ensure model is initialized
        if target not in self.models:
            logger.info(f"🔄 Initializing model for {target}")
            await self.initialize_models(target)
        
        if target not in self.models:
            logger.error(f"❌ No model available for {target}")
            return {
                'error': f'No model available for {target}',
                'pic50': None,
                'ic50_nm': None,
                'confidence': 0.0,
                'similarity': 0.0,
                'model_type': 'error'
            }
        
        try:
            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return {
                    'error': 'Invalid SMILES string',
                    'pic50': None,
                    'ic50_nm': None,
                    'confidence': 0.0,
                    'similarity': 0.0,
                    'model_type': 'error'
                }
            
            # Calculate features
            features = self._calculate_molecule_features(mol)
            if features is None:
                return {
                    'error': 'Could not calculate molecular features',
                    'pic50': None,
                    'ic50_nm': None,
                    'confidence': 0.0,
                    'similarity': 0.0,
                    'model_type': 'error'
                }
            
            # Make prediction
            model_data = self.models[target]
            predicted_pic50 = model_data['model'].predict([features])[0]
            
            # Convert to IC50 in nM
            ic50_nm = 10 ** (9 - predicted_pic50)
            
            # Calculate similarity to training set
            similarity = chembl_manager.calculate_tanimoto_similarity(
                smiles, self.reference_smiles.get(target, [])
            )
            
            # Calculate confidence based on similarity and model performance
            base_confidence = model_data['performance']['test_r2']
            similarity_weight = similarity * 0.8 + 0.2  # Boost confidence for similar molecules
            confidence = min(base_confidence * similarity_weight, 1.0)
            
            return {
                'pic50': float(predicted_pic50),
                'ic50_nm': float(ic50_nm),
                'confidence': float(confidence),
                'similarity': float(similarity),
                'model_type': model_data.get('model_type', 'real_ml_rf'),
                'target_specific': True,
                'model_performance': model_data['performance'],
                'training_size': model_data['training_size']
            }
            
        except Exception as e:
            logger.error(f"❌ Error predicting IC50: {e}")
            return {
                'error': str(e),
                'pic50': None,
                'ic50_nm': None,
                'confidence': 0.0,
                'similarity': 0.0,
                'model_type': 'error'
            }
    
    def _calculate_molecule_features(self, mol) -> Optional[List[float]]:
        """Calculate molecular features for prediction"""
        try:
            from rdkit.Chem import Descriptors, Crippen
            
            features = [
                Descriptors.MolWt(mol),
                Crippen.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.FpDensityMorgan1(mol),  # Alternative to FractionCsp3
                Descriptors.qed(mol),
                Descriptors.BertzCT(mol)
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return None
    
    async def predict_ic50_async(self, smiles: str, target: str) -> Dict:
        """Async wrapper for IC50 prediction"""
        # Initialize model if not available
        if target not in self.models:
            success = await self.initialize_models(target)
            if not success:
                raise Exception(f"Failed to initialize model for {target}")
        
        # Make prediction using sync method
        return self.predict_ic50(smiles, target)
    
    def get_available_targets(self) -> List[str]:
        """Get list of available targets"""
        return chembl_manager.get_available_targets()
    
    def get_model_info(self, target: str) -> Dict:
        """Get information about trained model"""
        if target in self.models:
            return {
                'target': target,
                'available': True,
                'performance': self.models[target]['performance'],
                'training_size': self.models[target]['training_size']
            }
        else:
            return {
                'target': target,
                'available': False,
                'performance': None,
                'training_size': 0
            }

# Global predictor instance
real_predictor = RealChempropPredictor()