"""
True Chemprop Graph Neural Network Predictor
Implements actual GNN-based molecular property prediction using Chemprop library
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import tempfile
import csv
import joblib
from rdkit import Chem
from chemprop.train import cross_validate, train
from chemprop.args import TrainArgs, PredictArgs
from chembl_data_manager import chembl_manager

logger = logging.getLogger(__name__)

class ChempropGNNPredictor:
    """Real Chemprop Graph Neural Network predictor using molecular graphs"""
    
    def __init__(self, model_dir: str = "/app/backend/trained_gnn_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.models = {}
        self.training_data = {}
        self.reference_smiles = {}
        
        # GNN configuration
        self.model_config = {
            'hidden_size': 300,
            'depth': 3,
            'dropout': 0.1,
            'batch_size': 50,
            'epochs': 10,  # Start with fewer epochs for testing
            'ffn_hidden_size': 300,
            'ffn_num_layers': 2
        }
    
    async def initialize_models(self, target: str = "EGFR"):
        """Initialize GNN models for specific target"""
        logger.info(f"🎯 Initializing Chemprop GNN model for {target}")
        
        try:
            # Load or download training data
            training_data, reference_smiles = await chembl_manager.prepare_training_data(target)
            
            if len(training_data) < 100:
                logger.warning(f"❌ Insufficient training data for GNN: {len(training_data)} samples")
                return False
            
            self.training_data[target] = training_data
            self.reference_smiles[target] = reference_smiles
            
            # Try to load existing model
            model_file = self.model_dir / f"{target}_gnn_model.pkl"
            
            if model_file.exists():
                try:
                    self.models[target] = joblib.load(model_file)
                    logger.info(f"✅ Loaded cached GNN model for {target}")
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Error loading cached GNN model: {e}")
            
            # Train new GNN model
            success = await self._train_gnn_model(target, training_data)
            return success
            
        except Exception as e:
            logger.error(f"❌ Error initializing GNN model for {target}: {e}")
            return False
    
    async def _train_gnn_model(self, target: str, training_data: pd.DataFrame) -> bool:
        """Train Chemprop GNN model on ChEMBL data"""
        logger.info(f"🧠 Training Chemprop GNN for {target} with {len(training_data)} compounds")
        
        try:
            # Prepare training data in Chemprop format
            train_file = self._prepare_chemprop_data(training_data, target)
            
            # Set up training arguments
            train_args = TrainArgs()
            train_args.data_path = str(train_file)
            train_args.dataset_type = 'regression'
            train_args.save_dir = str(self.model_dir / f"{target}_gnn")
            train_args.epochs = self.model_config['epochs']
            train_args.hidden_size = self.model_config['hidden_size']
            train_args.depth = self.model_config['depth']
            train_args.dropout = self.model_config['dropout']
            train_args.batch_size = self.model_config['batch_size']
            train_args.ffn_hidden_size = self.model_config['ffn_hidden_size']
            train_args.ffn_num_layers = self.model_config['ffn_num_layers']
            train_args.split_type = 'random'
            train_args.split_sizes = [0.8, 0.1, 0.1]  # train, val, test
            train_args.metric = 'rmse'
            train_args.quiet = True
            
            logger.info(f"🔬 Starting GNN training with config:")
            logger.info(f"  📊 Hidden size: {train_args.hidden_size}")
            logger.info(f"  🏗️ Depth: {train_args.depth} layers")
            logger.info(f"  📦 Batch size: {train_args.batch_size}")
            logger.info(f"  🔄 Epochs: {train_args.epochs}")
            
            # Train the model
            from chemprop.train import train
            model_scores = train(args=train_args)
            
            # Extract performance metrics
            if model_scores:
                best_score = min(model_scores) if model_scores else float('inf')
                logger.info(f"📈 GNN Training completed - Best RMSE: {best_score:.3f}")
            else:
                logger.warning("⚠️ No training scores returned")
                best_score = None
            
            # Save model metadata
            model_metadata = {
                'target': target,
                'model_type': 'chemprop_gnn',
                'training_size': len(training_data),
                'config': self.model_config,
                'performance': {
                    'best_rmse': best_score,
                    'train_compounds': len(training_data)
                },
                'model_path': str(self.model_dir / f"{target}_gnn")
            }
            
            self.models[target] = model_metadata
            
            # Save metadata
            metadata_file = self.model_dir / f"{target}_gnn_metadata.pkl"
            joblib.dump(model_metadata, metadata_file)
            
            logger.info(f"✅ GNN model saved for {target}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training GNN model for {target}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _prepare_chemprop_data(self, data: pd.DataFrame, target: str) -> Path:
        """Prepare data in Chemprop CSV format"""
        
        # Create temporary CSV file for Chemprop
        temp_file = self.model_dir / f"{target}_training_data.csv"
        
        # Prepare data with SMILES and pIC50
        chemprop_data = []
        for _, row in data.iterrows():
            chemprop_data.append({
                'smiles': row['smiles'],
                'target': row['pic50']  # Chemprop expects 'target' column for regression
            })
        
        # Write to CSV
        df_chemprop = pd.DataFrame(chemprop_data)
        df_chemprop.to_csv(temp_file, index=False)
        
        logger.info(f"📝 Prepared Chemprop training data: {temp_file}")
        logger.info(f"   📊 {len(df_chemprop)} compounds for GNN training")
        
        return temp_file
    
    async def predict_ic50_gnn(self, smiles: str, target: str = "EGFR") -> Dict:
        """Predict IC50 using trained Chemprop GNN model"""
        
        # Ensure model is initialized
        if target not in self.models:
            logger.info(f"🔄 Initializing GNN model for {target}")
            await self.initialize_models(target)
        
        if target not in self.models:
            logger.error(f"❌ No GNN model available for {target}")
            return {
                'error': f'No GNN model available for {target}',
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
            
            # Prepare prediction data
            pred_file = self._prepare_prediction_data(smiles, target)
            
            # Set up prediction arguments
            predict_args = PredictArgs()
            predict_args.test_path = str(pred_file)
            predict_args.checkpoint_dir = self.models[target]['model_path']
            predict_args.preds_path = str(self.model_dir / f"{target}_predictions.csv")
            
            # Import prediction function
            from chemprop.train import make_predictions
            
            # Make prediction
            predictions = make_predictions(args=predict_args)
            
            if predictions and len(predictions) > 0:
                predicted_pic50 = predictions[0][0]  # First prediction, first target
                
                # Convert to IC50 in nM
                ic50_nm = 10 ** (9 - predicted_pic50)
                
                # Calculate similarity to training set
                similarity = chembl_manager.calculate_tanimoto_similarity(
                    smiles, self.reference_smiles.get(target, [])
                )
                
                # Calculate confidence based on similarity and model performance
                base_confidence = 0.8  # GNNs typically have high confidence
                similarity_weight = similarity * 0.7 + 0.3
                confidence = min(base_confidence * similarity_weight, 1.0)
                
                return {
                    'pic50': float(predicted_pic50),
                    'ic50_nm': float(ic50_nm),
                    'confidence': float(confidence),
                    'similarity': float(similarity),
                    'model_type': 'chemprop_gnn',
                    'target_specific': True,
                    'model_performance': self.models[target]['performance'],
                    'training_size': self.models[target]['training_size'],
                    'architecture': 'Graph Neural Network'
                }
            else:
                raise Exception("No predictions returned from GNN model")
            
        except Exception as e:
            logger.error(f"❌ Error in GNN prediction: {e}")
            return {
                'error': str(e),
                'pic50': None,
                'ic50_nm': None,
                'confidence': 0.0,
                'similarity': 0.0,
                'model_type': 'error'
            }
    
    def _prepare_prediction_data(self, smiles: str, target: str) -> Path:
        """Prepare single SMILES for prediction"""
        pred_file = self.model_dir / f"{target}_predict_input.csv"
        
        # Write single SMILES to CSV
        with open(pred_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['smiles'])
            writer.writerow([smiles])
        
        return pred_file
    
    def get_available_targets(self) -> List[str]:
        """Get list of available targets"""
        return chembl_manager.get_available_targets()
    
    def get_model_info(self, target: str) -> Dict:
        """Get information about trained GNN model"""
        if target in self.models:
            return {
                'target': target,
                'available': True,
                'model_type': 'chemprop_gnn',
                'architecture': 'Graph Neural Network',
                'performance': self.models[target]['performance'],
                'training_size': self.models[target]['training_size'],
                'config': self.models[target]['config']
            }
        else:
            return {
                'target': target,
                'available': False,
                'model_type': 'chemprop_gnn',
                'architecture': 'Graph Neural Network',
                'performance': None,
                'training_size': 0
            }

# Global GNN predictor instance
gnn_predictor = ChempropGNNPredictor()