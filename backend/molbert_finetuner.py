"""
Fine-tuned MolBERT Predictor using BenevolentAI Pretrained Weights
Implements fine-tuning on top of the pre-trained MolBERT model for improved IC50 predictions
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tempfile
import shutil

# Add MolBERT path to Python path
molbert_path = "/app/MolBERT"
if molbert_path not in sys.path:
    sys.path.append(molbert_path)

from molbert.apps.finetune import FinetuneSmilesMolbertApp
from molbert.models.finetune import FinetuneSmilesMolbertModel
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
from molbert.datasets.finetune import BertFinetuneSmilesDataset
import pytorch_lightning as pl
from argparse import Namespace

from chembl_data_manager import chembl_manager

logger = logging.getLogger(__name__)

class MolBERTFinetuner:
    """Fine-tuned MolBERT predictor using BenevolentAI pretrained weights"""
    
    def __init__(self, model_dir: str = "/app/backend/trained_molbert_finetuned_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Pretrained model path
        self.pretrained_model_path = "/app/models/molbert_pretrained/molbert_100epochs/checkpoints/last.ckpt"
        
        # Model storage
        self.models = {}
        self.training_data = {}
        self.reference_smiles = {}
        
        # Device configuration - prefer CPU for fine-tuning stability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️ MolBERT Fine-tuner using device: {self.device}")
        
        # Fine-tuning parameters
        self.max_seq_length = 512  # Match pretrained model
        self.batch_size = 16  # Smaller batch for fine-tuning
        self.learning_rate = 1e-5  # Lower learning rate for fine-tuning
        self.max_epochs = 10  # Fewer epochs for fine-tuning
        
        logger.info(f"🎯 MolBERT Fine-tuner initialized with pretrained weights from: {self.pretrained_model_path}")
    
    async def initialize_models(self, target: str = "EGFR"):
        """Initialize fine-tuned MolBERT models for specific target"""
        logger.info(f"🎯 Initializing fine-tuned MolBERT model for {target}")
        
        try:
            # Check if pretrained model exists
            if not os.path.exists(self.pretrained_model_path):
                logger.error(f"❌ Pretrained MolBERT model not found at {self.pretrained_model_path}")
                return False
            
            # Load or download training data
            training_data, reference_smiles = await chembl_manager.prepare_training_data(target)
            
            if len(training_data) < 50:
                logger.warning(f"❌ Insufficient training data for MolBERT fine-tuning: {len(training_data)} samples")
                return False
            
            self.training_data[target] = training_data
            self.reference_smiles[target] = reference_smiles
            
            # Try to load existing fine-tuned model
            model_file = self.model_dir / f"{target}_molbert_finetuned_model.pkl"
            
            if model_file.exists():
                try:
                    model_data = joblib.load(model_file)
                    self.models[target] = model_data
                    logger.info(f"✅ Loaded cached fine-tuned MolBERT model for {target}")
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Error loading cached fine-tuned model: {e}")
            
            # Fine-tune new MolBERT model
            success = await self._finetune_molbert_model(target, training_data)
            return success
            
        except Exception as e:
            logger.error(f"❌ Error initializing fine-tuned MolBERT model for {target}: {e}")
            return False
    
    async def _finetune_molbert_model(self, target: str, training_data: pd.DataFrame) -> bool:
        """Fine-tune MolBERT model on ChEMBL SMILES data"""
        logger.info(f"🤖 Fine-tuning MolBERT for {target} with {len(training_data)} compounds")
        
        try:
            # Prepare CSV files for fine-tuning
            temp_dir = Path(tempfile.mkdtemp())
            
            # Split data
            train_data, test_data = train_test_split(training_data, test_size=0.2, random_state=42)
            train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
            
            logger.info(f"📊 Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
            
            # Save data files with correct format
            train_file = temp_dir / "train.csv"
            val_file = temp_dir / "val.csv"
            test_file = temp_dir / "test.csv"
            
            # Prepare data with SMILES and IC50 columns
            train_data_formatted = pd.DataFrame({
                'smiles': train_data['smiles'],
                'ic50': train_data['pic50']  # Use pIC50 values
            })
            val_data_formatted = pd.DataFrame({
                'smiles': val_data['smiles'],
                'ic50': val_data['pic50']
            })
            test_data_formatted = pd.DataFrame({
                'smiles': test_data['smiles'],
                'ic50': test_data['pic50']
            })
            
            train_data_formatted.to_csv(train_file, index=False)
            val_data_formatted.to_csv(val_file, index=False)
            test_data_formatted.to_csv(test_file, index=False)
            
            logger.info(f"📁 Saved training files to {temp_dir}")
            
            # Create hyperparameters for fine-tuning
            hparams = Namespace(
                train_file=str(train_file),
                valid_file=str(val_file),
                test_file=str(test_file),
                pretrained_model_path=self.pretrained_model_path,
                label_column='ic50',
                mode='regression',
                output_size=1,
                max_seq_length=self.max_seq_length,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                max_epochs=self.max_epochs,
                freeze_level=0,  # Fine-tune all layers
                num_workers=2,
                tiny=False,
                vocab_size=42,  # From pretrained model hparams
                max_position_embeddings=512,
                default_root_dir=str(self.model_dir / target),
                gpus=1 if torch.cuda.is_available() else 0,
                fast_dev_run=False,
                deterministic=True
            )
            
            logger.info("🏗️ Creating fine-tuned MolBERT model...")
            
            # Initialize model
            model = FinetuneSmilesMolbertModel(hparams)
            
            # Load pretrained weights
            app = FinetuneSmilesMolbertApp()
            model = app.load_model_weights(model=model, checkpoint_file=self.pretrained_model_path)
            
            logger.info("✅ Loaded pretrained MolBERT weights successfully")
            
            # Create trainer
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                gpus=1 if torch.cuda.is_available() else 0,
                default_root_dir=str(self.model_dir / target),
                deterministic=True,
                logger=False,  # Disable logging for simplicity
                checkpoint_callback=False  # Disable checkpointing for now
            )
            
            logger.info(f"🚀 Starting MolBERT fine-tuning for {self.max_epochs} epochs...")
            
            # Fine-tune the model
            trainer.fit(model)
            
            logger.info("📊 Evaluating fine-tuned model...")
            
            # Test the model
            test_results = trainer.test(model)
            
            logger.info(f"🎯 Fine-tuning completed. Test results: {test_results}")
            
            # Extract metrics
            if test_results and len(test_results) > 0:
                test_r2 = test_results[0].get('test_r2', 0.0)
                test_rmse = test_results[0].get('test_rmse', 999.0)
            else:
                test_r2 = 0.0
                test_rmse = 999.0
            
            logger.info(f"🎯 Fine-tuned MolBERT Performance:")
            logger.info(f"  🎯 Test R²: {test_r2:.3f}")
            logger.info(f"  📏 Test RMSE: {test_rmse:.3f}")
            
            # Save fine-tuned model
            model_data = {
                'model': model,
                'target': target,
                'model_type': 'molbert_finetuned',
                'training_size': len(training_data),
                'performance': {
                    'test_r2': test_r2,
                    'test_rmse': test_rmse
                },
                'device': str(self.device),
                'architecture': 'Fine-tuned MolBERT',
                'pretrained_from': 'BenevolentAI MolBERT',
                'featurizer': SmilesIndexFeaturizer.bert_smiles_index_featurizer(self.max_seq_length)
            }
            
            self.models[target] = model_data
            
            model_file = self.model_dir / f"{target}_molbert_finetuned_model.pkl"
            joblib.dump(model_data, model_file)
            
            logger.info(f"✅ Fine-tuned MolBERT model saved for {target}")
            
            # Cleanup temporary files
            shutil.rmtree(temp_dir)
            logger.info("🧹 Cleaned up temporary files")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error fine-tuning MolBERT model for {target}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def predict_ic50_gnn(self, smiles: str, target: str = "EGFR") -> Dict:
        """Predict IC50 using fine-tuned MolBERT model"""
        
        # Ensure model is initialized
        if target not in self.models:
            logger.info(f"🔄 Initializing fine-tuned MolBERT model for {target}")
            await self.initialize_models(target)
        
        if target not in self.models:
            logger.error(f"❌ No fine-tuned MolBERT model available for {target}")
            return {
                'error': f'No fine-tuned MolBERT model available for {target}',
                'pic50': None,
                'ic50_nm': None,
                'confidence': 0.0,
                'similarity': 0.0,
                'model_type': 'error'
            }
        
        try:
            # Get model data
            model_data = self.models[target]
            model = model_data['model']
            featurizer = model_data['featurizer']
            
            # Set model to evaluation mode
            model.eval()
            
            # Featurize SMILES
            features, valid = featurizer.transform_single(smiles)
            
            if not valid:
                logger.error(f"❌ Invalid SMILES for featurization: {smiles}")
                return {
                    'error': f'Invalid SMILES: {smiles}',
                    'pic50': None,
                    'ic50_nm': None,
                    'confidence': 0.0,
                    'similarity': 0.0,
                    'model_type': 'error'
                }
            
            # Create batch
            batch = {
                'input_ids': torch.tensor(features).unsqueeze(0),  # Add batch dimension
                'attention_mask': torch.ones(len(features)).unsqueeze(0)  # Simple attention mask
            }
            
            with torch.no_grad():
                # Make prediction
                outputs = model(batch)
                predicted_pic50 = outputs[0].item()  # Extract scalar value
            
            # Convert to IC50 in nM
            ic50_nm = 10 ** (9 - predicted_pic50)
            
            # Calculate similarity to training set
            similarity = chembl_manager.calculate_tanimoto_similarity(
                smiles, self.reference_smiles.get(target, [])
            )
            
            # Calculate confidence based on similarity and model performance
            base_confidence = max(model_data['performance']['test_r2'], 0.1)
            similarity_weight = similarity * 0.9 + 0.1  # High confidence for pretrained model
            confidence = min(base_confidence * similarity_weight, 1.0)
            
            return {
                'pic50': float(predicted_pic50),
                'ic50_nm': float(ic50_nm),
                'confidence': float(confidence),
                'similarity': float(similarity),
                'model_type': 'molbert_finetuned',
                'target_specific': True,
                'architecture': 'Fine-tuned MolBERT',
                'pretrained_from': 'BenevolentAI MolBERT',
                'model_performance': model_data['performance'],
                'training_size': model_data['training_size'],
                'transformer_features': {
                    'vocabulary_size': 42,
                    'attention_heads': 12,
                    'transformer_layers': 12,
                    'sequence_length': self.max_seq_length,
                    'hidden_size': 768
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in fine-tuned MolBERT prediction: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'pic50': None,
                'ic50_nm': None,
                'confidence': 0.0,
                'similarity': 0.0,
                'model_type': 'error'
            }
    
    def get_model_info(self, target: str) -> Dict:
        """Get information about fine-tuned MolBERT model"""
        if target in self.models:
            return {
                'target': target,
                'available': True,
                'model_type': 'molbert_finetuned',
                'architecture': 'Fine-tuned MolBERT',
                'pretrained_from': 'BenevolentAI MolBERT',
                'performance': self.models[target]['performance'],
                'training_size': self.models[target]['training_size'],
                'transformer_features': {
                    'vocabulary_size': 42,
                    'attention_heads': 12,
                    'transformer_layers': 12,
                    'sequence_length': self.max_seq_length,
                    'hidden_size': 768
                }
            }
        else:
            return {
                'target': target,
                'available': False,
                'model_type': 'molbert_finetuned',
                'architecture': 'Fine-tuned MolBERT',
                'pretrained_from': 'BenevolentAI MolBERT',
                'performance': None,
                'training_size': 0
            }

# Global fine-tuned MolBERT predictor instance
molbert_finetuner = MolBERTFinetuner()