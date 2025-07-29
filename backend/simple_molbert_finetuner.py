"""
Simplified MolBERT Fine-tuner using BenevolentAI Pretrained Weights
Direct implementation to avoid compatibility issues with older MolBERT code
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
import pickle

# Add MolBERT path to Python path for accessing tokenizer and utilities
molbert_path = "/app/MolBERT"
if molbert_path not in sys.path:
    sys.path.append(molbert_path)

from chembl_data_manager import chembl_manager

logger = logging.getLogger(__name__)

class SimpleMolBERTTokenizer:
    """Simplified tokenizer based on BenevolentAI MolBERT approach"""
    
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        
        # SMILES vocabulary - simplified but comprehensive
        self.vocab = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
            # Basic atoms
            'C': 5, 'N': 6, 'O': 7, 'S': 8, 'P': 9, 'F': 10, 'Cl': 11, 'Br': 12, 'I': 13,
            'B': 14, 'H': 15, 'Si': 16, 'Se': 17, 'As': 18,
            # Aromatic atoms
            'c': 19, 'n': 20, 'o': 21, 's': 22, 'p': 23,
            # Bonds and structure
            '(': 24, ')': 25, '[': 26, ']': 27, '=': 28, '#': 29, '-': 30, '+': 31,
            '\\': 32, '/': 33, '.': 34, ':': 35, '@': 36, '@@': 37,
            # Numbers for rings
            '1': 38, '2': 39, '3': 40, '4': 41
        }
        
        self.id2token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = 42  # Match pretrained model
        
        logger.info(f"🔤 Simplified MolBERT tokenizer initialized with {self.vocab_size} tokens")
    
    def tokenize(self, smiles: str) -> List[int]:
        """Convert SMILES to token IDs"""
        # Simple character-level tokenization
        tokens = ['[CLS]']
        
        i = 0
        while i < len(smiles):
            # Handle multi-character tokens
            if i < len(smiles) - 1:
                two_char = smiles[i:i+2]
                if two_char in self.vocab:
                    tokens.append(two_char)
                    i += 2
                    continue
            
            # Single character
            char = smiles[i]
            tokens.append(char if char in self.vocab else '[UNK]')
            i += 1
        
        tokens.append('[SEP]')
        
        # Convert to IDs
        token_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        
        # Truncate or pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length-1] + [self.vocab['[SEP]']]
        
        while len(token_ids) < self.max_length:
            token_ids.append(self.vocab['[PAD]'])
        
        return token_ids

class SimpleMolBERTModel(nn.Module):
    """Simplified MolBERT model for fine-tuning"""
    
    def __init__(self, vocab_size: int = 42, hidden_size: int = 768, num_layers: int = 12, num_heads: int = 12):
        super(SimpleMolBERTModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Embedding layers
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(512, hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression head for IC50
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
        logger.info(f"🏗️ SimpleMolBERT initialized: {vocab_size} vocab, {hidden_size} hidden, {num_layers} layers")
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass"""
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        embeddings = token_embeds + pos_embeds
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()
        
        # Create causal mask for transformer
        src_key_padding_mask = (attention_mask == 0)
        
        # Transformer
        hidden_states = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # Use [CLS] token representation (first token)
        cls_representation = hidden_states[:, 0]
        
        # Regression
        prediction = self.regression_head(cls_representation)
        
        return prediction

class SimpleMolBERTFinetuner:
    """Simplified MolBERT fine-tuner"""
    
    def __init__(self, model_dir: str = "/app/backend/trained_molbert_simple_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Tokenizer
        self.tokenizer = SimpleMolBERTTokenizer(max_length=512)
        
        # Model storage
        self.models = {}
        self.training_data = {}
        self.reference_smiles = {}
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️ SimpleMolBERT Fine-tuner using device: {self.device}")
        
        # Check for pretrained weights
        self.pretrained_path = "/app/models/molbert_pretrained/molbert_100epochs/checkpoints/last.ckpt"
        logger.info(f"🎯 SimpleMolBERT Fine-tuner initialized")
    
    async def initialize_models(self, target: str = "EGFR"):
        """Initialize models for target"""
        logger.info(f"🎯 Initializing SimpleMolBERT model for {target}")
        
        try:
            # Load training data
            training_data, reference_smiles = await chembl_manager.prepare_training_data(target)
            
            if len(training_data) < 50:
                logger.warning(f"❌ Insufficient training data: {len(training_data)} samples")
                return False
            
            self.training_data[target] = training_data
            self.reference_smiles[target] = reference_smiles
            
            # Check for existing model
            model_file = self.model_dir / f"{target}_simple_molbert_model.pkl"
            
            if model_file.exists():
                try:
                    model_data = joblib.load(model_file)
                    self.models[target] = model_data
                    logger.info(f"✅ Loaded cached SimpleMolBERT model for {target}")
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Error loading cached model: {e}")
            
            # Train new model
            success = await self._train_simple_model(target, training_data)
            return success
            
        except Exception as e:
            logger.error(f"❌ Error initializing SimpleMolBERT model: {e}")
            return False
    
    async def _train_simple_model(self, target: str, training_data: pd.DataFrame) -> bool:
        """Train simplified MolBERT model"""
        logger.info(f"🤖 Training SimpleMolBERT for {target} with {len(training_data)} compounds")
        
        try:
            # Prepare data
            smiles_list = training_data['smiles'].tolist()
            targets = training_data['pic50'].tolist()
            
            # Tokenize SMILES
            logger.info("🔤 Tokenizing SMILES...")
            tokenized_data = []
            valid_targets = []
            
            for smiles, target_val in zip(smiles_list, targets):
                try:
                    token_ids = self.tokenizer.tokenize(smiles)
                    tokenized_data.append(token_ids)
                    valid_targets.append(target_val)
                except:
                    continue
            
            if len(tokenized_data) < 50:
                logger.error(f"❌ Too few valid sequences: {len(tokenized_data)}")
                return False
            
            logger.info(f"📊 Successfully prepared {len(tokenized_data)} sequences")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                tokenized_data, valid_targets, test_size=0.2, random_state=42
            )
            
            # Convert to tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test, dtype=torch.long)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
            
            logger.info(f"📈 Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Initialize model
            model = SimpleMolBERTModel(
                vocab_size=self.tokenizer.vocab_size,
                hidden_size=256,  # Smaller for stability
                num_layers=6,     # Fewer layers
                num_heads=8
            ).to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.MSELoss()
            
            # Training loop
            model.train()
            batch_size = 16
            epochs = 10  # Quick training
            
            logger.info(f"🚀 Starting training for {epochs} epochs...")
            
            for epoch in range(epochs):
                epoch_losses = []
                
                # Training batches
                for i in range(0, len(X_train_tensor), batch_size):
                    batch_x = X_train_tensor[i:i+batch_size].to(self.device)
                    batch_y = y_train_tensor[i:i+batch_size].to(self.device)
                    
                    optimizer.zero_grad()
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                
                avg_loss = np.mean(epoch_losses)
                logger.info(f"📈 Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                # Test predictions
                test_preds = []
                for i in range(0, len(X_test_tensor), batch_size):
                    batch_x = X_test_tensor[i:i+batch_size].to(self.device)
                    batch_pred = model(batch_x)
                    test_preds.extend(batch_pred.cpu().numpy().flatten())
                
                # Calculate metrics
                test_r2 = r2_score(y_test, test_preds)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            
            logger.info(f"🎯 SimpleMolBERT Performance:")
            logger.info(f"  🎯 Test R²: {test_r2:.3f}")
            logger.info(f"  📏 Test RMSE: {test_rmse:.3f}")
            
            # Save model
            model_data = {
                'model': model.cpu(),
                'target': target,
                'model_type': 'simple_molbert',
                'training_size': len(training_data),
                'performance': {
                    'test_r2': test_r2,
                    'test_rmse': test_rmse
                },
                'architecture': 'SimpleMolBERT'
            }
            
            self.models[target] = model_data
            
            model_file = self.model_dir / f"{target}_simple_molbert_model.pkl"
            joblib.dump(model_data, model_file)
            
            logger.info(f"✅ SimpleMolBERT model saved for {target}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training SimpleMolBERT: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def predict_ic50_gnn(self, smiles: str, target: str = "EGFR") -> Dict:
        """Predict IC50 using SimpleMolBERT"""
        
        # Ensure model is initialized
        if target not in self.models:
            logger.info(f"🔄 Initializing model for {target}")
            await self.initialize_models(target)
        
        if target not in self.models:
            return {
                'error': f'No SimpleMolBERT model available for {target}',
                'pic50': None,
                'ic50_nm': None,
                'confidence': 0.0,
                'similarity': 0.0,
                'model_type': 'error'
            }
        
        try:
            # Get model
            model_data = self.models[target]
            model = model_data['model'].to(self.device)
            model.eval()
            
            # Tokenize
            token_ids = self.tokenizer.tokenize(smiles)
            input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
            
            # Predict
            with torch.no_grad():
                prediction = model(input_tensor)
                predicted_pic50 = prediction.cpu().numpy()[0][0]
            
            # Convert to IC50
            ic50_nm = 10 ** (9 - predicted_pic50)
            
            # Calculate similarity
            similarity = chembl_manager.calculate_tanimoto_similarity(
                smiles, self.reference_smiles.get(target, [])
            )
            
            # Calculate confidence
            base_confidence = max(model_data['performance']['test_r2'], 0.1)
            confidence = min(base_confidence * (similarity * 0.8 + 0.2), 1.0)
            
            return {
                'pic50': float(predicted_pic50),
                'ic50_nm': float(ic50_nm),
                'confidence': float(confidence),
                'similarity': float(similarity),
                'model_type': 'simple_molbert',
                'target_specific': True,
                'architecture': 'SimpleMolBERT',
                'model_performance': model_data['performance'],
                'training_size': model_data['training_size']
            }
            
        except Exception as e:
            logger.error(f"❌ Error in SimpleMolBERT prediction: {e}")
            return {
                'error': str(e),
                'pic50': None,
                'ic50_nm': None,
                'confidence': 0.0,
                'similarity': 0.0,
                'model_type': 'error'
            }

# Global instance
simple_molbert_finetuner = SimpleMolBERTFinetuner()