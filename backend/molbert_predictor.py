"""
MolBERT Predictor for Molecular Property Prediction
Implements BERT-style transformer for SMILES molecular sequences
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from chembl_data_manager import chembl_manager
import re

logger = logging.getLogger(__name__)

class SMILESTokenizer:
    """Custom SMILES tokenizer for molecular sequences"""
    
    def __init__(self):
        # SMILES vocabulary including special tokens
        self.vocab = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
            # Atoms
            'C': 5, 'N': 6, 'O': 7, 'S': 8, 'P': 9, 'F': 10, 'Cl': 11, 'Br': 12, 'I': 13,
            'B': 14, 'H': 15, 'Si': 16, 'Se': 17, 'Te': 18, 'As': 19, 'At': 20,
            # Aromatic atoms
            'c': 21, 'n': 22, 'o': 23, 's': 24, 'p': 25,
            # Bonds and structure
            '(': 26, ')': 27, '[': 28, ']': 29, '=': 30, '#': 31, '-': 32, '+': 33,
            '\\': 34, '/': 35, '.': 36, ':': 37, '@': 38, '@@': 39,
            # Ring numbers
            '1': 40, '2': 41, '3': 42, '4': 43, '5': 44, '6': 45, '7': 46, '8': 47, '9': 48,
            # Common multi-character tokens
            'Cl': 49, 'Br': 50, '@@': 51, '[nH]': 52, '[NH]': 53, '[OH]': 54, '[CH]': 55,
            '[N+]': 56, '[O-]': 57, '[S+]': 58, '[Cl-]': 59, '[NH+]': 60, '[NH2+]': 61,
            '[NH3+]': 62, '[n+]': 63, '[nH+]': 64, '[o+]': 65, '[s+]': 66
        }
        
        self.id2token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        # Special token IDs
        self.pad_token_id = self.vocab['[PAD]']
        self.unk_token_id = self.vocab['[UNK]']
        self.cls_token_id = self.vocab['[CLS]']
        self.sep_token_id = self.vocab['[SEP]']
        self.mask_token_id = self.vocab['[MASK]']
        
        logger.info(f"🔤 SMILES tokenizer initialized with {self.vocab_size} tokens")
    
    def tokenize_smiles(self, smiles: str) -> List[str]:
        """Tokenize SMILES string into tokens"""
        try:
            # Handle multi-character tokens first
            smiles = smiles.replace('Cl', ' Cl ').replace('Br', ' Br ')
            smiles = smiles.replace('@@', ' @@ ')
            smiles = smiles.replace('[nH]', ' [nH] ').replace('[NH]', ' [NH] ')
            smiles = smiles.replace('[OH]', ' [OH] ').replace('[CH]', ' [CH] ')
            smiles = smiles.replace('[N+]', ' [N+] ').replace('[O-]', ' [O-] ')
            smiles = smiles.replace('[S+]', ' [S+] ').replace('[Cl-]', ' [Cl-] ')
            smiles = smiles.replace('[NH+]', ' [NH+] ').replace('[NH2+]', ' [NH2+] ')
            smiles = smiles.replace('[NH3+]', ' [NH3+] ').replace('[n+]', ' [n+] ')
            smiles = smiles.replace('[nH+]', ' [nH+] ').replace('[o+]', ' [o+] ')
            smiles = smiles.replace('[s+]', ' [s+] ')
            
            # Split by spaces and filter empty strings
            tokens = [token.strip() for token in smiles.split() if token.strip()]
            
            # Further split single-character tokens
            final_tokens = []
            for token in tokens:
                if len(token) == 1:
                    final_tokens.append(token)
                elif token in self.vocab:
                    final_tokens.append(token)
                else:
                    # Split unknown multi-character tokens
                    for char in token:
                        final_tokens.append(char)
            
            return final_tokens
            
        except Exception as e:
            logger.warning(f"Error tokenizing SMILES {smiles}: {e}")
            return []
    
    def encode(self, smiles: str, max_length: int = 128, add_special_tokens: bool = True) -> Dict:
        """Encode SMILES to token IDs"""
        tokens = self.tokenize_smiles(smiles)
        
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # Convert to IDs
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # Truncate if too long
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length-1] + [self.sep_token_id]
        
        # Pad if too short
        attention_mask = [1] * len(token_ids)
        while len(token_ids) < max_length:
            token_ids.append(self.pad_token_id)
            attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor(token_ids),
            'attention_mask': torch.tensor(attention_mask)
        }

class MolBERTModel(nn.Module):
    """MolBERT model for molecular property prediction"""
    
    def __init__(self, vocab_size: int, hidden_size: int = 256, num_layers: int = 6, 
                 num_heads: int = 8, max_position_embeddings: int = 128):
        super(MolBERTModel, self).__init__()
        
        # BERT configuration
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        
        # BERT backbone
        self.bert = BertModel(config)
        
        # Regression head for IC50 prediction
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through MolBERT"""
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation for sequence classification
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Predict IC50
        prediction = self.regression_head(cls_output)
        
        return prediction

class MolBERTPredictor:
    """MolBERT predictor for molecular property prediction"""
    
    def __init__(self, model_dir: str = "/app/backend/trained_molbert_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = SMILESTokenizer()
        
        # Model storage
        self.models = {}
        self.training_data = {}
        self.reference_smiles = {}
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️ MolBERT using device: {self.device}")
    
    async def initialize_models(self, target: str = "EGFR"):
        """Initialize MolBERT models for specific target"""
        logger.info(f"🎯 Initializing MolBERT model for {target}")
        
        try:
            # Load or download training data
            training_data, reference_smiles = await chembl_manager.prepare_training_data(target)
            
            if len(training_data) < 50:
                logger.warning(f"❌ Insufficient training data for MolBERT: {len(training_data)} samples")
                return False
            
            self.training_data[target] = training_data
            self.reference_smiles[target] = reference_smiles
            
            # Try to load existing model
            model_file = self.model_dir / f"{target}_molbert_model.pkl"
            
            if model_file.exists():
                try:
                    model_data = joblib.load(model_file)
                    self.models[target] = model_data
                    logger.info(f"✅ Loaded cached MolBERT model for {target}")
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Error loading cached MolBERT model: {e}")
            
            # Train new MolBERT model
            success = await self._train_molbert_model(target, training_data)
            return success
            
        except Exception as e:
            logger.error(f"❌ Error initializing MolBERT model for {target}: {e}")
            return False
    
    async def _train_molbert_model(self, target: str, training_data: pd.DataFrame) -> bool:
        """Train MolBERT model on ChEMBL SMILES data"""
        logger.info(f"🤖 Training MolBERT for {target} with {len(training_data)} compounds")
        
        try:
            # Prepare tokenized data
            smiles_list = training_data['smiles'].tolist()
            targets = training_data['pic50'].tolist()
            
            logger.info("🔤 Tokenizing SMILES sequences...")
            tokenized_data = []
            valid_targets = []
            
            for i, (smiles, target_val) in enumerate(zip(smiles_list, targets)):
                if i % 200 == 0:
                    logger.info(f"  Tokenized {i}/{len(smiles_list)} molecules...")
                
                encoded = self.tokenizer.encode(smiles, max_length=128)
                if encoded['input_ids'].sum() > self.tokenizer.cls_token_id + self.tokenizer.sep_token_id:  # Valid encoding
                    tokenized_data.append(encoded)
                    valid_targets.append(target_val)
            
            if len(tokenized_data) < 50:
                logger.error(f"❌ Too few valid tokenized sequences: {len(tokenized_data)}")
                return False
            
            logger.info(f"📊 Successfully tokenized {len(tokenized_data)} SMILES sequences")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                tokenized_data, valid_targets, test_size=0.2, random_state=42
            )
            
            logger.info(f"📈 Train: {len(X_train)}, Test: {len(X_test)} sequences")
            
            # Initialize MolBERT model
            model = MolBERTModel(
                vocab_size=self.tokenizer.vocab_size,
                hidden_size=256,
                num_layers=6,
                num_heads=8
            ).to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
            criterion = nn.MSELoss()
            
            logger.info(f"🏗️ MolBERT Architecture:")
            logger.info(f"  🔤 Vocabulary size: {self.tokenizer.vocab_size}")
            logger.info(f"  🧠 Hidden size: 256")
            logger.info(f"  📚 Transformer layers: 6")
            logger.info(f"  👁️ Attention heads: 8")
            
            # Training loop with checkpointing
            model.train()
            best_test_loss = float('inf')
            
            logger.info("🚀 Starting MolBERT incremental training...")
            
            # Shorter epochs for incremental training (5 epochs at a time)
            epochs = 5  # Reduced from 20 - can be run multiple times
            batch_size = 16  # Small batch size for memory efficiency
            
            # Check for existing checkpoint
            checkpoint_file = self.model_dir / f"{target}_molbert_checkpoint.pkl"
            start_epoch = 0
            
            if checkpoint_file.exists():
                try:
                    checkpoint = joblib.load(checkpoint_file)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch']
                    best_test_loss = checkpoint['best_test_loss']
                    logger.info(f"📁 Resumed from checkpoint at epoch {start_epoch}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load checkpoint: {e}")
            
            for epoch in range(start_epoch, start_epoch + epochs):
                # Training
                model.train()
                train_losses = []
                
                # Process in batches
                for i in range(0, len(X_train), batch_size):
                    batch_data = X_train[i:i + batch_size]
                    batch_targets = torch.FloatTensor(y_train[i:i + batch_size]).unsqueeze(1).to(self.device)
                    
                    # Prepare batch tensors
                    input_ids = torch.stack([item['input_ids'] for item in batch_data]).to(self.device)
                    attention_mask = torch.stack([item['attention_mask'] for item in batch_data]).to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, batch_targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                
                # Evaluation every 2 epochs (more frequent for shorter runs)
                if epoch % 2 == 0:
                    model.eval()
                    test_losses = []
                    test_preds = []
                    test_targets = []
                    
                    with torch.no_grad():
                        for i in range(0, len(X_test), batch_size):
                            batch_data = X_test[i:i + batch_size]
                            batch_targets = y_test[i:i + batch_size]
                            
                            input_ids = torch.stack([item['input_ids'] for item in batch_data]).to(self.device)
                            attention_mask = torch.stack([item['attention_mask'] for item in batch_data]).to(self.device)
                            
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            
                            batch_targets_tensor = torch.FloatTensor(batch_targets).unsqueeze(1).to(self.device)
                            loss = criterion(outputs, batch_targets_tensor)
                            
                            test_losses.append(loss.item())
                            test_preds.extend(outputs.cpu().numpy().flatten())
                            test_targets.extend(batch_targets)
                    
                    avg_train_loss = np.mean(train_losses)
                    avg_test_loss = np.mean(test_losses)
                    
                    # Calculate R²
                    test_r2 = r2_score(test_targets, test_preds)
                    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
                    
                    logger.info(f"  Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Test Loss={avg_test_loss:.4f}, R²={test_r2:.3f}, RMSE={test_rmse:.3f}")
                    
                    if avg_test_loss < best_test_loss:
                        best_test_loss = avg_test_loss
                    
                    # Save checkpoint after each evaluation
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_test_loss': best_test_loss,
                        'test_r2': test_r2,
                        'test_rmse': test_rmse
                    }
                    joblib.dump(checkpoint, checkpoint_file)
                    logger.info(f"💾 Checkpoint saved at epoch {epoch}")
                    
                    model.train()
            
            # Final evaluation
            logger.info("📊 Final MolBERT evaluation...")
            model.eval()
            
            # Training predictions
            train_preds = []
            train_targets = []
            
            with torch.no_grad():
                for i in range(0, len(X_train), batch_size):
                    batch_data = X_train[i:i + batch_size]
                    batch_targets = y_train[i:i + batch_size]
                    
                    input_ids = torch.stack([item['input_ids'] for item in batch_data]).to(self.device)
                    attention_mask = torch.stack([item['attention_mask'] for item in batch_data]).to(self.device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    train_preds.extend(outputs.cpu().numpy().flatten())
                    train_targets.extend(batch_targets)
            
            # Test predictions
            test_preds = []
            test_targets = []
            
            with torch.no_grad():
                for i in range(0, len(X_test), batch_size):
                    batch_data = X_test[i:i + batch_size]
                    batch_targets = y_test[i:i + batch_size]
                    
                    input_ids = torch.stack([item['input_ids'] for item in batch_data]).to(self.device)
                    attention_mask = torch.stack([item['attention_mask'] for item in batch_data]).to(self.device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    test_preds.extend(outputs.cpu().numpy().flatten())
                    test_targets.extend(batch_targets)
            
            # Calculate final metrics
            train_r2 = r2_score(train_targets, train_preds)
            test_r2 = r2_score(test_targets, test_preds)
            test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
            
            logger.info(f"🎯 MolBERT Final Performance:")
            logger.info(f"  🎯 Train R²: {train_r2:.3f}")
            logger.info(f"  🎯 Test R²: {test_r2:.3f}")
            logger.info(f"  📏 Test RMSE: {test_rmse:.3f}")
            
            # Save model
            model_data = {
                'model': model.cpu(),  # Move to CPU for saving
                'target': target,
                'model_type': 'molbert',
                'training_size': len(training_data),
                'performance': {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_rmse': test_rmse
                },
                'device': str(self.device),
                'architecture': 'MolBERT Transformer'
            }
            
            self.models[target] = model_data
            
            model_file = self.model_dir / f"{target}_molbert_model.pkl"
            joblib.dump(model_data, model_file)
            
            logger.info(f"✅ MolBERT model saved for {target}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training MolBERT model for {target}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def predict_ic50_gnn(self, smiles: str, target: str = "EGFR") -> Dict:
        """Predict IC50 using trained MolBERT model"""
        
        # Ensure model is initialized
        if target not in self.models:
            logger.info(f"🔄 Initializing MolBERT model for {target}")
            await self.initialize_models(target)
        
        if target not in self.models:
            logger.error(f"❌ No MolBERT model available for {target}")
            return {
                'error': f'No MolBERT model available for {target}',
                'pic50': None,
                'ic50_nm': None,
                'confidence': 0.0,
                'similarity': 0.0,
                'model_type': 'error'
            }
        
        try:
            # Tokenize SMILES
            encoded = self.tokenizer.encode(smiles, max_length=128)
            
            # Load model and make prediction
            model_data = self.models[target]
            model = model_data['model'].to(self.device)
            model.eval()
            
            with torch.no_grad():
                input_ids = encoded['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = encoded['attention_mask'].unsqueeze(0).to(self.device)
                
                prediction = model(input_ids=input_ids, attention_mask=attention_mask)
                predicted_pic50 = prediction.cpu().numpy()[0][0]
            
            # Convert to IC50 in nM
            ic50_nm = 10 ** (9 - predicted_pic50)
            
            # Calculate similarity to training set
            similarity = chembl_manager.calculate_tanimoto_similarity(
                smiles, self.reference_smiles.get(target, [])
            )
            
            # Calculate confidence based on similarity and model performance
            base_confidence = model_data['performance']['test_r2']
            similarity_weight = similarity * 0.85 + 0.15  # High confidence for transformer
            confidence = min(base_confidence * similarity_weight, 1.0)
            
            return {
                'pic50': float(predicted_pic50),
                'ic50_nm': float(ic50_nm),
                'confidence': float(confidence),
                'similarity': float(similarity),
                'model_type': 'molbert',
                'target_specific': True,
                'architecture': 'MolBERT Transformer',
                'model_performance': model_data['performance'],
                'training_size': model_data['training_size'],
                'transformer_features': {
                    'vocabulary_size': self.tokenizer.vocab_size,
                    'attention_heads': 8,
                    'transformer_layers': 6,
                    'sequence_length': 128
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in MolBERT prediction: {e}")
            return {
                'error': str(e),
                'pic50': None,
                'ic50_nm': None,
                'confidence': 0.0,
                'similarity': 0.0,
                'model_type': 'error'
            }
    
    def get_available_targets(self) -> List[str]:
        """Get list of available targets"""
        return chembl_manager.get_available_targets()
    
    def get_model_info(self, target: str) -> Dict:
        """Get information about trained MolBERT model"""
        if target in self.models:
            return {
                'target': target,
                'available': True,
                'model_type': 'molbert',
                'architecture': 'MolBERT Transformer',
                'performance': self.models[target]['performance'],
                'training_size': self.models[target]['training_size'],
                'transformer_features': {
                    'vocabulary_size': self.tokenizer.vocab_size,
                    'attention_heads': 8,
                    'transformer_layers': 6,
                    'sequence_length': 128
                }
            }
        else:
            return {
                'target': target,
                'available': False,
                'model_type': 'molbert',
                'architecture': 'MolBERT Transformer',
                'performance': None,
                'training_size': 0
            }

# Global MolBERT predictor instance
molbert_predictor = MolBERTPredictor()