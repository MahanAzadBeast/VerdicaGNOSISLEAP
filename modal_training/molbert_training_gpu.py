"""
GPU-optimized MolBERT Trainer for Modal.com
Adapted from existing MolBERT predictor with GPU optimizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import logging
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class GPUMolBERTTrainer:
    """GPU-optimized MolBERT trainer for Modal deployment"""
    
    def __init__(self, target="EGFR", max_epochs=50, batch_size=64, 
                 learning_rate=0.0001, progress_callback=None):
        self.target = target
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.progress_callback = progress_callback
        
        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️ Using device: {self.device}")
        
        # Mixed precision training
        self.scaler = GradScaler('cuda') if torch.cuda.is_available() else None
        
        # Initialize components
        self._init_tokenizer()
        self._init_model()
        
    def _init_tokenizer(self):
        """Initialize SMILES tokenizer (from existing code)"""
        # Copy your existing SMILESTokenizer class here
        from molbert_predictor import SMILESTokenizer
        self.tokenizer = SMILESTokenizer()
        
    def _init_model(self):
        """Initialize MolBERT model (from existing code)"""
        # Copy your existing MolBERTModel class here
        from molbert_predictor import MolBERTModel
        
        self.model = MolBERTModel(
            vocab_size=self.tokenizer.vocab_size,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            max_seq_length=128
        ).to(self.device)
        
        # Optimizer with GPU optimizations
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
    def _send_progress(self, status, message, progress, **kwargs):
        """Send progress update via callback"""
        if self.progress_callback:
            self.progress_callback(status, message, progress, **kwargs)
        
    def load_data(self):
        """Load ChEMBL data for target"""
        logger.info(f"📥 Loading ChEMBL data for {self.target}")
        self._send_progress("loading_data", f"Loading ChEMBL data for {self.target}", 10)
        
        # Import your existing data loading logic
        from chembl_data_manager import chembl_manager
        
        # Get training data
        data = chembl_manager.get_ic50_data(self.target)
        smiles_list = data['canonical_smiles'].tolist()
        ic50_values = data['pIC50'].tolist()
        
        logger.info(f"📊 Loaded {len(smiles_list)} compounds for {self.target}")
        return smiles_list, ic50_values
        
    def prepare_data(self, smiles_list, ic50_values):
        """Prepare and tokenize data"""
        logger.info("🔤 Tokenizing SMILES sequences...")
        self._send_progress("tokenizing", "Tokenizing molecular data", 20)
        
        # Tokenize SMILES
        tokenized_data = []
        valid_ic50 = []
        
        for smiles, ic50 in zip(smiles_list, ic50_values):
            try:
                tokens = self.tokenizer.tokenize(smiles)
                if tokens:
                    tokenized_data.append(tokens)
                    valid_ic50.append(ic50)
            except Exception as e:
                logger.warning(f"Failed to tokenize {smiles}: {e}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            tokenized_data, valid_ic50, test_size=0.2, random_state=42
        )
        
        logger.info(f"📊 Training set: {len(X_train)}, Test set: {len(X_test)}")
        return X_train, X_test, np.array(y_train), np.array(y_test)
        
    def train_epoch(self, X_train, y_train, epoch):
        """Train one epoch with GPU optimizations"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [X_train[idx] for idx in batch_indices]
            batch_targets = torch.FloatTensor([y_train[idx] for idx in batch_indices]).unsqueeze(1).to(self.device)
            
            try:
                # Prepare batch tensors
                input_ids = torch.stack([item['input_ids'] for item in batch_data]).to(self.device)
                attention_mask = torch.stack([item['attention_mask'] for item in batch_data]).to(self.device)
                
                self.optimizer.zero_grad()
                
                # Mixed precision forward pass
                if self.scaler:
                    with autocast('cuda'):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = nn.MSELoss()(outputs, batch_targets)
                    
                    # Scaled backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular training
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = nn.MSELoss()(outputs, batch_targets)
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Progress update every 10 batches
                if num_batches % 10 == 0:
                    progress = 30 + (epoch / self.max_epochs) * 60 + (i / len(X_train)) * (60 / self.max_epochs)
                    self._send_progress(
                        "training", 
                        f"Epoch {epoch+1}/{self.max_epochs}, Batch {num_batches}",
                        progress,
                        epoch=epoch+1,
                        batch=num_batches,
                        loss=loss.item()
                    )
                
            except Exception as e:
                logger.error(f"❌ Error in batch {i}: {e}")
                continue
        
        avg_loss = total_loss / max(1, num_batches)
        return avg_loss
        
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_test), self.batch_size):
                batch_data = X_test[i:i + self.batch_size]
                
                try:
                    input_ids = torch.stack([item['input_ids'] for item in batch_data]).to(self.device)
                    attention_mask = torch.stack([item['attention_mask'] for item in batch_data]).to(self.device)
                    
                    if self.scaler:
                        with autocast('cuda'):
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    predictions.extend(outputs.cpu().numpy().flatten())
                    
                except Exception as e:
                    logger.error(f"❌ Error in evaluation batch {i}: {e}")
                    predictions.extend([0.0] * len(batch_data))
        
        # Calculate metrics
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        return r2, rmse, predictions
        
    def train(self):
        """Main training loop"""
        start_time = time.time()
        
        # Load and prepare data
        smiles_list, ic50_values = self.load_data()
        X_train, X_test, y_train, y_test = self.prepare_data(smiles_list, ic50_values)
        
        self._send_progress("training_started", f"Starting {self.max_epochs} epochs", 30)
        
        best_r2 = float('-inf')
        best_model_state = None
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.max_epochs):
            logger.info(f"🔄 Epoch {epoch+1}/{self.max_epochs}")
            
            # Train epoch
            train_loss = self.train_epoch(X_train, y_train, epoch)
            
            # Evaluate
            r2, rmse, _ = self.evaluate(X_test, y_test)
            
            # Learning rate scheduling
            self.scheduler.step(train_loss)
            
            logger.info(f"📊 Epoch {epoch+1}: Loss={train_loss:.4f}, R²={r2:.4f}, RMSE={rmse:.4f}")
            
            # Save best model
            if r2 > best_r2:
                best_r2 = r2
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Progress update
            progress = 30 + ((epoch + 1) / self.max_epochs) * 60
            self._send_progress(
                "training", 
                f"Epoch {epoch+1} completed",
                progress,
                epoch=epoch+1,
                loss=train_loss,
                r2_score=r2,
                rmse=rmse,
                best_r2=best_r2
            )
            
            # Early stopping
            if patience_counter >= 10:
                logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_r2, final_rmse, final_predictions = self.evaluate(X_test, y_test)
        
        training_time = time.time() - start_time
        
        results = {
            'target': self.target,
            'final_r2': final_r2,
            'final_rmse': final_rmse,
            'best_r2': best_r2,
            'epochs_completed': epoch + 1,
            'training_time_minutes': training_time / 60,
            'training_size': len(X_train),
            'test_size': len(X_test),
            'device': str(self.device),
            'batch_size': self.batch_size
        }
        
        logger.info(f"✅ Training completed in {training_time/60:.1f} minutes")
        logger.info(f"📊 Final R²: {final_r2:.4f}, RMSE: {final_rmse:.4f}")
        
        return results
        
    def save_model(self, path):
        """Save trained model"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'target': self.target,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, path)
        logger.info(f"💾 Model saved to {path}")