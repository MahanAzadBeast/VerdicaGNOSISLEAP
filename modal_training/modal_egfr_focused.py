"""
Modal EGFR-Only Training - Using Existing Data Pipeline
Focus on EGFR target with your existing 1,635 samples
"""

import modal
import os

# Define Modal app for EGFR training
app = modal.App("molbert-egfr-training")

# Create Modal image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "torch>=2.1.0",
    "transformers>=4.21.0", 
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "joblib>=1.3.0",
    "requests>=2.31.0",
    "rdkit-pypi>=2022.9.5",
    "chembl-webresource-client>=0.10.8",
    "accelerate>=0.21.0"
])

# Create persistent volume for model storage
volume = modal.Volume.from_name("molbert-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",  # Updated syntax
    volumes={"/models": volume},
    timeout=14400,  # 4 hours max
    memory=32768,   # 32GB RAM
    cpu=8.0
)
def train_egfr_molbert(
    max_epochs: int = 50,
    batch_size: int = 64,  # Larger batch for A100
    learning_rate: float = 0.0001,
    webhook_url: str = None
):
    """
    Train MolBERT model specifically for EGFR using existing data pipeline
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.amp import autocast, GradScaler
    import logging
    import numpy as np
    from datetime import datetime
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    import joblib
    import pandas as pd
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def send_progress(status, message, progress, **kwargs):
        """Send progress updates to webhook"""
        if webhook_url:
            try:
                import requests
                data = {
                    "status": status,
                    "message": message, 
                    "progress": progress,
                    "target": "EGFR",
                    "timestamp": datetime.now().isoformat(),
                    **kwargs
                }
                requests.post(webhook_url, json=data, timeout=10)
                logger.info(f"📡 Progress sent: {status} - {message} ({progress}%)")
            except Exception as e:
                logger.error(f"Failed to send progress: {e}")
    
    logger.info(f"🚀 Starting EGFR MolBERT training on A100 GPU")
    logger.info(f"🖥️ GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "No GPU")
    
    send_progress("started", "Initializing EGFR training on A100", 5)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load EGFR data using the existing approach
        logger.info("📥 Loading EGFR IC50 data...")
        send_progress("loading_data", "Loading EGFR IC50 data from ChEMBL", 10)
        
        # Use the optimized ChEMBL query for EGFR
        from chembl_webresource_client.new_client import new_client
        
        # Get EGFR target (CHEMBL203 is the main EGFR target)
        activity = new_client.activity
        egfr_activities = activity.filter(
            target_chembl_id="CHEMBL203",  # EGFR
            type="IC50",
            value__isnull=False,
            units="nM",
            relation="="
        ).only([
            'canonical_smiles', 'standard_value', 'standard_units', 'pchembl_value'
        ])
        
        # Convert to list and process
        activities_list = list(egfr_activities)
        logger.info(f"📊 Retrieved {len(activities_list)} EGFR IC50 records")
        
        # Process data
        processed_data = []
        for activity in activities_list:
            if (activity.get('canonical_smiles') and 
                activity.get('standard_value') and 
                activity.get('pchembl_value')):
                
                processed_data.append({
                    'smiles': activity['canonical_smiles'],
                    'ic50_nm': float(activity['standard_value']),
                    'pic50': float(activity['pchembl_value'])
                })
        
        # Remove duplicates and filter
        df = pd.DataFrame(processed_data)
        df = df.drop_duplicates(subset=['smiles'])
        df = df[df['pic50'].between(4, 10)]  # Reasonable pIC50 range
        
        logger.info(f"📊 Processed {len(df)} unique EGFR compounds")
        send_progress("data_processed", f"Processed {len(df)} EGFR compounds", 20)
        
        if len(df) < 100:
            raise ValueError(f"Insufficient data: only {len(df)} samples")
        
        # Advanced SMILES tokenization
        logger.info("🔤 Advanced SMILES tokenization...")
        send_progress("tokenizing", "Tokenizing SMILES with advanced vocab", 25)
        
        # Create comprehensive SMILES vocabulary
        all_chars = set()
        for smiles in df['smiles']:
            all_chars.update(smiles)
        
        # Add special tokens
        vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + sorted(list(all_chars))
        char_to_id = {char: idx for idx, char in enumerate(vocab)}
        vocab_size = len(vocab)
        
        logger.info(f"📚 Vocabulary size: {vocab_size}")
        
        def tokenize_smiles_advanced(smiles_list, max_length=128):
            tokenized = []
            for smiles in smiles_list:
                # Add start/end tokens
                tokens = [char_to_id['<START>']]
                for char in smiles[:max_length-2]:
                    tokens.append(char_to_id.get(char, char_to_id['<UNK>']))
                tokens.append(char_to_id['<END>'])
                
                # Pad to max_length
                while len(tokens) < max_length:
                    tokens.append(char_to_id['<PAD>'])
                
                tokenized.append(tokens)
            
            return torch.tensor(tokenized, dtype=torch.long)
        
        # Tokenize data
        X = tokenize_smiles_advanced(df['smiles'].tolist())
        y = torch.tensor(df['pic50'].values, dtype=torch.float32)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        logger.info(f"📊 Training: {len(X_train)}, Test: {len(X_test)}")
        send_progress("data_split", f"Training: {len(X_train)}, Test: {len(X_test)}", 30)
        
        # Enhanced MolBERT Model
        class EnhancedMolBERT(nn.Module):
            def __init__(self, vocab_size, hidden_dim=512, num_layers=8, num_heads=8, max_seq_len=128):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
                self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, hidden_dim))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                
                # Regression head with dropout
                self.regression_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 4, 1)
                )
                
            def forward(self, x):
                # Create padding mask
                padding_mask = (x == 0)
                
                # Embedding + positional encoding
                seq_len = x.size(1)
                x = self.embedding(x) + self.pos_encoding[:seq_len].unsqueeze(0)
                
                # Transformer
                x = self.transformer(x, src_key_padding_mask=padding_mask)
                
                # Global average pooling (ignoring padding)
                mask = (~padding_mask).float().unsqueeze(-1)
                x = (x * mask).sum(dim=1) / mask.sum(dim=1)
                
                return self.regression_head(x)
        
        # Initialize model
        model = EnhancedMolBERT(vocab_size=vocab_size).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        criterion = nn.MSELoss()
        scaler = GradScaler() if torch.cuda.is_available() else None
        
        logger.info(f"🧠 Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
        send_progress("model_initialized", "Enhanced MolBERT model initialized", 35)
        
        # Training loop
        best_r2 = float('-inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training phase
            model.train()
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size].to(device)
                batch_y = y_train[i:i+batch_size].unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                
                if scaler:
                    with autocast(device_type='cuda'):
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Evaluation phase
            model.eval()
            with torch.no_grad():
                test_predictions = []
                for i in range(0, len(X_test), batch_size):
                    batch_X = X_test[i:i+batch_size].to(device)
                    
                    if scaler:
                        with autocast():
                            outputs = model(batch_X)
                    else:
                        outputs = model(batch_X)
                    
                    test_predictions.extend(outputs.cpu().numpy().flatten())
                
                r2 = r2_score(y_test.numpy(), test_predictions)
                rmse = np.sqrt(mean_squared_error(y_test.numpy(), test_predictions))
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Save best model
            if r2 > best_r2:
                best_r2 = r2
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Progress update
            progress = 35 + ((epoch + 1) / max_epochs) * 60
            send_progress(
                "training", 
                f"Epoch {epoch+1}/{max_epochs} - R²: {r2:.4f}",
                progress,
                epoch=epoch+1,
                loss=avg_loss,
                r2_score=r2,
                rmse=rmse,
                best_r2=best_r2,
                learning_rate=optimizer.param_groups[0]['lr']
            )
            
            logger.info(f"📊 Epoch {epoch+1}: Loss={avg_loss:.4f}, R²={r2:.4f}, RMSE={rmse:.4f}, Best R²={best_r2:.4f}")
            
            # Early stopping
            if patience_counter >= 10:
                logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Load best model and save
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Save final model
        model_save_path = "/models/EGFR_molbert_enhanced.pkl"
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'char_to_id': char_to_id,
            'vocab_size': vocab_size,
            'best_r2': best_r2,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'model_config': {
                'hidden_dim': 512,
                'num_layers': 8,
                'num_heads': 8,
                'max_seq_len': 128
            }
        }, model_save_path)
        
        results = {
            'target': 'EGFR',
            'final_r2': best_r2,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs_completed': epoch + 1,
            'vocab_size': vocab_size,
            'model_path': model_save_path,
            'model_parameters': sum(p.numel() for p in model.parameters())
        }
        
        send_progress("completed", f"EGFR training completed - R²: {best_r2:.4f}", 100, 
                     results=results)
        
        logger.info(f"✅ EGFR training completed! Best R²: {best_r2:.4f}")
        logger.info(f"💾 Model saved to: {model_save_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ EGFR training failed: {e}")
        send_progress("failed", f"Training failed: {str(e)}", -1)
        raise

@app.local_entrypoint()
def main(webhook_url: str = None):
    """
    Deploy EGFR training
    """
    print("🚀 Starting EGFR MolBERT training on A100...")
    results = train_egfr_molbert.remote(webhook_url=webhook_url)
    print("✅ EGFR training completed!")
    print(f"📊 Results: {results}")
    return results