"""
FIXED Modal MolBERT Training - Direct Implementation
Embeds the training code directly to avoid import issues
"""

import modal
import os

# Define Modal app
app = modal.App("molbert-training-fixed")

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
    gpu=modal.gpu.A100(count=1),  # Single A100 GPU
    volumes={"/models": volume},
    timeout=14400,  # 4 hours max
    memory=32768,   # 32GB RAM
    cpu=8.0
)
def train_molbert_gpu(
    target: str = "EGFR",
    max_epochs: int = 50,
    batch_size: int = 32,  # Conservative batch size 
    webhook_url: str = None
):
    """
    Train MolBERT model on Modal GPU - Self-contained version
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import logging
    import json
    import time
    import numpy as np
    from datetime import datetime
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    import joblib
    
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
                    "target": target,
                    "timestamp": datetime.now().isoformat(),
                    **kwargs
                }
                requests.post(webhook_url, json=data, timeout=10)
                logger.info(f"📡 Progress sent: {status} - {message} ({progress}%)")
            except Exception as e:
                logger.error(f"Failed to send progress: {e}")
    
    logger.info(f"🚀 Starting MolBERT training on Modal GPU for {target}")
    logger.info(f"🖥️ GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "No GPU")
    
    send_progress("started", f"Initializing {target} training on Modal GPU", 5)
    
    try:
        # Simple MolBERT implementation embedded directly
        class SimpleMolBERT(nn.Module):
            def __init__(self, vocab_size=100, hidden_dim=256, num_layers=6, num_heads=8):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_dim)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(hidden_dim, num_heads, batch_first=True),
                    num_layers
                )
                self.regression_head = nn.Linear(hidden_dim, 1)
                
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                x = x.mean(dim=1)  # Pool over sequence
                return self.regression_head(x)
        
        # Load ChEMBL data
        logger.info(f"📥 Loading ChEMBL data for {target}")
        send_progress("loading_data", f"Loading ChEMBL data for {target}", 10)
        
        # Simulate data loading (replace with actual ChEMBL API call)
        from chembl_webresource_client.new_client import new_client
        
        # Get target data
        target_search = new_client.target
        targets_query = target_search.search(target)
        
        if not targets_query:
            raise ValueError(f"No targets found for {target}")
        
        target_chembl_id = targets_query[0]['target_chembl_id']
        logger.info(f"Found target: {target_chembl_id}")
        
        # Get bioactivities
        activity = new_client.activity
        bioactivities = activity.filter(
            target_chembl_id=target_chembl_id,
            type="IC50",
            value__isnull=False,
            units="nM"
        ).only([
            'canonical_smiles', 'value', 'units'
        ])
        
        # Convert to list and filter
        bioactivities_list = list(bioactivities)
        logger.info(f"📊 Found {len(bioactivities_list)} bioactivity records")
        
        if len(bioactivities_list) < 100:
            logger.warning(f"Low data count: {len(bioactivities_list)} samples")
        
        # Create simple training data
        smiles_list = []
        ic50_values = []
        
        for bio in bioactivities_list[:1000]:  # Limit to 1000 samples
            if bio.get('canonical_smiles') and bio.get('value'):
                smiles_list.append(bio['canonical_smiles'])
                # Convert to pIC50
                ic50_nm = float(bio['value'])
                pic50 = -np.log10(ic50_nm * 1e-9)  # Convert nM to M then to pIC50
                ic50_values.append(pic50)
        
        logger.info(f"📊 Prepared {len(smiles_list)} training samples")
        send_progress("data_loaded", f"Loaded {len(smiles_list)} compounds for {target}", 20)
        
        # Simple tokenization (character-level)
        def tokenize_smiles(smiles):
            # Create simple character mapping
            chars = set()
            for s in smiles_list:
                chars.update(s)
            char_to_id = {c: i+1 for i, c in enumerate(sorted(chars))}
            char_to_id['<PAD>'] = 0
            
            # Tokenize
            tokens = []
            for s in smiles:
                token_ids = [char_to_id.get(c, 0) for c in s[:50]]  # Max length 50
                # Pad to 50
                while len(token_ids) < 50:
                    token_ids.append(0)
                tokens.append(token_ids)
            
            return torch.tensor(tokens), len(char_to_id)
        
        logger.info("🔤 Tokenizing SMILES sequences...")
        X, vocab_size = tokenize_smiles(smiles_list)
        y = torch.tensor(ic50_values, dtype=torch.float32)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"📊 Training set: {len(X_train)}, Test set: {len(X_test)}")
        send_progress("training_started", f"Starting training with {len(X_train)} samples", 30)
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleMolBERT(vocab_size=vocab_size).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        best_r2 = float('-inf')
        best_model_state = None
        
        for epoch in range(max_epochs):
            model.train()
            total_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size].to(device)
                batch_y = y_train[i:i+batch_size].unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                test_X = X_test.to(device)
                test_y = y_test.numpy()
                predictions = model(test_X).cpu().numpy().flatten()
                
                r2 = r2_score(test_y, predictions)
                rmse = np.sqrt(mean_squared_error(test_y, predictions))
            
            avg_loss = total_loss / (len(X_train) // batch_size)
            
            if r2 > best_r2:
                best_r2 = r2
                best_model_state = model.state_dict().copy()
            
            # Progress update
            progress = 30 + ((epoch + 1) / max_epochs) * 60
            send_progress(
                "training", 
                f"Epoch {epoch+1}/{max_epochs} completed",
                progress,
                epoch=epoch+1,
                loss=avg_loss,
                r2_score=r2,
                rmse=rmse,
                best_r2=best_r2
            )
            
            logger.info(f"📊 Epoch {epoch+1}: Loss={avg_loss:.4f}, R²={r2:.4f}, RMSE={rmse:.4f}")
            
            if epoch % 10 == 0:  # Save checkpoint every 10 epochs
                checkpoint_path = f"/models/{target}_checkpoint_epoch_{epoch}.pkl"
                torch.save({
                    'model_state_dict': best_model_state,
                    'epoch': epoch,
                    'best_r2': best_r2,
                    'vocab_size': vocab_size
                }, checkpoint_path)
        
        # Final model save
        final_model_path = f"/models/{target}_molbert_final.pkl"
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': vocab_size,
            'target': target,
            'final_r2': best_r2,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }, final_model_path)
        
        results = {
            'target': target,
            'final_r2': best_r2,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs_completed': max_epochs,
            'vocab_size': vocab_size,
            'model_path': final_model_path
        }
        
        send_progress("completed", f"Training completed for {target}", 100, 
                     results=results, model_path=final_model_path)
        
        logger.info(f"✅ Training completed! Best R²: {best_r2:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        send_progress("failed", f"Training failed: {str(e)}", -1)
        raise

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1),
    volumes={"/models": volume},
    timeout=21600,  # 6 hours for multi-target
    memory=32768,
    cpu=8.0
)
def train_all_targets(
    targets: list = ["EGFR", "BRAF", "CDK2", "PARP1", "BCL2", "VEGFR2"],
    max_epochs: int = 20,  # Reduced for faster completion
    batch_size: int = 32,
    webhook_url: str = None
):
    """
    Train MolBERT for all targets in sequence
    """
    import logging
    logger = logging.getLogger(__name__)
    
    results = {}
    total_targets = len(targets)
    
    for i, target in enumerate(targets):
        logger.info(f"🎯 Training target {i+1}/{total_targets}: {target}")
        
        try:
            # Use local execution to avoid remote call issues
            result = train_molbert_gpu.local(
                target=target,
                max_epochs=max_epochs,
                batch_size=batch_size,
                webhook_url=webhook_url
            )
            
            results[target] = result
            logger.info(f"✅ Completed {target} - R²: {result.get('final_r2', 'N/A')}")
            
        except Exception as e:
            logger.error(f"❌ Failed training {target}: {e}")
            results[target] = {"error": str(e)}
    
    return results

@app.local_entrypoint()
def main(
    target: str = "EGFR",
    all_targets: bool = False,
    webhook_url: str = None
):
    """
    Main entry point for Modal deployment
    """
    if all_targets:
        print(f"🚀 Starting multi-target training on Modal...")
        results = train_all_targets.remote(webhook_url=webhook_url)
    else:
        print(f"🚀 Starting {target} training on Modal...")
        results = train_molbert_gpu.remote(
            target=target,
            webhook_url=webhook_url
        )
    
    print(f"✅ Training completed!")
    print(f"📊 Results: {results}")
    return results