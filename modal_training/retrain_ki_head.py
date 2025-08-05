"""
Retrain Ki Head Only - Medium Fix for Ki Prediction Calibration
"""

import modal
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from transformers import AutoTokenizer
import sys
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = modal.App("retrain-ki-head")

# Setup Modal environment
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "torch>=2.0.0",
    "transformers>=4.21.0", 
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "scikit-learn>=1.2.0"
])

# Modal volume with training data
expanded_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=False)

# Define the model architecture (matching original)
class GnosisIModel(nn.Module):
    """Gnosis I - Ligand Activity Predictor Model"""
    
    def __init__(self, num_targets):
        super().__init__()
        
        # Molecular encoder using ChemBERTa (already trained)
        from transformers import AutoModel
        self.chemberta = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        
        # Target embedding
        self.target_embedding = nn.Embedding(num_targets, 256)
        
        # Combined features
        combined_dim = self.chemberta.config.hidden_size + 256  # 768 + 256 = 1024
        
        # Separate prediction heads for each assay type
        self.ic50_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        self.ki_head = nn.Sequential(  # This is what we'll retrain
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        self.ec50_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask, target_ids, assay_type='ic50'):
        # Encode molecular structure
        mol_output = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        mol_features = mol_output.pooler_output
        
        # Encode target
        target_features = self.target_embedding(target_ids)
        
        # Combine features
        combined = torch.cat([mol_features, target_features], dim=-1)
        
        # Select appropriate head
        if assay_type.lower() == 'ic50':
            return self.ic50_head(combined)
        elif assay_type.lower() == 'ki':
            return self.ki_head(combined)
        elif assay_type.lower() == 'ec50':
            return self.ec50_head(combined)
        else:
            raise ValueError(f"Unknown assay_type: {assay_type}")

@app.function(
    image=image,
    volumes={"/vol/expanded": expanded_volume},
    gpu="T4",
    timeout=7200,  # 2 hours
    memory=16384   # 16GB RAM
)
def retrain_ki_head():
    """Retrain only the Ki prediction head using real Ki data"""
    
    logger.info("🚀 STARTING KI HEAD RETRAINING")
    logger.info("=" * 60)
    
    # 1. LOAD TRAINING DATA
    logger.info("📊 Loading Ki training data from Modal volume...")
    
    data_path = Path("/vol/expanded/gnosis_model1_binding_training.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    # Load full dataset
    df = pd.read_csv(data_path)
    logger.info(f"Total dataset: {len(df):,} records")
    
    # Filter for Ki data only
    ki_data = df[df['assay_type'] == 'KI'].copy()  # Note: might be 'KI' not 'Ki'
    if len(ki_data) == 0:
        ki_data = df[df['assay_type'] == 'Ki'].copy()  # Try alternate spelling
    
    logger.info(f"Ki data found: {len(ki_data):,} records")
    
    if len(ki_data) < 1000:
        logger.warning(f"⚠️ Only {len(ki_data)} Ki records found - this may not be sufficient")
        # Show available assay types
        available_types = df['assay_type'].value_counts()
        logger.info(f"Available assay types: {dict(available_types)}")
    
    # 2. LOAD EXISTING MODEL
    logger.info("🔧 Loading existing model checkpoint...")
    
    # Copy model file from local to Modal (if needed)
    import subprocess
    subprocess.run(["cp", "/app/backend/models/gnosis_model1_best.pt", "/tmp/model.pt"])
    
    checkpoint = torch.load("/tmp/model.pt", map_location='cpu', weights_only=False)
    target_list = checkpoint['target_list']
    target_encoder = checkpoint['target_encoder']
    
    logger.info(f"Model targets: {len(target_list)}")
    logger.info(f"Model R² score: {checkpoint.get('test_r2', 'Unknown')}")
    
    # 3. INITIALIZE MODEL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = GnosisIModel(len(target_list))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    
    # 4. FREEZE ALL PARAMETERS EXCEPT Ki HEAD
    logger.info("🔒 Freezing all parameters except Ki head...")
    
    for name, param in model.named_parameters():
        if 'ki_head' in name:
            param.requires_grad = True
            logger.info(f"  ✅ Trainable: {name}")
        else:
            param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # 5. PREPARE Ki TRAINING DATA
    logger.info("📝 Preparing Ki training data...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    
    # Prepare training samples
    train_data = []
    for idx, row in ki_data.iterrows():
        if row['target_name'] in target_list:  # Only use targets the model knows
            train_data.append({
                'smiles': row['SMILES'],
                'target': row['target_name'], 
                'activity': row['affinity_nm'],  # Ki in nM
                'pActivity': -np.log10(row['affinity_nm'] * 1e-9)  # Convert to pKi
            })
    
    logger.info(f"Prepared {len(train_data)} Ki training samples")
    
    if len(train_data) < 500:
        logger.error(f"❌ Insufficient Ki training data: {len(train_data)} samples")
        return {"error": "Insufficient Ki training data"}
    
    # Split train/validation
    train_samples, val_samples = train_test_split(train_data, test_size=0.2, random_state=42)
    logger.info(f"Training samples: {len(train_samples)}, Validation: {len(val_samples)}")
    
    # 6. TRAINING SETUP
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], 
                                lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # 7. TRAINING LOOP
    logger.info("🔥 Starting Ki head training...")
    
    best_val_r2 = -float('inf')
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(50):  # Max 50 epochs
        model.train()
        train_losses = []
        train_preds = []
        train_actuals = []
        
        # Training batch processing
        batch_size = 32
        for i in range(0, len(train_samples), batch_size):
            batch = train_samples[i:i+batch_size]
            
            # Tokenize SMILES
            smiles_list = [item['smiles'] for item in batch]
            tokens = tokenizer(smiles_list, padding=True, truncation=True, 
                             max_length=512, return_tensors='pt')
            
            # Prepare targets
            targets = [item['target'] for item in batch]
            target_ids = torch.tensor([target_encoder.transform([t])[0] for t in targets])
            
            # Prepare labels
            labels = torch.tensor([item['pActivity'] for item in batch], dtype=torch.float32)
            
            # Move to device
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            target_ids = target_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, target_ids, assay_type='ki')
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_losses.append(loss.item())
            train_preds.extend(outputs.squeeze().detach().cpu().numpy())
            train_actuals.extend(labels.cpu().numpy())
        
        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_actuals = []
        
        with torch.no_grad():
            for i in range(0, len(val_samples), batch_size):
                batch = val_samples[i:i+batch_size]
                
                smiles_list = [item['smiles'] for item in batch]
                tokens = tokenizer(smiles_list, padding=True, truncation=True,
                                 max_length=512, return_tensors='pt')
                
                targets = [item['target'] for item in batch]
                target_ids = torch.tensor([target_encoder.transform([t])[0] for t in targets])
                labels = torch.tensor([item['pActivity'] for item in batch], dtype=torch.float32)
                
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)
                target_ids = target_ids.to(device)
                labels = labels.to(device)
                
                outputs = model(input_ids, attention_mask, target_ids, assay_type='ki')
                loss = criterion(outputs.squeeze(), labels)
                
                val_losses.append(loss.item())
                val_preds.extend(outputs.squeeze().cpu().numpy())
                val_actuals.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_r2 = r2_score(train_actuals, train_preds)
        val_r2 = r2_score(val_actuals, val_preds)
        train_rmse = np.sqrt(mean_squared_error(train_actuals, train_preds))
        val_rmse = np.sqrt(mean_squared_error(val_actuals, val_preds))
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        logger.info(f"Epoch {epoch+1:2d}: Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, "
                   f"Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'target_encoder': target_encoder,
                'target_list': target_list,
                'ki_val_r2': val_r2,
                'ki_val_rmse': val_rmse,
                'num_targets': len(target_list)
            }, '/tmp/gnosis_model1_ki_fixed.pt')
            
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # 8. VALIDATION & TESTING
    logger.info("🧪 Final validation...")
    
    # Test on a few known compounds
    test_compounds = [
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "EGFR", "Aspirin"),  # Should be weak
        ("CCN(CC)CCCC(C)NC1=C2C=C(C=CC2=NC=C1)Cl", "EGFR", "Chloroquine"),  # Should be moderate
    ]
    
    model.eval()
    logger.info("🔍 Testing on known compounds:")
    
    with torch.no_grad():
        for smiles, target, name in test_compounds:
            if target in target_list:
                tokens = tokenizer([smiles], padding=True, truncation=True, 
                                 max_length=512, return_tensors='pt')
                target_id = torch.tensor([target_encoder.transform([target])[0]])
                
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)
                target_id = target_id.to(device)
                
                # Test Ki prediction
                ki_pred = model(input_ids, attention_mask, target_id, assay_type='ki')
                ki_pActivity = ki_pred.item()
                ki_uM = (10**(-ki_pActivity)) * 1e6
                
                logger.info(f"  {name} vs {target}: Ki = {ki_uM:.3f} μM (pKi = {ki_pActivity:.3f})")
    
    logger.info("✅ Ki head retraining completed!")
    logger.info(f"Best validation R²: {best_val_r2:.4f}")
    
    # Copy model back to local
    subprocess.run(["cp", "/tmp/gnosis_model1_ki_fixed.pt", "/tmp/result_model.pt"])
    
    return {
        "success": True,
        "best_val_r2": best_val_r2,
        "training_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "epochs_trained": epoch + 1,
        "model_saved": "/tmp/gnosis_model1_ki_fixed.pt"
    }

@app.local_entrypoint()
def main():
    result = retrain_ki_head.remote()
    print("🎉 RETRAINING RESULT:")
    print(result)
    return result