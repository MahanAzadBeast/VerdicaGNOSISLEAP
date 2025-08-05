"""
Real Ki Training - Train Ki head on actual experimental Ki data
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
from transformers import AutoTokenizer, AutoModel
import pickle
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = modal.App("real-ki-training")

# Setup Modal environment with GPU
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "torch>=2.0.0",
    "transformers>=4.21.0", 
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "scikit-learn>=1.2.0"
])

# Modal volume with training data
expanded_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=False)

@app.function(
    image=image,
    volumes={"/vol/expanded": expanded_volume},
    gpu="T4",
    timeout=10800,  # 3 hours
    memory=16384    # 16GB RAM
)
def train_real_ki_head():
    """Train Ki head using real experimental Ki data"""
    
    logger.info("🧬 TRAINING Ki HEAD WITH REAL EXPERIMENTAL DATA")
    logger.info("=" * 65)
    
    # 1. LOAD AND ANALYZE REAL Ki DATA
    logger.info("📊 Loading comprehensive training dataset...")
    
    data_path = Path("/vol/expanded/gnosis_model1_binding_training.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    # Load full dataset
    df = pd.read_csv(data_path)
    logger.info(f"Total dataset: {len(df):,} records")
    logger.info(f"Available columns: {list(df.columns)}")
    
    # Analyze assay types
    assay_distribution = df['assay_type'].value_counts()
    logger.info(f"Assay distribution: {dict(assay_distribution)}")
    
    # Extract Ki data (try different possible column names)
    ki_data = None
    for ki_variant in ['KI', 'Ki', 'ki']:
        potential_ki = df[df['assay_type'] == ki_variant]
        if len(potential_ki) > 0:
            ki_data = potential_ki.copy()
            logger.info(f"✅ Found {len(ki_data):,} Ki records with assay_type='{ki_variant}'")
            break
    
    if ki_data is None or len(ki_data) < 1000:
        logger.error(f"❌ Insufficient Ki data found: {len(ki_data) if ki_data is not None else 0}")
        logger.info("Available assay types:", df['assay_type'].unique()[:10])
        return {"error": "Insufficient Ki training data"}
    
    # 2. LOAD EXISTING MODEL STRUCTURE
    logger.info("🔧 Setting up model architecture...")
    
    # Copy model from local filesystem to Modal temporary space
    import subprocess
    import os
    
    # Create local copy in Modal environment
    local_model_path = "/tmp/gnosis_model.pt"
    subprocess.run(["cp", "/app/backend/models/gnosis_model1_best.pt", local_model_path], 
                   cwd="/app", capture_output=True)
    
    if not os.path.exists(local_model_path):
        logger.error("❌ Failed to copy model file to Modal environment")
        return {"error": "Model file not accessible"}
    
    # Load checkpoint
    checkpoint = torch.load(local_model_path, map_location='cpu', weights_only=False)
    target_list = checkpoint['target_list']
    target_encoder = checkpoint['target_encoder']
    
    logger.info(f"Model loaded - {len(target_list)} targets available")
    logger.info(f"Target encoder: {type(target_encoder)}")
    
    # 3. PREPARE Ki TRAINING DATA
    logger.info("📝 Preparing Ki training samples...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Prepare valid training samples
    valid_samples = []
    target_counts = {}
    
    for idx, row in ki_data.iterrows():
        target_name = row['target_name']
        
        # Check if target is in model's target list
        if target_name in target_list:
            # Convert Ki from nM to pKi
            ki_nm = float(row['affinity_nm'])
            
            # Skip unrealistic values
            if 0.001 <= ki_nm <= 1e6:  # 1 pM to 1 mM range
                pki = -np.log10(ki_nm * 1e-9)  # Convert nM to M, then pKi
                
                valid_samples.append({
                    'smiles': row['SMILES'],
                    'target': target_name,
                    'ki_nm': ki_nm,
                    'pki': pki,
                    'data_source': row.get('data_source', 'unknown')
                })
                
                target_counts[target_name] = target_counts.get(target_name, 0) + 1
    
    logger.info(f"Prepared {len(valid_samples)} valid Ki training samples")
    
    # Show target distribution
    logger.info("Top targets with Ki data:")
    sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
    for target, count in sorted_targets[:10]:
        logger.info(f"  {target}: {count} samples")
    
    if len(valid_samples) < 1000:
        logger.error(f"❌ Not enough valid Ki samples: {len(valid_samples)}")
        return {"error": f"Only {len(valid_samples)} valid Ki samples"}
    
    # 4. DEFINE SIMPLIFIED MODEL ARCHITECTURE
    class KiHeadTrainer(nn.Module):
        """Simplified model for training only Ki head"""
        
        def __init__(self, target_list, pretrained_checkpoint):
            super().__init__()
            
            # Load ChemBERTa
            self.chemberta = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
            
            # Target embedding  
            self.target_embedding = nn.Embedding(len(target_list), 256)
            
            # Load pre-trained target embeddings if available
            if 'target_embedding.weight' in pretrained_checkpoint['model_state_dict']:
                self.target_embedding.weight.data = pretrained_checkpoint['model_state_dict']['target_embedding.weight']
                logger.info("✅ Loaded pre-trained target embeddings")
            
            # Ki prediction head (to be trained)
            combined_dim = 768 + 256  # ChemBERTa + target embedding
            self.ki_head = nn.Sequential(
                nn.Linear(combined_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
            
            # Initialize Ki head with small random weights
            for module in self.ki_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
        def forward(self, input_ids, attention_mask, target_ids):
            # Encode molecule
            mol_output = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
            mol_features = mol_output.pooler_output
            
            # Encode target
            target_features = self.target_embedding(target_ids)
            
            # Combine and predict
            combined = torch.cat([mol_features, target_features], dim=-1)
            return self.ki_head(combined)
    
    # 5. INITIALIZE TRAINING
    logger.info("🏗️ Initializing Ki head training...")
    
    model = KiHeadTrainer(target_list, checkpoint)
    model.to(device)
    
    # Freeze ChemBERTa and target embeddings - only train Ki head
    for name, param in model.named_parameters():
        if 'ki_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable Ki head parameters: {trainable_params:,}")
    
    # 6. SPLIT AND PREPARE DATA
    train_samples, val_samples = train_test_split(valid_samples, test_size=0.2, 
                                                  random_state=42, stratify=None)
    logger.info(f"Training: {len(train_samples)}, Validation: {len(val_samples)}")
    
    # Training setup
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                                 lr=3e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          patience=3, factor=0.7)
    
    # 7. TRAINING LOOP
    logger.info("🔥 Starting real Ki training...")
    
    best_val_r2 = -float('inf')
    patience_counter = 0
    max_patience = 8
    batch_size = 16  # Smaller batch size for GPU memory
    
    for epoch in range(30):  # Maximum epochs
        model.train()
        train_losses = []
        train_preds = []
        train_actuals = []
        
        # Shuffle training data
        np.random.shuffle(train_samples)
        
        # Training batches
        for i in range(0, len(train_samples), batch_size):
            batch = train_samples[i:i+batch_size]
            
            try:
                # Prepare batch data
                smiles_list = [item['smiles'] for item in batch]
                targets = [item['target'] for item in batch]
                labels = [item['pki'] for item in batch]
                
                # Tokenize SMILES
                tokens = tokenizer(smiles_list, padding=True, truncation=True, 
                                 max_length=256, return_tensors='pt')
                
                # Encode targets
                target_ids = []
                for target in targets:
                    try:
                        encoded = target_encoder.transform([target])[0]
                        target_ids.append(encoded)
                    except:
                        # Skip invalid targets
                        continue
                
                if len(target_ids) != len(batch):
                    continue  # Skip batch if target encoding failed
                
                target_ids = torch.tensor(target_ids, dtype=torch.long)
                labels = torch.tensor(labels, dtype=torch.float32)
                
                # Move to device
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)
                target_ids = target_ids.to(device)
                labels = labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask, target_ids)
                loss = criterion(outputs.squeeze(), labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Track metrics
                train_losses.append(loss.item())
                train_preds.extend(outputs.squeeze().detach().cpu().numpy())
                train_actuals.extend(labels.cpu().numpy())
                
            except Exception as e:
                logger.warning(f"Batch error: {e}")
                continue
        
        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_actuals = []
        
        with torch.no_grad():
            for i in range(0, len(val_samples), batch_size):
                batch = val_samples[i:i+batch_size]
                
                try:
                    smiles_list = [item['smiles'] for item in batch]
                    targets = [item['target'] for item in batch]
                    labels = [item['pki'] for item in batch]
                    
                    tokens = tokenizer(smiles_list, padding=True, truncation=True,
                                     max_length=256, return_tensors='pt')
                    
                    target_ids = []
                    for target in targets:
                        try:
                            encoded = target_encoder.transform([target])[0] 
                            target_ids.append(encoded)
                        except:
                            continue
                    
                    if len(target_ids) != len(batch):
                        continue
                    
                    target_ids = torch.tensor(target_ids, dtype=torch.long)
                    labels = torch.tensor(labels, dtype=torch.float32)
                    
                    input_ids = tokens['input_ids'].to(device)
                    attention_mask = tokens['attention_mask'].to(device)
                    target_ids = target_ids.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(input_ids, attention_mask, target_ids)
                    loss = criterion(outputs.squeeze(), labels)
                    
                    val_losses.append(loss.item())
                    val_preds.extend(outputs.squeeze().cpu().numpy())
                    val_actuals.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    continue
        
        # Calculate metrics
        if len(train_preds) > 0 and len(val_preds) > 0:
            train_r2 = r2_score(train_actuals, train_preds) 
            val_r2 = r2_score(val_actuals, val_preds)
            train_rmse = np.sqrt(mean_squared_error(train_actuals, train_preds))
            val_rmse = np.sqrt(mean_squared_error(val_actuals, val_preds))
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            logger.info(f"Epoch {epoch+1:2d}: Train R²={train_r2:.4f} ({train_rmse:.3f}), "
                       f"Val R²={val_r2:.4f} ({val_rmse:.3f}), LR={optimizer.param_groups[0]['lr']:.2e}")
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                patience_counter = 0
                
                # Update original checkpoint with trained Ki head
                updated_checkpoint = checkpoint.copy()
                updated_checkpoint['model_state_dict']['ki_head.0.weight'] = model.ki_head[0].weight.cpu()
                updated_checkpoint['model_state_dict']['ki_head.0.bias'] = model.ki_head[0].bias.cpu()
                updated_checkpoint['model_state_dict']['ki_head.3.weight'] = model.ki_head[3].weight.cpu()
                updated_checkpoint['model_state_dict']['ki_head.3.bias'] = model.ki_head[3].bias.cpu()
                
                # Add training metadata
                if 'metadata' not in updated_checkpoint:
                    updated_checkpoint['metadata'] = {}
                
                updated_checkpoint['metadata']['ki_training'] = {
                    'method': 'real_experimental_data',
                    'training_samples': len(train_samples),
                    'validation_r2': float(val_r2),
                    'validation_rmse': float(val_rmse),
                    'epoch': epoch + 1
                }
                
                # Save trained model
                torch.save(updated_checkpoint, '/tmp/gnosis_model1_ki_trained.pt')
                logger.info(f"💾 Saved best Ki model (R²={val_r2:.4f})")
                
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        else:
            logger.warning(f"Epoch {epoch+1}: No valid predictions generated")
    
    # 8. FINAL TESTING
    logger.info("🧪 Testing trained Ki head...")
    
    # Test predictions with known compounds
    test_compounds = [
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "EGFR", "Aspirin"),
        ("CCN(CC)CCCC(C)NC1=C2C=C(C=CC2=NC=C1)Cl", "EGFR", "Chloroquine"),
    ]
    
    model.eval()
    with torch.no_grad():
        logger.info("🔬 Testing on validation compounds:")
        for smiles, target, name in test_compounds:
            if target in target_list:
                try:
                    tokens = tokenizer([smiles], padding=True, truncation=True, 
                                     max_length=256, return_tensors='pt')
                    target_id = torch.tensor([target_encoder.transform([target])[0]])
                    
                    input_ids = tokens['input_ids'].to(device)
                    attention_mask = tokens['attention_mask'].to(device)
                    target_id = target_id.to(device)
                    
                    ki_pred = model(input_ids, attention_mask, target_id)
                    pki = ki_pred.item()
                    ki_nm = 10**(-pki) * 1e9
                    ki_um = ki_nm / 1000
                    
                    logger.info(f"  {name} vs {target}: Ki = {ki_um:.3f} μM (pKi = {pki:.3f})")
                    
                except Exception as e:
                    logger.warning(f"  {name} prediction failed: {e}")
    
    logger.info("✅ REAL Ki TRAINING COMPLETED!")
    logger.info(f"🏆 Best validation R²: {best_val_r2:.4f}")
    
    return {
        "success": True,
        "best_val_r2": best_val_r2,
        "training_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "epochs_trained": epoch + 1,
        "model_path": "/tmp/gnosis_model1_ki_trained.pt",
        "target_distribution": dict(sorted_targets[:10])
    }

@app.local_entrypoint()
def main():
    print("🧬 Starting Real Ki Head Training...")
    result = train_real_ki_head.remote()
    print("\n" + "="*60)
    print("🎉 REAL Ki TRAINING RESULTS:")
    print("="*60)
    for key, value in result.items():
        print(f"{key}: {value}")
    return result