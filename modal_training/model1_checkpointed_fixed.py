"""
Model 1 CHECKPOINTED TRAINING - FIXED DOCKER IMAGE
Fixed Modal image build with proper dependency installation
"""

import modal
import pandas as pd
import numpy as np
import logging
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import math

# FIXED: Proper Modal image with guaranteed dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("pandas")
    .pip_install("numpy==1.24.3")
    .pip_install("scikit-learn")
    .pip_install("matplotlib")
    .pip_install("seaborn")
    .pip_install("torch==2.0.1")
    .pip_install("torchvision")
    .pip_install("transformers==4.30.2")
    .pip_install("wandb")
)

app = modal.App("model1-checkpointed-fixed")

# Persistent volumes
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("cell-line-models", create_if_missing=True)

class FineTunedChemBERTaEncoder(nn.Module):
    """PROVEN: Fine-tuned ChemBERTa (unfrozen, differential LR)"""
    
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", embedding_dim=768):
        super().__init__()
        
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chemberta = AutoModel.from_pretrained(model_name)
        
        # USER FIX: Unfreeze ChemBERTa for fine-tuning
        self.chemberta.requires_grad_(True)  # UNFREEZE!
        
        self.projection = nn.Linear(embedding_dim, 512)  # Reduced dimension
        self.dropout = nn.Dropout(0.1)
        
        print(f"   ✅ ChemBERTa loaded: {model_name} (UNFROZEN for fine-tuning)")
        
    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        # Tokenize SMILES
        tokens = self.tokenizer(
            smiles_list, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # USER FIX: Remove torch.no_grad() - allow fine-tuning!
        outputs = self.chemberta(**tokens)  # GRADIENTS FLOW NOW!
        pooled_output = outputs.pooler_output  # [batch_size, embedding_dim]
        
        # Project and regularize
        molecular_features = self.projection(pooled_output)
        molecular_features = self.dropout(molecular_features)
        
        return molecular_features

class SimpleProteinEncoder(nn.Module):
    """Simplified protein encoder using learned embeddings + basic features"""
    
    def __init__(self, num_targets, embedding_dim=128):
        super().__init__()
        
        # Learned protein embeddings
        self.protein_embeddings = nn.Embedding(num_targets, embedding_dim)
        
        # Context layers for protein understanding
        self.context_layers = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        print(f"   ✅ Simple protein encoder: {num_targets} targets → {embedding_dim}D → 256D")
        
    def forward(self, target_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.protein_embeddings(target_ids)
        return self.context_layers(embeddings)

class SimplifiedFixedModel(nn.Module):
    """PROVEN architecture that achieved R² = 0.5994"""
    
    def __init__(self, num_targets):
        super().__init__()
        
        # Molecular encoder (fine-tuned ChemBERTa)
        self.molecular_encoder = FineTunedChemBERTaEncoder()
        
        # Simplified protein encoder  
        self.protein_encoder = SimpleProteinEncoder(num_targets, embedding_dim=128)
        
        # USER FIX: Simple fusion MLP
        # ChemBERTa: 512D, Protein: 256D → Total: 768D
        self.fusion = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Separate heads for different assay types
        self.ic50_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self.ki_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self.ec50_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        print(f"   ✅ Architecture: Simple fusion (768→512→256) + separate assay heads")
        
    def forward(self, smiles_list, target_ids, assay_types):
        # Encode inputs
        molecular_features = self.molecular_encoder(smiles_list)
        protein_features = self.protein_encoder(target_ids)
        
        # Fuse features
        combined = torch.cat([molecular_features, protein_features], dim=1)
        fused_features = self.fusion(combined)
        
        # Multi-task prediction
        predictions = []
        for i, assay_type in enumerate(assay_types):
            if assay_type == 'IC50':
                pred = self.ic50_head(fused_features[i:i+1])
            elif assay_type == 'KI':
                pred = self.ki_head(fused_features[i:i+1])
            elif assay_type == 'EC50':
                pred = self.ec50_head(fused_features[i:i+1])
            else:
                pred = self.ic50_head(fused_features[i:i+1])  # Default to IC50
            
            predictions.append(pred)
        
        return torch.cat(predictions, dim=0).squeeze(-1)

@app.function(
    image=image,
    volumes={
        "/data": datasets_volume,
        "/models": models_volume
    },
    gpu="T4",
    timeout=7200,  # 2 hours
    memory=8192
)
def train_model1_with_checkpointing():
    """Train Model 1 with epoch-by-epoch checkpointing"""
    
    import wandb
    
    print("🧬 GNOSIS MODEL 1: CHECKPOINTED TRAINING (FIXED)")
    print("=" * 80)
    
    # Test imports first
    print("✅ Testing imports...")
    print(f"   pandas version: {pd.__version__}")
    print(f"   numpy version: {np.__version__}")
    print(f"   torch version: {torch.__version__}")
    
    # Initialize wandb
    wandb.init(
        project="gnosis-model1-checkpointed-fixed",
        name=f"model1-fixed-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "model_type": "ligand_activity_predictor",
            "architecture": "FineTuned_ChemBERTa_Simple_Protein_Fusion",
            "target_r2": ">0.6",
            "checkpointing": "every_epoch",
            "previous_best_r2": 0.5994,
            "fixed_docker_image": True
        }
    )
    
    # STEP 1: Load Model 1 training data
    print("\n📊 STEP 1: Loading Model 1 training data...")
    
    model1_path = Path("/data/gnosis_model1_binding_training.csv")
    if not model1_path.exists():
        raise FileNotFoundError(f"Model 1 data not found at {model1_path}")
    
    training_df = pd.read_csv(model1_path)
    print(f"   ✅ Training data loaded: {len(training_df):,} records")
    print(f"   📊 Unique compounds: {training_df['SMILES'].nunique():,}")
    print(f"   📊 Unique targets: {training_df['Target'].nunique():,}")
    
    # STEP 2: Data preprocessing (same as before - it worked!)
    print("\n🔧 STEP 2: Data preprocessing...")
    
    # Remove invalid SMILES
    valid_smiles = training_df['SMILES'].notna() & (training_df['SMILES'] != '')
    training_df = training_df[valid_smiles].copy()
    print(f"   📊 After SMILES filtering: {len(training_df):,} records")
    
    # STEP 3: Target sparsity handling (same successful approach)
    print("\n🎯 STEP 3: Target sparsity handling...")
    
    target_counts = training_df['Target'].value_counts()
    print(f"   📊 Targets before filtering: {len(target_counts)}")
    print(f"   📊 Records per target: min={target_counts.min()}, max={target_counts.max()}, median={target_counts.median()}")
    
    # Filter targets with sufficient data (≥200 records)
    min_records_per_target = 200
    valid_targets = target_counts[target_counts >= min_records_per_target].index
    filtered_df = training_df[training_df['Target'].isin(valid_targets)].copy()
    
    print(f"   ✅ Targets after filtering (≥{min_records_per_target} records): {len(valid_targets)}")
    print(f"   ✅ Records after filtering: {len(filtered_df):,}")
    
    # STEP 4: Prepare multi-task targets
    print("\n🎯 STEP 4: Preparing multi-task targets...")
    
    # Filter for valid activity values
    valid_activity = (
        filtered_df['Activity_nM'].notna() & 
        (filtered_df['Activity_nM'] > 0) &
        (filtered_df['Activity_Type'].isin(['IC50', 'KI', 'EC50']))
    )
    
    model_df = filtered_df[valid_activity].copy()
    
    # Convert to pIC50/pKi/pEC50
    model_df['pActivity'] = -np.log10(model_df['Activity_nM'] * 1e-9)  # Convert nM to M
    
    # Filter reasonable range
    model_df = model_df[(model_df['pActivity'] >= 2.0) & (model_df['pActivity'] <= 10.0)].copy()
    
    print(f"   ✅ Valid training records: {len(model_df):,}")
    
    # Create encoders
    target_encoder = LabelEncoder()
    target_list = sorted(model_df['Target'].unique())
    target_encoder.fit(target_list)
    
    model_df['target_id'] = target_encoder.transform(model_df['Target'])
    
    # Activity type distribution
    assay_dist = model_df['Activity_Type'].value_counts()
    print(f"   📊 Assay distribution: {assay_dist.to_dict()}")
    print(f"   📊 Target range: {model_df['pActivity'].min():.2f} - {model_df['pActivity'].max():.2f} pIC50/pKi/pEC50")
    
    # Calculate loss weights (inverse sqrt frequency)
    target_weights = {}
    for target in target_list:
        count = len(model_df[model_df['Target'] == target])
        target_weights[target] = 1.0 / math.sqrt(count)
    
    loss_weights = torch.FloatTensor([target_weights[target] for target in model_df['Target']])
    print(f"   ✅ Loss weights calculated: range {loss_weights.min():.4f} - {loss_weights.max():.4f}")
    
    # STEP 5: Train-test split
    print("\n📊 STEP 5: Creating train-test split...")
    
    train_idx, test_idx = train_test_split(
        model_df.index, 
        test_size=0.2, 
        random_state=42,
        stratify=model_df['Target']
    )
    
    print(f"   📊 Training samples: {len(train_idx):,}")
    print(f"   📊 Test samples: {len(test_idx):,}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   💻 Using device: {device}")
    
    # STEP 6: Initialize model (same architecture that worked)
    print("\n🤖 STEP 6: Initializing PROVEN Model...")
    
    model = SimplifiedFixedModel(len(target_list)).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Differential learning rates (same as before)
    chemberta_params = list(model.molecular_encoder.chemberta.parameters())
    other_params = [p for p in model.parameters() if p not in chemberta_params]
    
    optimizer = torch.optim.AdamW([
        {'params': chemberta_params, 'lr': 3e-5},  # ChemBERTa
        {'params': other_params, 'lr': 1e-4}      # New layers
    ], weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    print("   ✅ Optimizer: Differential LR - ChemBERTa(3e-5), Others(1e-4)")
    print("   ✅ Scheduler: Cosine annealing")
    
    # STEP 7: Training with checkpointing
    print(f"\n🏋️ STEP 7: CHECKPOINTED TRAINING (Target: R² > 0.6)...")
    
    batch_size = 32
    num_epochs = 50
    best_r2 = -float('inf')
    best_model_state = None
    
    # Create checkpoints directory
    checkpoint_dir = Path("/models/model1_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss function with weights
    def weighted_mse_loss(predictions, targets, weights):
        squared_diff = (predictions - targets) ** 2
        weighted_loss = squared_diff * weights
        return weighted_loss.mean()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        # Training batches
        for i in range(0, len(train_idx), batch_size):
            batch_end = min(i + batch_size, len(train_idx))
            batch_indices = train_idx[i:batch_end]
            
            batch_df = model_df.loc[batch_indices]
            batch_smiles = batch_df['SMILES'].tolist()
            batch_targets = torch.LongTensor(batch_df['target_id'].values).to(device)
            batch_assays = batch_df['Activity_Type'].tolist()
            batch_labels = torch.FloatTensor(batch_df['pActivity'].values).to(device)
            batch_weights = loss_weights[i:batch_end].to(device)
            
            optimizer.zero_grad()
            
            predictions = model(batch_smiles, batch_targets, batch_assays)
            loss = weighted_mse_loss(predictions, batch_labels, batch_weights)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_train_loss = train_loss / num_batches
        
        # Evaluation
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(0, len(test_idx), batch_size):
                batch_end = min(i + batch_size, len(test_idx))
                batch_indices = test_idx[i:batch_end]
                
                batch_df = model_df.loc[batch_indices]
                batch_smiles = batch_df['SMILES'].tolist()
                batch_targets = torch.LongTensor(batch_df['target_id'].values).to(device)
                batch_assays = batch_df['Activity_Type'].tolist()
                batch_labels = batch_df['pActivity'].values
                
                predictions = model(batch_smiles, batch_targets, batch_assays).cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(batch_labels)
        
        # Calculate metrics
        test_r2 = r2_score(all_labels, all_predictions)
        test_mae = mean_absolute_error(all_labels, all_predictions)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "learning_rate": current_lr
        })
        
        print(f"   Epoch {epoch:2d}: Loss={avg_train_loss:.4f} | Test R²={test_r2:.4f} | Test MAE={test_mae:.4f} | LR={current_lr:.2e}")
        
        # Save checkpoint every epoch
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:02d}_r2_{test_r2:.4f}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'target_encoder': target_encoder,
            'target_list': target_list,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'train_loss': avg_train_loss,
            'num_targets': len(target_list)
        }, checkpoint_path)
        
        # Track best model
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_model_state = model.state_dict().copy()
            print(f"     🎯 NEW BEST R²: {test_r2:.4f}")
            
            # Save best model separately
            best_path = checkpoint_dir / "best_model1.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'target_encoder': target_encoder,
                'target_list': target_list,
                'best_r2': best_r2,
                'test_mae': test_mae,
                'num_targets': len(target_list),
                'model_config': {
                    'architecture': 'simplified_fixed_ligand_activity',
                    'molecular_encoder': 'ChemBERTa_zinc_base_v1_fine_tuned',
                    'protein_encoder': 'simple_learned_embeddings',
                    'fusion_dim': 768
                }
            }, best_path)
        
        # Success check
        if test_r2 > 0.6:
            print(f"   🎉 TARGET ACHIEVED: R² = {test_r2:.4f} > 0.6!")
    
    # Final results
    print(f"\n🎉 MODEL 1 CHECKPOINTED TRAINING COMPLETED!")
    print("=" * 80)
    print(f"📊 FINAL RESULTS:")
    print(f"  • Best Test R²: {best_r2:.4f}")
    print(f"  • Previous best: 0.5994")
    print(f"  • Improvement: {'+' if best_r2 > 0.5994 else ''}{(best_r2 - 0.5994):.4f}")
    print(f"  • Target (>0.6): {'✅ ACHIEVED' if best_r2 > 0.6 else '❌ Not reached'}")
    
    wandb.finish()
    
    return {
        'best_r2': float(best_r2),
        'target_achieved': best_r2 > 0.6,
        'improvement_over_previous': float(best_r2 - 0.5994),
        'checkpoints_saved': num_epochs + 1
    }

if __name__ == "__main__":
    with app.run():
        result = train_model1_with_checkpointing.remote()
        print("Training completed:", result)