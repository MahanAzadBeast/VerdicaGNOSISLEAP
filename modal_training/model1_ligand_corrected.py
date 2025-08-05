"""
Model 1 LIGAND ACTIVITY PREDICTOR - CORRECTED DATASET STRUCTURE
Fixed column names: target_name, assay_type, etc.
Using existing pIC50/pKi/pEC50 columns
"""

import modal
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import json
import math

# Fixed Modal image without WandB
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("pandas")
    .pip_install("numpy==1.24.3")
    .pip_install("scikit-learn")
    .pip_install("torch==2.0.1")
    .pip_install("transformers==4.30.2")
)

app = modal.App("model1-ligand-corrected")

# Volumes
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("cell-line-models", create_if_missing=True)

class FineTunedChemBERTaEncoder(nn.Module):
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", embedding_dim=768):
        super().__init__()
        
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chemberta = AutoModel.from_pretrained(model_name)
        self.chemberta.requires_grad_(True)
        
        self.projection = nn.Linear(embedding_dim, 512)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            smiles_list, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        )
        
        device = next(self.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        outputs = self.chemberta(**tokens)
        pooled_output = outputs.pooler_output
        
        molecular_features = self.projection(pooled_output)
        molecular_features = self.dropout(molecular_features)
        
        return molecular_features

class SimpleProteinEncoder(nn.Module):
    def __init__(self, num_targets, embedding_dim=128):
        super().__init__()
        
        self.protein_embeddings = nn.Embedding(num_targets, embedding_dim)
        
        self.context_layers = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
    def forward(self, target_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.protein_embeddings(target_ids)
        return self.context_layers(embeddings)

class LigandActivityModel(nn.Module):
    def __init__(self, num_targets):
        super().__init__()
        
        self.molecular_encoder = FineTunedChemBERTaEncoder()
        self.protein_encoder = SimpleProteinEncoder(num_targets, embedding_dim=128)
        
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
        
    def forward(self, smiles_list, target_ids, assay_types):
        molecular_features = self.molecular_encoder(smiles_list)
        protein_features = self.protein_encoder(target_ids)
        
        combined = torch.cat([molecular_features, protein_features], dim=1)
        fused_features = self.fusion(combined)
        
        predictions = []
        for i, assay_type in enumerate(assay_types):
            if assay_type == 'IC50':
                pred = self.ic50_head(fused_features[i:i+1])
            elif assay_type == 'Ki':
                pred = self.ki_head(fused_features[i:i+1])
            elif assay_type == 'EC50':
                pred = self.ec50_head(fused_features[i:i+1])
            else:
                pred = self.ic50_head(fused_features[i:i+1])  # Default
            
            predictions.append(pred)
        
        return torch.cat(predictions, dim=0).squeeze(-1)

@app.function(
    image=image,
    volumes={
        "/data": datasets_volume,
        "/models": models_volume
    },
    gpu="T4",
    timeout=7200,
    memory=8192
)
def train_ligand_activity_model():
    print("🧬 GNOSIS MODEL 1: LIGAND ACTIVITY PREDICTOR (CORRECTED)")
    print("=" * 80)
    
    # Load data
    model1_path = Path("/data/gnosis_model1_binding_training.csv")
    df = pd.read_csv(model1_path)
    print(f"   ✅ Loaded: {len(df):,} ligand-target records")
    print(f"   📊 Unique compounds: {df['SMILES'].nunique():,}")
    print(f"   📊 Unique targets: {df['target_name'].nunique():,}")
    
    # Preprocessing
    valid_smiles = df['SMILES'].notna() & (df['SMILES'] != '')
    df = df[valid_smiles].copy()
    print(f"   📊 After SMILES filtering: {len(df):,} records")
    
    # Target sparsity handling
    target_counts = df['target_name'].value_counts()
    print(f"   📊 Targets before filtering: {len(target_counts)}")
    print(f"   📊 Records per target: min={target_counts.min()}, max={target_counts.max()}, median={target_counts.median()}")
    
    # Filter targets with sufficient data (≥200 records)
    min_records_per_target = 200
    valid_targets = target_counts[target_counts >= min_records_per_target].index
    filtered_df = df[df['target_name'].isin(valid_targets)].copy()
    
    print(f"   ✅ Targets after filtering (≥{min_records_per_target} records): {len(valid_targets)}")
    print(f"   ✅ Records after filtering: {len(filtered_df):,}")
    
    # Prepare activity data - use existing computed values
    model_df = filtered_df.copy()
    
    # Create unified activity column based on assay type
    model_df['pActivity'] = np.nan
    
    # Fill pActivity based on assay type
    ic50_mask = (model_df['assay_type'] == 'IC50') & model_df['pIC50'].notna()
    ki_mask = (model_df['assay_type'] == 'Ki') & model_df['pKi'].notna()
    ec50_mask = (model_df['assay_type'] == 'EC50') & model_df['pEC50'].notna()
    
    model_df.loc[ic50_mask, 'pActivity'] = model_df.loc[ic50_mask, 'pIC50']
    model_df.loc[ki_mask, 'pActivity'] = model_df.loc[ki_mask, 'pKi']
    model_df.loc[ec50_mask, 'pActivity'] = model_df.loc[ec50_mask, 'pEC50']
    
    # Filter valid activities
    model_df = model_df[model_df['pActivity'].notna()].copy()
    model_df = model_df[(model_df['pActivity'] >= 2.0) & (model_df['pActivity'] <= 10.0)].copy()
    
    print(f"   ✅ Valid training records: {len(model_df):,}")
    
    # Create encoders
    target_encoder = LabelEncoder()
    target_list = sorted(model_df['target_name'].unique())
    target_encoder.fit(target_list)
    model_df['target_id'] = target_encoder.transform(model_df['target_name'])
    
    # Activity type distribution
    assay_dist = model_df['assay_type'].value_counts()
    print(f"   📊 Assay distribution: {assay_dist.to_dict()}")
    print(f"   📊 pActivity range: {model_df['pActivity'].min():.2f} - {model_df['pActivity'].max():.2f}")
    
    # Calculate loss weights
    target_weights = {}
    for target in target_list:
        count = len(model_df[model_df['target_name'] == target])
        target_weights[target] = 1.0 / math.sqrt(count)
    
    loss_weights = torch.FloatTensor([target_weights[target] for target in model_df['target_name']])
    print(f"   ✅ Loss weights calculated: range {loss_weights.min():.4f} - {loss_weights.max():.4f}")
    
    # Train-test split
    train_idx, test_idx = train_test_split(
        model_df.index, 
        test_size=0.2, 
        random_state=42,
        stratify=model_df['target_name']
    )
    
    print(f"   📊 Training samples: {len(train_idx):,}")
    print(f"   📊 Test samples: {len(test_idx):,}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LigandActivityModel(len(target_list)).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model parameters: {total_params:,}")
    
    # FIXED parameter grouping
    chemberta_param_names = set()
    for name, param in model.molecular_encoder.chemberta.named_parameters():
        chemberta_param_names.add(f"molecular_encoder.chemberta.{name}")
    
    chemberta_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if name in chemberta_param_names:
            chemberta_params.append(param)
        else:
            other_params.append(param)
    
    print(f"   ✅ ChemBERTa params: {len(chemberta_params)}, Other params: {len(other_params)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': chemberta_params, 'lr': 3e-5},
        {'params': other_params, 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Training
    batch_size = 32
    num_epochs = 50
    best_r2 = -float('inf')
    training_log = []
    
    checkpoint_dir = Path("/models/model1_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🏋️ TRAINING (Target: R² > 0.6, Previous: 0.5994)...")
    
    def weighted_mse_loss(predictions, targets, weights):
        squared_diff = (predictions - targets) ** 2
        weighted_loss = squared_diff * weights
        return weighted_loss.mean()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(train_idx), batch_size):
            batch_end = min(i + batch_size, len(train_idx))
            batch_indices = train_idx[i:batch_end]
            
            batch_df = model_df.loc[batch_indices]
            batch_smiles = batch_df['SMILES'].tolist()
            batch_targets = torch.LongTensor(batch_df['target_id'].values).to(device)
            batch_assays = batch_df['assay_type'].tolist()
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
                batch_assays = batch_df['assay_type'].tolist()
                batch_labels = batch_df['pActivity'].values
                
                predictions = model(batch_smiles, batch_targets, batch_assays).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(batch_labels)
        
        test_r2 = r2_score(all_labels, all_predictions)
        test_mae = mean_absolute_error(all_labels, all_predictions)
        current_lr = optimizer.param_groups[0]['lr']
        
        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "learning_rate": current_lr,
            "timestamp": datetime.now().isoformat()
        }
        training_log.append(log_entry)
        
        print(f"   Epoch {epoch:2d}: Loss={avg_train_loss:.4f} | R²={test_r2:.4f} | MAE={test_mae:.4f} | LR={current_lr:.2e}")
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:02d}_r2_{test_r2:.4f}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'target_encoder': target_encoder,
            'target_list': target_list,
            'test_r2': test_r2,
            'num_targets': len(target_list)
        }, checkpoint_path)
        
        if test_r2 > best_r2:
            best_r2 = test_r2
            print(f"     🎯 NEW BEST R²: {test_r2:.4f}")
            
            best_path = checkpoint_dir / "best_model1.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'target_encoder': target_encoder,
                'target_list': target_list,
                'best_r2': best_r2,
                'num_targets': len(target_list)
            }, best_path)
        
        if test_r2 > 0.6:
            print(f"   🎉 TARGET ACHIEVED: R² = {test_r2:.4f} > 0.6!")
    
    # Final results
    print(f"\n🎉 MODEL 1 TRAINING COMPLETED!")
    print(f"  • Best Test R²: {best_r2:.4f}")
    print(f"  • Previous best: 0.5994")
    print(f"  • Improvement: {'+' if best_r2 > 0.5994 else ''}{(best_r2 - 0.5994):.4f}")
    print(f"  • Target (>0.6): {'✅ ACHIEVED' if best_r2 > 0.6 else '❌ Not reached'}")
    
    # Save final log
    log_path = checkpoint_dir / "training_log_final.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    return {
        'best_r2': float(best_r2),
        'target_achieved': best_r2 > 0.6,
        'improvement_over_previous': float(best_r2 - 0.5994),
        'checkpoints_saved': num_epochs + 1
    }

if __name__ == "__main__":
    with app.run():
        result = train_ligand_activity_model.remote()
        print("Training completed:", result)