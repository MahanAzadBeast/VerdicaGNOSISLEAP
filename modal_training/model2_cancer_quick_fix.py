"""
Model 2 CANCER IC50 TRAINING - QUICK FIX
Fixed the parameter grouping issue
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

# Simple Modal image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas")
    .pip_install("numpy")
    .pip_install("scikit-learn")
    .pip_install("torch")
    .pip_install("transformers")
)

app = modal.App("model2-cancer-quick-fix")

# Volumes
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("cell-line-models", create_if_missing=True)

class ChemBERTaEncoder(nn.Module):
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

class GenomicEncoder(nn.Module):
    def __init__(self, num_genomic_features, embedding_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(num_genomic_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, genomic_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(genomic_features)

class CancerIC50Model(nn.Module):
    def __init__(self, num_genomic_features):
        super().__init__()
        
        self.molecular_encoder = ChemBERTaEncoder()
        self.genomic_encoder = GenomicEncoder(num_genomic_features)
        
        self.fusion = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.cancer_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, smiles_list, genomic_features):
        molecular_features = self.molecular_encoder(smiles_list)
        genomic_features = self.genomic_encoder(genomic_features)
        
        combined = torch.cat([molecular_features, genomic_features], dim=1)
        fused_features = self.fusion(combined)
        
        cancer_pred = self.cancer_head(fused_features)
        return cancer_pred.squeeze(-1)

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
def train_cancer_ic50_model():
    print("🧬 GNOSIS MODEL 2: CANCER IC50 PREDICTION (QUICK FIX)")
    print("=" * 80)
    
    # Load data
    gdsc_path = Path("/data/gnosis_model2_cytotox_training.csv")
    df = pd.read_csv(gdsc_path)
    print(f"   ✅ Loaded: {len(df):,} cancer records")
    
    # Simple preprocessing
    valid_smiles = df['SMILES'].notna() & (df['SMILES'] != '')
    df = df[valid_smiles].copy()
    df = df[(df['pic50_cancer'] >= 2.0) & (df['pic50_cancer'] <= 10.0)].copy()
    
    print(f"   ✅ After preprocessing: {len(df):,} records")
    print(f"   📊 pIC50 range: {df['pic50_cancer'].min():.2f} - {df['pic50_cancer'].max():.2f}")
    
    # Create features
    genomic_cols = [col for col in df.columns if col.startswith('genomic_')]
    genomic_data = df[genomic_cols].fillna(0).values
    
    cell_line_encoder = LabelEncoder()
    cell_line_encoded = cell_line_encoder.fit_transform(df['cell_line_id'])
    
    tissue_encoder = LabelEncoder()
    tissue_encoded = tissue_encoder.fit_transform(df['tissue_type'])
    
    all_features = np.column_stack([
        genomic_data,
        cell_line_encoded.reshape(-1, 1),
        tissue_encoded.reshape(-1, 1)
    ])
    
    num_features = all_features.shape[1]
    print(f"   📊 Genomic features: {num_features}")
    
    # Train-test split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=pd.cut(df['pic50_cancer'], bins=5, labels=False)
    )
    
    train_genomic = torch.FloatTensor(all_features[train_df.index.values])
    test_genomic = torch.FloatTensor(all_features[test_df.index.values])
    train_labels = torch.FloatTensor(train_df['pic50_cancer'].values)
    test_labels = torch.FloatTensor(test_df['pic50_cancer'].values)
    
    print(f"   📊 Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CancerIC50Model(num_features).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model parameters: {total_params:,}")
    
    # FIXED: Proper parameter grouping using named parameters
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
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.MSELoss()
    
    # Training
    batch_size = 32
    num_epochs = 50
    best_r2 = -float('inf')
    training_log = []
    
    checkpoint_dir = Path("/models/model2_cancer_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🏋️ TRAINING (Target R² > 0.6)...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(train_df), batch_size):
            batch_end = min(i + batch_size, len(train_df))
            batch_indices = train_df.index[i:batch_end].values
            
            batch_smiles = train_df.loc[batch_indices, 'SMILES'].tolist()
            batch_genomic = train_genomic[i:batch_end].to(device)
            batch_labels = train_labels[i:batch_end].to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_smiles, batch_genomic)
            loss = criterion(predictions, batch_labels)
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
            for i in range(0, len(test_df), batch_size):
                batch_end = min(i + batch_size, len(test_df))
                batch_indices = test_df.index[i:batch_end].values
                
                batch_smiles = test_df.loc[batch_indices, 'SMILES'].tolist()
                batch_genomic = test_genomic[i:batch_end].to(device)
                batch_labels = test_labels[i:batch_end]
                
                predictions = model(batch_smiles, batch_genomic).cpu()
                all_predictions.extend(predictions.tolist())
                all_labels.extend(batch_labels.tolist())
        
        test_r2 = r2_score(all_labels, all_predictions)
        test_mae = mean_absolute_error(all_labels, all_predictions)
        current_lr = optimizer.param_groups[0]['lr']
        
        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "cancer_r2": test_r2,
            "cancer_mae": test_mae,
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
            'cancer_r2': test_r2,
            'cancer_mae': test_mae,
            'num_genomic_features': num_features
        }, checkpoint_path)
        
        if test_r2 > best_r2:
            best_r2 = test_r2
            print(f"     🎯 NEW BEST R²: {test_r2:.4f}")
            
            best_path = checkpoint_dir / "best_cancer_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'cancer_r2': best_r2,
                'num_genomic_features': num_features
            }, best_path)
        
        if test_r2 > 0.6:
            print(f"   🎉 TARGET ACHIEVED: R² = {test_r2:.4f} > 0.6!")
    
    # Final results
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"  • Best Cancer R²: {best_r2:.4f}")
    print(f"  • Target (>0.6): {'✅ ACHIEVED' if best_r2 > 0.6 else '❌ Not reached'}")
    
    # Save final log
    log_path = checkpoint_dir / "training_log_final.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    return {
        'best_cancer_r2': float(best_r2),
        'target_achieved': best_r2 > 0.6,
        'checkpoints_saved': num_epochs + 1
    }

if __name__ == "__main__":
    with app.run():
        result = train_cancer_ic50_model.remote()
        print("Training completed:", result)