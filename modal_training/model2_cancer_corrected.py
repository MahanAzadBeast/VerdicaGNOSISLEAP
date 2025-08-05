"""
Model 2 CANCER-ONLY TRAINING - CORRECTED DATASET STRUCTURE
All 55,100 records are already cancer-only from GDSC
Using existing pic50_cancer column and rich genomic features
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

# Modal image without WandB
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas")
    .pip_install("numpy")
    .pip_install("scikit-learn")
    .pip_install("matplotlib")
    .pip_install("seaborn")
    .pip_install("torch")
    .pip_install("torchvision")
    .pip_install("transformers")
)

app = modal.App("model2-cancer-corrected")

# Persistent volumes
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("cell-line-models", create_if_missing=True)

class ChemBERTaEncoder(nn.Module):
    """ChemBERTa encoder for molecular features"""
    
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", embedding_dim=768):
        super().__init__()
        
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chemberta = AutoModel.from_pretrained(model_name)
        
        # Fine-tune ChemBERTa
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
    """Rich genomic encoder using all available features"""
    
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
    """Cancer IC50 prediction model with rich genomic context"""
    
    def __init__(self, num_genomic_features):
        super().__init__()
        
        self.molecular_encoder = ChemBERTaEncoder()
        self.genomic_encoder = GenomicEncoder(num_genomic_features)
        
        # Fusion layer: 512 (molecular) + 256 (genomic) = 768
        self.fusion = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cancer IC50 prediction head
        self.cancer_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        print("   ✅ Architecture: Cancer IC50 prediction with rich genomic context")
        
    def forward(self, smiles_list, genomic_features):
        # Encode inputs
        molecular_features = self.molecular_encoder(smiles_list)
        genomic_features = self.genomic_encoder(genomic_features)
        
        # Fuse features
        combined = torch.cat([molecular_features, genomic_features], dim=1)
        fused_features = self.fusion(combined)
        
        # Predict cancer IC50
        cancer_pred = self.cancer_head(fused_features)
        
        return cancer_pred.squeeze(-1)

def create_genomic_features(df: pd.DataFrame) -> torch.Tensor:
    """Create rich genomic features using all available columns"""
    
    # Get all genomic columns
    genomic_cols = [col for col in df.columns if col.startswith('genomic_')]
    print(f"   📊 Found {len(genomic_cols)} genomic feature columns")
    
    # Fill NaN values with 0 for genomic features
    genomic_data = df[genomic_cols].fillna(0).values
    
    # Add cell line encoding
    cell_line_encoder = LabelEncoder()
    cell_line_encoded = cell_line_encoder.fit_transform(df['cell_line_id'])
    
    # Add tissue type encoding
    tissue_encoder = LabelEncoder()
    tissue_encoded = tissue_encoder.fit_transform(df['tissue_type'])
    
    # Combine all features
    all_features = np.column_stack([
        genomic_data,
        cell_line_encoded.reshape(-1, 1),
        tissue_encoded.reshape(-1, 1)
    ])
    
    num_features = all_features.shape[1]
    print(f"   📊 Total genomic features: {num_features} ({len(genomic_cols)} genomic + cell_line + tissue)")
    
    return torch.FloatTensor(all_features), num_features

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
def train_cancer_ic50_model():
    """Train cancer IC50 prediction model with rich genomic features"""
    
    print("🧬 GNOSIS MODEL 2: CANCER IC50 PREDICTION (CORRECTED)")
    print("=" * 80)
    
    # Test imports first
    print("✅ Testing imports...")
    print(f"   pandas version: {pd.__version__}")
    print(f"   numpy version: {np.__version__}")
    print(f"   torch version: {torch.__version__}")
    
    # Create training log file
    training_log = []
    
    # STEP 1: Load cancer data
    print("\n📊 STEP 1: Loading GDSC cancer data...")
    gdsc_path = Path("/data/gnosis_model2_cytotox_training.csv")
    
    if not gdsc_path.exists():
        raise FileNotFoundError(f"GDSC data not found at {gdsc_path}")
    
    df = pd.read_csv(gdsc_path)
    print(f"   ✅ Total cancer records loaded: {len(df):,}")
    print(f"   📊 Unique compounds: {df['SMILES'].nunique():,}")
    print(f"   📊 Unique cell lines: {df['cell_line_id'].nunique():,}")
    print(f"   📊 Tissue types: {df['tissue_type'].nunique():,}")
    
    # STEP 2: Data preprocessing
    print("\n🔧 STEP 2: Cancer data preprocessing...")
    
    # Remove invalid SMILES
    valid_smiles = df['SMILES'].notna() & (df['SMILES'] != '')
    df = df[valid_smiles].copy()
    
    # Use existing pic50_cancer values (already computed!)
    # Remove outliers (keep reasonable pIC50 range)
    df = df[(df['pic50_cancer'] >= 2.0) & (df['pic50_cancer'] <= 10.0)].copy()
    
    print(f"   ✅ After preprocessing: {len(df):,} records")
    print(f"   📊 pIC50 range: {df['pic50_cancer'].min():.2f} - {df['pic50_cancer'].max():.2f}")
    print(f"   📊 pIC50 mean±std: {df['pic50_cancer'].mean():.2f}±{df['pic50_cancer'].std():.2f}")
    
    # STEP 3: Create rich genomic features
    print("\n🎯 STEP 3: Creating molecular and genomic features...")
    
    # Create genomic features
    genomic_features, num_genomic_features = create_genomic_features(df)
    
    # STEP 4: Train-test split
    print("\n📊 STEP 4: Creating train-test split...")
    
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=pd.cut(df['pic50_cancer'], bins=5, labels=False)  # Stratify by pIC50 ranges
    )
    
    print(f"   📊 Training samples: {len(train_df):,}")
    print(f"   📊 Test samples: {len(test_df):,}")
    
    # Create feature tensors
    train_genomic = genomic_features[train_df.index.values]
    test_genomic = genomic_features[test_df.index.values]
    
    train_labels = torch.FloatTensor(train_df['pic50_cancer'].values)
    test_labels = torch.FloatTensor(test_df['pic50_cancer'].values)
    
    # STEP 5: Initialize model
    print(f"\n🤖 STEP 5: Initializing Cancer IC50 Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   💻 Using device: {device}")
    
    model = CancerIC50Model(num_genomic_features).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer with differential learning rates
    chemberta_params = list(model.molecular_encoder.chemberta.parameters())
    other_params = [p for p in model.parameters() if p not in chemberta_params]
    
    optimizer = torch.optim.AdamW([
        {'params': chemberta_params, 'lr': 3e-5},  # Lower LR for pretrained
        {'params': other_params, 'lr': 1e-4}      # Higher LR for new layers
    ], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Loss function
    criterion = nn.MSELoss()
    
    print("   ✅ Optimizer: Differential LR - ChemBERTa(3e-5), Others(1e-4)")
    print("   ✅ Scheduler: Cosine annealing")
    print("   ✅ Target: Cancer IC50 R² > 0.6")
    
    # STEP 6: Training loop with epoch checkpointing
    print(f"\n🏋️ STEP 6: CANCER IC50 TRAINING (Target R² > 0.6)...")
    
    batch_size = 32
    num_epochs = 50
    best_r2 = -float('inf')
    best_model_state = None
    
    # Create checkpoints directory
    checkpoint_dir = Path("/models/model2_cancer_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        # Training batches
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
            
            # Gradient clipping for stability
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
        
        # Calculate metrics
        test_r2 = r2_score(all_labels, all_predictions)
        test_mae = mean_absolute_error(all_labels, all_predictions)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics (save to file instead of wandb)
        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "cancer_r2": test_r2,
            "cancer_mae": test_mae,
            "learning_rate": current_lr,
            "timestamp": datetime.now().isoformat()
        }
        training_log.append(log_entry)
        
        print(f"   Epoch {epoch:2d}: Loss={avg_train_loss:.4f} | Cancer R²={test_r2:.4f} | MAE={test_mae:.4f} | LR={current_lr:.2e}")
        
        # Save checkpoint every epoch
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:02d}_r2_{test_r2:.4f}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'cancer_r2': test_r2,
            'cancer_mae': test_mae,
            'train_loss': avg_train_loss,
            'num_genomic_features': num_genomic_features
        }, checkpoint_path)
        
        # Track best model
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_model_state = model.state_dict().copy()
            print(f"     🎯 NEW BEST Cancer R²: {test_r2:.4f}")
            
            # Save best model separately
            best_path = checkpoint_dir / "best_cancer_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'cancer_r2': best_r2,
                'cancer_mae': test_mae,
                'num_genomic_features': num_genomic_features,
                'model_config': {
                    'architecture': 'cancer_ic50_rich_genomics',
                    'molecular_encoder': 'ChemBERTa_zinc_base_v1',
                    'genomic_encoder': 'rich_genomic_features',
                    'genomic_features': num_genomic_features
                }
            }, best_path)
        
        # Save training log every 5 epochs
        if epoch % 5 == 0:
            log_path = checkpoint_dir / "training_log.json"
            with open(log_path, 'w') as f:
                json.dump(training_log, f, indent=2)
        
        # Early success check
        if test_r2 > 0.6:
            print(f"   🎉 TARGET ACHIEVED: Cancer R² = {test_r2:.4f} > 0.6!")
    
    # Final results
    print(f"\n🎉 CANCER IC50 MODEL TRAINING COMPLETED!")
    print("=" * 80)
    print(f"📊 FINAL RESULTS:")
    print(f"  • Best Cancer R²: {best_r2:.4f}")
    print(f"  • Target (>0.6): {'✅ ACHIEVED' if best_r2 > 0.6 else '❌ Not reached'}")
    print(f"  • Training samples: {len(train_df):,}")
    print(f"  • Test samples: {len(test_df):,}")
    print(f"  • Genomic features: {num_genomic_features}")
    
    # Save final training log
    log_path = checkpoint_dir / "training_log_final.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    return {
        'best_cancer_r2': float(best_r2),
        'target_achieved': best_r2 > 0.6,
        'checkpoints_saved': num_epochs + 1,
        'final_mae': float(test_mae),
        'genomic_features': num_genomic_features
    }

if __name__ == "__main__":
    with app.run():
        result = train_cancer_ic50_model.remote()
        print("Training completed:", result)