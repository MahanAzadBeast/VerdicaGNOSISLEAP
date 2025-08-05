"""
Model 2 CANCER-ONLY TRAINING - FIXED DOCKER IMAGE
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
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas")
    .pip_install("numpy")
    .pip_install("scikit-learn")
    .pip_install("matplotlib")
    .pip_install("seaborn")
    .pip_install("torch")
    .pip_install("torchvision")
    .pip_install("transformers")
    .pip_install("wandb")
)

app = modal.App("model2-cancer-only-fixed")

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
    """Simplified genomic encoder for cancer cell lines"""
    
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

class CancerOnlyModel(nn.Module):
    """SIMPLIFIED: Cancer-only IC50 prediction model"""
    
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
        
        # Single cancer IC50 prediction head
        self.cancer_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        print("   ✅ Architecture: Cancer-only IC50 prediction (simplified)")
        
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
    """Create simple genomic features from cell line names"""
    
    # Simple features based on cell line characteristics
    cell_lines = df['cell_line'].unique()
    cell_line_map = {cl: idx for idx, cl in enumerate(cell_lines)}
    
    # One-hot encode cell lines (simplified genomic representation)
    num_cell_lines = len(cell_lines)
    genomic_features = np.zeros((len(df), num_cell_lines))
    
    for i, cell_line in enumerate(df['cell_line']):
        genomic_features[i, cell_line_map[cell_line]] = 1.0
    
    print(f"   📊 Genomic features: {num_cell_lines} cell lines (one-hot encoded)")
    
    return torch.FloatTensor(genomic_features), num_cell_lines

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
def train_cancer_only_model():
    """Train cancer-only IC50 prediction model with epoch checkpointing"""
    
    import wandb
    
    print("🧬 GNOSIS MODEL 2: CANCER-ONLY TRAINING (FIXED)")
    print("=" * 80)
    
    # Test imports first
    print("✅ Testing imports...")
    print(f"   pandas version: {pd.__version__}")
    print(f"   numpy version: {np.__version__}")
    print(f"   torch version: {torch.__version__}")
    
    # Initialize wandb
    wandb.init(
        project="gnosis-model2-cancer-only-fixed",
        name=f"cancer-only-fixed-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "model_type": "cancer_only_ic50",
            "architecture": "ChemBERTa + Genomic + Simple_Fusion",
            "target_r2": ">0.6",
            "focus": "cancer_cells_only",
            "fixed_docker_image": True
        }
    )
    
    # STEP 1: Load cancer data only
    print("\n📊 STEP 1: Loading GDSC cancer data...")
    gdsc_path = Path("/data/gnosis_model2_cytotox_training.csv")
    
    if not gdsc_path.exists():
        raise FileNotFoundError(f"GDSC data not found at {gdsc_path}")
    
    df = pd.read_csv(gdsc_path)
    print(f"   ✅ Total records loaded: {len(df):,}")
    
    # Filter to cancer data only (remove any normal cell data)
    cancer_df = df[df['dataset_type'] == 'cancer'].copy()
    print(f"   ✅ Cancer-only records: {len(cancer_df):,}")
    print(f"   📊 Unique compounds: {cancer_df['SMILES'].nunique():,}")
    print(f"   📊 Unique cell lines: {cancer_df['cell_line'].nunique():,}")
    
    # STEP 2: Data preprocessing
    print("\n🔧 STEP 2: Cancer data preprocessing...")
    
    # Remove invalid SMILES
    valid_smiles = cancer_df['SMILES'].notna() & (cancer_df['SMILES'] != '')
    cancer_df = cancer_df[valid_smiles].copy()
    
    # Convert IC50 to pIC50 (log transformation)
    cancer_df['pIC50'] = -np.log10(cancer_df['IC50_nM'] * 1e-9)  # Convert nM to M, then -log10
    
    # Remove outliers (keep reasonable pIC50 range)
    cancer_df = cancer_df[(cancer_df['pIC50'] >= 2.0) & (cancer_df['pIC50'] <= 10.0)].copy()
    
    print(f"   ✅ After preprocessing: {len(cancer_df):,} records")
    print(f"   📊 pIC50 range: {cancer_df['pIC50'].min():.2f} - {cancer_df['pIC50'].max():.2f}")
    print(f"   📊 pIC50 mean±std: {cancer_df['pIC50'].mean():.2f}±{cancer_df['pIC50'].std():.2f}")
    
    # STEP 3: Create features
    print("\n🎯 STEP 3: Creating molecular and genomic features...")
    
    # Create genomic features
    genomic_features, num_genomic_features = create_genomic_features(cancer_df)
    
    # STEP 4: Train-test split
    print("\n📊 STEP 4: Creating train-test split...")
    
    train_df, test_df = train_test_split(
        cancer_df, 
        test_size=0.2, 
        random_state=42,
        stratify=pd.cut(cancer_df['pIC50'], bins=5, labels=False)  # Stratify by pIC50 ranges
    )
    
    print(f"   📊 Training samples: {len(train_df):,}")
    print(f"   📊 Test samples: {len(test_df):,}")
    
    # Create feature tensors
    train_genomic = genomic_features[train_df.index.values]
    test_genomic = genomic_features[test_df.index.values]
    
    train_labels = torch.FloatTensor(train_df['pIC50'].values)
    test_labels = torch.FloatTensor(test_df['pIC50'].values)
    
    # STEP 5: Initialize model
    print(f"\n🤖 STEP 5: Initializing Cancer-Only Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   💻 Using device: {device}")
    
    model = CancerOnlyModel(num_genomic_features).to(device)
    
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
    print(f"\n🏋️ STEP 6: CANCER-ONLY TRAINING (Target R² > 0.6)...")
    
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
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "cancer_r2": test_r2,
            "cancer_mae": test_mae,
            "learning_rate": current_lr
        })
        
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
                    'architecture': 'cancer_only_ic50',
                    'molecular_encoder': 'ChemBERTa_zinc_base_v1',
                    'genomic_encoder': 'one_hot_cell_lines',
                    'fusion_dim': 768
                }
            }, best_path)
        
        # Early success check
        if test_r2 > 0.6:
            print(f"   🎉 TARGET ACHIEVED: Cancer R² = {test_r2:.4f} > 0.6!")
    
    # Final results
    print(f"\n🎉 CANCER-ONLY MODEL 2 TRAINING COMPLETED!")
    print("=" * 80)
    print(f"📊 FINAL RESULTS:")
    print(f"  • Best Cancer R²: {best_r2:.4f}")
    print(f"  • Target (>0.6): {'✅ ACHIEVED' if best_r2 > 0.6 else '❌ Not reached'}")
    print(f"  • Training samples: {len(train_df):,}")
    print(f"  • Test samples: {len(test_df):,}")
    
    wandb.finish()
    
    return {
        'best_cancer_r2': float(best_r2),
        'target_achieved': best_r2 > 0.6,
        'checkpoints_saved': num_epochs + 1
    }

if __name__ == "__main__":
    with app.run():
        result = train_cancer_only_model.remote()
        print("Training completed:", result)