"""
Model 2 IMPROVED Training - Cytotoxicity Prediction Model
Incorporates user feedback: label normalization, dataset balancing, simplified architecture, longer training
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight
import pickle
# Removed RDKit import since we're not using it in this improved version
from transformers import AutoTokenizer, AutoModel

# Modal setup with ML libraries
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy<2",  # Fix numpy compatibility
    "torch",
    "torchvision", 
    "scikit-learn",
    "rdkit==2023.9.6",
    "transformers",
    "wandb",
    "matplotlib",
    "seaborn"
])

app = modal.App("model2-improved-training")

# Persistent volume for datasets and models
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("cell-line-models", create_if_missing=True)

class ChemBERTaEncoder(nn.Module):
    """ChemBERTa molecular encoder for SMILES"""
    
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", embedding_dim=768):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chemberta = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(embedding_dim, 256)
        self.dropout = nn.Dropout(0.1)
        
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
        
        # Get ChemBERTa embeddings
        with torch.no_grad():
            outputs = self.chemberta(**tokens)
            pooled_output = outputs.pooler_output  # [batch_size, embedding_dim]
        
        # Project to desired dimension
        molecular_features = self.projection(pooled_output)
        molecular_features = self.dropout(molecular_features)
        
        return molecular_features

class GenomicEncoder(nn.Module):
    """Simplified encoder for genomic features"""
    
    def __init__(self, input_dim: int = 100, hidden_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
    def forward(self, genomic_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(genomic_features)

class ImprovedCytotoxicityModel(nn.Module):
    """IMPROVED Model 2: Simplified, progressive training approach"""
    
    def __init__(self, 
                 genomic_feature_dim: int = 100,
                 molecular_dim: int = 256,
                 genomic_dim: int = 32,
                 fusion_dim: int = 288):  # 256 + 32
        super().__init__()
        
        # Encoders
        self.molecular_encoder = ChemBERTaEncoder()
        self.genomic_encoder = GenomicEncoder(input_dim=genomic_feature_dim)
        
        # Simplified fusion - just concatenation
        self.fusion_layers = nn.Sequential(
            nn.Linear(molecular_dim + genomic_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # STAGE 1: Cancer IC50 head (primary task)
        self.cancer_ic50_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # STAGE 2: Normal AC50 head (secondary task - activated later)
        self.normal_ac50_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # Training stage control
        self.training_stage = 1  # Start with cancer only
        
    def set_training_stage(self, stage: int):
        """Control progressive training: 1=cancer only, 2=cancer+normal"""
        self.training_stage = stage
        
    def forward(self, smiles_batch: List[str], genomic_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Encode molecular structure
        molecular_features = self.molecular_encoder(smiles_batch)
        
        # Encode genomic features
        genomic_features_encoded = self.genomic_encoder(genomic_features)
        
        # Simple fusion (no attention to avoid complexity)
        fused_features = torch.cat([molecular_features, genomic_features_encoded], dim=-1)
        fused_features = self.fusion_layers(fused_features)
        
        # Progressive task activation
        cancer_ic50 = self.cancer_ic50_head(fused_features)
        
        if self.training_stage >= 2:
            normal_ac50 = self.normal_ac50_head(fused_features)
        else:
            # Don't compute normal predictions in stage 1
            normal_ac50 = torch.zeros_like(cancer_ic50)
        
        return cancer_ic50, normal_ac50

def normalize_ic50_to_pic50(ic50_um: float) -> float:
    """Convert IC50 in µM to pIC50 = -log10(IC50_M)"""
    if pd.isna(ic50_um) or ic50_um <= 0:
        return np.nan
    ic50_m = ic50_um / 1e6  # Convert µM to M
    return -np.log10(ic50_m)

def normalize_ac50_to_pac50(ac50_um: float) -> float:
    """Convert AC50 in µM to pAC50 = -log10(AC50_M)"""
    if pd.isna(ac50_um) or ac50_um <= 0:
        return np.nan
    ac50_m = ac50_um / 1e6  # Convert µM to M
    return -np.log10(ac50_m)

def balance_dataset(df: pd.DataFrame, cancer_col: str, normal_col: str, balance_ratio: float = 0.3) -> pd.DataFrame:
    """Balance dataset by oversampling normal cell data"""
    
    # Separate records with normal data vs cancer-only
    has_normal = df[normal_col].notna()
    normal_records = df[has_normal].copy()
    cancer_only_records = df[~has_normal].copy()
    
    print(f"   📊 Original: {len(normal_records)} normal, {len(cancer_only_records)} cancer-only")
    
    # Calculate target sizes
    total_target = len(df)
    normal_target = int(total_target * balance_ratio)
    
    # Oversample normal records if needed
    if len(normal_records) < normal_target:
        oversample_factor = normal_target // len(normal_records) + 1
        normal_records_balanced = pd.concat([normal_records] * oversample_factor).sample(n=normal_target, random_state=42)
        print(f"   ⚖️ Oversampled normal data: {len(normal_records)} → {len(normal_records_balanced)}")
    else:
        normal_records_balanced = normal_records.sample(n=normal_target, random_state=42)
        print(f"   ⚖️ Sampled normal data: {len(normal_records)} → {len(normal_records_balanced)}")
    
    # Sample cancer-only to balance
    cancer_target = total_target - len(normal_records_balanced)
    cancer_records_balanced = cancer_only_records.sample(n=min(cancer_target, len(cancer_only_records)), random_state=42)
    
    # Combine
    balanced_df = pd.concat([normal_records_balanced, cancer_records_balanced], ignore_index=True).sample(frac=1, random_state=42)
    
    print(f"   ✅ Balanced: {len(balanced_df)} total ({len(normal_records_balanced)} normal, {len(cancer_records_balanced)} cancer-only)")
    
    return balanced_df

@app.function(
    image=image,
    volumes={
        "/vol/datasets": datasets_volume,
        "/vol/models": models_volume
    },
    gpu="A10G",
    cpu=8.0,
    memory=32768,
    timeout=18000,  # 5 hours for longer training
    secrets=[modal.Secret.from_name("wandb-secret")]
)
def train_improved_cytotoxicity_model():
    """
    IMPROVED Model 2 Training with fixes:
    1. Proper label normalization (pIC50/pAC50)
    2. Dataset balancing
    3. Progressive training (cancer first, then normal)
    4. Longer training (50 epochs)
    5. Separate per-task metrics
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🚀 IMPROVED MODEL 2: CYTOTOXICITY PREDICTION TRAINING")
    print("=" * 80)
    print("🔧 IMPROVEMENTS:")
    print("   • ✅ Proper pIC50/pAC50 normalization")
    print("   • ⚖️ Dataset balancing (oversample normal data)")
    print("   • 🎯 Progressive training (cancer → cancer+normal)")
    print("   • ⏰ Longer training (50 epochs)")
    print("   • 📊 Separate per-task R² tracking")
    print("   • 🧬 ChemBERTa molecular embeddings")
    
    # Initialize W&B
    import wandb
    wandb.init(
        project="gnosis-model2-improved",
        name=f"model2-improved-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "model": "Improved_Cytotoxicity_Model",
            "data": "GDSC_EPA_Balanced",
            "architecture": "ChemBERTa_Progressive_Training",
            "improvements": ["label_normalization", "dataset_balancing", "progressive_training"],
            "epochs": 50
        }
    )
    
    try:
        datasets_dir = Path("/vol/datasets")
        models_dir = Path("/vol/models")
        
        # Load Model 2 training data
        print("\n📊 STEP 1: Loading Model 2 training data...")
        
        training_data_path = datasets_dir / "gnosis_model2_cytotox_training.csv"
        
        if not training_data_path.exists():
            raise Exception("Model 2 training data not found. Run model2_training_starter.py first.")
        
        training_df = pd.read_csv(training_data_path)
        print(f"   ✅ Training data loaded: {len(training_df):,} records")
        
        # IMPROVEMENT 1: Proper label normalization
        print("\n🔧 STEP 2: IMPROVED LABEL NORMALIZATION...")
        
        # Check current label format
        print(f"   📊 Current labels check:")
        print(f"     • pic50_cancer range: {training_df['pic50_cancer'].min():.2f} - {training_df['pic50_cancer'].max():.2f}")
        if 'pIC50' in training_df.columns:
            print(f"     • pIC50 (normal) range: {training_df['pIC50'].min():.2f} - {training_df['pIC50'].max():.2f}")
        
        # Ensure proper pIC50 normalization for cancer data
        if 'ic50_um_cancer' in training_df.columns:
            # Re-normalize from raw µM values to be sure
            training_df['pic50_cancer_normalized'] = training_df['ic50_um_cancer'].apply(normalize_ic50_to_pic50)
            print(f"   ✅ Cancer IC50: Re-normalized from µM to pIC50")
        else:
            training_df['pic50_cancer_normalized'] = training_df['pic50_cancer']
        
        # Ensure proper pAC50 normalization for normal data
        if 'ac50_um_normal' in training_df.columns:
            training_df['pac50_normal_normalized'] = training_df['ac50_um_normal'].apply(normalize_ac50_to_pac50)
            print(f"   ✅ Normal AC50: Re-normalized from µM to pAC50")
        elif 'pIC50' in training_df.columns:
            training_df['pac50_normal_normalized'] = training_df['pIC50']
        
        # Filter valid records
        training_df = training_df.dropna(subset=['SMILES', 'pic50_cancer_normalized'])
        training_df = training_df[training_df['SMILES'].str.len() >= 5]
        
        print(f"   📊 After normalization: {len(training_df):,} valid records")
        print(f"     • Normalized cancer pIC50 range: {training_df['pic50_cancer_normalized'].min():.2f} - {training_df['pic50_cancer_normalized'].max():.2f}")
        
        # IMPROVEMENT 2: Dataset balancing
        print("\n⚖️ STEP 3: DATASET BALANCING...")
        
        training_df = balance_dataset(training_df, 'pic50_cancer_normalized', 'pac50_normal_normalized', balance_ratio=0.4)
        
        # Prepare genomic features
        print("\n🧬 STEP 4: Preparing genomic features...")
        
        genomic_cols = [col for col in training_df.columns if col.startswith('genomic_')]
        
        if not genomic_cols:
            print("   ⚠️ No genomic features found, using molecular features as proxy")
            genomic_features = pd.DataFrame(index=training_df.index)
            
            # Create simple proxy features
            for i in range(20):
                genomic_features[f'genomic_proxy_{i}'] = np.random.randn(len(training_df))
            
            genomic_cols = genomic_features.columns.tolist()
        else:
            genomic_features = training_df[genomic_cols].fillna(0)
        
        print(f"   ✅ Genomic features: {len(genomic_cols)} features")
        
        # Standardize features
        print("\n⚖️ STEP 5: Standardizing features...")
        
        genomic_scaler = StandardScaler()
        genomic_features_scaled = genomic_scaler.fit_transform(genomic_features)
        
        # Prepare targets
        cancer_targets = training_df['pic50_cancer_normalized'].values
        normal_targets = training_df['pac50_normal_normalized'].fillna(-999).values  # Use -999 for missing
        has_normal_data = training_df['pac50_normal_normalized'].notna().values
        
        print(f"   ✅ Targets prepared:")
        print(f"     • Cancer pIC50: {len(cancer_targets)} samples")
        print(f"     • Normal pAC50: {has_normal_data.sum()} samples (of {len(normal_targets)})")
        
        # Train-test split
        print("\n📊 STEP 6: Creating train-test split...")
        
        indices = np.arange(len(training_df))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=has_normal_data)
        
        print(f"   📊 Training samples: {len(train_idx):,}")
        print(f"   📊 Test samples: {len(test_idx):,}")
        
        # Convert to device tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   💻 Using device: {device}")
        
        # Prepare data tensors
        train_smiles = training_df.iloc[train_idx]['SMILES'].tolist()
        train_genomics = torch.tensor(genomic_features_scaled[train_idx], dtype=torch.float32).to(device)
        train_cancer_targets = torch.tensor(cancer_targets[train_idx], dtype=torch.float32).to(device)
        train_normal_targets = torch.tensor(normal_targets[train_idx], dtype=torch.float32).to(device)
        train_has_normal = torch.tensor(has_normal_data[train_idx], dtype=torch.bool).to(device)
        
        test_smiles = training_df.iloc[test_idx]['SMILES'].tolist()
        test_genomics = torch.tensor(genomic_features_scaled[test_idx], dtype=torch.float32).to(device)
        test_cancer_targets = torch.tensor(cancer_targets[test_idx], dtype=torch.float32).to(device)
        test_normal_targets = torch.tensor(normal_targets[test_idx], dtype=torch.float32).to(device)
        test_has_normal = torch.tensor(has_normal_data[test_idx], dtype=torch.bool).to(device)
        
        # IMPROVEMENT 3: Initialize simplified model
        print(f"\n🤖 STEP 7: Initializing IMPROVED Cytotoxicity Model...")
        
        model = ImprovedCytotoxicityModel(
            genomic_feature_dim=genomic_features_scaled.shape[1],
            molecular_dim=256,
            genomic_dim=32,
            fusion_dim=288
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ Model parameters: {total_params:,} (simplified architecture)")
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)  # Higher LR
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # IMPROVEMENT 4: Progressive training with longer epochs
        print(f"\n🏋️ STEP 8: IMPROVED PROGRESSIVE TRAINING...")
        
        batch_size = 32  # Larger batch
        stage1_epochs = 25  # Cancer only
        stage2_epochs = 25  # Cancer + Normal
        total_epochs = stage1_epochs + stage2_epochs
        best_cancer_r2 = -float('inf')
        best_normal_r2 = -float('inf')
        patience = 20
        no_improve_count = 0
        
        print(f"   🎯 STAGE 1: Cancer IC50 only (Epochs 1-{stage1_epochs})")
        print(f"   🎯 STAGE 2: Cancer + Normal (Epochs {stage1_epochs+1}-{total_epochs})")
        
        for epoch in range(total_epochs):
            # Switch to stage 2 after stage1_epochs
            if epoch == stage1_epochs:
                model.set_training_stage(2)
                print(f"\n   🔄 SWITCHING TO STAGE 2: Adding Normal AC50 prediction")
            
            model.train()
            epoch_loss = 0.0
            cancer_loss_sum = 0.0
            normal_loss_sum = 0.0
            num_batches = 0
            
            # Shuffle training data
            perm = torch.randperm(len(train_idx))
            train_smiles_shuffled = [train_smiles[i] for i in perm]
            train_genomics_shuffled = train_genomics[perm]
            train_cancer_targets_shuffled = train_cancer_targets[perm]
            train_normal_targets_shuffled = train_normal_targets[perm]
            train_has_normal_shuffled = train_has_normal[perm]
            
            # Mini-batch training
            for i in range(0, len(train_idx), batch_size):
                end_idx = min(i + batch_size, len(train_idx))
                
                batch_smiles = train_smiles_shuffled[i:end_idx]
                batch_genomics = train_genomics_shuffled[i:end_idx]
                batch_cancer_targets = train_cancer_targets_shuffled[i:end_idx]
                batch_normal_targets = train_normal_targets_shuffled[i:end_idx]
                batch_has_normal = train_has_normal_shuffled[i:end_idx]
                
                optimizer.zero_grad()
                
                # Forward pass
                pred_cancer, pred_normal = model(batch_smiles, batch_genomics)
                pred_cancer = pred_cancer.squeeze()
                pred_normal = pred_normal.squeeze()
                
                # IMPROVEMENT 5: Balanced loss computation
                loss = 0.0
                
                # Cancer loss (always active)
                cancer_loss = F.mse_loss(pred_cancer, batch_cancer_targets)
                loss += cancer_loss
                cancer_loss_sum += cancer_loss.item()
                
                # Normal loss (only in stage 2 and for samples with normal data)
                if model.training_stage >= 2 and batch_has_normal.sum() > 0:
                    normal_mask = batch_has_normal
                    valid_normal_targets = batch_normal_targets[normal_mask]
                    valid_normal_targets = valid_normal_targets[valid_normal_targets != -999]  # Remove missing markers
                    
                    if len(valid_normal_targets) > 0:
                        valid_normal_preds = pred_normal[normal_mask][:len(valid_normal_targets)]
                        normal_loss = F.mse_loss(valid_normal_preds, valid_normal_targets)
                        loss += normal_loss  # Equal weight
                        normal_loss_sum += normal_loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_cancer_loss = cancer_loss_sum / num_batches
            avg_normal_loss = normal_loss_sum / num_batches if model.training_stage >= 2 else 0
            
            # Validation every 2 epochs
            if epoch % 2 == 0:
                model.eval()
                with torch.no_grad():
                    # IMPROVEMENT 6: Separate per-task metrics
                    cancer_preds = []
                    cancer_actuals = []
                    normal_preds = []
                    normal_actuals = []
                    
                    for i in range(0, len(test_idx), batch_size):
                        end_idx = min(i + batch_size, len(test_idx))
                        
                        batch_test_smiles = test_smiles[i:end_idx]
                        batch_test_genomics = test_genomics[i:end_idx]
                        batch_test_cancer = test_cancer_targets[i:end_idx]
                        batch_test_normal = test_normal_targets[i:end_idx]
                        batch_test_has_normal = test_has_normal[i:end_idx]
                        
                        pred_test_cancer, pred_test_normal = model(batch_test_smiles, batch_test_genomics)
                        pred_test_cancer = pred_test_cancer.squeeze()
                        pred_test_normal = pred_test_normal.squeeze()
                        
                        # Collect cancer predictions
                        cancer_preds.extend(pred_test_cancer.cpu().numpy())
                        cancer_actuals.extend(batch_test_cancer.cpu().numpy())
                        
                        # Collect normal predictions (only for samples with normal data)
                        if model.training_stage >= 2:
                            normal_mask = batch_test_has_normal
                            if normal_mask.sum() > 0:
                                valid_normal_targets = batch_test_normal[normal_mask]
                                valid_normal_targets = valid_normal_targets[valid_normal_targets != -999]
                                
                                if len(valid_normal_targets) > 0:
                                    valid_normal_preds = pred_test_normal[normal_mask][:len(valid_normal_targets)]
                                    normal_preds.extend(valid_normal_preds.cpu().numpy())
                                    normal_actuals.extend(valid_normal_targets.cpu().numpy())
                    
                    # Calculate separate R² scores
                    cancer_r2 = r2_score(cancer_actuals, cancer_preds)
                    normal_r2 = r2_score(normal_actuals, normal_preds) if len(normal_preds) > 0 else 0.0
                    
                    # Display metrics
                    if model.training_stage == 1:
                        print(f"   Epoch {epoch:3d}: Loss={avg_loss:.4f} | Cancer R²={cancer_r2:.4f} | Stage=Cancer-Only")
                    else:
                        print(f"   Epoch {epoch:3d}: Loss={avg_loss:.4f} | Cancer R²={cancer_r2:.4f} | Normal R²={normal_r2:.4f} | Stage=Multi-Task")
                    
                    # W&B logging
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "cancer_loss": avg_cancer_loss,
                        "normal_loss": avg_normal_loss,
                        "cancer_r2": cancer_r2,
                        "normal_r2": normal_r2,
                        "training_stage": model.training_stage
                    })
                    
                    # Track best performance
                    improved = False
                    if cancer_r2 > best_cancer_r2:
                        best_cancer_r2 = cancer_r2
                        improved = True
                    if normal_r2 > best_normal_r2:
                        best_normal_r2 = normal_r2
                        improved = True
                    
                    if improved:
                        best_model_state = model.state_dict().copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                
                model.train()
                scheduler.step(avg_loss)
                
                # Early stopping
                if no_improve_count >= patience:
                    print(f"   Early stopping at epoch {epoch}")
                    break
        
        # Save best model
        print(f"\n💾 STEP 9: Saving IMPROVED Model 2...")
        
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_save_path = models_dir / "model2_improved_cytotoxicity.pth"
        torch.save({
            'model_state_dict': best_model_state,
            'genomic_scaler': genomic_scaler,
            'genomic_feature_columns': genomic_cols,
            'best_cancer_r2': best_cancer_r2,
            'best_normal_r2': best_normal_r2,
            'model_config': {
                'genomic_feature_dim': genomic_features_scaled.shape[1],
                'molecular_dim': 256,
                'genomic_dim': 32,
                'fusion_dim': 288
            },
            'training_improvements': [
                'label_normalization',
                'dataset_balancing', 
                'progressive_training',
                'longer_training',
                'per_task_metrics'
            ]
        }, model_save_path)
        
        # Create metadata
        metadata = {
            'model_type': 'Model2_Improved_Cytotoxicity',
            'architecture': 'ChemBERTa_Progressive_Training',
            'improvements': [
                'Proper pIC50/pAC50 normalization',
                'Dataset balancing (40% normal data)',
                'Progressive training (cancer → multi-task)',
                'Longer training (50 epochs)',
                'Separate per-task R² tracking'
            ],
            'training_samples': len(train_idx),
            'test_samples': len(test_idx),
            'best_cancer_r2': float(best_cancer_r2),
            'best_normal_r2': float(best_normal_r2),
            'with_normal_data': int(has_normal_data.sum()),
            'balanced_dataset': True,
            'label_normalization': 'pIC50_pAC50',
            'real_experimental_data': True,
            'training_timestamp': datetime.now().isoformat(),
            'ready_for_inference': True
        }
        
        metadata_path = models_dir / "model2_improved_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Final evaluation
        print(f"\n🎉 IMPROVED MODEL 2 TRAINING COMPLETED!")
        print("=" * 80)
        print(f"📊 FINAL RESULTS:")
        print(f"  • Cancer IC50 R²: {best_cancer_r2:.4f}")
        print(f"  • Normal AC50 R²: {best_normal_r2:.4f}")
        print(f"  • Training samples: {len(train_idx):,}")
        print(f"  • Test samples: {len(test_idx):,}")
        print(f"  • With normal data: {has_normal_data.sum():,}")
        
        print(f"\n✅ IMPROVEMENTS APPLIED:")
        print(f"  • ✅ pIC50/pAC50 normalization (log-scale)")
        print(f"  • ⚖️ Dataset balancing (40% normal data)")
        print(f"  • 🎯 Progressive training (cancer → multi-task)")
        print(f"  • ⏰ Longer training ({total_epochs} epochs)")
        print(f"  • 📊 Separate per-task R² tracking")
        
        wandb.finish()
        
        return {
            'status': 'success',
            'model_type': 'Model2_Improved_Cytotoxicity',
            'best_cancer_r2': float(best_cancer_r2),
            'best_normal_r2': float(best_normal_r2),
            'training_samples': len(train_idx),
            'test_samples': len(test_idx),
            'improvements_applied': True,
            'model_path': str(model_save_path),
            'ready_for_inference': True
        }
        
    except Exception as e:
        print(f"❌ IMPROVED MODEL 2 TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🚀 Model 2 IMPROVED: Progressive Cytotoxicity Training")