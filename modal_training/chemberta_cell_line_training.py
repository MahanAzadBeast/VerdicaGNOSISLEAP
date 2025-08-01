"""
ChemBERTa-based Cell Line Response Model
Multi-modal architecture: ChemBERTa drug embeddings + Genomic features → IC50 prediction
"""

import modal
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import pickle
import warnings
warnings.filterwarnings("ignore")

# Import transformers for ChemBERTa
from transformers import AutoTokenizer, AutoModel
import torch.optim as optim

# Modal setup with comprehensive ML and transformer libraries
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "torch==2.0.1",
    "transformers==4.35.0",
    "tokenizers==0.14.1",
    "scikit-learn==1.3.0",
    "pandas==2.0.3",
    "numpy==1.24.3",
    "scipy==1.11.3",
    "rdkit-pypi==2023.3.2",
    "tqdm==4.64.0"
])

app = modal.App("chemberta-cell-line-training")

# Persistent volumes
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("chemberta-cell-line-models", create_if_missing=True)

class ChemBERTaDrugEncoder(nn.Module):
    """Drug encoder using pretrained ChemBERTa"""
    
    def __init__(self, chemberta_model_name: str = "seyonec/ChemBERTa-zinc-base-v1", freeze_chemberta: bool = False):
        super().__init__()
        
        self.chemberta_model_name = chemberta_model_name
        self.freeze_chemberta = freeze_chemberta
        
        # Load pretrained ChemBERTa
        self.tokenizer = AutoTokenizer.from_pretrained(chemberta_model_name)
        self.chemberta = AutoModel.from_pretrained(chemberta_model_name)
        
        # Freeze ChemBERTa weights if specified
        if freeze_chemberta:
            for param in self.chemberta.parameters():
                param.requires_grad = False
        
        # Get ChemBERTa hidden size
        self.chemberta_hidden_size = self.chemberta.config.hidden_size  # Usually 768
        
        # Projection layer to reduce dimensionality
        self.projection = nn.Sequential(
            nn.Linear(self.chemberta_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
    def forward(self, smiles_batch: List[str]) -> torch.Tensor:
        """
        Encode batch of SMILES using ChemBERTa
        Args:
            smiles_batch: List of SMILES strings
        Returns:
            torch.Tensor: [batch_size, 256] drug embeddings
        """
        # Tokenize SMILES
        encoded = self.tokenizer(
            smiles_batch,
            padding=True,
            truncation=True,
            max_length=512,  # ChemBERTa max length
            return_tensors='pt'
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Get ChemBERTa embeddings
        with torch.set_grad_enabled(not self.freeze_chemberta):
            outputs = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Project to lower dimension
        drug_embedding = self.projection(cls_embedding)  # [batch_size, 256]
        
        return drug_embedding

class GenomicEncoder(nn.Module):
    """Cell line genomic features encoder"""
    
    def __init__(self, genomic_dim: int = 100, output_dim: int = 256):
        super().__init__()
        
        self.genomic_dim = genomic_dim
        self.output_dim = output_dim
        
        # Multi-layer encoder with batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(genomic_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, output_dim)
        )
        
    def forward(self, genomic_features: torch.Tensor) -> torch.Tensor:
        """
        Encode genomic features
        Args:
            genomic_features: [batch_size, genomic_dim]
        Returns:
            torch.Tensor: [batch_size, output_dim] genomic embeddings
        """
        return self.encoder(genomic_features)

class FusionLayer(nn.Module):
    """Fusion layer for drug and genomic embeddings"""
    
    def __init__(self, drug_dim: int = 256, genomic_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        
        # Concatenation fusion
        concat_dim = drug_dim + genomic_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.output_dim = hidden_dim // 4
        
    def forward(self, drug_embedding: torch.Tensor, genomic_embedding: torch.Tensor) -> torch.Tensor:
        """
        Fuse drug and genomic embeddings
        Args:
            drug_embedding: [batch_size, drug_dim]
            genomic_embedding: [batch_size, genomic_dim]
        Returns:
            torch.Tensor: [batch_size, output_dim] fused embedding
        """
        # Concatenate embeddings
        concatenated = torch.cat([drug_embedding, genomic_embedding], dim=1)
        
        # Apply fusion layers
        fused = self.fusion(concatenated)
        
        return fused

class ChemBERTaCellLineModel(nn.Module):
    """
    Complete ChemBERTa-based Cell Line Response Model
    Architecture: ChemBERTa drug encoder + Genomic encoder + Fusion + Regression
    """
    
    def __init__(self, 
                 genomic_dim: int = 100, 
                 chemberta_model: str = "seyonec/ChemBERTa-zinc-base-v1",
                 freeze_chemberta: bool = False):
        super().__init__()
        
        self.genomic_dim = genomic_dim
        self.chemberta_model = chemberta_model
        
        # Drug encoder (ChemBERTa)
        self.drug_encoder = ChemBERTaDrugEncoder(chemberta_model, freeze_chemberta)
        
        # Genomic encoder
        self.genomic_encoder = GenomicEncoder(genomic_dim, output_dim=256)
        
        # Fusion layer
        self.fusion_layer = FusionLayer(drug_dim=256, genomic_dim=256, hidden_dim=512)
        
        # Regression head for log(IC50) prediction
        self.regression_head = nn.Sequential(
            nn.Linear(self.fusion_layer.output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single output for log(IC50)
        )
        
    def forward(self, smiles_batch: List[str], genomic_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            smiles_batch: List of SMILES strings
            genomic_features: [batch_size, genomic_dim] tensor
        Returns:
            torch.Tensor: [batch_size, 1] log(IC50) predictions
        """
        # Encode drug using ChemBERTa
        drug_embedding = self.drug_encoder(smiles_batch)  # [batch_size, 256]
        
        # Encode genomic features
        genomic_embedding = self.genomic_encoder(genomic_features)  # [batch_size, 256]
        
        # Fuse embeddings
        fused_embedding = self.fusion_layer(drug_embedding, genomic_embedding)
        
        # Predict log(IC50)
        log_ic50_pred = self.regression_head(fused_embedding)
        
        return log_ic50_pred

class CellLineDataset(Dataset):
    """Dataset for ChemBERTa-based cell line training"""
    
    def __init__(self, smiles_list: List[str], genomic_features: np.ndarray, targets: np.ndarray):
        self.smiles_list = smiles_list
        self.genomic_features = torch.tensor(genomic_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        return (
            self.smiles_list[idx],
            self.genomic_features[idx],
            self.targets[idx]
        )

def collate_fn(batch):
    """Custom collate function for variable length SMILES"""
    smiles, genomic_features, targets = zip(*batch)
    
    return (
        list(smiles),  # Keep as list for ChemBERTa tokenizer
        torch.stack(genomic_features),
        torch.stack(targets)
    )

@app.function(
    image=image,
    volumes={
        "/vol/datasets": datasets_volume,
        "/vol/models": models_volume
    },
    gpu="A100",  # Need GPU for ChemBERTa
    cpu=8.0,
    memory=64384,  # 64GB for large transformer model
    timeout=14400  # 4 hours
)
def train_chemberta_cell_line_model():
    """
    Train ChemBERTa-based Cell Line Response Model
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 CHEMBERTA-BASED CELL LINE RESPONSE MODEL TRAINING")
    print("=" * 80)
    print("🎯 Architecture: ChemBERTa + Genomics → IC50 prediction")
    
    try:
        datasets_dir = Path("/vol/datasets")
        models_dir = Path("/vol/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load comprehensive GDSC dataset
        print("\n📊 STEP 1: Loading comprehensive GDSC dataset...")
        print("-" * 60)
        
        dataset_path = datasets_dir / "gdsc_comprehensive_training_data.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Comprehensive GDSC dataset not found: {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        print(f"   ✅ Dataset loaded: {len(df):,} records")
        print(f"   📊 Unique cell lines: {df['CELL_LINE_ID'].nunique():,}")
        print(f"   📊 Unique drugs: {df['DRUG_ID'].nunique():,}")
        print(f"   📊 Unique SMILES: {df['SMILES'].nunique():,}")
        
        # Step 2: Prepare features
        print(f"\n🔧 STEP 2: Preparing features for ChemBERTa training...")
        print("-" * 60)
        
        # SMILES for ChemBERTa
        smiles_list = df['SMILES'].astype(str).tolist()
        print(f"   📊 SMILES prepared: {len(smiles_list):,}")
        
        # Genomic features
        genomic_columns = [col for col in df.columns 
                          if any(suffix in col for suffix in ['_mutation', '_cnv', '_expression'])]
        
        if not genomic_columns:
            raise ValueError("No genomic features found in dataset")
        
        genomic_data = df[genomic_columns].copy()
        
        # Handle missing values and convert dtypes
        for col in genomic_columns:
            genomic_data[col] = pd.to_numeric(genomic_data[col], errors='coerce').fillna(0.0)
        
        genomic_features = genomic_data.values.astype(np.float32)
        print(f"   📊 Genomic features: {genomic_features.shape}")
        print(f"   📊 Feature types: {len([col for col in genomic_columns if '_mutation' in col])} mutations, "
              f"{len([col for col in genomic_columns if '_cnv' in col])} CNVs, "
              f"{len([col for col in genomic_columns if '_expression' in col])} expression")
        
        # Standardize genomic features
        genomic_scaler = RobustScaler()  # More robust to outliers
        genomic_features = genomic_scaler.fit_transform(genomic_features)
        
        # Targets (log IC50)
        ic50_values = pd.to_numeric(df['IC50_nM'], errors='coerce').fillna(1000.0).values
        log_ic50_values = np.log10(ic50_values).astype(np.float32)
        
        # Filter reasonable range
        valid_mask = (log_ic50_values >= -1) & (log_ic50_values <= 8)  # 0.1 nM to 100 mM
        
        smiles_list = [smiles_list[i] for i in range(len(smiles_list)) if valid_mask[i]]
        genomic_features = genomic_features[valid_mask]
        log_ic50_values = log_ic50_values[valid_mask]
        df_filtered = df[valid_mask].reset_index(drop=True)
        
        print(f"   📊 After filtering: {len(smiles_list):,} samples")
        print(f"   📊 Log IC50 range: {log_ic50_values.min():.2f} - {log_ic50_values.max():.2f}")
        
        # Step 3: Cell line stratified split
        print(f"\n📋 STEP 3: Creating cell line stratified split...")
        print("-" * 60)
        
        # Use GroupShuffleSplit to ensure cell lines don't appear in both train and test
        unique_cell_lines = df_filtered['CELL_LINE_ID'].unique()
        cell_line_groups = df_filtered['CELL_LINE_ID'].values
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(smiles_list, log_ic50_values, groups=cell_line_groups))
        
        # Further split train into train/val
        train_smiles = [smiles_list[i] for i in train_idx]
        train_genomics = genomic_features[train_idx]
        train_targets = log_ic50_values[train_idx]
        train_groups = cell_line_groups[train_idx]
        
        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)  # 0.25 of train = 0.2 of total
        train_train_idx, val_idx = next(gss_val.split(train_smiles, train_targets, groups=train_groups))
        
        final_train_smiles = [train_smiles[i] for i in train_train_idx]
        final_train_genomics = train_genomics[train_train_idx]
        final_train_targets = train_targets[train_train_idx]
        
        val_smiles = [train_smiles[i] for i in val_idx]
        val_genomics = train_genomics[val_idx]
        val_targets = train_targets[val_idx]
        
        test_smiles = [smiles_list[i] for i in test_idx]
        test_genomics = genomic_features[test_idx]
        test_targets = log_ic50_values[test_idx]
        
        print(f"   📊 Training samples: {len(final_train_smiles):,}")
        print(f"   📊 Validation samples: {len(val_smiles):,}")
        print(f"   📊 Test samples: {len(test_smiles):,}")
        print(f"   📊 Unique cell lines - Train: {len(set(cell_line_groups[train_idx[train_train_idx]]))}, "
              f"Val: {len(set(cell_line_groups[train_idx[val_idx]]))}, "
              f"Test: {len(set(cell_line_groups[test_idx]))}")
        
        # Step 4: Create datasets and dataloaders
        print(f"\n🗂️ STEP 4: Creating datasets and dataloaders...")
        print("-" * 60)
        
        train_dataset = CellLineDataset(final_train_smiles, final_train_genomics, final_train_targets)
        val_dataset = CellLineDataset(val_smiles, val_genomics, val_targets)
        test_dataset = CellLineDataset(test_smiles, test_genomics, test_targets)
        
        # Smaller batch size for transformer model
        batch_size = 8  # Reduced for GPU memory
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        print(f"   📊 Batch size: {batch_size}")
        print(f"   📊 Training batches: {len(train_loader)}")
        print(f"   📊 Validation batches: {len(val_loader)}")
        print(f"   📊 Test batches: {len(test_loader)}")
        
        # Step 5: Initialize ChemBERTa model
        print(f"\n🤖 STEP 5: Initializing ChemBERTa-based model...")
        print("-" * 60)
        
        model = ChemBERTaCellLineModel(
            genomic_dim=genomic_features.shape[1],
            chemberta_model="seyonec/ChemBERTa-zinc-base-v1",
            freeze_chemberta=False  # Allow finetuning
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   🖥️ Device: {device}")
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   📊 Trainable parameters: {trainable_params:,}")
        print(f"   📊 ChemBERTa model: {model.chemberta_model}")
        
        # Step 6: Training setup
        print(f"\n⚙️ STEP 6: Setting up training...")
        print("-" * 60)
        
        # Optimizer with different learning rates for ChemBERTa and other components
        chemberta_params = list(model.drug_encoder.chemberta.parameters())
        other_params = [p for p in model.parameters() if not any(p is cp for cp in chemberta_params)]
        
        optimizer = optim.AdamW([
            {'params': chemberta_params, 'lr': 1e-5},  # Lower LR for pretrained ChemBERTa
            {'params': other_params, 'lr': 1e-4}      # Higher LR for other components
        ], weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        criterion = nn.MSELoss()
        
        print(f"   ✅ Optimizer: AdamW with different LRs (ChemBERTa: 1e-5, Others: 1e-4)")
        print(f"   ✅ Scheduler: ReduceLROnPlateau")
        print(f"   ✅ Loss: MSE")
        
        # Step 7: Training loop
        print(f"\n🚀 STEP 7: Training ChemBERTa Cell Line Model...")
        print("-" * 60)
        
        num_epochs = 100
        best_val_loss = float('inf')
        best_val_r2 = -float('inf')
        patience = 20
        patience_counter = 0
        
        training_history = []
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets_epoch = []
            
            for batch_idx, (batch_smiles, batch_genomics, batch_targets) in enumerate(train_loader):
                batch_genomics = batch_genomics.to(device)
                batch_targets = batch_targets.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(batch_smiles, batch_genomics)
                loss = criterion(predictions, batch_targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_predictions.extend(predictions.detach().cpu().numpy())
                train_targets_epoch.extend(batch_targets.detach().cpu().numpy())
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"     Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}: Loss = {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            train_r2 = r2_score(train_targets_epoch, train_predictions)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets_epoch = []
            
            with torch.no_grad():
                for batch_smiles, batch_genomics, batch_targets in val_loader:
                    batch_genomics = batch_genomics.to(device)
                    batch_targets = batch_targets.to(device).unsqueeze(1)
                    
                    predictions = model(batch_smiles, batch_genomics)
                    loss = criterion(predictions, batch_targets)
                    
                    val_loss += loss.item()
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets_epoch.extend(batch_targets.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_r2 = r2_score(val_targets_epoch, val_predictions)
            val_pearson = pearsonr(np.array(val_targets_epoch).flatten(), np.array(val_predictions).flatten())[0]
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), models_dir / "chemberta_cell_line_best.pth")
            else:
                patience_counter += 1
            
            # Log progress
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_r2': train_r2,
                'val_loss': avg_val_loss,
                'val_r2': val_r2,
                'val_pearson': val_pearson,
                'lr_chemberta': optimizer.param_groups[0]['lr'],
                'lr_others': optimizer.param_groups[1]['lr']
            })
            
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f}, Train R² = {train_r2:.4f}")
                print(f"                Val Loss = {avg_val_loss:.4f}, Val R² = {val_r2:.4f}, Val Pearson = {val_pearson:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch + 1}")
                break
        
        # Step 8: Final evaluation
        print(f"\n📊 STEP 8: Final evaluation on test set...")
        print("-" * 60)
        
        # Load best model
        model.load_state_dict(torch.load(models_dir / "chemberta_cell_line_best.pth"))
        model.eval()
        
        test_predictions = []
        test_targets_final = []
        
        with torch.no_grad():
            for batch_smiles, batch_genomics, batch_targets in test_loader:
                batch_genomics = batch_genomics.to(device)
                batch_targets = batch_targets.to(device).unsqueeze(1)
                
                predictions = model(batch_smiles, batch_genomics)
                
                test_predictions.extend(predictions.cpu().numpy())
                test_targets_final.extend(batch_targets.cpu().numpy())
        
        # Calculate final metrics
        test_r2 = r2_score(test_targets_final, test_predictions)
        test_rmse = np.sqrt(mean_squared_error(test_targets_final, test_predictions))
        test_mae = mean_absolute_error(test_targets_final, test_predictions)
        test_pearson = pearsonr(np.array(test_targets_final).flatten(), np.array(test_predictions).flatten())[0]
        
        print(f"   📊 Final Test R²: {test_r2:.4f}")
        print(f"   📊 Final Test RMSE: {test_rmse:.4f} log(IC50) units")
        print(f"   📊 Final Test MAE: {test_mae:.4f} log(IC50) units")
        print(f"   📊 Final Test Pearson: {test_pearson:.4f}")
        print(f"   📊 Best Validation R²: {best_val_r2:.4f}")
        
        # Step 9: Save model and artifacts
        print(f"\n💾 STEP 9: Saving model and artifacts...")
        print("-" * 60)
        
        # Save final model
        torch.save(model.state_dict(), models_dir / "chemberta_cell_line_final.pth")
        
        # Save scaler
        with open(models_dir / "genomic_scaler.pkl", 'wb') as f:
            pickle.dump(genomic_scaler, f)
        
        # Save training history
        pd.DataFrame(training_history).to_csv(models_dir / "training_history.csv", index=False)
        
        # Save comprehensive metadata
        metadata = {
            'model_type': 'ChemBERTa_Cell_Line_Response_Model',
            'architecture': 'ChemBERTa_Drug_Encoder_Plus_Genomic_Encoder',
            'training_timestamp': datetime.now().isoformat(),
            'chemberta_model': model.chemberta_model,
            'training_data': {
                'total_samples': len(df),
                'filtered_samples': len(smiles_list),
                'training_samples': len(final_train_smiles),
                'validation_samples': len(val_smiles),
                'test_samples': len(test_smiles),
                'unique_cell_lines': df_filtered['CELL_LINE_ID'].nunique(),
                'unique_drugs': df_filtered['DRUG_ID'].nunique(),
                'unique_smiles': df_filtered['SMILES'].nunique(),
                'genomic_features': genomic_features.shape[1]
            },
            'model_architecture': {
                'drug_encoder': f'ChemBERTa ({model.chemberta_model})',
                'genomic_encoder': 'Multi_Layer_MLP_with_BatchNorm',
                'fusion': 'Concatenation_with_Dropout',
                'regression_head': 'MLP_with_Dropout',
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'chemberta_hidden_size': model.drug_encoder.chemberta_hidden_size
            },
            'training_config': {
                'epochs_completed': len(training_history),
                'batch_size': batch_size,
                'learning_rates': {'chemberta': 1e-5, 'others': 1e-4},
                'optimizer': 'AdamW',
                'scheduler': 'ReduceLROnPlateau',
                'loss_function': 'MSE',
                'early_stopping_patience': patience
            },
            'performance': {
                'test_r2': float(test_r2),
                'test_rmse': float(test_rmse),
                'test_mae': float(test_mae),
                'test_pearson': float(test_pearson),
                'best_validation_r2': float(best_val_r2),
                'best_validation_loss': float(best_val_loss)
            },
            'data_sources': {
                'drug_sensitivity': 'GDSC_Comprehensive',
                'genomics': 'Synthetic_Cancer_Genomics',
                'molecular': 'ChemBERTa_Embeddings'
            }
        }
        
        metadata_path = models_dir / "chemberta_cell_line_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Best model: chemberta_cell_line_best.pth")
        print(f"   ✅ Final model: chemberta_cell_line_final.pth")
        print(f"   ✅ Genomic scaler: genomic_scaler.pkl")
        print(f"   ✅ Training history: training_history.csv")
        print(f"   ✅ Metadata: {metadata_path}")
        
        # Final report
        print(f"\n🎉 CHEMBERTA CELL LINE MODEL TRAINING COMPLETED!")
        print("=" * 80)
        print(f"🧬 ChemBERTa + Genomics Architecture Successfully Trained")
        print(f"📊 Final Performance Metrics:")
        print(f"  • Test R²: {test_r2:.4f} ⭐")
        print(f"  • Test RMSE: {test_rmse:.4f} log(IC50) units")
        print(f"  • Test MAE: {test_mae:.4f} log(IC50) units")
        print(f"  • Test Pearson: {test_pearson:.4f}")
        print(f"  • Best Val R²: {best_val_r2:.4f}")
        print(f"📋 Model Capabilities:")
        print(f"  • ✅ ChemBERTa molecular embeddings")
        print(f"  • ✅ Comprehensive genomic features")
        print(f"  • ✅ Cell line stratified training")
        print(f"  • ✅ Multi-modal fusion architecture")
        print(f"🚀 Model ready for production deployment!")
        
        return {
            'status': 'success',
            'model_path': str(models_dir / "chemberta_cell_line_best.pth"),
            'metadata_path': str(metadata_path),
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'test_pearson': float(test_pearson),
            'best_val_r2': float(best_val_r2),
            'training_samples': len(final_train_smiles),
            'test_samples': len(test_smiles),
            'genomic_features': genomic_features.shape[1],
            'epochs_completed': len(training_history),
            'chemberta_integration': True,
            'model_ready': True
        }
        
    except Exception as e:
        print(f"❌ CHEMBERTA CELL LINE MODEL TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🧬 ChemBERTa-based Cell Line Response Model Training")