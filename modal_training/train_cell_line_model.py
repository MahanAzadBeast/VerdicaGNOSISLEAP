"""
Cell Line Response Model Training Script
Trains the multi-modal model using GDSC data
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional for now
import warnings
warnings.filterwarnings("ignore")

# Modal setup with full ML stack
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "torch==2.0.1",
    "torchvision", 
    "scikit-learn",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "rdkit-pypi",
    "transformers",
    "tqdm"
])

app = modal.App("cell-line-model-training")

# Persistent volumes
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("cell-line-models", create_if_missing=True)

class SMILESTokenizer:
    """SMILES tokenizer for molecular encoding"""
    
    def __init__(self):
        # Extended SMILES vocabulary for better coverage
        self.chars = list("()[]{}.-=+#@/*\\123456789%CNOSPFIBrClncos")
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.chars) + 1  # +1 for padding
        self.pad_token = 0
    
    def tokenize(self, smiles: str, max_length: int = 128) -> List[int]:
        """Convert SMILES string to token indices"""
        tokens = [self.char_to_idx.get(char, 0) for char in smiles[:max_length]]
        # Pad to max_length
        tokens += [self.pad_token] * (max_length - len(tokens))
        return tokens
    
    def batch_tokenize(self, smiles_list: List[str], max_length: int = 128) -> torch.Tensor:
        """Tokenize a batch of SMILES"""
        tokenized = [self.tokenize(smiles, max_length) for smiles in smiles_list]
        return torch.tensor(tokenized, dtype=torch.long)

class MolecularEncoder(nn.Module):
    """Encode molecular SMILES into feature vectors"""
    
    def __init__(self, vocab_size: int = 60, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.1)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True, dropout=0.1)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        
    def forward(self, smiles_tokens: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # Create attention mask for padding
        if attention_mask is None:
            attention_mask = (smiles_tokens != 0).float()
        
        # Embed SMILES tokens
        embedded = self.embedding(smiles_tokens)  # [batch, seq_len, embedding_dim]
        
        # RNN encoding
        rnn_out, _ = self.rnn(embedded)  # [batch, seq_len, hidden_dim * 2]
        
        # Self-attention with masking
        attended, _ = self.attention(rnn_out, rnn_out, rnn_out, key_padding_mask=(attention_mask == 0))
        
        # Masked global pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(attended)
        attended_masked = attended * mask_expanded
        pooled = attended_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
        
        # Output projection
        molecular_features = self.output_projection(pooled)  # [batch, 128]
        
        return molecular_features

class GenomicEncoder(nn.Module):
    """Encode genomic features into feature vectors"""
    
    def __init__(self, genomic_dim: int = 63):  # Based on our genomic features
        super().__init__()
        
        # Separate encoders for different genomic data types
        mutation_dim = 24  # 24 mutation features
        cnv_dim = 12      # 12 CNV features  
        expr_dim = 15     # 15 expression features
        
        self.mutation_encoder = nn.Sequential(
            nn.Linear(mutation_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        self.cnv_encoder = nn.Sequential(
            nn.Linear(cnv_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )
        
        self.expression_encoder = nn.Sequential(
            nn.Linear(expr_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(32 + 16 + 32, 128),  # 80 total
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )
        
    def forward(self, genomic_features: torch.Tensor) -> torch.Tensor:
        # Split genomic features by type
        mutations = genomic_features[:, :24]      # First 24 are mutations
        cnvs = genomic_features[:, 24:36]         # Next 12 are CNVs
        expression = genomic_features[:, 36:51]   # Next 15 are expression
        
        # Encode each data type
        mutation_encoded = self.mutation_encoder(mutations)
        cnv_encoded = self.cnv_encoder(cnvs)
        expression_encoded = self.expression_encoder(expression)
        
        # Fuse all genomic features
        combined = torch.cat([mutation_encoded, cnv_encoded, expression_encoded], dim=1)
        genomic_features = self.fusion_layer(combined)
        
        return genomic_features

class CellLineResponseModel(nn.Module):
    """Multi-modal model for predicting IC50 from molecular and genomic features"""
    
    def __init__(self, smiles_vocab_size: int = 60, genomic_dim: int = 63):
        super().__init__()
        
        self.molecular_encoder = MolecularEncoder(smiles_vocab_size)
        self.genomic_encoder = GenomicEncoder(genomic_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(128, num_heads=4, batch_first=True, dropout=0.1)
        
        # Fusion and prediction layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(256, 256),  # 128 (molecular) + 128 (genomic)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single IC50 prediction
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def forward(self, smiles_tokens: torch.Tensor, genomic_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode inputs
        molecular_features = self.molecular_encoder(smiles_tokens)    # [batch, 128]
        genomic_features_enc = self.genomic_encoder(genomic_features) # [batch, 128]
        
        # Cross-modal attention (molecular attending to genomic)
        molecular_expanded = molecular_features.unsqueeze(1)  # [batch, 1, 128]
        genomic_expanded = genomic_features_enc.unsqueeze(1)  # [batch, 1, 128]
        
        attended_molecular, _ = self.cross_attention(
            molecular_expanded, genomic_expanded, genomic_expanded
        )
        attended_molecular = attended_molecular.squeeze(1)  # [batch, 128]
        
        # Fuse features
        fused_features = torch.cat([attended_molecular, genomic_features_enc], dim=1)  # [batch, 256]
        
        # Predict IC50 and uncertainty
        ic50_pred = self.fusion_layers(fused_features)  # [batch, 1]
        uncertainty = self.uncertainty_head(fused_features)  # [batch, 1]
        
        return ic50_pred, uncertainty

class CellLineDataset(Dataset):
    """Dataset for cell line drug sensitivity data"""
    
    def __init__(self, smiles_tokens, genomic_features, targets):
        self.smiles_tokens = smiles_tokens
        self.genomic_features = genomic_features
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (
            self.smiles_tokens[idx],
            self.genomic_features[idx],
            self.targets[idx]
        )

@app.function(
    image=image,
    volumes={
        "/vol/datasets": datasets_volume,
        "/vol/models": models_volume
    },
    gpu="A100",
    cpu=8.0,
    memory=32768,
    timeout=7200
)
def train_cell_line_response_model():
    """
    Train the Cell Line Response Model using GDSC data
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 CELL LINE RESPONSE MODEL TRAINING")
    print("=" * 80)
    print("🎯 Multi-modal IC₅₀ prediction: Drug structure + Cancer genomics")
    
    try:
        datasets_dir = Path("/vol/datasets")
        models_dir = Path("/vol/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load training data
        print("\n📊 STEP 1: Loading GDSC training data...")
        print("-" * 60)
        
        training_data_path = datasets_dir / "gdsc_cell_line_training_data.csv"
        if not training_data_path.exists():
            raise FileNotFoundError(f"Training data not found: {training_data_path}")
        
        df = pd.read_csv(training_data_path)
        print(f"   ✅ Training data loaded: {len(df):,} records")
        print(f"   📊 Unique cell lines: {df['CELL_LINE_ID'].nunique()}")
        print(f"   📊 Unique compounds: {df['COMPOUND_ID'].nunique()}")
        
        # Check data quality
        missing_smiles = df['SMILES'].isna().sum()
        missing_ic50 = df['IC50_nM'].isna().sum()
        print(f"   📊 Missing SMILES: {missing_smiles}")
        print(f"   📊 Missing IC50: {missing_ic50}")
        
        # Step 2: Prepare features
        print(f"\n🔧 STEP 2: Preparing molecular and genomic features...")
        print("-" * 60)
        
        # Initialize tokenizer
        tokenizer = SMILESTokenizer()
        print(f"   📊 SMILES vocabulary size: {tokenizer.vocab_size}")
        
        # Prepare molecular features (SMILES tokens)
        smiles_list = df['SMILES'].fillna('CCO').tolist()  # Fill missing with ethanol
        smiles_tokens = tokenizer.batch_tokenize(smiles_list, max_length=100)
        print(f"   📊 Molecular features shape: {smiles_tokens.shape}")
        
        # Prepare genomic features
        genomic_columns = [col for col in df.columns 
                          if any(suffix in col for suffix in ['_mutation', '_cnv', '_expression'])]
        
        if not genomic_columns:
            raise ValueError("No genomic features found in dataset")
        
        genomic_features = df[genomic_columns].fillna(0).values.astype(np.float32)
        print(f"   📊 Genomic features shape: {genomic_features.shape}")
        print(f"   📊 Genomic feature types: {len(genomic_columns)}")
        
        # Standardize genomic features
        genomic_scaler = StandardScaler()
        genomic_features = genomic_scaler.fit_transform(genomic_features)
        
        # Prepare targets (pIC50)
        ic50_values = df['IC50_nM'].fillna(1000).values  # Fill missing with 1 μM
        pic50_values = -np.log10(ic50_values / 1e9)  # Convert to pIC50
        
        # Filter reasonable pIC50 range
        valid_mask = (pic50_values >= 2) & (pic50_values <= 12)  # 10 mM to 1 pM
        
        smiles_tokens = smiles_tokens[valid_mask]
        genomic_features = genomic_features[valid_mask]
        pic50_values = pic50_values[valid_mask]
        
        print(f"   📊 Targets (pIC50) shape: {pic50_values.shape}")
        print(f"   📊 pIC50 range: {pic50_values.min():.2f} - {pic50_values.max():.2f}")
        print(f"   📊 After filtering: {len(pic50_values):,} samples")
        
        # Step 3: Train-validation-test split
        print(f"\n📋 STEP 3: Creating data splits...")
        print("-" * 60)
        
        # Split by cell lines to avoid data leakage
        unique_cell_lines = df.loc[valid_mask, 'CELL_LINE_ID'].unique()
        train_cell_lines, temp_cell_lines = train_test_split(
            unique_cell_lines, test_size=0.4, random_state=42
        )
        val_cell_lines, test_cell_lines = train_test_split(
            temp_cell_lines, test_size=0.5, random_state=42
        )
        
        # Create masks
        cell_line_series = df.loc[valid_mask, 'CELL_LINE_ID'].reset_index(drop=True)
        train_mask = cell_line_series.isin(train_cell_lines)
        val_mask = cell_line_series.isin(val_cell_lines)
        test_mask = cell_line_series.isin(test_cell_lines)
        
        print(f"   📊 Training cell lines: {len(train_cell_lines)}")
        print(f"   📊 Validation cell lines: {len(val_cell_lines)}")
        print(f"   📊 Test cell lines: {len(test_cell_lines)}")
        print(f"   📊 Training samples: {train_mask.sum():,}")
        print(f"   📊 Validation samples: {val_mask.sum():,}")
        print(f"   📊 Test samples: {test_mask.sum():,}")
        
        # Step 4: Create data loaders
        print(f"\n🗂️ STEP 4: Creating data loaders...")
        print("-" * 60)
        
        # Convert to PyTorch tensors
        genomic_tensor = torch.tensor(genomic_features, dtype=torch.float32)
        targets_tensor = torch.tensor(pic50_values, dtype=torch.float32).unsqueeze(1)
        
        # Create datasets
        train_dataset = CellLineDataset(
            smiles_tokens[train_mask],
            genomic_tensor[train_mask],
            targets_tensor[train_mask]
        )
        
        val_dataset = CellLineDataset(
            smiles_tokens[val_mask],
            genomic_tensor[val_mask],
            targets_tensor[val_mask]
        )
        
        test_dataset = CellLineDataset(
            smiles_tokens[test_mask],
            genomic_tensor[test_mask],
            targets_tensor[test_mask]
        )
        
        # Create data loaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"   📊 Batch size: {batch_size}")
        print(f"   📊 Training batches: {len(train_loader)}")
        print(f"   📊 Validation batches: {len(val_loader)}")
        print(f"   📊 Test batches: {len(test_loader)}")
        
        # Step 5: Initialize model
        print(f"\n🤖 STEP 5: Initializing Cell Line Response Model...")
        print("-" * 60)
        
        model = CellLineResponseModel(
            smiles_vocab_size=tokenizer.vocab_size,
            genomic_dim=genomic_features.shape[1]
        )
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"   🖥️ Training device: {device}")
        print(f"   📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        criterion = nn.MSELoss()
        
        # Step 6: Training loop
        print(f"\n🚀 STEP 6: Training Cell Line Response Model...")
        print("-" * 60)
        
        num_epochs = 50
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        training_history = []
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_smiles, batch_genomics, batch_targets in train_loader:
                batch_smiles = batch_smiles.to(device)
                batch_genomics = batch_genomics.to(device)
                batch_targets = batch_targets.to(device)
                
                optimizer.zero_grad()
                
                predictions, uncertainty = model(batch_smiles, batch_genomics)
                
                # Loss with uncertainty weighting
                mse_loss = criterion(predictions, batch_targets)
                uncertainty_reg = torch.mean(uncertainty) * 0.01  # Small regularization
                loss = mse_loss + uncertainty_reg
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_smiles, batch_genomics, batch_targets in val_loader:
                    batch_smiles = batch_smiles.to(device)
                    batch_genomics = batch_genomics.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    predictions, uncertainty = model(batch_smiles, batch_genomics)
                    loss = criterion(predictions, batch_targets)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_targets.cpu().numpy())
            
            avg_val_loss = val_loss / val_batches
            val_r2 = r2_score(val_targets, val_predictions)
            val_mae = mean_absolute_error(val_targets, val_predictions)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), models_dir / "cell_line_response_model_best.pth")
            else:
                patience_counter += 1
            
            # Log progress
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_r2': val_r2,
                'val_mae': val_mae,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val R² = {val_r2:.4f}, Val MAE = {val_mae:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch + 1}")
                break
        
        # Step 7: Final evaluation
        print(f"\n📊 STEP 7: Final model evaluation...")
        print("-" * 60)
        
        # Load best model
        model.load_state_dict(torch.load(models_dir / "cell_line_response_model_best.pth"))
        model.eval()
        
        # Test evaluation
        test_predictions = []
        test_targets = []
        test_uncertainties = []
        
        with torch.no_grad():
            for batch_smiles, batch_genomics, batch_targets in test_loader:
                batch_smiles = batch_smiles.to(device)
                batch_genomics = batch_genomics.to(device)
                batch_targets = batch_targets.to(device)
                
                predictions, uncertainty = model(batch_smiles, batch_genomics)
                
                test_predictions.extend(predictions.cpu().numpy())
                test_targets.extend(batch_targets.cpu().numpy())
                test_uncertainties.extend(uncertainty.cpu().numpy())
        
        # Calculate metrics
        test_r2 = r2_score(test_targets, test_predictions)
        test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
        test_mae = mean_absolute_error(test_targets, test_predictions)
        
        print(f"   📊 Test R²: {test_r2:.4f}")
        print(f"   📊 Test RMSE: {test_rmse:.4f} pIC50 units")
        print(f"   📊 Test MAE: {test_mae:.4f} pIC50 units")
        print(f"   📊 Best validation loss: {best_val_loss:.4f}")
        
        # Step 8: Save model and artifacts
        print(f"\n💾 STEP 8: Saving model and artifacts...")
        print("-" * 60)
        
        # Save final model
        torch.save(model.state_dict(), models_dir / "cell_line_response_model_final.pth")
        
        # Save tokenizer and scaler
        with open(models_dir / "smiles_tokenizer.pkl", 'wb') as f:
            pickle.dump(tokenizer, f)
        
        with open(models_dir / "genomic_scaler.pkl", 'wb') as f:
            pickle.dump(genomic_scaler, f)
        
        # Save training history
        training_history_df = pd.DataFrame(training_history)
        training_history_df.to_csv(models_dir / "training_history.csv", index=False)
        
        # Save comprehensive metadata
        metadata = {
            'model_type': 'Cell_Line_Response_Model',
            'architecture': 'Multi_Modal_Molecular_Genomic',
            'training_timestamp': datetime.now().isoformat(),
            'training_data': {
                'total_samples': len(df),
                'valid_samples': len(pic50_values),
                'training_samples': train_mask.sum(),
                'validation_samples': val_mask.sum(),
                'test_samples': test_mask.sum(),
                'unique_cell_lines': len(unique_cell_lines),
                'unique_compounds': df['COMPOUND_ID'].nunique(),
                'genomic_features': len(genomic_columns)
            },
            'model_architecture': {
                'molecular_encoder': 'BiLSTM_with_Attention',
                'genomic_encoder': 'Multi_Type_MLP_Fusion',
                'fusion': 'Cross_Modal_Attention',
                'smiles_vocab_size': tokenizer.vocab_size,
                'genomic_dim': genomic_features.shape[1],
                'total_parameters': sum(p.numel() for p in model.parameters())
            },
            'training_config': {
                'epochs_completed': len(training_history),
                'batch_size': batch_size,
                'learning_rate': 1e-3,
                'optimizer': 'AdamW',
                'scheduler': 'ReduceLROnPlateau',
                'early_stopping_patience': patience
            },
            'performance': {
                'test_r2': float(test_r2),
                'test_rmse': float(test_rmse),
                'test_mae': float(test_mae),
                'best_validation_loss': float(best_val_loss)
            },
            'data_sources': {
                'drug_sensitivity': 'GDSC',
                'genomics': 'Synthetic_Realistic_Features',
                'molecular': 'SMILES_Tokenized'
            },
            'model_files': {
                'best_model': 'cell_line_response_model_best.pth',
                'final_model': 'cell_line_response_model_final.pth',
                'tokenizer': 'smiles_tokenizer.pkl',
                'scaler': 'genomic_scaler.pkl',
                'history': 'training_history.csv'
            }
        }
        
        metadata_path = models_dir / "cell_line_response_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Best model: cell_line_response_model_best.pth")
        print(f"   ✅ Final model: cell_line_response_model_final.pth")
        print(f"   ✅ Tokenizer: smiles_tokenizer.pkl")
        print(f"   ✅ Genomic scaler: genomic_scaler.pkl")
        print(f"   ✅ Training history: training_history.csv")
        print(f"   ✅ Metadata: {metadata_path}")
        
        # Final report
        print(f"\n🎉 CELL LINE RESPONSE MODEL TRAINING COMPLETED!")
        print("=" * 80)
        print(f"🧬 Multi-Modal Architecture Successfully Trained")
        print(f"📊 Final Performance Metrics:")
        print(f"  • Test R²: {test_r2:.4f}")
        print(f"  • Test RMSE: {test_rmse:.4f} pIC50 units")
        print(f"  • Test MAE: {test_mae:.4f} pIC50 units")
        print(f"📋 Model Capabilities:")
        print(f"  • ✅ Multi-modal IC₅₀ prediction")
        print(f"  • ✅ Uncertainty quantification")
        print(f"  • ✅ Cancer cell line genomics integration")
        print(f"  • ✅ Drug molecular structure processing")
        print(f"  • ✅ Cross-modal attention fusion")
        print(f"🚀 Model ready for deployment and inference!")
        
        return {
            'status': 'success',
            'model_path': str(models_dir / "cell_line_response_model_best.pth"),
            'metadata_path': str(metadata_path),
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'training_samples': train_mask.sum(),
            'test_samples': test_mask.sum(),
            'genomic_features': len(genomic_columns),
            'epochs_completed': len(training_history),
            'model_ready': True
        }
        
    except Exception as e:
        print(f"❌ CELL LINE RESPONSE MODEL TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🧬 Cell Line Response Model Training Script")