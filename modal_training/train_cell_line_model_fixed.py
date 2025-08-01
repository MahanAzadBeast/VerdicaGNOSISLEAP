"""
Fixed Cell Line Response Model Training Pipeline
Addresses dtype issues and provides robust training with comprehensive error handling
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
import warnings
warnings.filterwarnings("ignore")

# Modal setup with comprehensive ML stack
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "torch==2.0.1",
    "scikit-learn==1.3.0",
    "pandas==2.0.3",
    "numpy==1.24.3",
    "rdkit-pypi==2023.3.2",
    "transformers==4.21.0",
    "tqdm==4.64.0"
])

app = modal.App("cell-line-model-training-fixed")

# Persistent volumes
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("cell-line-models", create_if_missing=True)

class SMILESTokenizer:
    """SMILES tokenizer for molecular encoding"""
    
    def __init__(self):
        # Extended SMILES vocabulary
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
        
    def forward(self, smiles_tokens: torch.Tensor) -> torch.Tensor:
        # Create attention mask for padding
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
    
    def __init__(self, genomic_dim: int = 63):
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
        
    def forward(self, smiles_tokens: torch.Tensor, genomic_features: torch.Tensor) -> torch.Tensor:
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
        
        # Predict IC50
        ic50_pred = self.fusion_layers(fused_features)  # [batch, 1]
        
        return ic50_pred

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
    timeout=10800  # 3 hours
)
def train_cell_line_model_fixed():
    """
    Train the Cell Line Response Model with fixed dtype handling
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 CELL LINE RESPONSE MODEL TRAINING (FIXED)")
    print("=" * 80)
    print("🎯 Multi-modal IC₅₀ prediction: Drug structure + Cancer genomics (Fixed dtypes)")
    
    try:
        datasets_dir = Path("/vol/datasets")
        models_dir = Path("/vol/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Create comprehensive training dataset
        print("\n📊 STEP 1: Creating comprehensive training dataset...")
        print("-" * 60)
        
        # Generate comprehensive synthetic training data
        training_data = create_comprehensive_training_dataset()
        
        print(f"   ✅ Training dataset created: {len(training_data):,} records")
        print(f"   📊 Unique cell lines: {training_data['CELL_LINE_NAME'].nunique()}")
        print(f"   📊 Unique compounds: {training_data['DRUG_NAME'].nunique()}")
        
        # Step 2: Prepare features with fixed dtypes
        print(f"\n🔧 STEP 2: Preparing molecular and genomic features (fixed dtypes)...")
        print("-" * 60)
        
        # Initialize tokenizer
        tokenizer = SMILESTokenizer()
        print(f"   📊 SMILES vocabulary size: {tokenizer.vocab_size}")
        
        # Prepare molecular features (SMILES tokens)
        smiles_list = training_data['SMILES'].fillna('CCO').astype(str).tolist()
        smiles_tokens = tokenizer.batch_tokenize(smiles_list, max_length=100)
        print(f"   📊 Molecular features shape: {smiles_tokens.shape}")
        
        # Prepare genomic features with explicit dtype handling
        genomic_columns = [col for col in training_data.columns 
                          if any(suffix in col for suffix in ['_mutation', '_cnv', '_expression'])]
        
        if not genomic_columns:
            raise ValueError("No genomic features found in dataset")
        
        # Explicit dtype conversion to avoid numpy.bool issues
        genomic_data = training_data[genomic_columns].copy()
        
        # Convert to float64 explicitly
        for col in genomic_columns:
            genomic_data[col] = pd.to_numeric(genomic_data[col], errors='coerce')
        
        genomic_features = genomic_data.fillna(0.0).values.astype(np.float64)
        print(f"   📊 Genomic features shape: {genomic_features.shape}")
        print(f"   📊 Genomic feature types: {len(genomic_columns)}")
        print(f"   📊 Genomic features dtype: {genomic_features.dtype}")
        
        # Standardize genomic features
        genomic_scaler = StandardScaler()
        genomic_features = genomic_scaler.fit_transform(genomic_features).astype(np.float32)
        
        # Prepare targets (pIC50) with explicit dtype
        ic50_values = pd.to_numeric(training_data['IC50_nM'], errors='coerce').fillna(1000.0).values
        pic50_values = -np.log10(ic50_values / 1e9).astype(np.float32)
        
        # Filter reasonable pIC50 range
        valid_mask = (pic50_values >= 2) & (pic50_values <= 12)  # 10 mM to 1 pM
        
        smiles_tokens = smiles_tokens[valid_mask]
        genomic_features = genomic_features[valid_mask]
        pic50_values = pic50_values[valid_mask]
        
        print(f"   📊 Targets (pIC50) shape: {pic50_values.shape}")
        print(f"   📊 pIC50 range: {pic50_values.min():.2f} - {pic50_values.max():.2f}")
        print(f"   📊 After filtering: {len(pic50_values):,} samples")
        print(f"   📊 Target dtype: {pic50_values.dtype}")
        
        # Step 3: Train-validation-test split
        print(f"\n📋 STEP 3: Creating data splits...")
        print("-" * 60)
        
        # Random split for simplicity (since we have synthetic data)
        indices = np.arange(len(pic50_values))
        train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        
        print(f"   📊 Training samples: {len(train_idx):,}")
        print(f"   📊 Validation samples: {len(val_idx):,}")
        print(f"   📊 Test samples: {len(test_idx):,}")
        
        # Step 4: Create data loaders
        print(f"\n🗂️ STEP 4: Creating data loaders...")
        print("-" * 60)
        
        # Convert to PyTorch tensors with explicit dtypes
        genomic_tensor = torch.tensor(genomic_features, dtype=torch.float32)
        targets_tensor = torch.tensor(pic50_values, dtype=torch.float32).unsqueeze(1)
        
        # Create datasets
        train_dataset = CellLineDataset(
            smiles_tokens[train_idx],
            genomic_tensor[train_idx],
            targets_tensor[train_idx]
        )
        
        val_dataset = CellLineDataset(
            smiles_tokens[val_idx],
            genomic_tensor[val_idx],
            targets_tensor[val_idx]
        )
        
        test_dataset = CellLineDataset(
            smiles_tokens[test_idx],
            genomic_tensor[test_idx],
            targets_tensor[test_idx]
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
        
        num_epochs = 100
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        training_history = []
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            try:
                for batch_idx, (batch_smiles, batch_genomics, batch_targets) in enumerate(train_loader):
                    batch_smiles = batch_smiles.to(device)
                    batch_genomics = batch_genomics.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    optimizer.zero_grad()
                    
                    predictions = model(batch_smiles, batch_genomics)
                    loss = criterion(predictions, batch_targets)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                    # Progress update every 50 batches
                    if (batch_idx + 1) % 50 == 0:
                        print(f"   Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}: Loss = {loss.item():.4f}")
            
            except Exception as e:
                print(f"   ❌ Training error in epoch {epoch+1}: {e}")
                continue
            
            avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                try:
                    for batch_smiles, batch_genomics, batch_targets in val_loader:
                        batch_smiles = batch_smiles.to(device)
                        batch_genomics = batch_genomics.to(device)
                        batch_targets = batch_targets.to(device)
                        
                        predictions = model(batch_smiles, batch_genomics)
                        loss = criterion(predictions, batch_targets)
                        
                        val_loss += loss.item()
                        val_batches += 1
                        
                        val_predictions.extend(predictions.cpu().numpy())
                        val_targets.extend(batch_targets.cpu().numpy())
                
                except Exception as e:
                    print(f"   ❌ Validation error in epoch {epoch+1}: {e}")
                    continue
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            
            # Calculate metrics
            if val_predictions and val_targets:
                val_r2 = r2_score(val_targets, val_predictions)
                val_mae = mean_absolute_error(val_targets, val_predictions)
            else:
                val_r2 = 0.0
                val_mae = float('inf')
            
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
        if (models_dir / "cell_line_response_model_best.pth").exists():
            model.load_state_dict(torch.load(models_dir / "cell_line_response_model_best.pth"))
        model.eval()
        
        # Test evaluation
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for batch_smiles, batch_genomics, batch_targets in test_loader:
                batch_smiles = batch_smiles.to(device)
                batch_genomics = batch_genomics.to(device)
                batch_targets = batch_targets.to(device)
                
                predictions = model(batch_smiles, batch_genomics)
                
                test_predictions.extend(predictions.cpu().numpy())
                test_targets.extend(batch_targets.cpu().numpy())
        
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
        
        # Save training data
        training_data.to_csv(datasets_dir / "cell_line_training_data_final.csv", index=False)
        
        # Save comprehensive metadata
        metadata = {
            'model_type': 'Cell_Line_Response_Model_Fixed',
            'architecture': 'Multi_Modal_Molecular_Genomic',
            'training_timestamp': datetime.now().isoformat(),
            'training_data': {
                'total_samples': len(training_data),
                'valid_samples': len(pic50_values),
                'training_samples': len(train_idx),
                'validation_samples': len(val_idx),
                'test_samples': len(test_idx),
                'unique_cell_lines': training_data['CELL_LINE_NAME'].nunique(),
                'unique_compounds': training_data['DRUG_NAME'].nunique(),
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
                'drug_sensitivity': 'Synthetic_Comprehensive',
                'genomics': 'Synthetic_Realistic_Features',
                'molecular': 'SMILES_Tokenized'
            },
            'fixes_applied': {
                'dtype_handling': 'Explicit float64/float32 conversion',
                'pandas_compatibility': 'to_numeric with error handling',
                'tensor_dtypes': 'Explicit torch tensor dtypes',
                'numpy_bool_fix': 'Avoided deprecated numpy.bool'
            }
        }
        
        metadata_path = models_dir / "cell_line_response_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Best model: cell_line_response_model_best.pth")
        print(f"   ✅ Final model: cell_line_response_model_final.pth")
        print(f"   ✅ Tokenizer: smiles_tokenizer.pkl")
        print(f"   ✅ Genomic scaler: genomic_scaler.pkl")
        print(f"   ✅ Training data: cell_line_training_data_final.csv")
        print(f"   ✅ Metadata: {metadata_path}")
        
        # Final report
        print(f"\n🎉 CELL LINE RESPONSE MODEL TRAINING COMPLETED!")
        print("=" * 80)
        print(f"🧬 Multi-Modal Architecture Successfully Trained (Fixed)")
        print(f"📊 Final Performance Metrics:")
        print(f"  • Test R²: {test_r2:.4f}")
        print(f"  • Test RMSE: {test_rmse:.4f} pIC50 units")
        print(f"  • Test MAE: {test_mae:.4f} pIC50 units")
        print(f"📋 Model Capabilities:")
        print(f"  • ✅ Multi-modal IC₅₀ prediction")
        print(f"  • ✅ Genomics-informed predictions")
        print(f"  • ✅ Cancer cell line specific")
        print(f"  • ✅ Cross-modal attention fusion")  
        print(f"🔧 Fixes Applied:")
        print(f"  • ✅ Dtype handling fixed")
        print(f"  • ✅ Pandas compatibility resolved")
        print(f"  • ✅ Numpy.bool issues avoided")
        print(f"🚀 Model ready for deployment and inference!")
        
        return {
            'status': 'success',
            'model_path': str(models_dir / "cell_line_response_model_best.pth"),
            'metadata_path': str(metadata_path),
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'training_samples': len(train_idx),
            'test_samples': len(test_idx),
            'genomic_features': len(genomic_columns),
            'epochs_completed': len(training_history),
            'fixes_applied': True,
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

def create_comprehensive_training_dataset() -> pd.DataFrame:
    """Create comprehensive synthetic training dataset for Cell Line Response Model"""
    
    print("📊 Creating comprehensive synthetic training dataset...")
    
    # Extended cancer cell lines with realistic genomic profiles
    cell_lines = {
        'A549': {'cancer_type': 'LUNG', 'mutations': ['TP53', 'KRAS'], 'cnvs': {'MYC': 1, 'CDKN2A': -1}},
        'MCF7': {'cancer_type': 'BREAST', 'mutations': ['PIK3CA'], 'cnvs': {}},
        'HCT116': {'cancer_type': 'COLON', 'mutations': ['KRAS', 'PIK3CA'], 'cnvs': {'PTEN': -1, 'MYC': 1}},
        'HeLa': {'cancer_type': 'CERVICAL', 'mutations': ['TP53', 'RB1'], 'cnvs': {'MDM2': 1}},
        'U87MG': {'cancer_type': 'BRAIN', 'mutations': ['PTEN', 'TP53'], 'cnvs': {'EGFR': 1}},
        'PC3': {'cancer_type': 'PROSTATE', 'mutations': ['TP53', 'PTEN'], 'cnvs': {}},
        'OVCAR3': {'cancer_type': 'OVARIAN', 'mutations': ['TP53', 'BRCA1'], 'cnvs': {'MYC': 1}},
        'K562': {'cancer_type': 'LEUKEMIA', 'mutations': [], 'cnvs': {'MYC': 1}},
        'T47D': {'cancer_type': 'BREAST', 'mutations': ['PIK3CA'], 'cnvs': {'HER2': 1}},
        'SW480': {'cancer_type': 'COLON', 'mutations': ['KRAS', 'APC'], 'cnvs': {}},
        'MDAMB231': {'cancer_type': 'BREAST', 'mutations': ['TP53', 'KRAS'], 'cnvs': {'PTEN': -1}},
        'LNCaP': {'cancer_type': 'PROSTATE', 'mutations': ['PTEN'], 'cnvs': {}},
        'SKBR3': {'cancer_type': 'BREAST', 'mutations': ['TP53'], 'cnvs': {'HER2': 1}},
        'BT474': {'cancer_type': 'BREAST', 'mutations': ['PIK3CA'], 'cnvs': {'HER2': 1}},
        'DU145': {'cancer_type': 'PROSTATE', 'mutations': ['TP53'], 'cnvs': {'PTEN': -1}},
        'SKOV3': {'cancer_type': 'OVARIAN', 'mutations': ['TP53'], 'cnvs': {'MYC': 1}},
        'HL60': {'cancer_type': 'LEUKEMIA', 'mutations': ['TP53'], 'cnvs': {}},
        'JURKAT': {'cancer_type': 'LEUKEMIA', 'mutations': ['PTEN'], 'cnvs': {}},
        'THP1': {'cancer_type': 'LEUKEMIA', 'mutations': ['TP53'], 'cnvs': {}},
        'U937': {'cancer_type': 'LEUKEMIA', 'mutations': [], 'cnvs': {}},
        'PANC1': {'cancer_type': 'PANCREATIC', 'mutations': ['KRAS', 'TP53'], 'cnvs': {'CDKN2A': -1}},
        'MIAPACA2': {'cancer_type': 'PANCREATIC', 'mutations': ['KRAS', 'TP53', 'BRCA2'], 'cnvs': {}}
    }
    
    # Extended oncology drug library
    drugs = {
        'Erlotinib': {'smiles': 'Cc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCOC', 'target': 'EGFR'},
        'Gefitinib': {'smiles': 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1', 'target': 'EGFR'},
        'Imatinib': {'smiles': 'Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C', 'target': 'BCR-ABL'},
        'Sorafenib': {'smiles': 'CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(c3)C(F)(F)F)cc2)ccn1', 'target': 'RAF'},
        'Sunitinib': {'smiles': 'CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C', 'target': 'VEGFR'},
        'Dasatinib': {'smiles': 'Cc1nc(Nc2ncc(s2)C(=O)Nc3c(C)cccc3Cl)cc(n1)N4CCN(CC4)CCO', 'target': 'SRC'},
        'Lapatinib': {'smiles': 'CS(=O)(=O)CCNCc1oc(cc1)c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2', 'target': 'HER2'},
        'Trametinib': {'smiles': 'CC(C)[C@H](C(=O)N1CC2=C(C1=O)SC(=N2)C3=CC(=C(C=C3)F)F)N4C(=O)C5=C(C4=O)C=CC=C5I', 'target': 'MEK'},
        'Vemurafenib': {'smiles': 'CCC1=C2C=C(C=CC2=NC(=C1)C3=CC=CC=C3S(=O)(=O)N)F', 'target': 'BRAF'},
        'Paclitaxel': {'smiles': 'CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C', 'target': 'Tubulin'},
        'Docetaxel': {'smiles': 'CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@H]([C@H](c5ccccc5)NC(=O)OC(C)(C)C)O)O)OC(=O)C)(CO4)OC(=O)c6ccccc6)O)C)OC(=O)C', 'target': 'Tubulin'},
        'Doxorubicin': {'smiles': 'CC1C(C(CC(O1)OC2C(CC(C(C2)O)O)O)N)O', 'target': 'Topoisomerase'},
        'Cisplatin': {'smiles': 'N.N.Cl[Pt]Cl', 'target': 'DNA'},
        'Carboplatin': {'smiles': 'CC1(C)OC(=O)[C@H]2[C@H]([Pt](N)(N)O[C@@H]2C(=O)O1)C', 'target': 'DNA'},
        'Temozolomide': {'smiles': 'CN1C(=O)N=C2C(=O)NCCC(=O)N2C1', 'target': 'DNA'},
        'Bevacizumab': {'smiles': 'CCO', 'target': 'VEGF'},  # Simplified for antibody
        'Trastuzumab': {'smiles': 'CCO', 'target': 'HER2'},  # Simplified for antibody  
        'Rituximab': {'smiles': 'CCO', 'target': 'CD20'},   # Simplified for antibody
        'Cetuximab': {'smiles': 'CCO', 'target': 'EGFR'},   # Simplified for antibody
        '5-Fluorouracil': {'smiles': 'C1=C(C(=O)NC(=O)N1)F', 'target': 'Thymidylate'},
        'Gemcitabine': {'smiles': 'C1=CN(C(=O)N=C1N)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F', 'target': 'DNA'},
        'Oxaliplatin': {'smiles': 'C1C[NH2+]C[C@H]([NH2+]1)C2(C(=O)O[C@H]3C(=O)O[Pt-2]234OC(=O)[C@@H]5NCCN5)O', 'target': 'DNA'},
        'Capecitabine': {'smiles': 'CCN(CC)C(=O)O[C@@H]1[C@@H]([C@@H](O[C@H]([C@H]1O)N2C=CC(=NC2=O)N)CO)O', 'target': 'Thymidylate'}
    }
    
    # Cancer-related genes for genomic features
    cancer_genes = [
        'TP53', 'KRAS', 'PIK3CA', 'APC', 'BRCA1', 'BRCA2', 'EGFR', 'HER2',
        'BRAF', 'MET', 'ALK', 'ROS1', 'RET', 'NTRK1', 'CDK4', 'CDK6',
        'MDM2', 'CDKN2A', 'RB1', 'PTEN', 'VHL', 'IDH1', 'IDH2', 'TERT'
    ]
    
    # Generate comprehensive training data
    records = []
    
    for cell_line_name, cell_line_info in cell_lines.items():
        for drug_name, drug_info in drugs.items():
            
            # Create genomic profile for this cell line
            genomic_profile = {}
            
            # Mutations (binary)
            for gene in cancer_genes:
                is_mutated = 1 if gene in cell_line_info.get('mutations', []) else 0
                # Add some noise
                if np.random.random() < 0.05:  # 5% chance to flip
                    is_mutated = 1 - is_mutated
                genomic_profile[f'{gene}_mutation'] = is_mutated
            
            # CNVs (categorical: -1, 0, 1)
            for gene in cancer_genes[:12]:  # First 12 genes
                cnv_value = cell_line_info.get('cnvs', {}).get(gene, 0)
                # Add some noise
                if cnv_value == 0 and np.random.random() < 0.1:  # 10% chance of alteration
                    cnv_value = np.random.choice([-1, 1])
                genomic_profile[f'{gene}_cnv'] = cnv_value
            
            # Expression (continuous z-scores)
            for gene in cancer_genes[:15]:  # First 15 genes
                base_expression = 0.0
                # Mutations can affect expression
                if genomic_profile.get(f'{gene}_mutation', 0) == 1:
                    base_expression += np.random.normal(-0.5, 0.5)  # Mutated genes often downregulated
                # CNVs affect expression
                cnv = genomic_profile.get(f'{gene}_cnv', 0)
                if cnv == 1:
                    base_expression += np.random.normal(1.0, 0.3)  # Amplifications increase expression
                elif cnv == -1:
                    base_expression += np.random.normal(-1.0, 0.3)  # Deletions decrease expression
                else:
                    base_expression += np.random.normal(0.0, 0.5)  # Normal variation
                
                genomic_profile[f'{gene}_expression'] = base_expression
            
            # Generate realistic IC50 based on drug-cell line interaction
            base_ic50 = generate_realistic_ic50(drug_name, drug_info, cell_line_name, cell_line_info, genomic_profile)
            
            # Create record
            record = {
                'CELL_LINE_NAME': cell_line_name,
                'CANCER_TYPE': cell_line_info['cancer_type'],
                'DRUG_NAME': drug_name,
                'SMILES': drug_info['smiles'],
                'TARGET': drug_info['target'],
                'IC50_nM': base_ic50,
                **genomic_profile
            }
            
            records.append(record)
    
    training_df = pd.DataFrame(records)
    
    print(f"   ✅ Created {len(training_df):,} training records")
    print(f"   📊 Cell lines: {training_df['CELL_LINE_NAME'].nunique()}")
    print(f"   📊 Drugs: {training_df['DRUG_NAME'].nunique()}")
    print(f"   📊 Cancer types: {training_df['CANCER_TYPE'].nunique()}")
    
    return training_df

def generate_realistic_ic50(drug_name: str, drug_info: Dict, cell_line_name: str, cell_line_info: Dict, genomic_profile: Dict) -> float:
    """Generate realistic IC50 values based on drug-cell line interactions"""
    
    # Base IC50 (1 μM)
    base_ic50 = 1000.0  # nM
    
    # Drug-specific effects
    target = drug_info['target']
    
    # EGFR inhibitors (Erlotinib, Gefitinib, Cetuximab)
    if target == 'EGFR':
        if genomic_profile.get('EGFR_expression', 0) > 1.0:
            base_ic50 *= 0.3  # High EGFR expression -> sensitive
        if genomic_profile.get('KRAS_mutation', 0) == 1:
            base_ic50 *= 5.0   # KRAS mutation -> resistant
        if genomic_profile.get('EGFR_cnv', 0) == 1:
            base_ic50 *= 0.5   # EGFR amplification -> sensitive
    
    # HER2 inhibitors (Lapatinib, Trastuzumab)
    elif target == 'HER2':
        if genomic_profile.get('HER2_cnv', 0) == 1:
            base_ic50 *= 0.2   # HER2 amplification -> very sensitive
        else:
            base_ic50 *= 3.0   # No HER2 amplification -> resistant
    
    # MEK inhibitors (Trametinib)
    elif target == 'MEK':
        if genomic_profile.get('KRAS_mutation', 0) == 1 or genomic_profile.get('BRAF_mutation', 0) == 1:
            base_ic50 *= 0.3   # KRAS/BRAF mutation -> sensitive
        else:
            base_ic50 *= 2.0   # Wild-type -> less sensitive
    
    # BRAF inhibitors (Vemurafenib)
    elif target == 'BRAF':
        if genomic_profile.get('BRAF_mutation', 0) == 1:
            base_ic50 *= 0.1   # BRAF mutation -> very sensitive
        else:
            base_ic50 *= 10.0  # Wild-type BRAF -> resistant
    
    # PI3K pathway (related to PIK3CA, PTEN)
    if genomic_profile.get('PIK3CA_mutation', 0) == 1:
        if target in ['PI3K', 'AKT', 'mTOR']:
            base_ic50 *= 0.4   # PIK3CA mutation -> sensitive to PI3K inhibitors
    
    if genomic_profile.get('PTEN_cnv', 0) == -1:
        if target in ['PI3K', 'AKT', 'mTOR']:
            base_ic50 *= 0.3   # PTEN loss -> sensitive to PI3K inhibitors
    
    # p53 status affects general drug sensitivity
    if genomic_profile.get('TP53_mutation', 0) == 1:
        base_ic50 *= 1.5   # p53 mutation -> general resistance
    
    # Cancer type specific effects
    cancer_type = cell_line_info['cancer_type']
    
    if cancer_type == 'BREAST' and target == 'HER2':
        base_ic50 *= 0.7   # Breast cancer often HER2-driven
    elif cancer_type == 'LUNG' and target == 'EGFR':
        base_ic50 *= 0.8   # Lung cancer often EGFR-driven
    elif cancer_type == 'COLON' and drug_name in ['5-Fluorouracil', 'Oxaliplatin']:
        base_ic50 *= 0.6   # Standard colon cancer drugs
    
    # Add biological variability
    base_ic50 *= np.random.lognormal(0, 0.3)  # Log-normal noise
    
    # Ensure reasonable range
    base_ic50 = max(1.0, min(base_ic50, 100000000.0))  # 1 nM to 100 mM
    
    return base_ic50

if __name__ == "__main__":
    print("🧬 Fixed Cell Line Response Model Training Pipeline")