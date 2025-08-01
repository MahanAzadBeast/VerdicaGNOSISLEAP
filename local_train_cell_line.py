"""
Local Cell Line Response Model Training
Simple training implementation that works locally without Modal dependencies
"""

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
    
    def __init__(self, vocab_size: int = 60, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
    def forward(self, smiles_tokens: torch.Tensor) -> torch.Tensor:
        # Create attention mask for padding
        attention_mask = (smiles_tokens != 0).float()
        
        # Embed SMILES tokens
        embedded = self.embedding(smiles_tokens)  # [batch, seq_len, embedding_dim]
        
        # RNN encoding
        rnn_out, _ = self.rnn(embedded)  # [batch, seq_len, hidden_dim * 2]
        
        # Simple average pooling with masking
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(rnn_out)
        rnn_masked = rnn_out * mask_expanded
        pooled = rnn_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
        
        # Output projection
        molecular_features = self.output_projection(pooled)  # [batch, 64]
        
        return molecular_features

class GenomicEncoder(nn.Module):
    """Encode genomic features into feature vectors"""
    
    def __init__(self, genomic_dim: int = 51):  # Reduced from 63
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(genomic_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )
        
    def forward(self, genomic_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(genomic_features)

class CellLineResponseModel(nn.Module):
    """Simplified multi-modal model for predicting IC50"""
    
    def __init__(self, smiles_vocab_size: int = 60, genomic_dim: int = 51):
        super().__init__()
        
        self.molecular_encoder = MolecularEncoder(smiles_vocab_size)
        self.genomic_encoder = GenomicEncoder(genomic_dim)
        
        # Simple fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(128, 128),  # 64 (molecular) + 64 (genomic)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Single IC50 prediction
        )
        
    def forward(self, smiles_tokens: torch.Tensor, genomic_features: torch.Tensor) -> torch.Tensor:
        # Encode inputs
        molecular_features = self.molecular_encoder(smiles_tokens)    # [batch, 64]
        genomic_features_enc = self.genomic_encoder(genomic_features) # [batch, 64]
        
        # Simple concatenation fusion
        fused_features = torch.cat([molecular_features, genomic_features_enc], dim=1)  # [batch, 128]
        
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

def create_training_dataset() -> pd.DataFrame:
    """Create synthetic training dataset"""
    
    print("📊 Creating training dataset...")
    
    # Simplified cell lines
    cell_lines = {
        'A549': {'cancer_type': 'LUNG', 'mutations': ['TP53', 'KRAS']},
        'MCF7': {'cancer_type': 'BREAST', 'mutations': ['PIK3CA']},
        'HCT116': {'cancer_type': 'COLON', 'mutations': ['KRAS', 'PIK3CA']},
        'HeLa': {'cancer_type': 'CERVICAL', 'mutations': ['TP53']},
        'U87MG': {'cancer_type': 'BRAIN', 'mutations': ['PTEN']},
        'PC3': {'cancer_type': 'PROSTATE', 'mutations': ['TP53', 'PTEN']},
        'K562': {'cancer_type': 'LEUKEMIA', 'mutations': []},
        'SKBR3': {'cancer_type': 'BREAST', 'mutations': ['TP53']}
    }
    
    # Simplified drugs
    drugs = {
        'Erlotinib': {'smiles': 'Cc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCOC', 'target': 'EGFR'},
        'Imatinib': {'smiles': 'Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C', 'target': 'BCR-ABL'},
        'Sorafenib': {'smiles': 'CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(c3)C(F)(F)F)cc2)ccn1', 'target': 'RAF'},
        'Trametinib': {'smiles': 'CC(C)[C@H](C(=O)N1CC2=C(C1=O)SC(=N2)C3=CC(=C(C=C3)F)F)N4C(=O)C5=C(C4=O)C=CC=C5I', 'target': 'MEK'},
        'Paclitaxel': {'smiles': 'CCO', 'target': 'Tubulin'},  # Simplified
        'Doxorubicin': {'smiles': 'CCN', 'target': 'DNA'},    # Simplified
    }
    
    # Key cancer genes (reduced set)
    cancer_genes = [
        'TP53', 'KRAS', 'PIK3CA', 'EGFR', 'HER2', 'BRAF', 'MET', 'ALK',
        'PTEN', 'BRCA1', 'BRCA2', 'CDK4', 'CDK6', 'MDM2', 'RB1'
    ]
    
    records = []
    
    for cell_line_name, cell_line_info in cell_lines.items():
        for drug_name, drug_info in drugs.items():
            
            # Create genomic profile
            genomic_profile = {}
            
            # Mutations (15 genes)
            for gene in cancer_genes:
                is_mutated = 1 if gene in cell_line_info.get('mutations', []) else 0
                # Add noise
                if np.random.random() < 0.05:
                    is_mutated = 1 - is_mutated
                genomic_profile[f'{gene}_mutation'] = is_mutated
            
            # CNVs (12 genes)
            for gene in cancer_genes[:12]:
                cnv_value = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
                genomic_profile[f'{gene}_cnv'] = cnv_value
            
            # Expression (12 genes)
            for gene in cancer_genes[:12]:
                expr_value = np.random.normal(0, 1)
                genomic_profile[f'{gene}_expression'] = expr_value
            
            # Generate IC50 based on simple rules
            base_ic50 = 1000.0  # 1 μM baseline
            
            # Drug-specific effects
            if drug_info['target'] == 'EGFR':
                if genomic_profile.get('KRAS_mutation', 0) == 1:
                    base_ic50 *= 5.0  # KRAS mutation -> resistance
                if genomic_profile.get('EGFR_cnv', 0) == 1:
                    base_ic50 *= 0.5  # EGFR amplification -> sensitivity
            
            elif drug_info['target'] == 'MEK':
                if genomic_profile.get('KRAS_mutation', 0) == 1:
                    base_ic50 *= 0.3  # KRAS mutation -> sensitivity
            
            # p53 effects
            if genomic_profile.get('TP53_mutation', 0) == 1:
                base_ic50 *= 1.5  # p53 mutation -> general resistance
            
            # Add variability
            base_ic50 *= np.random.lognormal(0, 0.3)
            base_ic50 = max(10.0, min(base_ic50, 100000.0))  # Reasonable range
            
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
    
    return pd.DataFrame(records)

def train_cell_line_model():
    """Train the Cell Line Response Model locally"""
    
    print("🧬 CELL LINE RESPONSE MODEL LOCAL TRAINING")
    print("=" * 80)
    
    try:
        # Create output directory
        output_dir = Path("/tmp/cell_line_models")
        output_dir.mkdir(exist_ok=True)
        
        # Step 1: Create training data
        print("\n📊 STEP 1: Creating training dataset...")
        training_data = create_training_dataset()
        print(f"   ✅ Training data: {len(training_data):,} records")
        
        # Step 2: Prepare features
        print("\n🔧 STEP 2: Preparing features...")
        
        # Tokenizer
        tokenizer = SMILESTokenizer()
        
        # Molecular features
        smiles_list = training_data['SMILES'].astype(str).tolist()
        smiles_tokens = tokenizer.batch_tokenize(smiles_list, max_length=80)
        
        # Genomic features
        genomic_columns = [col for col in training_data.columns 
                          if any(suffix in col for suffix in ['_mutation', '_cnv', '_expression'])]
        
        genomic_data = training_data[genomic_columns].copy()
        for col in genomic_columns:
            genomic_data[col] = pd.to_numeric(genomic_data[col], errors='coerce').fillna(0.0)
        
        genomic_features = genomic_data.values.astype(np.float32)
        
        # Standardize genomic features
        scaler = StandardScaler()
        genomic_features = scaler.fit_transform(genomic_features)
        
        # Targets
        ic50_values = pd.to_numeric(training_data['IC50_nM'], errors='coerce').fillna(1000.0).values
        pic50_values = -np.log10(ic50_values / 1e9).astype(np.float32)
        
        print(f"   📊 SMILES tokens: {smiles_tokens.shape}")
        print(f"   📊 Genomic features: {genomic_features.shape}")
        print(f"   📊 Targets: {pic50_values.shape}")
        
        # Step 3: Train-test split
        print("\n📋 STEP 3: Train-test split...")
        indices = np.arange(len(pic50_values))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        print(f"   📊 Training: {len(train_idx)} samples")
        print(f"   📊 Testing: {len(test_idx)} samples")
        
        # Step 4: Create data loaders
        print("\n🗂️ STEP 4: Creating data loaders...")
        
        genomic_tensor = torch.tensor(genomic_features, dtype=torch.float32)
        targets_tensor = torch.tensor(pic50_values, dtype=torch.float32).unsqueeze(1)
        
        train_dataset = CellLineDataset(
            smiles_tokens[train_idx],
            genomic_tensor[train_idx],
            targets_tensor[train_idx]
        )
        
        test_dataset = CellLineDataset(
            smiles_tokens[test_idx],
            genomic_tensor[test_idx],
            targets_tensor[test_idx]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Step 5: Initialize model
        print("\n🤖 STEP 5: Initializing model...")
        
        model = CellLineResponseModel(
            smiles_vocab_size=tokenizer.vocab_size,
            genomic_dim=genomic_features.shape[1]
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"   🖥️ Device: {device}")
        print(f"   📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        # Step 6: Training loop
        print("\n🚀 STEP 6: Training model...")
        
        num_epochs = 50
        best_test_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_smiles, batch_genomics, batch_targets in train_loader:
                batch_smiles = batch_smiles.to(device)
                batch_genomics = batch_genomics.to(device) 
                batch_targets = batch_targets.to(device)
                
                optimizer.zero_grad()
                predictions = model(batch_smiles, batch_genomics)
                loss = criterion(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Testing
            model.eval()
            test_loss = 0.0
            test_predictions = []
            test_targets = []
            
            with torch.no_grad():
                for batch_smiles, batch_genomics, batch_targets in test_loader:
                    batch_smiles = batch_smiles.to(device)
                    batch_genomics = batch_genomics.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    predictions = model(batch_smiles, batch_genomics)
                    loss = criterion(predictions, batch_targets)
                    
                    test_loss += loss.item()
                    test_predictions.extend(predictions.cpu().numpy())
                    test_targets.extend(batch_targets.cpu().numpy())
            
            test_loss /= len(test_loader)
            test_r2 = r2_score(test_targets, test_predictions)
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), output_dir / "best_model.pth")
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1:2d}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Test R² = {test_r2:.4f}")
        
        # Step 7: Final evaluation
        print("\n📊 STEP 7: Final evaluation...")
        
        model.load_state_dict(torch.load(output_dir / "best_model.pth"))
        model.eval()
        
        final_predictions = []
        final_targets = []
        
        with torch.no_grad():
            for batch_smiles, batch_genomics, batch_targets in test_loader:
                batch_smiles = batch_smiles.to(device)
                batch_genomics = batch_genomics.to(device)
                batch_targets = batch_targets.to(device)
                
                predictions = model(batch_smiles, batch_genomics)
                final_predictions.extend(predictions.cpu().numpy())
                final_targets.extend(batch_targets.cpu().numpy())
        
        final_r2 = r2_score(final_targets, final_predictions)
        final_rmse = np.sqrt(mean_squared_error(final_targets, final_predictions))
        final_mae = mean_absolute_error(final_targets, final_predictions)
        
        print(f"   📊 Final Test R²: {final_r2:.4f}")
        print(f"   📊 Final Test RMSE: {final_rmse:.4f}")
        print(f"   📊 Final Test MAE: {final_mae:.4f}")
        
        # Step 8: Save artifacts
        print("\n💾 STEP 8: Saving artifacts...")
        
        # Save tokenizer and scaler
        with open(output_dir / "tokenizer.pkl", 'wb') as f:
            pickle.dump(tokenizer, f)
        
        with open(output_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save training data
        training_data.to_csv(output_dir / "training_data.csv", index=False)
        
        # Save metadata
        metadata = {
            'model_type': 'Cell_Line_Response_Model_Local',
            'training_timestamp': datetime.now().isoformat(),
            'performance': {
                'test_r2': float(final_r2),
                'test_rmse': float(final_rmse),
                'test_mae': float(final_mae)
            },
            'training_data': {
                'total_samples': len(training_data),
                'training_samples': len(train_idx),
                'test_samples': len(test_idx),
                'genomic_features': genomic_features.shape[1]
            },
            'model_files': {
                'model': 'best_model.pth',
                'tokenizer': 'tokenizer.pkl',
                'scaler': 'scaler.pkl',
                'training_data': 'training_data.csv'
            }
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Model saved: {output_dir}/best_model.pth")
        print(f"   ✅ Tokenizer saved: {output_dir}/tokenizer.pkl")
        print(f"   ✅ Scaler saved: {output_dir}/scaler.pkl")
        print(f"   ✅ Metadata saved: {output_dir}/metadata.json")
        
        print(f"\n🎉 CELL LINE RESPONSE MODEL TRAINING COMPLETED!")
        print("=" * 80)
        print(f"📊 Final Performance:")
        print(f"  • Test R²: {final_r2:.4f}")
        print(f"  • Test RMSE: {final_rmse:.4f}")
        print(f"  • Test MAE: {final_mae:.4f}")
        print(f"🚀 Model ready for use!")
        
        return {
            'status': 'success',
            'test_r2': final_r2,
            'test_rmse': final_rmse,
            'test_mae': final_mae,
            'model_path': str(output_dir / "best_model.pth"),
            'metadata_path': str(output_dir / "metadata.json")
        }
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}

if __name__ == "__main__":
    result = train_cell_line_model()
    print("Training result:", result)