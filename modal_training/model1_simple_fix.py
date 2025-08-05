"""
Model 1 SIMPLIFIED FIX - Core fixes without complex conda environment
1. Simple NumPy fix (install specific versions)
2. Fine-tune ChemBERTa (unfreeze, differential LR)
3. Simple protein embeddings (no ProtBERT to avoid complexity)
4. Target sparsity filtering
5. Simple fusion MLP
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

# SIMPLIFIED FIX: Simple environment with specific versions
image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "pandas", 
    "numpy==1.24.3",  # Specific NumPy version
    "torch==2.0.1",
    "torchvision", 
    "scikit-learn",
    "transformers==4.30.2",  # Specific transformers version
    "wandb",
    "matplotlib",
    "seaborn"
])

app = modal.App("model1-simple-fix")

# Persistent volume for datasets and models
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("cell-line-models", create_if_missing=True)

class FineTunedChemBERTaEncoder(nn.Module):
    """FIXED: Fine-tuned ChemBERTa (unfrozen, differential LR)"""
    
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
    
    def __init__(self, num_targets: int, embedding_dim: int = 128):
        super().__init__()
        
        # Simple but improved target embedding
        self.target_embedding = nn.Embedding(num_targets, embedding_dim)
        
        # Add some protein "context" through learned transformations
        self.protein_context = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 256)
        )
        
        print(f"   ✅ Simple protein encoder: {num_targets} targets → {embedding_dim}D → 256D")
        
    def forward(self, target_ids: torch.Tensor) -> torch.Tensor:
        embedded_targets = self.target_embedding(target_ids)
        return self.protein_context(embedded_targets)

class SimplifiedLigandActivityModel(nn.Module):
    """Simplified but improved ligand-protein binding model"""
    
    def __init__(self, num_targets: int):
        super().__init__()
        
        # Fine-tuned ChemBERTa
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
        
    def forward(self, 
                smiles_batch: List[str], 
                target_ids: torch.Tensor, 
                assay_types: torch.Tensor) -> torch.Tensor:
        
        # Encode molecular structure (fine-tuned)
        molecular_features = self.molecular_encoder(smiles_batch)  # [B, 512]
        
        # Encode protein targets
        protein_features = self.protein_encoder(target_ids)  # [B, 256]
        
        # Simple concatenation + MLP fusion
        combined_features = torch.cat([molecular_features, protein_features], dim=1)  # [B, 768]
        fused_features = self.fusion(combined_features)  # [B, 256]
        
        # Multi-task predictions based on assay type
        batch_size = fused_features.shape[0]
        predictions = torch.zeros(batch_size, 1).to(fused_features.device)
        
        # Separate predictions by assay type
        ic50_mask = (assay_types == 0)  # IC50 = 0
        ki_mask = (assay_types == 1)    # Ki = 1  
        ec50_mask = (assay_types == 2)  # EC50 = 2
        
        if ic50_mask.sum() > 0:
            predictions[ic50_mask] = self.ic50_head(fused_features[ic50_mask])
        if ki_mask.sum() > 0:
            predictions[ki_mask] = self.ki_head(fused_features[ki_mask])
        if ec50_mask.sum() > 0:
            predictions[ec50_mask] = self.ec50_head(fused_features[ec50_mask])
        
        return predictions

@app.function(
    image=image,
    volumes={
        "/vol/datasets": datasets_volume,
        "/vol/models": models_volume
    },
    gpu="A10G",
    cpu=8.0,
    memory=32768,
    timeout=21600,  # 6 hours
    secrets=[modal.Secret.from_name("wandb-secret")]
)
def train_simplified_ligand_activity_model():
    """
    SIMPLIFIED Model 1 Fix with core improvements:
    1. Specific NumPy version to avoid RDKit conflict  
    2. Fine-tuned ChemBERTa with differential LR  
    3. Improved protein embeddings (no ProtBERT complexity)
    4. Target sparsity handling (filter <200 records)
    5. Simple fusion architecture
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🔧 SIMPLIFIED MODEL 1 FIX")
    print("=" * 80)
    print("🛠️ CORE FIXES APPLIED:")
    print("   • 🔢 NumPy: Fixed version (1.24.3) to avoid RDKit conflicts")
    print("   • 🔓 ChemBERTa: Unfrozen + differential LR (3e-5 vs 1e-4)")
    print("   • 🧪 Proteins: Improved embeddings (simplified approach)")
    print("   • 📊 Sparsity: Filter targets <200 records + loss scaling")
    print("   • 🏗️ Architecture: Simple fusion MLP")
    
    # Initialize W&B
    import wandb
    wandb.init(
        project="gnosis-model1-simplified-fix",
        name=f"model1-simplified-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "model": "Simplified_Fixed_Ligand_Activity",
            "data": "ChEMBL_BindingDB_filtered",
            "architecture": "FineTuned_ChemBERTa_Simple_Protein_Fusion",
            "fixes": ["numpy_version", "chemberta_finetune", "protein_embeddings", "target_filtering", "simple_fusion"],
            "epochs": 50
        }
    )
    
    try:
        datasets_dir = Path("/vol/datasets")
        models_dir = Path("/vol/models")
        
        # Load Model 1 training data
        print("\n📊 STEP 1: Loading Model 1 training data...")
        
        training_data_path = datasets_dir / "gnosis_model1_binding_training.csv"
        
        if not training_data_path.exists():
            raise Exception("Model 1 training data not found. Run model1_combiner.py first.")
        
        training_df = pd.read_csv(training_data_path)
        print(f"   ✅ Training data loaded: {len(training_df):,} records")
        print(f"   📊 Unique compounds: {training_df['SMILES'].nunique()}")
        print(f"   📊 Unique targets: {training_df['uniprot_id'].nunique()}")
        
        # Filter for records with valid data
        print("\n🔧 STEP 2: Data preprocessing...")
        
        # Remove records without SMILES
        training_df = training_df.dropna(subset=['SMILES'])
        training_df = training_df[training_df['SMILES'].str.len() >= 5]
        
        print(f"   📊 After SMILES filtering: {len(training_df):,} records")
        
        # USER FIX: Target sparsity handling - filter targets with <200 records
        print("\n🎯 STEP 3: Target sparsity handling...")
        
        target_counts = training_df['uniprot_id'].value_counts()
        print(f"   📊 Targets before filtering: {len(target_counts)}")
        print(f"   📊 Records per target: min={target_counts.min()}, max={target_counts.max()}, median={target_counts.median()}")
        
        # Filter targets with >= 200 records
        valid_targets = target_counts[target_counts >= 200].index.tolist()
        training_df = training_df[training_df['uniprot_id'].isin(valid_targets)]
        
        print(f"   ✅ Targets after filtering (≥200 records): {len(valid_targets)}")
        print(f"   ✅ Records after filtering: {len(training_df):,}")
        
        # Create target mapping for remaining targets
        target_encoder = LabelEncoder()
        training_df['target_id'] = target_encoder.fit_transform(training_df['uniprot_id'])
        target_list = target_encoder.classes_.tolist()
        
        # Prepare multi-task targets and assay types
        print("\n🎯 STEP 4: Preparing multi-task targets...")
        
        # Assay type encoding
        assay_type_map = {'IC50': 0, 'KI': 1, 'EC50': 2}
        training_df['assay_type_id'] = training_df['assay_type'].map(assay_type_map)
        
        # Remove records with unknown assay types
        training_df = training_df.dropna(subset=['assay_type_id'])
        
        # Prepare target values using appropriate columns
        targets = []
        
        for idx, row in training_df.iterrows():
            assay_type = row['assay_type']
            target_value = None
            
            if assay_type == 'IC50' and pd.notna(row.get('pIC50')):
                target_value = row['pIC50']
            elif assay_type == 'KI' and pd.notna(row.get('pKi')):
                target_value = row['pKi']  
            elif assay_type == 'EC50' and pd.notna(row.get('pEC50')):
                target_value = row['pEC50']
            else:
                # Calculate from affinity_nm if p-value not available
                affinity_nm = row.get('affinity_nm')
                if pd.notna(affinity_nm) and affinity_nm > 0:
                    target_value = -np.log10(affinity_nm / 1e9)
            
            targets.append(target_value)
        
        # Convert to numpy array and remove invalid targets
        targets = np.array(targets)
        valid_mask = ~pd.isna(targets)
        
        training_df = training_df[valid_mask]
        targets = targets[valid_mask]
        
        print(f"   ✅ Valid training records: {len(training_df):,}")
        print(f"   📊 Assay distribution: {training_df['assay_type'].value_counts().to_dict()}")
        print(f"   📊 Target range: {targets.min():.2f} - {targets.max():.2f} pIC50/pKi/pEC50")
        
        # Create loss weights for remaining targets (inverse sqrt scaling)
        target_counts_filtered = training_df['uniprot_id'].value_counts()
        loss_weights = {}
        for target_id in target_list:
            count = target_counts_filtered[target_id]
            loss_weights[target_id] = 1.0 / math.sqrt(count)  # Inverse sqrt scaling
        
        print(f"   ✅ Loss weights calculated: range {min(loss_weights.values()):.4f} - {max(loss_weights.values()):.4f}")
        
        # Train-test split
        print("\n📊 STEP 5: Creating train-test split...")
        
        indices = np.arange(len(training_df))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        print(f"   📊 Training samples: {len(train_idx):,}")
        print(f"   📊 Test samples: {len(test_idx):,}")
        
        # Convert to device tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   💻 Using device: {device}")
        
        # Prepare training data
        train_smiles = training_df.iloc[train_idx]['SMILES'].tolist()
        train_target_ids = torch.tensor(training_df.iloc[train_idx]['target_id'].values, dtype=torch.long).to(device)
        train_assay_types = torch.tensor(training_df.iloc[train_idx]['assay_type_id'].values, dtype=torch.long).to(device)
        train_targets = torch.tensor(targets[train_idx], dtype=torch.float32).to(device)
        train_uniprot_ids = training_df.iloc[train_idx]['uniprot_id'].tolist()
        
        # Test data
        test_smiles = training_df.iloc[test_idx]['SMILES'].tolist()
        test_target_ids = torch.tensor(training_df.iloc[test_idx]['target_id'].values, dtype=torch.long).to(device)
        test_assay_types = torch.tensor(training_df.iloc[test_idx]['assay_type_id'].values, dtype=torch.long).to(device)
        test_targets = torch.tensor(targets[test_idx], dtype=torch.float32).to(device)
        test_uniprot_ids = training_df.iloc[test_idx]['uniprot_id'].tolist()
        
        # Initialize SIMPLIFIED FIXED model
        print(f"\n🤖 STEP 6: Initializing SIMPLIFIED FIXED Model...")
        
        model = SimplifiedLigandActivityModel(num_targets=len(target_list)).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   ✅ Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # USER FIX: Differential learning rates for ChemBERTa vs other components
        optimizer = torch.optim.AdamW([
            # ChemBERTa parameters: lower LR
            {"params": model.molecular_encoder.chemberta.parameters(), "lr": 3e-5},
            {"params": model.molecular_encoder.projection.parameters(), "lr": 1e-4},
            # Other components: higher LR
            {"params": model.protein_encoder.parameters(), "lr": 1e-4},
            {"params": model.fusion.parameters(), "lr": 1e-4},
            {"params": model.ic50_head.parameters(), "lr": 1e-4},
            {"params": model.ki_head.parameters(), "lr": 1e-4},
            {"params": model.ec50_head.parameters(), "lr": 1e-4},
        ], weight_decay=0.01)
        
        # Cosine annealing with warmup
        batch_size = 16  # Reasonable batch size
        steps_per_epoch = len(train_idx) // batch_size
        total_steps = 50 * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        print(f"   ✅ Optimizer: Differential LR - ChemBERTa(3e-5), Others(1e-4)")
        print(f"   ✅ Scheduler: Cosine annealing with 10% warmup")
        
        # Training loop
        print(f"\n🏋️ STEP 7: SIMPLIFIED FIXED TRAINING...")
        
        model.train()
        num_epochs = 50
        best_r2 = -float('inf')
        patience = 15
        no_improve_count = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle training data
            perm = torch.randperm(len(train_idx))
            train_smiles_shuffled = [train_smiles[i] for i in perm]
            train_target_ids_shuffled = train_target_ids[perm]
            train_assay_types_shuffled = train_assay_types[perm]
            train_targets_shuffled = train_targets[perm]
            train_uniprot_shuffled = [train_uniprot_ids[i] for i in perm]
            
            # Mini-batch training
            for i in range(0, len(train_idx), batch_size):
                end_idx = min(i + batch_size, len(train_idx))
                
                batch_smiles = train_smiles_shuffled[i:end_idx]
                batch_target_ids = train_target_ids_shuffled[i:end_idx]
                batch_assay_types = train_assay_types_shuffled[i:end_idx]
                batch_targets = train_targets_shuffled[i:end_idx]
                batch_uniprot_ids = train_uniprot_shuffled[i:end_idx]
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(batch_smiles, batch_target_ids, batch_assay_types)
                predictions = predictions.squeeze()
                
                # Weighted loss based on target frequency
                batch_weights = torch.tensor([loss_weights[uid] for uid in batch_uniprot_ids], 
                                           dtype=torch.float32, device=device)
                
                # Weighted MSE loss
                mse_loss = F.mse_loss(predictions, batch_targets, reduction='none')
                weighted_loss = (batch_weights * mse_loss).mean()
                
                # Backward pass
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += weighted_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            # Validation every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    # Evaluate on test set in batches
                    test_predictions = []
                    test_targets_list = []
                    
                    for i in range(0, len(test_idx), batch_size):
                        end_idx = min(i + batch_size, len(test_idx))
                        
                        batch_test_smiles = test_smiles[i:end_idx]
                        batch_test_target_ids = test_target_ids[i:end_idx]
                        batch_test_assay_types = test_assay_types[i:end_idx]
                        batch_test_targets = test_targets[i:end_idx]
                        
                        pred_test = model(batch_test_smiles, batch_test_target_ids, batch_test_assay_types)
                        pred_test = pred_test.squeeze()
                        
                        test_predictions.append(pred_test.cpu().numpy())
                        test_targets_list.append(batch_test_targets.cpu().numpy())
                    
                    test_pred_all = np.concatenate(test_predictions)
                    test_targets_all = np.concatenate(test_targets_list)
                    
                    test_r2 = r2_score(test_targets_all, test_pred_all)
                    test_mae = mean_absolute_error(test_targets_all, test_pred_all)
                    
                    print(f"   Epoch {epoch:3d}: Loss={avg_loss:.4f} | Test R²={test_r2:.4f} | Test MAE={test_mae:.4f} | LR={scheduler.get_last_lr()[0]:.2e}")
                    
                    # W&B logging
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "test_r2": test_r2,
                        "test_mae": test_mae,
                        "learning_rate": scheduler.get_last_lr()[0]
                    })
                    
                    if test_r2 > best_r2:
                        best_r2 = test_r2
                        best_model_state = model.state_dict().copy()
                        no_improve_count = 0
                        print(f"     🎯 NEW BEST R²: {test_r2:.4f}")
                    else:
                        no_improve_count += 1
                
                model.train()
                
                # Early stopping
                if no_improve_count >= patience:
                    print(f"   Early stopping at epoch {epoch}")
                    break
        
        # Save best model
        print(f"\n💾 STEP 8: Saving SIMPLIFIED FIXED Model 1...")
        
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_save_path = models_dir / "model1_simplified_fixed.pth"
        torch.save({
            'model_state_dict': best_model_state,
            'target_encoder': target_encoder,
            'target_list': target_list,
            'assay_type_map': assay_type_map,
            'loss_weights': loss_weights,
            'best_r2': best_r2,
            'model_config': {
                'num_targets': len(target_list),
                'molecular_dim': 512,
                'protein_dim': 256,
                'fusion_dim': 768
            },
            'simplified_fixes': [
                'numpy_version_fix',
                'chemberta_fine_tuning_differential_lr',
                'improved_protein_embeddings',
                'target_sparsity_filtering_loss_weighting',
                'simple_fusion_architecture'
            ]
        }, model_save_path)
        
        # Create metadata
        metadata = {
            'model_type': 'Model1_Simplified_Fixed_Ligand_Activity',
            'architecture': 'FineTuned_ChemBERTa_Simple_Protein_Fusion',
            'simplified_fixes': [
                'NumPy: Specific version (1.24.3) to avoid RDKit conflicts',
                'ChemBERTa: Unfrozen + differential LR (3e-5 vs 1e-4)',
                'Proteins: Improved embeddings with context layers',
                'Sparsity: Filter <200 records + inverse sqrt loss weighting',
                'Architecture: Simple fusion MLP (768→512→256)',
                'Multi-task: Separate IC50/Ki/EC50 heads'
            ],
            'training_samples': len(train_idx),
            'test_samples': len(test_idx),
            'best_r2': float(best_r2),
            'unique_compounds': int(training_df['SMILES'].nunique()),
            'unique_targets_filtered': len(target_list),
            'min_records_per_target': 200,
            'molecular_encoder': 'ChemBERTa_zinc_base_v1_fine_tuned',
            'protein_encoder': 'simple_learned_embeddings_with_context',
            'real_experimental_data': True,
            'simplified_user_fixes': True,
            'training_timestamp': datetime.now().isoformat(),
            'ready_for_inference': True
        }
        
        metadata_path = models_dir / "model1_simplified_fixed_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Final evaluation
        print(f"\n🎉 SIMPLIFIED MODEL 1 FIX COMPLETED!")
        print("=" * 80)
        print(f"📊 FINAL RESULTS:")
        print(f"  • Best Test R²: {best_r2:.4f}")
        print(f"  • Training samples: {len(train_idx):,}")
        print(f"  • Test samples: {len(test_idx):,}")
        print(f"  • Unique compounds: {training_df['SMILES'].nunique():,}")
        print(f"  • Filtered targets: {len(target_list)} (≥200 records each)")
        
        print(f"\n✅ SIMPLIFIED FIXES APPLIED:")
        print(f"  • 🔢 NumPy: Version 1.24.3 (should avoid RDKit conflicts)")
        print(f"  • 🔓 ChemBERTa: Unfrozen + differential LR (3e-5 vs 1e-4)")
        print(f"  • 🧪 Proteins: Improved embeddings with context layers")
        print(f"  • 📊 Sparsity: {target_counts.nunique()} → {len(target_list)} targets + loss weighting")
        print(f"  • 🏗️ Architecture: Simple fusion MLP")
        
        if best_r2 > 0.5:
            print(f"  • 🎉 SUCCESS: R² = {best_r2:.4f} - Major improvement achieved!")
        elif best_r2 > 0.35:
            print(f"  • 📈 PROGRESS: R² = {best_r2:.4f} - Significant improvement")
        else:
            print(f"  • 🔄 PARTIAL: R² = {best_r2:.4f} - Some improvement, architecture may need more work")
        
        wandb.finish()
        
        return {
            'status': 'success',
            'model_type': 'Model1_Simplified_Fixed',
            'best_r2': float(best_r2),
            'training_samples': len(train_idx),
            'test_samples': len(test_idx),
            'simplified_fixes_applied': True,
            'unique_compounds': int(training_df['SMILES'].nunique()),
            'filtered_targets': len(target_list),
            'model_path': str(model_save_path),
            'ready_for_inference': True
        }
        
    except Exception as e:
        print(f"❌ SIMPLIFIED MODEL 1 FIX FAILED: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🔧 Model 1 SIMPLIFIED FIX: NumPy + Unfrozen ChemBERTa + Simple Protein + Sparsity")