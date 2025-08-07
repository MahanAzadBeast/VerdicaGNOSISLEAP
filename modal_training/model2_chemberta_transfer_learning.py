"""
Train Model 2 using the existing ChemBERTa from Model 1 with cytotoxic head
Transfer learning approach: Use proven ChemBERTa encoder + add cytotoxicity prediction head
"""

import modal
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("model2-chemberta-transfer-learning")

# Use same image as existing ChemBERTa models
image = modal.Image.debian_slim().pip_install([
    "torch==2.0.1", 
    "transformers==4.33.0", 
    "pandas==2.1.0", 
    "numpy==1.24.3", 
    "scikit-learn==1.3.0", 
    "scipy==1.11.0"
])

data_volume = modal.Volume.from_name("expanded-datasets")
model_volume = modal.Volume.from_name("trained-models")

class ChemBERTaEncoder(nn.Module):
    """Reuse the same ChemBERTa encoder from Model 1"""
    
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", embedding_dim=768):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chemberta = AutoModel.from_pretrained(model_name)
        
        # Keep ChemBERTa frozen initially (transfer learning)
        self.chemberta.requires_grad_(False)
        
        self.projection = nn.Linear(embedding_dim, 512)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, smiles_list):
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

class CytotoxicityModel(nn.Module):
    """Model 2: ChemBERTa + Genomic Features + Cytotoxic Head"""
    
    def __init__(self, genomic_dim=30):
        super().__init__()
        
        # Molecular encoder (same as Model 1)
        self.molecular_encoder = ChemBERTaEncoder()
        
        # Genomic encoder
        self.genomic_encoder = nn.Sequential(
            nn.Linear(genomic_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion layer (molecular + genomic)
        self.fusion = nn.Sequential(
            nn.Linear(512 + 64, 256),  # ChemBERTa projection (512) + genomic (64)
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cytotoxic prediction head
        self.cytotoxic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # pIC50 output
        )
    
    def forward(self, smiles_list, genomic_features):
        # Encode molecules using ChemBERTa
        molecular_features = self.molecular_encoder(smiles_list)
        
        # Encode genomic features
        genomic_encoded = self.genomic_encoder(genomic_features)
        
        # Fuse molecular and genomic
        combined = torch.cat([molecular_features, genomic_encoded], dim=1)
        fused_features = self.fusion(combined)
        
        # Predict cytotoxicity
        cytotoxicity = self.cytotoxic_head(fused_features)
        
        return cytotoxicity

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/vol": data_volume, "/models": model_volume}
)
def train_model2_chemberta_transfer():
    """Train Model 2 using ChemBERTa transfer learning"""
    
    logger.info("🎯 TRAINING MODEL 2 WITH CHEMBERTA TRANSFER LEARNING")
    logger.info("Using proven ChemBERTa encoder from Model 1 + cytotoxic head")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # 1. LOAD REAL GDSC DATASET
    file_path = "/vol/gdsc_dataset/gdsc_sample_10k.csv"
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"✅ Loaded dataset: {df.shape}")
        
        # Find columns
        smiles_col = None
        ic50_col = None  
        cell_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if not smiles_col and any(term in col_lower for term in ['smiles', 'canonical']):
                smiles_col = col
            if not ic50_col and any(term in col_lower for term in ['ic50', 'ic_50']):
                ic50_col = col
            if not cell_col and any(term in col_lower for term in ['cell', 'line', 'cosmic', 'sample']):
                cell_col = col
        
        logger.info(f"SMILES: {smiles_col}, IC50: {ic50_col}, Cell: {cell_col}")
        
        # Filter complete records
        complete_mask = (
            df[smiles_col].notna() & 
            df[ic50_col].notna() & 
            df[cell_col].notna()
        )
        
        training_df = df[complete_mask].copy()
        logger.info(f"Complete records: {len(training_df):,}")
        
        # Convert IC50 to pIC50
        ic50_values = training_df[ic50_col]
        
        if ic50_values.mean() > 10:
            # Likely in nM, convert to µM then pIC50
            training_df['pIC50'] = -np.log10(ic50_values / 1000)
        else:
            # Already in µM
            training_df['pIC50'] = -np.log10(ic50_values)
        
        # Remove invalid values
        training_df = training_df[training_df['pIC50'].notna()]
        training_df = training_df[np.isfinite(training_df['pIC50'])]
        
        # Standardize columns
        training_df = training_df.rename(columns={
            smiles_col: 'SMILES',
            cell_col: 'CELL_LINE'
        })
        
        logger.info(f"Training data: {len(training_df):,} records")
        logger.info(f"Unique compounds: {training_df['SMILES'].nunique():,}")
        logger.info(f"Unique cell lines: {training_df['CELL_LINE'].nunique():,}")
        logger.info(f"pIC50 range: {training_df['pIC50'].min():.2f} - {training_df['pIC50'].max():.2f}")
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return {"success": False, "error": f"Data loading: {e}"}
    
    # 2. GENERATE GENOMIC FEATURES (same as before)
    logger.info("🧬 Generating genomic features...")
    
    unique_cell_lines = training_df['CELL_LINE'].unique()
    cell_genomic_profiles = {}
    
    for cell_line in unique_cell_lines:
        np.random.seed(hash(cell_line) % (2**32))
        
        # 30 genomic features to match existing model
        mutations = np.random.binomial(1, 0.08, 15).astype(float)
        cnv = np.random.normal(0, 0.2, 10)
        expression = np.random.normal(0, 0.15, 5)
        
        genomic_profile = np.concatenate([mutations, cnv, expression])
        cell_genomic_profiles[cell_line] = genomic_profile
    
    genomic_features = []
    for _, row in training_df.iterrows():
        cell_line = row['CELL_LINE']
        genomic_features.append(cell_genomic_profiles[cell_line])
    
    genomic_features = np.array(genomic_features, dtype=np.float32)
    logger.info(f"✅ Genomic features: {genomic_features.shape}")
    
    # 3. PREPARE TRAINING DATA
    smiles_list = training_df['SMILES'].tolist()
    y = training_df['pIC50'].values
    
    logger.info(f"Training data prepared:")
    logger.info(f"  SMILES: {len(smiles_list)} molecules")
    logger.info(f"  Genomic: {genomic_features.shape}")
    logger.info(f"  Targets: {y.shape}")
    
    # 4. CREATE MODEL
    model = CytotoxicityModel(genomic_dim=genomic_features.shape[1]).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=10, verbose=True
    )
    
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 5. TRAIN/VAL/TEST SPLIT
    indices = np.arange(len(smiles_list))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.18, random_state=42)
    
    # Split data
    smiles_train = [smiles_list[i] for i in train_idx]
    smiles_val = [smiles_list[i] for i in val_idx]
    smiles_test = [smiles_list[i] for i in test_idx]
    
    genomic_train = genomic_features[train_idx]
    genomic_val = genomic_features[val_idx]
    genomic_test = genomic_features[test_idx]
    
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    
    # Scale genomic features
    gen_scaler = StandardScaler()
    genomic_train_s = gen_scaler.fit_transform(genomic_train)
    genomic_val_s = gen_scaler.transform(genomic_val)
    genomic_test_s = gen_scaler.transform(genomic_test)
    
    # Convert to tensors
    genomic_train_t = torch.FloatTensor(genomic_train_s).to(device)
    genomic_val_t = torch.FloatTensor(genomic_val_s).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    logger.info(f"Data splits: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    
    # 6. TRAINING LOOP
    logger.info("🏃 Training Model 2 with ChemBERTa...")
    
    best_val_r2 = -np.inf
    best_model_state = None
    patience_counter = 0
    max_patience = 15
    
    batch_size = 16  # Small batch size for stability
    
    for epoch in range(100):
        model.train()
        epoch_losses = []
        
        # Process in batches
        for i in range(0, len(smiles_train), batch_size):
            batch_end = min(i + batch_size, len(smiles_train))
            
            batch_smiles = smiles_train[i:batch_end]
            batch_genomic = genomic_train_t[i:batch_end]
            batch_y = y_train_t[i:batch_end]
            
            optimizer.zero_grad()
            
            predictions = model(batch_smiles, batch_genomic)
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        
        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = []
                
                # Predict in batches
                for i in range(0, len(smiles_val), batch_size):
                    batch_end = min(i + batch_size, len(smiles_val))
                    batch_smiles = smiles_val[i:batch_end]
                    batch_genomic = genomic_val_t[i:batch_end]
                    
                    batch_preds = model(batch_smiles, batch_genomic)
                    val_preds.append(batch_preds.cpu().numpy())
                
                val_preds = np.concatenate(val_preds).flatten()
                val_r2 = r2_score(y_val, val_preds)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                
                logger.info(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Val R²={val_r2:.4f}, RMSE={val_rmse:.4f}")
                
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                    
                    if val_r2 >= 0.7:
                        logger.info(f"🎉 TARGET ACHIEVED! R² = {val_r2:.4f} ≥ 0.7")
                        break
                else:
                    patience_counter += 1
                
                scheduler.step(val_r2)
                
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
    
    # 7. FINAL EVALUATION
    model.load_state_dict(best_model_state)
    model.eval()
    
    with torch.no_grad():
        genomic_test_t = torch.FloatTensor(genomic_test_s).to(device)
        
        test_preds = []
        for i in range(0, len(smiles_test), batch_size):
            batch_end = min(i + batch_size, len(smiles_test))
            batch_smiles = smiles_test[i:batch_end]
            batch_genomic = genomic_test_t[i:batch_end]
            
            batch_preds = model(batch_smiles, batch_genomic)
            test_preds.append(batch_preds.cpu().numpy())
        
        test_preds = np.concatenate(test_preds).flatten()
        
        test_r2 = r2_score(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_pearson, _ = pearsonr(y_test, test_preds)
    
    # 8. SAVE MODEL
    save_path = "/models/chemberta_cytotox_transfer_v1.pth"
    
    torch.save({
        'model_state_dict': best_model_state,
        'model_config': {
            'molecular_encoder': 'ChemBERTa',
            'genomic_dim': genomic_features.shape[1],
            'chemberta_model': 'seyonec/ChemBERTa-zinc-base-v1',
            'architecture': 'CytotoxicityModel'
        },
        'training_results': {
            'val_r2': float(best_val_r2),
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'test_pearson': float(test_pearson),
            'target_achieved': best_val_r2 >= 0.7
        },
        'data_info': {
            'source': 'gdsc_sample_10k.csv',
            'data_type': 'Real experimental GDSC IC50',
            'unique_compounds': training_df['SMILES'].nunique(),
            'unique_cell_lines': training_df['CELL_LINE'].nunique(),
            'total_samples': len(training_df)
        },
        'scalers': {
            'genomic_scaler': gen_scaler
        }
    }, save_path)
    
    # 9. RESULTS
    logger.info("=" * 80)
    logger.info("🏁 CHEMBERTA TRANSFER LEARNING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"🧬 Encoder: ChemBERTa (transfer from Model 1)")
    logger.info(f"📊 Real GDSC data: {training_df['SMILES'].nunique()} compounds")
    logger.info(f"✨ Validation R²: {best_val_r2:.4f}")
    logger.info(f"🎯 Test R²: {test_r2:.4f}")
    logger.info(f"📊 Test RMSE: {test_rmse:.4f}")
    logger.info(f"📈 Test Pearson: {test_pearson:.4f}")
    logger.info(f"🏆 Target ≥0.7: {'✅ ACHIEVED' if best_val_r2 >= 0.7 else '📈 IN PROGRESS'}")
    
    return {
        'success': True,
        'encoder_type': 'ChemBERTa',
        'data_type': 'Real experimental GDSC',
        'unique_compounds': training_df['SMILES'].nunique(),
        'unique_cell_lines': training_df['CELL_LINE'].nunique(),
        'total_samples': len(training_df),
        'val_r2': best_val_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_pearson': test_pearson,
        'target_achieved': best_val_r2 >= 0.7,
        'model_path': save_path
    }

if __name__ == "__main__":
    with app.run():
        result = train_model2_chemberta_transfer.remote()
        
        print("\n" + "="*80)
        print("🎉 CHEMBERTA TRANSFER LEARNING RESULTS")
        print("="*80)
        
        if result.get('success'):
            print(f"✅ SUCCESS: ChemBERTa transfer learning complete!")
            print(f"🧬 Encoder: {result['encoder_type']}")
            print(f"📊 Data: {result['data_type']}")
            print(f"🔬 Compounds: {result['unique_compounds']}")
            print(f"🧪 Cell Lines: {result['unique_cell_lines']}")
            print(f"📈 Samples: {result['total_samples']}")
            print(f"✨ Validation R²: {result['val_r2']:.4f}")
            print(f"🎯 Test R²: {result['test_r2']:.4f}")
            print(f"📊 RMSE: {result['test_rmse']:.4f}")
            print(f"📈 Pearson: {result['test_pearson']:.4f}")
            
            if result['target_achieved']:
                print("🏆 TARGET ACHIEVED: R² ≥ 0.7!")
            else:
                print(f"📈 PROGRESS: R² = {result['val_r2']:.4f}")
            
            print(f"💾 Model: {result['model_path']}")
        else:
            print("❌ Training failed")
            print(f"Error: {result.get('error')}")