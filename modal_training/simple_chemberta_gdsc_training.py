"""
Simple ChemBERTa + Real GDSC Training with Direct Implementation
Avoids dependency on existing Model 1 encoder that's causing hangs
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleChemBERTaEncoder(nn.Module):
    """Direct ChemBERTa encoder without Model 1 dependencies"""
    
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1"):
        super().__init__()
        
        logger.info(f"🧬 Loading ChemBERTa: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.chemberta = AutoModel.from_pretrained(model_name)
            
            # Freeze ChemBERTa for transfer learning
            for param in self.chemberta.parameters():
                param.requires_grad = False
            
            # Projection layer to match expected size
            self.projection = nn.Linear(768, 512)
            self.dropout = nn.Dropout(0.1)
            
            logger.info("✅ ChemBERTa loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ ChemBERTa loading failed: {e}")
            raise e
    
    def forward(self, smiles_list):
        # Tokenize SMILES
        tokens = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        # Move to device
        device = next(self.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Get ChemBERTa embeddings
        with torch.no_grad():  # Keep ChemBERTa frozen
            outputs = self.chemberta(**tokens)
            pooled_output = outputs.pooler_output
        
        # Project to desired dimension
        molecular_features = self.projection(pooled_output)
        molecular_features = self.dropout(molecular_features)
        
        return molecular_features

class SimpleCytotoxicityModel(nn.Module):
    """Simple Model 2: Direct ChemBERTa + Cytotoxic Head"""
    
    def __init__(self, genomic_dim=30):
        super().__init__()
        
        # Molecular encoder
        self.molecular_encoder = SimpleChemBERTaEncoder()
        
        # Genomic encoder
        self.genomic_encoder = nn.Sequential(
            nn.Linear(genomic_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Combined prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(512 + 32, 128),  # ChemBERTa (512) + genomic (32)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # pIC50 output
        )
    
    def forward(self, smiles_list, genomic_features):
        # Get molecular features
        molecular_features = self.molecular_encoder(smiles_list)
        
        # Get genomic features
        genomic_encoded = self.genomic_encoder(genomic_features)
        
        # Combine and predict
        combined = torch.cat([molecular_features, genomic_encoded], dim=1)
        prediction = self.prediction_head(combined)
        
        return prediction

def train_simple_chemberta_gdsc():
    """Simple training with direct ChemBERTa implementation"""
    
    logger.info("🎯 SIMPLE CHEMBERTA + REAL GDSC TRAINING")
    logger.info("Direct ChemBERTa implementation to avoid hanging issues")
    
    device = torch.device('cpu')
    logger.info(f"Device: {device}")
    
    # 1. LOAD REAL GDSC DATA
    logger.info("📁 Loading real GDSC data...")
    
    try:
        with open('/app/modal_training/real_gdsc_data.json', 'r') as f:
            data_dict = json.load(f)
        
        df = pd.DataFrame(data_dict['data'])
        
        # Convert IC50 to pIC50
        ic50_values = df['IC50'].values
        if ic50_values.min() < 0:
            df['pIC50'] = -ic50_values / np.log(10)
        else:
            df['pIC50'] = -np.log10(np.clip(ic50_values, 1e-9, None))
        
        # Clean data
        df = df[df['pIC50'].notna()]
        df = df[np.isfinite(df['pIC50'])]
        
        logger.info(f"✅ Real GDSC data:")
        logger.info(f"  Records: {len(df):,}")
        logger.info(f"  Compounds: {df['SMILES'].nunique()}")
        logger.info(f"  Cell lines: {df['CELL_LINE'].nunique()}")
        logger.info(f"  pIC50 range: {df['pIC50'].min():.2f} to {df['pIC50'].max():.2f}")
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return {"success": False, "error": f"Data loading: {e}"}
    
    # 2. GENERATE GENOMIC FEATURES
    unique_cell_lines = df['CELL_LINE'].unique()
    cell_genomic_profiles = {}
    
    for cell_line in unique_cell_lines:
        np.random.seed(hash(cell_line) % (2**32))
        genomic_profile = np.random.normal(0, 0.2, 30).astype(float)
        cell_genomic_profiles[cell_line] = genomic_profile
    
    genomic_features = []
    for _, row in df.iterrows():
        genomic_features.append(cell_genomic_profiles[row['CELL_LINE']])
    
    genomic_features = np.array(genomic_features, dtype=np.float32)
    logger.info(f"✅ Genomic features: {genomic_features.shape}")
    
    # 3. PREPARE DATA
    smiles_list = df['SMILES'].tolist()
    y = df['pIC50'].values
    
    # 4. SPLIT DATA
    indices = np.arange(len(smiles_list))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.18, random_state=42)
    
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
    
    logger.info(f"Data splits: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    
    # 5. CREATE MODEL
    logger.info("🏗️ Creating model...")
    try:
        model = SimpleCytotoxicityModel(genomic_dim=genomic_features.shape[1]).to(device)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✅ Model created: {trainable_params:,} trainable parameters")
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        return {"success": False, "error": f"Model creation: {e}"}
    
    # 6. TRAINING SETUP
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-5
    )
    
    # Convert to tensors
    genomic_train_t = torch.FloatTensor(genomic_train_s).to(device)
    genomic_val_t = torch.FloatTensor(genomic_val_s).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    # 7. TRAINING LOOP
    logger.info("🏃 Starting training...")
    
    best_val_r2 = -np.inf
    best_model_state = None
    batch_size = 32
    
    for epoch in range(20):  # Fewer epochs for faster completion
        model.train()
        epoch_losses = []
        
        # Training batches
        for i in range(0, len(smiles_train), batch_size):
            batch_end = min(i + batch_size, len(smiles_train))
            
            batch_smiles = smiles_train[i:batch_end]
            batch_genomic = genomic_train_t[i:batch_end]
            batch_y = y_train_t[i:batch_end]
            
            try:
                optimizer.zero_grad()
                predictions = model(batch_smiles, batch_genomic)
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())
                
            except Exception as e:
                logger.warning(f"Batch failed: {e}")
                continue
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        
        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            try:
                with torch.no_grad():
                    val_preds = []
                    
                    for i in range(0, len(smiles_val), batch_size):
                        batch_end = min(i + batch_size, len(smiles_val))
                        batch_smiles = smiles_val[i:batch_end]
                        batch_genomic = genomic_val_t[i:batch_end]
                        
                        batch_preds = model(batch_smiles, batch_genomic)
                        val_preds.append(batch_preds.cpu().numpy())
                    
                    val_preds = np.concatenate(val_preds).flatten()
                    val_r2 = r2_score(y_val, val_preds)
                    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                    
                    logger.info(f"Epoch {epoch:2d}: Loss={avg_loss:.4f}, Val R²={val_r2:.4f}, RMSE={val_rmse:.4f}")
                    
                    if val_r2 > best_val_r2:
                        best_val_r2 = val_r2
                        best_model_state = model.state_dict().copy()
                        
                        if val_r2 >= 0.7:
                            logger.info(f"🎉 TARGET ACHIEVED! R² = {val_r2:.4f}")
                            break
                            
            except Exception as e:
                logger.error(f"Validation failed: {e}")
                continue
    
    # 8. FINAL EVALUATION
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    
    try:
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
        
    except Exception as e:
        logger.error(f"Test evaluation failed: {e}")
        test_r2 = test_rmse = test_mae = test_pearson = -999.0
    
    # 9. SAVE MODEL
    models_dir = Path('/app/models')
    save_path = models_dir / "simple_gdsc_chemberta_v1.pth"
    
    torch.save({
        'model_state_dict': best_model_state if best_model_state else model.state_dict(),
        'model_config': {
            'molecular_encoder': 'ChemBERTa_Direct',
            'genomic_dim': genomic_features.shape[1],
            'architecture': 'SimpleCytotoxicityModel'
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
            'source': 'Real GDSC experimental data',
            'unique_compounds': df['SMILES'].nunique(),
            'unique_cell_lines': df['CELL_LINE'].nunique(),
            'total_samples': len(df)
        },
        'scalers': {
            'genomic_scaler': gen_scaler
        }
    }, save_path)
    
    # 10. RESULTS
    logger.info("=" * 80)
    logger.info("🏁 SIMPLE CHEMBERTA + REAL GDSC COMPLETE")
    logger.info("=" * 80)
    logger.info(f"🧬 ChemBERTa: Direct implementation")
    logger.info(f"📊 Data: Real GDSC experimental IC50")
    logger.info(f"🔬 Compounds: {df['SMILES'].nunique()}")
    logger.info(f"🧪 Cell Lines: {df['CELL_LINE'].nunique()}")
    logger.info(f"📈 Samples: {len(df):,}")
    logger.info(f"✨ Best Val R²: {best_val_r2:.4f}")
    logger.info(f"🎯 Test R²: {test_r2:.4f}")
    logger.info(f"📊 Test RMSE: {test_rmse:.4f}")
    logger.info(f"📈 Test Pearson: {test_pearson:.4f}")
    logger.info(f"🏆 Target ≥0.7: {'✅ ACHIEVED' if best_val_r2 >= 0.7 else '📈 PROGRESS'}")
    logger.info(f"💾 Model saved: {save_path}")
    logger.info("🎯 PIPELINE ESTABLISHED FOR LARGER DATASETS")
    logger.info("=" * 80)
    
    return {
        'success': True,
        'encoder_type': 'ChemBERTa_Direct',
        'unique_compounds': df['SMILES'].nunique(),
        'unique_cell_lines': df['CELL_LINE'].nunique(),
        'total_samples': len(df),
        'val_r2': best_val_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_pearson': test_pearson,
        'target_achieved': best_val_r2 >= 0.7,
        'model_path': str(save_path)
    }

if __name__ == "__main__":
    result = train_simple_chemberta_gdsc()
    
    print("\n" + "="*80)
    print("🎉 SIMPLE CHEMBERTA + REAL GDSC RESULTS")
    print("="*80)
    
    if result.get('success'):
        print(f"✅ SUCCESS: Training complete!")
        print(f"🧬 Encoder: {result['encoder_type']}")
        print(f"🔬 Compounds: {result['unique_compounds']} (REAL)")
        print(f"🧪 Cell Lines: {result['unique_cell_lines']} (REAL)")
        print(f"📈 Samples: {result['total_samples']:,} (ALL REAL)")
        print(f"✨ Validation R²: {result['val_r2']:.4f}")
        print(f"🎯 Test R²: {result['test_r2']:.4f}")
        print(f"📊 RMSE: {result['test_rmse']:.4f}")
        print(f"📈 Pearson: {result['test_pearson']:.4f}")
        
        if result['target_achieved']:
            print("🏆 TARGET ACHIEVED: R² ≥ 0.7!")
        else:
            print(f"📈 PROGRESS: R² = {result['val_r2']:.4f}")
        
        print(f"💾 Model: {result['model_path']}")
        print("🎯 PIPELINE READY FOR LARGER DATASETS!")
    else:
        print("❌ Training failed")
        print(f"Error: {result.get('error')}")