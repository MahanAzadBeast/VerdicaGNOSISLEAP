"""
Model 2: Cytotoxicity Prediction using Unified GDSC Database with ChemBERT
Target: R² > 0.7 using real unified GDSC data with matching SMILES
Strategy: Use actual dimensions from expanded-datasets volume
"""

import modal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import pickle

# Molecular processing
try:
    from rdkit import Chem
    from rdkit.Chem import Scaffolds
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available - using alternative splits")

# Transformer model
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Modal app configuration  
app = modal.App("unified-gdsc-chemberta-training")

# Production image
image = modal.Image.debian_slim().pip_install([
    "torch==2.0.1",
    "transformers==4.33.0",
    "pandas==2.1.0", 
    "numpy==1.24.3",
    "scikit-learn==1.3.0",
    "scipy==1.11.0",
    "rdkit-pypi==2022.9.5",
    "tokenizers==0.13.3",
])

# Modal volumes
data_volume = modal.Volume.from_name("expanded-datasets")
model_volume = modal.Volume.from_name("trained-models")

class UnifiedGDSCDataLoader:
    """Load and process the unified GDSC database"""
    
    def __init__(self):
        self.data = None
        self.molecular_dim = None
        self.genomic_dim = None
        
    def load_unified_gdsc_data(self):
        """Load the unified GDSC database with matching SMILES"""
        
        logger.info("📊 LOADING UNIFIED GDSC DATABASE")
        logger.info("=" * 60)
        
        # Load the unified dataset
        gdsc_path = "/vol/gdsc_comprehensive_training_data.csv"
        assert os.path.exists(gdsc_path), f"Unified GDSC data not found: {gdsc_path}"
        
        self.data = pd.read_csv(gdsc_path)
        logger.info(f"✅ Loaded unified GDSC: {len(self.data):,} records")
        
        # Check data structure
        logger.info(f"📋 Columns: {len(self.data.columns)} total")
        logger.info(f"   Sample columns: {list(self.data.columns[:10])}")
        
        # Verify essential columns exist
        required_columns = ['SMILES', 'pIC50', 'CELL_LINE_NAME']
        missing_cols = [col for col in required_columns if col not in self.data.columns]
        
        if missing_cols:
            # Try alternative column names
            alt_mapping = {
                'SMILES': ['smiles', 'Smiles', 'canonical_smiles'],
                'pIC50': ['pic50', 'IC50', 'ic50', 'ln_ic50', 'LN_IC50'],
                'CELL_LINE_NAME': ['cell_line', 'Cell_Line', 'COSMIC_ID', 'cell_line_name']
            }
            
            for required_col, alternatives in alt_mapping.items():
                if required_col not in self.data.columns:
                    found_alt = None
                    for alt in alternatives:
                        if alt in self.data.columns:
                            found_alt = alt
                            break
                    
                    if found_alt:
                        self.data[required_col] = self.data[found_alt]
                        logger.info(f"   Mapped {found_alt} → {required_col}")
                    else:
                        logger.error(f"❌ Cannot find column for {required_col}")
                        logger.error(f"   Available columns: {list(self.data.columns)}")
                        raise ValueError(f"Missing required column: {required_col}")
        
        # Data summary
        logger.info(f"✅ Data loaded successfully:")
        logger.info(f"   Unique SMILES: {self.data['SMILES'].nunique():,}")
        logger.info(f"   Unique cell lines: {self.data['CELL_LINE_NAME'].nunique()}")
        logger.info(f"   pIC50 range: {self.data['pIC50'].min():.2f} - {self.data['pIC50'].max():.2f}")
        
        return self.data
    
    def analyze_features(self):
        """Analyze the feature structure in the dataset"""
        
        logger.info("🔍 ANALYZING FEATURE STRUCTURE")
        
        # Find potential genomic features
        genomic_features = []
        for col in self.data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['mutation', 'cnv', 'expression', 'methylation', 'protein']):
                genomic_features.append(col)
        
        logger.info(f"🧬 Found {len(genomic_features)} genomic feature columns")
        
        # Categorize features
        mutation_cols = [col for col in genomic_features if 'mutation' in col.lower()]
        cnv_cols = [col for col in genomic_features if 'cnv' in col.lower()]
        expression_cols = [col for col in genomic_features if 'expression' in col.lower()]
        other_cols = [col for col in genomic_features if col not in mutation_cols + cnv_cols + expression_cols]
        
        logger.info(f"   Mutation features: {len(mutation_cols)}")
        logger.info(f"   CNV features: {len(cnv_cols)}")
        logger.info(f"   Expression features: {len(expression_cols)}")
        logger.info(f"   Other genomic features: {len(other_cols)}")
        
        self.genomic_dim = len(genomic_features)
        
        return {
            'genomic_features': genomic_features,
            'mutation_features': mutation_cols,
            'cnv_features': cnv_cols,
            'expression_features': expression_cols,
            'total_genomic_dim': len(genomic_features)
        }
    
    def preprocess_data(self):
        """Clean and preprocess the unified GDSC data"""
        
        logger.info("🧹 PREPROCESSING UNIFIED GDSC DATA")
        
        if self.data is None:
            logger.error("❌ No data to preprocess!")
            return None
            
        df = self.data.copy()
        initial_count = len(df)
        
        # 1. Remove rows with missing essential data
        essential_cols = ['SMILES', 'pIC50', 'CELL_LINE_NAME']
        df = df.dropna(subset=essential_cols)
        logger.info(f"✅ After removing NaN: {len(df):,} records ({100*len(df)/initial_count:.1f}%)")
        
        # 2. Filter pIC50 values to reasonable range
        df = df[(df['pIC50'] >= 3.0) & (df['pIC50'] <= 12.0)]
        logger.info(f"✅ After pIC50 filter: {len(df):,} records ({100*len(df)/initial_count:.1f}%)")
        
        # 3. Remove duplicates based on SMILES + cell line
        pre_dedup = len(df)
        df = df.drop_duplicates(subset=['SMILES', 'CELL_LINE_NAME'], keep='first')
        logger.info(f"✅ After deduplication: {len(df):,} records (-{pre_dedup-len(df):,})")
        
        # 4. Ensure SMILES are valid (basic check)
        valid_smiles_mask = df['SMILES'].str.len() > 5  # Basic validity check
        df = df[valid_smiles_mask]
        logger.info(f"✅ After SMILES validation: {len(df):,} records")
        
        # 5. Final data summary
        final_count = len(df)
        logger.info("📊 PREPROCESSING COMPLETE")
        logger.info(f"   Initial records: {initial_count:,}")
        logger.info(f"   Final records: {final_count:,}")
        logger.info(f"   Data retention: {100*final_count/initial_count:.1f}%")
        logger.info(f"   Unique SMILES: {df['SMILES'].nunique():,}")
        logger.info(f"   Unique cell lines: {df['CELL_LINE_NAME'].nunique():,}")
        logger.info(f"   pIC50 range: {df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")
        
        return df

class ChemBERTaMolecularEncoder:
    """ChemBERTa encoder for molecular features"""
    
    def __init__(self):
        self.model_name = "DeepChem/ChemBERTa-77M-MLM"
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_encoder()
    
    def setup_encoder(self):
        """Setup ChemBERTa encoder"""
        
        logger.info("🧬 SETTING UP CHEMBERTA ENCODER")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Test to get actual output dimensions
            test_smiles = ["CCO", "c1ccccc1"]  # Simple test molecules
            test_features = self.encode_smiles_batch(test_smiles)
            self.output_dim = test_features.shape[1]
            
            logger.info(f"✅ ChemBERTa encoder ready")
            logger.info(f"   Model: {self.model_name}")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Output dimension: {self.output_dim}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup ChemBERTa: {e}")
            raise
    
    def encode_smiles_batch(self, smiles_list, batch_size=32):
        """Encode SMILES to molecular features"""
        
        if len(smiles_list) == 0:
            return np.array([]).reshape(0, self.output_dim if hasattr(self, 'output_dim') else 768)
        
        logger.info(f"🧬 Encoding {len(smiles_list):,} SMILES...")
        
        features = []
        
        try:
            for i in range(0, len(smiles_list), batch_size):
                batch_smiles = smiles_list[i:i+batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_smiles,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use [CLS] token embedding
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    features.append(cls_embeddings)
            
            result = np.vstack(features) if features else np.array([]).reshape(0, -1)
            logger.info(f"✅ Molecular encoding complete: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"❌ ChemBERTa encoding failed: {e}")
            raise

class CytotoxicityModel(nn.Module):
    """Neural network for cytotoxicity prediction"""
    
    def __init__(self, molecular_dim, genomic_dim, hidden_dim=512):
        super().__init__()
        
        self.molecular_dim = molecular_dim
        self.genomic_dim = genomic_dim
        
        # Molecular encoder (optional processing)
        self.molecular_encoder = nn.Sequential(
            nn.Linear(molecular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Genomic encoder
        self.genomic_encoder = nn.Sequential(
            nn.Linear(genomic_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion layers
        combined_dim = hidden_dim + 128
        self.fusion_layers = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(0.3),
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, molecular_features, genomic_features):
        """Forward pass"""
        
        # Process molecular features
        mol_encoded = self.molecular_encoder(molecular_features)
        
        # Process genomic features  
        gen_encoded = self.genomic_encoder(genomic_features)
        
        # Combine and predict
        combined = torch.cat([mol_encoded, gen_encoded], dim=1)
        prediction = self.fusion_layers(combined)
        
        return prediction

@app.function(
    image=image,
    gpu="A10G",
    timeout=21600,  # 6 hours
    volumes={"/vol": data_volume, "/models": model_volume}
)
def train_unified_gdsc_cytotox_model():
    """
    Train cytotoxicity model using unified GDSC database
    Target: R² > 0.7
    """
    
    logger.info("🎯 UNIFIED GDSC CYTOTOXICITY TRAINING")
    logger.info("=" * 80)
    logger.info("TARGET: R² > 0.7 using unified GDSC with matching SMILES")
    logger.info("STRATEGY: ChemBERTa + genomic features + deep neural network")
    logger.info("=" * 80)
    
    # 1. LOAD UNIFIED GDSC DATA
    logger.info("1️⃣ LOADING UNIFIED GDSC DATA")
    
    data_loader = UnifiedGDSCDataLoader()
    raw_data = data_loader.load_unified_gdsc_data()
    
    # 2. ANALYZE FEATURE STRUCTURE
    logger.info("2️⃣ ANALYZING FEATURE STRUCTURE")
    
    feature_info = data_loader.analyze_features()
    genomic_features = feature_info['genomic_features']
    
    # 3. PREPROCESS DATA
    logger.info("3️⃣ PREPROCESSING DATA")
    
    clean_data = data_loader.preprocess_data()
    if clean_data is None or len(clean_data) < 1000:
        logger.error("❌ Insufficient data after preprocessing!")
        return {"error": "Insufficient data"}
    
    # 4. SETUP MOLECULAR ENCODER
    logger.info("4️⃣ SETTING UP CHEMBERTA ENCODER")
    
    molecular_encoder = ChemBERTaMolecularEncoder()
    
    # 5. ENCODE MOLECULAR FEATURES
    logger.info("5️⃣ ENCODING MOLECULAR FEATURES")
    
    unique_smiles = clean_data['SMILES'].unique()
    molecular_features_dict = {}
    molecular_features_array = molecular_encoder.encode_smiles_batch(list(unique_smiles))
    
    for smiles, features in zip(unique_smiles, molecular_features_array):
        molecular_features_dict[smiles] = features
    
    molecular_dim = molecular_features_array.shape[1]
    logger.info(f"✅ Encoded {len(unique_smiles):,} unique SMILES → {molecular_dim}-dim features")
    
    # 6. PREPARE TRAINING DATA
    logger.info("6️⃣ PREPARING TRAINING DATA")
    
    # Get molecular features for all samples
    X_molecular = np.array([molecular_features_dict[smiles] for smiles in clean_data['SMILES']])
    
    # Get genomic features
    if genomic_features:
        X_genomic = clean_data[genomic_features].fillna(0).values
        genomic_dim = X_genomic.shape[1]
        logger.info(f"✅ Using {genomic_dim} genomic features")
    else:
        # Create dummy genomic features if none available
        genomic_dim = 50
        X_genomic = np.random.randn(len(clean_data), genomic_dim) * 0.1
        logger.warning("⚠️ No genomic features found - using dummy features")
    
    # Targets
    y = clean_data['pIC50'].values
    
    logger.info(f"📊 Training data prepared:")
    logger.info(f"   Molecular features: {X_molecular.shape}")
    logger.info(f"   Genomic features: {X_genomic.shape}")
    logger.info(f"   Targets: {y.shape}")
    
    # 7. CREATE TRAIN/VAL/TEST SPLITS
    logger.info("7️⃣ CREATING DATA SPLITS")
    
    # 80/10/10 split
    X_mol_temp, X_mol_test, X_gen_temp, X_gen_test, y_temp, y_test = train_test_split(
        X_molecular, X_genomic, y, test_size=0.1, random_state=42
    )
    
    X_mol_train, X_mol_val, X_gen_train, X_gen_val, y_train, y_val = train_test_split(
        X_mol_temp, X_gen_temp, y_temp, test_size=0.111, random_state=42  # 0.111 * 0.9 ≈ 0.1
    )
    
    logger.info(f"✅ Data splits created:")
    logger.info(f"   Train: {len(y_train):,} samples")
    logger.info(f"   Val: {len(y_val):,} samples")
    logger.info(f"   Test: {len(y_test):,} samples")
    
    # 8. SCALE FEATURES
    logger.info("8️⃣ SCALING FEATURES")
    
    # Scale molecular features
    mol_scaler = StandardScaler()
    X_mol_train_scaled = mol_scaler.fit_transform(X_mol_train)
    X_mol_val_scaled = mol_scaler.transform(X_mol_val)
    X_mol_test_scaled = mol_scaler.transform(X_mol_test)
    
    # Scale genomic features
    gen_scaler = StandardScaler()
    X_gen_train_scaled = gen_scaler.fit_transform(X_gen_train)
    X_gen_val_scaled = gen_scaler.transform(X_gen_val)
    X_gen_test_scaled = gen_scaler.transform(X_gen_test)
    
    # 9. CREATE MODEL
    logger.info("9️⃣ CREATING CYTOTOXICITY MODEL")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CytotoxicityModel(
        molecular_dim=molecular_dim,
        genomic_dim=genomic_dim,
        hidden_dim=512
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"✅ Model created on {device}")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Trainable parameters: {trainable_params:,}")
    
    # 10. TRAINING SETUP
    logger.info("🔟 TRAINING SETUP")
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Convert to tensors
    X_mol_train_t = torch.FloatTensor(X_mol_train_scaled).to(device)
    X_gen_train_t = torch.FloatTensor(X_gen_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    X_mol_val_t = torch.FloatTensor(X_mol_val_scaled).to(device)
    X_gen_val_t = torch.FloatTensor(X_gen_val_scaled).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # 11. TRAINING LOOP
    logger.info("1️⃣1️⃣ TRAINING MODEL")
    
    best_val_r2 = -np.inf
    best_model_state = None
    patience_counter = 0
    patience = 20
    
    for epoch in range(200):  # More epochs for better convergence
        # Training
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_mol_train_t, X_gen_train_t)
        loss = criterion(predictions, y_train_t)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_mol_val_t, X_gen_val_t)
                val_r2 = r2_score(y_val, val_predictions.cpu().numpy())
                val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions.cpu().numpy()))
                val_pearson, _ = pearsonr(y_val, val_predictions.cpu().numpy().flatten())
                
                logger.info(f"Epoch {epoch+1:3d}: Loss={loss:.4f}, Val R²={val_r2:.4f}, RMSE={val_rmse:.4f}, r={val_pearson:.4f}")
                
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                    
                    if val_r2 >= 0.7:
                        logger.info(f"🎉 TARGET ACHIEVED! Val R² = {val_r2:.4f} ≥ 0.7")
                else:
                    patience_counter += 1
                
                scheduler.step(val_r2)
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
    
    # 12. FINAL EVALUATION
    logger.info("1️⃣2️⃣ FINAL EVALUATION")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        # Test set evaluation
        X_mol_test_t = torch.FloatTensor(X_mol_test_scaled).to(device)
        X_gen_test_t = torch.FloatTensor(X_gen_test_scaled).to(device)
        test_predictions = model(X_mol_test_t, X_gen_test_t)
        
        test_r2 = r2_score(y_test, test_predictions.cpu().numpy())
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions.cpu().numpy()))
        test_mae = mean_absolute_error(y_test, test_predictions.cpu().numpy())
        test_pearson, _ = pearsonr(y_test, test_predictions.cpu().numpy().flatten())
        test_spearman, _ = spearmanr(y_test, test_predictions.cpu().numpy().flatten())
    
    # 13. SAVE MODEL
    logger.info("1️⃣3️⃣ SAVING MODEL")
    
    model_save_path = "/models/unified_gdsc_cytotox_model.pth"
    
    save_dict = {
        'model_state_dict': best_model_state,
        'model_config': {
            'molecular_dim': molecular_dim,
            'genomic_dim': genomic_dim,
            'hidden_dim': 512,
            'architecture': 'chemberta_cytotox_unified'
        },
        'training_results': {
            'best_val_r2': float(best_val_r2),
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'test_pearson': float(test_pearson),
            'test_spearman': float(test_spearman),
            'target_achieved': best_val_r2 >= 0.7
        },
        'data_info': {
            'total_samples': len(clean_data),
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'unique_smiles': len(unique_smiles),
            'molecular_dim': molecular_dim,
            'genomic_dim': genomic_dim,
            'genomic_features': genomic_features
        },
        'scalers': {
            'molecular_scaler': mol_scaler,
            'genomic_scaler': gen_scaler
        },
        'feature_info': feature_info
    }
    
    torch.save(save_dict, model_save_path)
    
    # 14. RESULTS SUMMARY
    logger.info("🏁 UNIFIED GDSC CYTOTOXICITY TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"🏆 VALIDATION R²: {best_val_r2:.4f}")
    logger.info(f"🧪 TEST R²: {test_r2:.4f}")
    logger.info(f"📊 TEST RMSE: {test_rmse:.4f}")
    logger.info(f"📊 TEST MAE: {test_mae:.4f}")
    logger.info(f"📊 TEST Pearson r: {test_pearson:.4f}")
    logger.info(f"📊 TEST Spearman ρ: {test_spearman:.4f}")
    logger.info(f"🎯 TARGET (R² ≥ 0.7): {'✅ ACHIEVED' if best_val_r2 >= 0.7 else '📈 IN PROGRESS'}")
    logger.info(f"📦 Model saved: {model_save_path}")
    logger.info("=" * 80)
    
    return {
        'val_r2': best_val_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_pearson': test_pearson,
        'test_spearman': test_spearman,
        'target_achieved': best_val_r2 >= 0.7,
        'model_path': model_save_path,
        'data_samples': len(clean_data),
        'molecular_dim': molecular_dim,
        'genomic_dim': genomic_dim,
        'approach': 'unified_gdsc_chemberta_cytotox'
    }

if __name__ == "__main__":
    logger.info("🧬 UNIFIED GDSC CYTOTOXICITY TRAINING")
    logger.info("🎯 TARGET: R² > 0.7 with unified GDSC database")
    
    with app.run():
        result = train_unified_gdsc_cytotox_model.remote()
        
        logger.info("🎉 TRAINING COMPLETED!")
        logger.info(f"📊 Results: {result}")
        
        if result.get('target_achieved'):
            logger.info("🏆 SUCCESS: R² > 0.7 TARGET ACHIEVED!")
        else:
            logger.info(f"📈 Progress: R² = {result.get('val_r2', 0):.4f}")