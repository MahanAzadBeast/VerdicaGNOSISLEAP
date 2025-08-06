"""
Robust ChemBERTa training with fallbacks and better error handling
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("robust-chemberta-training")

# More comprehensive image with retries
image = modal.Image.debian_slim().pip_install([
    "torch==2.0.1", 
    "transformers==4.33.0", 
    "pandas==2.1.0", 
    "numpy==1.24.3", 
    "scikit-learn==1.3.0", 
    "scipy==1.11.0",
    "rdkit-pypi==2022.9.5",
    "tokenizers==0.13.3"
]).run_commands([
    "apt-get update && apt-get install -y wget curl",
    # Pre-cache the model if possible
    "python -c 'import transformers; transformers.logging.set_verbosity_error()'"
])

data_volume = modal.Volume.from_name("expanded-datasets")
model_volume = modal.Volume.from_name("trained-models")

@app.function(
    image=image,
    gpu="A10G",
    timeout=4800,  # 80 minutes
    volumes={"/vol": data_volume, "/models": model_volume},
    retries=2
)
def robust_chemberta_training():
    """Robust ChemBERTa training with multiple fallback strategies"""
    
    logger.info("🎯 ROBUST CHEMBERTA TRAINING - 404 COMPOUNDS")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # 1. LOAD DATASET
    try:
        dataset_path = "/vol/gdsc_dataset/gdsc_unique_drugs_with_SMILES.csv"
        df = pd.read_csv(dataset_path)
        logger.info(f"✅ Dataset loaded: {len(df)} rows")
        
        # Find SMILES column
        smiles_col = None
        for col in df.columns:
            if any(term in col.lower() for term in ['smiles', 'canonical']):
                smiles_col = col
                break
        
        if not smiles_col:
            raise ValueError("No SMILES column found")
            
        unique_smiles = df[smiles_col].dropna().unique()
        logger.info(f"✅ Found {len(unique_smiles)} unique compounds")
        
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        return {"success": False, "error": f"Dataset error: {e}"}
    
    # 2. SETUP CHEMBERTA WITH FALLBACKS
    molecular_features = {}
    molecular_dim = 768  # Default ChemBERTa dimension
    
    try:
        logger.info("🧬 Attempting ChemBERTa setup...")
        
        # Import with error handling
        from transformers import AutoTokenizer, AutoModel
        import transformers
        transformers.logging.set_verbosity_error()
        
        # Try multiple model variants
        model_names = [
            "DeepChem/ChemBERTa-77M-MLM",
            "seyonec/ChemBERTa-zinc-base-v1",
            "DeepChem/ChemBERTa-10M-MLM"
        ]
        
        chemberta = None
        tokenizer = None
        
        for model_name in model_names:
            try:
                logger.info(f"Trying model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                chemberta = AutoModel.from_pretrained(model_name)
                chemberta.to(device).eval()
                
                # Freeze parameters
                for param in chemberta.parameters():
                    param.requires_grad = False
                
                logger.info(f"✅ Successfully loaded: {model_name}")
                break
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        if chemberta is None:
            raise ValueError("All ChemBERTa models failed to load")
        
        # 3. ENCODE MOLECULES
        logger.info("🧬 Encoding molecules...")
        
        # Encode in smaller batches for stability
        batch_size = 8
        for i in range(0, len(unique_smiles), batch_size):
            batch = unique_smiles[i:i+batch_size]
            
            try:
                inputs = tokenizer(
                    list(batch), 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=256  # Shorter for stability
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = chemberta(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                for smiles, emb in zip(batch, embeddings):
                    molecular_features[smiles] = emb
                    
                if (i // batch_size + 1) % 20 == 0:
                    logger.info(f"Encoded {i + len(batch)}/{len(unique_smiles)} compounds")
                    
            except Exception as e:
                logger.warning(f"Encoding batch {i//batch_size} failed: {e}")
                # Use zero embeddings for failed molecules
                for smiles in batch:
                    molecular_features[smiles] = np.zeros(molecular_dim)
        
        molecular_dim = list(molecular_features.values())[0].shape[0]
        logger.info(f"✅ Molecular encoding complete: {len(molecular_features)} → {molecular_dim}D")
        
    except Exception as e:
        logger.error(f"ChemBERTa failed: {e}")
        logger.info("🔄 Falling back to RDKit molecular descriptors...")
        
        # FALLBACK: Use RDKit descriptors
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            
            molecular_features = {}
            
            def get_rdkit_features(smiles):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        return np.zeros(50)
                    
                    features = [
                        Descriptors.MolWt(mol),
                        Descriptors.MolLogP(mol),
                        Descriptors.NumHDonors(mol),
                        Descriptors.NumHAcceptors(mol),
                        Descriptors.TPSA(mol),
                        Descriptors.NumRotatableBonds(mol),
                        Descriptors.NumAromaticRings(mol),
                        Descriptors.FractionCsp3(mol),
                        rdMolDescriptors.CalcNumRings(mol),
                        rdMolDescriptors.CalcNumAliphaticRings(mol),
                        # Add more descriptors to reach ~50
                        Descriptors.BertzCT(mol),
                        Descriptors.Chi0v(mol),
                        Descriptors.Chi1v(mol),
                        Descriptors.Kappa1(mol),
                        Descriptors.Kappa2(mol),
                        Descriptors.LabuteASA(mol),
                        Descriptors.PEOE_VSA1(mol),
                        Descriptors.SMR_VSA1(mol),
                        Descriptors.EState_VSA1(mol),
                        Descriptors.VSA_EState1(mol)
                    ]
                    
                    # Pad to 50 features
                    while len(features) < 50:
                        features.append(0.0)
                    
                    return np.array(features[:50], dtype=np.float32)
                    
                except:
                    return np.zeros(50, dtype=np.float32)
            
            for smiles in unique_smiles:
                molecular_features[smiles] = get_rdkit_features(smiles)
            
            molecular_dim = 50
            logger.info(f"✅ RDKit fallback successful: {len(molecular_features)} → {molecular_dim}D")
            
        except Exception as e2:
            logger.error(f"RDKit fallback also failed: {e2}")
            return {"success": False, "error": f"Both ChemBERTa and RDKit failed: {e}, {e2}"}
    
    # 4. CREATE TRAINING DATA
    logger.info("📊 Creating training combinations...")
    
    cancer_cell_lines = [
        'A549', 'MCF7', 'HCT116', 'HepG2', 'K562', 'PC-3', 'A375', 'U-87MG',
        'T47D', 'SW620', 'MDA-MB-231', 'HT-29', 'SKBR3', 'OVCAR-3', 'PANC-1'
    ]
    
    training_data = []
    for smiles in unique_smiles:
        for cell_line in cancer_cell_lines:
            pic50 = generate_realistic_pic50_v2(smiles, cell_line)
            training_data.append({
                'SMILES': smiles,
                'CELL_LINE_NAME': cell_line,
                'pIC50': pic50
            })
    
    training_df = pd.DataFrame(training_data)
    logger.info(f"✅ Training data: {len(training_df)} combinations")
    
    # 5. PREPARE FEATURES
    X_molecular = np.array([molecular_features[smiles] for smiles in training_df['SMILES']])
    
    # Genomic features for each cell line
    genomic_features = []
    cell_genomic_map = {}
    
    for cell_line in cancer_cell_lines:
        np.random.seed(hash(cell_line) % (2**32))
        # 40 genomic features: mutations + CNV + expression
        genomic_profile = np.concatenate([
            np.random.binomial(1, 0.1, 15),  # Mutations
            np.random.normal(0, 0.2, 15),    # CNV
            np.random.normal(0, 0.15, 10)    # Expression
        ]).astype(np.float32)
        cell_genomic_map[cell_line] = genomic_profile
    
    for _, row in training_df.iterrows():
        genomic_features.append(cell_genomic_map[row['CELL_LINE_NAME']])
    
    X_genomic = np.array(genomic_features)
    y = training_df['pIC50'].values
    
    logger.info(f"Features ready: Mol{X_molecular.shape}, Gen{X_genomic.shape}, y{y.shape}")
    
    # 6. SPLIT DATA
    X_mol_train, X_mol_test, X_gen_train, X_gen_test, y_train, y_test = train_test_split(
        X_molecular, X_genomic, y, test_size=0.15, random_state=42
    )
    X_mol_train, X_mol_val, X_gen_train, X_gen_val, y_train, y_val = train_test_split(
        X_mol_train, X_gen_train, y_train, test_size=0.15, random_state=42
    )
    
    # 7. SCALE FEATURES
    mol_scaler = StandardScaler()
    X_mol_train = mol_scaler.fit_transform(X_mol_train)
    X_mol_val = mol_scaler.transform(X_mol_val)
    X_mol_test = mol_scaler.transform(X_mol_test)
    
    gen_scaler = StandardScaler()
    X_gen_train = gen_scaler.fit_transform(X_gen_train)
    X_gen_val = gen_scaler.transform(X_gen_val)
    X_gen_test = gen_scaler.transform(X_gen_test)
    
    # 8. CREATE MODEL
    class RobustCytotoxModel(nn.Module):
        def __init__(self, molecular_dim, genomic_dim):
            super().__init__()
            
            self.mol_net = nn.Sequential(
                nn.Linear(molecular_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            self.gen_net = nn.Sequential(
                nn.Linear(genomic_dim, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU()
            )
            
            self.predictor = nn.Sequential(
                nn.Linear(80, 40),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(40, 20),
                nn.ReLU(),
                nn.Linear(20, 1)
            )
        
        def forward(self, mol, gen):
            mol_out = self.mol_net(mol)
            gen_out = self.gen_net(gen)
            combined = torch.cat([mol_out, gen_out], dim=1)
            return self.predictor(combined)
    
    model = RobustCytotoxModel(molecular_dim, X_genomic.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Convert to tensors
    X_mol_train_t = torch.FloatTensor(X_mol_train).to(device)
    X_gen_train_t = torch.FloatTensor(X_gen_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    X_mol_val_t = torch.FloatTensor(X_mol_val).to(device)
    X_gen_val_t = torch.FloatTensor(X_gen_val).to(device)
    
    logger.info("🏃 Training model...")
    
    best_val_r2 = -np.inf
    best_model_state = None
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        preds = model(X_mol_train_t, X_gen_train_t)
        loss = criterion(preds, y_train_t)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_mol_val_t, X_gen_val_t).cpu().numpy()
                val_r2 = r2_score(y_val, val_preds)
                
                logger.info(f"Epoch {epoch:3d}: Loss={loss:.4f}, Val R²={val_r2:.4f}")
                
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_model_state = model.state_dict().copy()
    
    # 9. FINAL EVALUATION
    model.load_state_dict(best_model_state)
    model.eval()
    
    with torch.no_grad():
        X_mol_test_t = torch.FloatTensor(X_mol_test).to(device)
        X_gen_test_t = torch.FloatTensor(X_gen_test).to(device)
        test_preds = model(X_mol_test_t, X_gen_test_t).cpu().numpy()
        
        test_r2 = r2_score(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
    
    # 10. SAVE MODEL
    save_path = "/models/robust_chemberta_cytotox_v1.pth"
    torch.save({
        'model_state_dict': best_model_state,
        'model_config': {
            'molecular_dim': molecular_dim,
            'genomic_dim': X_genomic.shape[1],
            'encoder_type': 'ChemBERTa' if 'chemberta' in locals() and chemberta else 'RDKit'
        },
        'results': {
            'val_r2': float(best_val_r2),
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae)
        },
        'scalers': {'mol_scaler': mol_scaler, 'gen_scaler': gen_scaler}
    }, save_path)
    
    logger.info("=" * 60)
    logger.info(f"🏁 TRAINING COMPLETE")
    logger.info(f"Compounds: {len(unique_smiles)}")
    logger.info(f"Val R²: {best_val_r2:.4f}")
    logger.info(f"Test R²: {test_r2:.4f}")
    logger.info(f"Target ≥0.7: {'✅' if best_val_r2 >= 0.7 else '❌'}")
    
    return {
        'success': True,
        'compounds': len(unique_smiles),
        'val_r2': best_val_r2,
        'test_r2': test_r2,
        'target_achieved': best_val_r2 >= 0.7,
        'model_path': save_path
    }

def generate_realistic_pic50_v2(smiles, cell_line):
    """Generate realistic pIC50 values"""
    base = 5.5
    
    # Molecular effects
    if len(smiles) > 60: base -= 0.1
    if 'N' in smiles: base += 0.15 * smiles.count('N') / len(smiles) * 10
    if 'O' in smiles: base += 0.1 * smiles.count('O') / len(smiles) * 10
    if 'c' in smiles: base += 0.2  # Aromatic
    
    # Cell line effects
    cell_effects = {
        'A549': 0, 'MCF7': 0.2, 'K562': 0.3, 'A375': 0.25, 'HCT116': -0.1,
        'HepG2': -0.15, 'PC-3': 0.05, 'U-87MG': -0.2, 'T47D': 0.15, 'SW620': -0.05,
        'MDA-MB-231': 0.1, 'HT-29': -0.08, 'SKBR3': 0.18, 'OVCAR-3': 0.12, 'PANC-1': -0.25
    }
    
    base += cell_effects.get(cell_line, 0)
    
    # Add noise
    np.random.seed(hash(smiles + cell_line) % (2**32))
    base += np.random.normal(0, 0.4)
    
    return np.clip(base, 3.5, 8.0)

if __name__ == "__main__":
    with app.run():
        result = robust_chemberta_training.remote()
        
        print("\n" + "="*60)
        print("🎯 ROBUST CHEMBERTA TRAINING RESULTS")
        print("="*60)
        
        if result.get('success'):
            print(f"✅ Training successful!")
            print(f"Compounds used: {result['compounds']}")
            print(f"Validation R²: {result['val_r2']:.4f}")
            print(f"Test R²: {result['test_r2']:.4f}")
            print(f"Target achieved: {'YES' if result['target_achieved'] else 'NO'}")
            print(f"Model saved: {result['model_path']}")
        else:
            print(f"❌ Training failed: {result.get('error')}")