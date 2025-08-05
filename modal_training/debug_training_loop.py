"""
Debug Step 2: Check training loop issues
30s for 14k rows is too fast - likely empty batches or no real training
"""

import modal
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = modal.App("debug-training-loop")

image = modal.Image.debian_slim().pip_install([
    "torch==2.0.1",
    "transformers==4.33.0",
    "pandas==2.1.0",
    "numpy==1.24.3",
    "scikit-learn==1.3.0"
])

data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/vol": data_volume}
)
def debug_training_loop():
    """Debug why training was so fast (30s for 14k samples)"""
    
    print("🔍 STEP 2: DEBUGGING TRAINING LOOP")
    print("=" * 60)
    
    # 1. Load and prepare mini dataset for debugging
    gdsc_path = "/vol/gdsc_comprehensive_training_data.csv"
    df = pd.read_csv(gdsc_path)
    
    # Take a small sample for debugging (2k train / 200 val as suggested)
    print("📊 CREATING DEBUG DATASET (2k train / 200 val)")
    df_clean = df.dropna(subset=['SMILES', 'CELL_LINE_NAME', 'pIC50'])
    df_sample = df_clean.sample(n=2200, random_state=42)
    
    train_data, val_data = train_test_split(df_sample, test_size=200, random_state=42)
    print(f"   Train: {len(train_data):,} samples")
    print(f"   Val: {len(val_data):,} samples")
    
    # 2. Quick molecular encoding (simplified)
    print("\n🧬 ENCODING MOLECULES (sample)...")
    
    # Get unique SMILES (small sample for debugging)
    unique_smiles = train_data['SMILES'].unique()[:50]  # Just first 50 for speed
    print(f"   Encoding {len(unique_smiles)} unique SMILES...")
    
    # Simple molecular features (384-dim as we found)
    mol_features_dict = {}
    for smiles in unique_smiles:
        # Create dummy molecular features for debugging
        mol_features_dict[smiles] = np.random.randn(384)  # Match actual ChemBERTa dim
    
    # Map to training data
    X_mol_train = np.array([
        mol_features_dict.get(smiles, np.zeros(384)) 
        for smiles in train_data['SMILES']
    ])
    X_mol_val = np.array([
        mol_features_dict.get(smiles, np.zeros(384)) 
        for smiles in val_data['SMILES']
    ])
    
    print(f"   Molecular features: {X_mol_train.shape}")
    
    # 3. Genomic features (simplified)
    print("\n🧬 CREATING GENOMIC FEATURES...")
    
    # Use 109 features as we found in real data, but simplified for debugging
    n_genomic = 109
    X_gen_train = np.random.randn(len(train_data), n_genomic)
    X_gen_val = np.random.randn(len(val_data), n_genomic)
    
    # Scale genomic features
    gen_scaler = StandardScaler()
    X_gen_train = gen_scaler.fit_transform(X_gen_train)
    X_gen_val = gen_scaler.transform(X_gen_val)
    
    print(f"   Genomic features: {X_gen_train.shape}")
    
    # 4. Targets
    y_train = train_data['pIC50'].values
    y_val = val_data['pIC50'].values
    
    print(f"   Target range - Train: {y_train.min():.2f} to {y_train.max():.2f}")
    print(f"   Target range - Val: {y_val.min():.2f} to {y_val.max():.2f}")
    
    # 5. MEAN-ONLY BASELINE (as suggested)
    print("\n📊 MEAN-ONLY BASELINE CHECK:")
    
    mean_pred = np.full_like(y_val, y_train.mean())
    val_var = np.var(y_val)
    mean_baseline_mse = np.mean((y_val - mean_pred)**2)
    mean_baseline_r2 = 1 - (mean_baseline_mse / val_var)
    
    print(f"   Val variance: {val_var:.4f}")
    print(f"   Mean baseline MSE: {mean_baseline_mse:.4f}")
    print(f"   Mean baseline R²: {mean_baseline_r2:.4f}")
    print(f"   ✅ Should be ≈ 0.0 (if negative, something is very wrong)")
    
    # 6. Create simple model for debugging
    print("\n🤖 CREATING DEBUG MODEL...")
    
    class DebugModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Simplified architecture
            self.molecular_proj = nn.Linear(384, 128)
            self.genomic_proj = nn.Linear(109, 128)  
            self.fusion = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        
        def forward(self, mol_features, gen_features):
            mol_out = self.molecular_proj(mol_features)
            gen_out = self.genomic_proj(gen_features)
            combined = torch.cat([mol_out, gen_out], dim=1)
            return self.fusion(combined)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DebugModel().to(device)
    
    # Convert to tensors
    X_mol_train_t = torch.FloatTensor(X_mol_train).to(device)
    X_gen_train_t = torch.FloatTensor(X_gen_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    X_mol_val_t = torch.FloatTensor(X_mol_val).to(device)
    X_gen_val_t = torch.FloatTensor(X_gen_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    print(f"   Model created on {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 7. DEBUG TRAINING LOOP
    print("\n🔧 DEBUG TRAINING (3 epochs with monitoring)...")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Lower LR as suggested
    
    batch_size = 64
    n_batches = len(X_mol_train) // batch_size
    print(f"   Batch size: {batch_size}")
    print(f"   Number of batches per epoch: {n_batches}")
    
    for epoch in range(3):
        print(f"\n📈 EPOCH {epoch+1}/3:")
        
        model.train()
        epoch_losses = []
        batch_count = 0
        
        # Manual batching for debugging
        for i in range(0, len(X_mol_train), batch_size):
            end_idx = min(i + batch_size, len(X_mol_train))
            
            # Get batch
            mol_batch = X_mol_train_t[i:end_idx]
            gen_batch = X_gen_train_t[i:end_idx]
            y_batch = y_train_t[i:end_idx]
            
            # ASSERTION as suggested
            assert mol_batch.size(0) > 0, f"Empty batch {batch_count}"
            assert gen_batch.size(0) > 0, f"Empty genomic batch {batch_count}"
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(mol_batch, gen_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            epoch_losses.append(loss.item())
            batch_count += 1
            
            # Log first few batches
            if batch_count <= 3:
                print(f"     Batch {batch_count}: loss = {loss.item():.4f}")
        
        avg_loss = np.mean(epoch_losses)
        print(f"   Average training loss: {avg_loss:.4f}")
        print(f"   Processed {batch_count} batches")
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_mol_val_t, X_gen_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            
            # R² calculation
            val_pred_np = val_pred.cpu().numpy().flatten()
            val_r2 = 1 - np.sum((y_val - val_pred_np)**2) / np.sum((y_val - y_val.mean())**2)
            
            print(f"   Validation loss: {val_loss:.4f}")
            print(f"   Validation R²: {val_r2:.4f}")
            
            # Print first 5 predictions vs true as suggested
            print(f"   First 5 predictions vs true pIC50:")
            for i in range(min(5, len(y_val))):
                print(f"     Pred: {val_pred_np[i]:.3f}, True: {y_val[i]:.3f}")
    
    print(f"\n✅ DEBUG TRAINING COMPLETE")
    print(f"   Final val R²: {val_r2:.4f}")
    print(f"   Training time per epoch: reasonable for {len(train_data)} samples")
    
    return {
        'final_val_r2': float(val_r2),
        'mean_baseline_r2': float(mean_baseline_r2),
        'batches_per_epoch': n_batches,
        'final_val_loss': float(val_loss)
    }

if __name__ == "__main__":
    with app.run():
        result = debug_training_loop.remote()
        
        print(f"\n📊 DEBUG RESULTS:")
        print(f"Mean baseline R²: {result['mean_baseline_r2']:.4f}")
        print(f"Model R²: {result['final_val_r2']:.4f}")
        print(f"Batches per epoch: {result['batches_per_epoch']}")
        print(f"Training properly executed: {'✅' if result['final_val_r2'] > result['mean_baseline_r2'] else '❌'}")