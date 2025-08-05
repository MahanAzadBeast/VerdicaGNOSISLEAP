"""
Debug dimension issues in GNOSIS transfer learning
"""

import modal
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

app = modal.App("debug-dimensions")

image = modal.Image.debian_slim().pip_install([
    "torch==2.0.1",
    "transformers==4.33.0",
    "pandas==2.1.0",
    "numpy==1.24.3",
])

data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": data_volume}
)
def debug_model_dimensions():
    """Debug the actual dimensions we're getting"""
    
    print("🔍 DEBUGGING MODEL DIMENSIONS")
    print("=" * 50)
    
    # 1. Load the GDSC data
    gdsc_path = "/vol/gdsc_comprehensive_training_data.csv"
    df = pd.read_csv(gdsc_path)
    print(f"📊 GDSC data: {len(df):,} records")
    
    # Check genomic features
    genomic_cols = [col for col in df.columns if '_mutation' in col or '_cnv' in col or '_expression' in col]
    print(f"🧬 Genomic feature columns: {len(genomic_cols)}")
    
    # Get sample data
    sample_smiles = df['SMILES'].dropna().head(5).tolist()
    print(f"🧪 Sample SMILES: {len(sample_smiles)} molecules")
    
    # 2. Test ChemBERTa encoder
    print("\n🧬 TESTING CHEMBERTA ENCODER")
    
    model_name = "DeepChem/ChemBERTa-77M-MLM"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Encode sample SMILES
    inputs = tokenizer(
        sample_smiles,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        molecular_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    print(f"✅ ChemBERTa output shape: {molecular_features.shape}")
    print(f"   Expected: (batch_size, 768)")
    print(f"   Actual: ({molecular_features.shape[0]}, {molecular_features.shape[1]})")
    
    # 3. Test genomic features
    print("\n🧬 TESTING GENOMIC FEATURES")
    
    # Count actual genomic features available
    mutation_cols = [col for col in df.columns if '_mutation' in col]
    cnv_cols = [col for col in df.columns if '_cnv' in col]
    expression_cols = [col for col in df.columns if '_expression' in col]
    
    print(f"📋 Feature breakdown:")
    print(f"   Mutation features: {len(mutation_cols)}")
    print(f"   CNV features: {len(cnv_cols)}")
    print(f"   Expression features: {len(expression_cols)}")
    print(f"   Total genomic: {len(genomic_cols)}")
    
    # Create sample genomic features
    sample_genomic = df[genomic_cols].head(5).values
    print(f"✅ Genomic features shape: {sample_genomic.shape}")
    
    # 4. Test model dimensions
    print(f"\n🔧 DIMENSION ANALYSIS")
    mol_dim = molecular_features.shape[1]
    gen_dim = sample_genomic.shape[1]
    combined_dim = mol_dim + gen_dim
    
    print(f"   Molecular dimension: {mol_dim}")
    print(f"   Genomic dimension: {gen_dim}")
    print(f"   Combined dimension: {combined_dim}")
    print(f"   Expected combined: 768 + 128 = 896")
    
    # 5. Recommendations
    if mol_dim != 768:
        print(f"⚠️  ChemBERTa dimension mismatch: {mol_dim} != 768")
        
    if gen_dim > 128:
        print(f"⚠️  Too many genomic features: {gen_dim} > 128")
        print("   Recommendation: Reduce genomic features or adjust model")
    
    return {
        'molecular_dim': mol_dim,
        'genomic_dim': gen_dim,
        'combined_dim': combined_dim,
        'genomic_feature_count': len(genomic_cols)
    }

if __name__ == "__main__":
    with app.run():
        result = debug_model_dimensions.remote()
        
        print(f"\n📊 FINAL DIMENSIONS:")
        print(f"   Molecular: {result['molecular_dim']}")  
        print(f"   Genomic: {result['genomic_dim']}")
        print(f"   Combined: {result['combined_dim']}")
        print(f"   Genomic features: {result['genomic_feature_count']}")