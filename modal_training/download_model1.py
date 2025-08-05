"""
Download and Load Model 1 for Backend Integration
Download the best Model 1 checkpoint and prepare it for inference
"""

import modal
import torch
import pickle
from pathlib import Path
import json

image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "torch==2.0.1", 
    "transformers==4.30.2",
    "pandas",
    "numpy==1.24.3",
    "scikit-learn"
])

app = modal.App("download-model1")

models_volume = modal.Volume.from_name("cell-line-models", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/models": models_volume}
)
def download_best_model1():
    """Download the best Model 1 checkpoint and metadata"""
    
    print("📥 DOWNLOADING BEST MODEL 1 CHECKPOINT")
    print("=" * 60)
    
    # Load the best model checkpoint
    best_model_path = Path("/models/model1_checkpoints/best_model1.pt")
    epoch_31_path = Path("/models/model1_checkpoints/epoch_31_r2_0.6281.pt")
    
    if epoch_31_path.exists():
        print(f"✅ Loading best performing model: epoch_31_r2_0.6281.pt")
        checkpoint = torch.load(epoch_31_path, map_location='cpu')
        model_path = epoch_31_path
    elif best_model_path.exists():
        print(f"✅ Loading best model checkpoint: best_model1.pt")
        checkpoint = torch.load(best_model_path, map_location='cpu')
        model_path = best_model_path
    else:
        raise FileNotFoundError("No Model 1 checkpoint found!")
    
    # Extract metadata
    metadata = {
        'model_name': 'Gnosis I',
        'model_type': 'ligand_activity_predictor',
        'architecture': 'FineTuned_ChemBERTa_Simple_Protein_Fusion',
        'r2_score': checkpoint.get('best_r2', checkpoint.get('test_r2', 0.0)),
        'epoch': checkpoint.get('epoch', 0),
        'num_targets': checkpoint.get('num_targets', 0),
        'target_list': checkpoint.get('target_list', []),
        'target_encoder': checkpoint.get('target_encoder', None)
    }
    
    print(f"📊 Model Metadata:")
    print(f"   • Model: {metadata['model_name']}")
    print(f"   • R² Score: {metadata['r2_score']:.4f}")
    print(f"   • Epoch: {metadata['epoch']}")
    print(f"   • Targets: {metadata['num_targets']}")
    print(f"   • Sample targets: {metadata['target_list'][:5] if metadata['target_list'] else 'None'}")
    
    # Save model weights and metadata as binary data to return
    model_weights = torch.load(model_path, map_location='cpu')
    
    return {
        'model_weights': model_weights,
        'metadata': metadata,
        'model_size_mb': model_path.stat().st_size / (1024*1024),
        'success': True
    }

if __name__ == "__main__":
    with app.run():
        result = download_best_model1.remote()
        print("Model 1 download completed:", result['success'])