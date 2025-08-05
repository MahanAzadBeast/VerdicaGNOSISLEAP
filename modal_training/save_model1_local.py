"""
Download and save Model 1 weights locally for backend use
"""

import modal
import torch
from pathlib import Path

image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "torch==2.0.1", 
    "scikit-learn"
])

app = modal.App("save-model1-local")
models_volume = modal.Volume.from_name("cell-line-models", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/models": models_volume}
)
def save_model1_locally():
    """Save the best Model 1 checkpoint locally"""
    
    # Load the best checkpoint
    epoch_31_path = Path("/models/model1_checkpoints/epoch_31_r2_0.6281.pt")
    best_model_path = Path("/models/model1_checkpoints/best_model1.pt")
    
    if epoch_31_path.exists():
        print(f"📥 Loading best performing model: epoch_31_r2_0.6281.pt")
        checkpoint = torch.load(epoch_31_path, map_location='cpu')
        source_path = epoch_31_path
    elif best_model_path.exists():
        print(f"📥 Loading backup model: best_model1.pt")
        checkpoint = torch.load(best_model_path, map_location='cpu')
        source_path = best_model_path
    else:
        raise FileNotFoundError("No Model 1 checkpoint found!")
    
    print(f"✅ Model loaded successfully")
    print(f"   R² Score: {checkpoint.get('best_r2', checkpoint.get('test_r2', 'N/A'))}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Targets: {checkpoint.get('num_targets', 'N/A')}")
    
    # Return the checkpoint data
    return {
        'checkpoint': checkpoint,
        'metadata': {
            'r2_score': checkpoint.get('best_r2', checkpoint.get('test_r2', 0.0)),
            'epoch': checkpoint.get('epoch', 0),
            'num_targets': checkpoint.get('num_targets', 0),
            'target_list': checkpoint.get('target_list', [])[:10],  # First 10 for display
            'model_size_mb': source_path.stat().st_size / (1024*1024)
        }
    }

if __name__ == "__main__":
    with app.run():
        result = save_model1_locally.remote()
        
        # Save locally
        local_model_path = Path("/app/backend/models")
        local_model_path.mkdir(exist_ok=True)
        
        checkpoint_path = local_model_path / "gnosis_model1_best.pt"
        torch.save(result['checkpoint'], checkpoint_path)
        
        print(f"\n✅ Model saved locally to: {checkpoint_path}")
        print(f"📊 Model info: {result['metadata']}")