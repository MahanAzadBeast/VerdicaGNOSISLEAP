"""
Modal.com MolBERT Training Migration
Deploy existing MolBERT training to Modal GPUs (A100/V100)
"""

import modal
import os
import sys
from pathlib import Path

# Define Modal app
app = modal.App("molbert-training")

# Create Modal image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "torch>=2.1.0",
    "transformers>=4.21.0", 
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "joblib>=1.3.0",
    "requests>=2.31.0",
    "rdkit-pypi>=2022.9.5",
    "chembl-webresource-client>=0.10.8",
    "accelerate>=0.21.0"
])

# Create persistent volume for model storage
volume = modal.Volume.from_name("molbert-models", create_if_missing=True)

# Mount the training code
code_mount = modal.Mount.from_local_dir(
    "/app/modal_training", 
    remote_path="/app/modal_training"
)

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1),  # Single A100 GPU
    volumes={"/models": volume},
    mounts={"/app": code_mount},
    timeout=14400,  # 4 hours max
    memory=32768,   # 32GB RAM
    cpu=8.0
)
def train_molbert_gpu(
    target: str = "EGFR",
    max_epochs: int = 50,
    batch_size: int = 64,  # Larger batch for A100
    learning_rate: float = 0.0001,
    webhook_url: str = None
):
    """
    Train MolBERT model on Modal GPU
    """
    import torch
    import logging
    import json
    import time
    import sys
    from datetime import datetime
    
    # Add the mounted code to Python path
    sys.path.append('/app/modal_training')
    sys.path.append('/app')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def send_progress(status, message, progress, **kwargs):
        """Send progress updates to webhook"""
        if webhook_url:
            try:
                import requests
                data = {
                    "status": status,
                    "message": message, 
                    "progress": progress,
                    "target": target,
                    "timestamp": datetime.now().isoformat(),
                    **kwargs
                }
                requests.post(webhook_url, json=data, timeout=10)
            except Exception as e:
                logger.error(f"Failed to send progress: {e}")
    
    logger.info(f"🚀 Starting MolBERT training on Modal GPU for {target}")
    logger.info(f"🖥️ GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "No GPU")
    
    send_progress("started", f"Initializing {target} training on Modal GPU", 5)
    
    try:
        # Import the existing MolBERT predictor directly 
        from molbert_predictor import MolBERTPredictor
        
        # Initialize predictor
        predictor = MolBERTPredictor()
        
        # Run training using existing method
        logger.info("🎯 Starting MolBERT incremental training...")
        send_progress("training_started", f"Training {target} with MolBERT", 10)
        
        # Use the existing training method
        results = predictor.train_molbert_model(target)
        
        # Save to persistent volume
        model_path = f"/models/{target}_molbert_final.pkl"
        
        # Copy trained model to persistent storage
        import shutil
        local_model_path = f"/app/backend/trained_molbert_models/{target}_molbert_model.pkl"
        if os.path.exists(local_model_path):
            shutil.copy2(local_model_path, model_path)
            logger.info(f"💾 Model saved to persistent volume: {model_path}")
        
        send_progress("completed", f"Training completed for {target}", 100, 
                     results=results, model_path=model_path)
        
        logger.info(f"✅ Training completed! Model saved to {model_path}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        send_progress("failed", f"Training failed: {str(e)}", -1)
        raise

@app.function(
    image=image,
    volumes={"/models": volume}
)
def download_trained_model(target: str = "EGFR"):
    """
    Download trained model from Modal volume
    """
    import joblib
    model_path = f"/models/{target}_molbert_final.pkl"
    
    if os.path.exists(model_path):
        # Read model data
        with open(model_path, 'rb') as f:
            model_data = f.read()
        return model_data
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1),
    volumes={"/models": volume},
    mounts={"/app": code_mount},
    timeout=21600,  # 6 hours for multi-target
    memory=32768,
    cpu=8.0
)
def train_all_targets(
    targets: list = ["EGFR", "BRAF", "CDK2", "PARP1", "BCL2", "VEGFR2"],
    max_epochs: int = 50,
    batch_size: int = 64,
    webhook_url: str = None
):
    """
    Train MolBERT for all targets in sequence
    """
    import logging
    logger = logging.getLogger(__name__)
    
    results = {}
    total_targets = len(targets)
    
    for i, target in enumerate(targets):
        logger.info(f"🎯 Training target {i+1}/{total_targets}: {target}")
        
        try:
            # Calculate progress (0-100 across all targets)
            base_progress = (i / total_targets) * 100
            
            def target_progress_callback(status, message, progress, **kwargs):
                # Adjust progress to account for multiple targets
                adjusted_progress = base_progress + (progress / total_targets)
                
                if webhook_url:
                    import requests
                    from datetime import datetime
                    data = {
                        "status": status,
                        "message": f"[{i+1}/{total_targets}] {target}: {message}",
                        "progress": adjusted_progress,
                        "current_target": target,
                        "completed_targets": i,
                        "total_targets": total_targets,
                        "timestamp": datetime.now().isoformat(),
                        **kwargs
                    }
                    try:
                        requests.post(webhook_url, json=data, timeout=10)
                    except:
                        pass
            
            # Train individual target using the local function
            result = train_molbert_gpu.local(
                target=target,
                max_epochs=max_epochs,
                batch_size=batch_size,
                webhook_url=webhook_url
            )
            
            results[target] = result
            logger.info(f"✅ Completed {target}")
            
        except Exception as e:
            logger.error(f"❌ Failed training {target}: {e}")
            results[target] = {"error": str(e)}
    
    return results

# Local deployment functions
@app.local_entrypoint()
def main(
    target: str = "EGFR",
    all_targets: bool = False,
    webhook_url: str = None
):
    """
    Main entry point for Modal deployment
    """
    if all_targets:
        print(f"🚀 Starting multi-target training on Modal...")
        results = train_all_targets.remote(webhook_url=webhook_url)
    else:
        print(f"🚀 Starting {target} training on Modal...")
        results = train_molbert_gpu.remote(
            target=target,
            webhook_url=webhook_url
        )
    
    print(f"✅ Training completed!")
    print(f"📊 Results: {results}")
    return results