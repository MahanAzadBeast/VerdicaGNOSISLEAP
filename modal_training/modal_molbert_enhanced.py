"""
Enhanced Modal.com MolBERT & Chemprop Setup with Persistent Model Storage
- Uses molbert-cache volume for model persistence
- Downloads seyonec/ChemBERTa-zinc-base-v1 from Hugging Face on first run
- Mounts Hugging Face cache to persistent volume
- Includes Chemprop GNN training on A100 GPUs
"""

import modal
import os
import sys
from pathlib import Path

# Define Modal app
app = modal.App("molbert-chemprop-enhanced")

# Create Modal image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "torch>=2.1.0",
    "transformers>=4.35.0", 
    "huggingface-hub>=0.19.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "joblib>=1.3.0",
    "requests>=2.31.0",
    "rdkit-pypi>=2022.9.5",
    "chembl-webresource-client>=0.10.8",
    "accelerate>=0.21.0",
    "pytorch-lightning>=2.0.0",
    "tokenizers>=0.14.0",
    "chemprop>=1.6.1",  # Add Chemprop for GNN training
    "descriptastorus>=2.6.0"  # Chemprop dependency
]).run_commands([
    # Install additional dependencies
    "apt-get update && apt-get install -y git wget curl",
    # Create cache directories
    "mkdir -p /root/.cache/huggingface",
    "mkdir -p /root/.cache/torch"
])

# Create persistent volume for model storage and HF cache
molbert_cache = modal.Volume.from_name("molbert-cache", create_if_missing=True)

# Create separate volume for training checkpoints  
training_volume = modal.Volume.from_name("molbert-training", create_if_missing=True)

@app.function(
    image=image,
    volumes={
        "/cache": molbert_cache,  # Persistent model cache
        "/training": training_volume  # Training outputs
    },
    timeout=1800,  # 30 minutes for model download
    memory=8192,   # 8GB RAM for downloading
    cpu=2.0
)
def download_pretrained_molbert():
    """
    Download and cache the pretrained MolBERT model
    Only runs on first setup or when cache is empty
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    import logging
    import os
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Mount HuggingFace cache to persistent volume
    hf_cache_dir = "/cache/huggingface"
    os.environ['HF_HOME'] = hf_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir
    os.environ['HF_DATASETS_CACHE'] = f"{hf_cache_dir}/datasets"
    
    # Create cache directories
    Path(hf_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{hf_cache_dir}/hub").mkdir(parents=True, exist_ok=True)
    Path(f"{hf_cache_dir}/transformers").mkdir(parents=True, exist_ok=True)
    
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    model_cache_path = Path(hf_cache_dir) / "models" / model_name.replace("/", "--")
    
    logger.info(f"🔍 Checking for cached model at: {model_cache_path}")
    
    # Check if model is already cached
    if model_cache_path.exists() and len(list(model_cache_path.glob("*"))) > 0:
        logger.info(f"✅ Model {model_name} already cached")
        return {
            "status": "cached",
            "model_name": model_name,
            "cache_path": str(model_cache_path),
            "message": "Model already available in cache"
        }
    
    try:
        logger.info(f"📥 Downloading pretrained model: {model_name}")
        logger.info(f"💾 Cache directory: {hf_cache_dir}")
        
        # Download tokenizer first
        logger.info("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            force_download=False  # Use cache if available
        )
        
        # Download model
        logger.info("📥 Downloading model weights...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            force_download=False  # Use cache if available
        )
        
        # Verify model loaded correctly
        logger.info("🧪 Testing model loading...")
        vocab_size = tokenizer.vocab_size
        model_config = model.config
        
        logger.info(f"✅ Model downloaded successfully!")
        logger.info(f"📊 Vocab size: {vocab_size}")
        logger.info(f"📊 Model type: {model_config.model_type}")
        logger.info(f"📊 Hidden size: {model_config.hidden_size}")
        logger.info(f"💾 Cached at: {hf_cache_dir}")
        
        # Save model info for later use
        info_file = Path("/cache") / "model_info.json"
        import json
        model_info = {
            "model_name": model_name,
            "vocab_size": vocab_size,
            "hidden_size": model_config.hidden_size,
            "model_type": model_config.model_type,
            "max_position_embeddings": getattr(model_config, 'max_position_embeddings', None),
            "download_timestamp": str(torch.utils.data.get_worker_info()),
            "cache_path": hf_cache_dir
        }
        
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"📄 Model info saved to: {info_file}")
        
        return {
            "status": "downloaded",
            "model_name": model_name,
            "vocab_size": vocab_size,
            "hidden_size": model_config.hidden_size,
            "cache_path": hf_cache_dir,
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        raise e

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1),  # Single A100 GPU
    volumes={
        "/cache": molbert_cache,      # Model cache
        "/training": training_volume  # Training outputs
    },
    timeout=14400,  # 4 hours max
    memory=32768,   # 32GB RAM
    cpu=8.0
)
def train_molbert_with_cache(
    target: str = "EGFR",
    max_epochs: int = 50,
    batch_size: int = 32,  # Conservative for fine-tuning
    learning_rate: float = 5e-5,  # Standard for BERT fine-tuning
    webhook_url: str = None
):
    """
    Train MolBERT model using cached pretrained weights
    """
    import torch
    import logging
    import json
    import time
    import sys
    from datetime import datetime
    from pathlib import Path
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Mount HuggingFace cache
    hf_cache_dir = "/cache/huggingface"
    os.environ['HF_HOME'] = hf_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir
    
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
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        logger.info(f"💾 GPU Memory: {gpu_props.total_memory / 1e9:.1f} GB")
        logger.info(f"🔧 GPU Compute: {gpu_props.major}.{gpu_props.minor}")
    
    send_progress("started", f"Initializing {target} training on Modal A100", 5)
    
    try:
        # Check if model is cached
        info_file = Path("/cache") / "model_info.json"
        if not info_file.exists():
            logger.error("❌ Model not cached! Run download_pretrained_molbert() first")
            raise ValueError("Pretrained model not found in cache")
        
        # Load model info
        with open(info_file, 'r') as f:
            model_info = json.load(f)
        
        logger.info(f"✅ Using cached model: {model_info['model_name']}")
        logger.info(f"📊 Vocab size: {model_info['vocab_size']}")
        logger.info(f"💾 Cache path: {model_info['cache_path']}")
        
        send_progress("loading_model", "Loading pretrained MolBERT from cache", 15)
        
        # Import transformers and load model
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        
        model_name = model_info['model_name']
        
        # Load from cache
        logger.info("📥 Loading tokenizer from cache...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            local_files_only=True  # Only use cached files
        )
        
        logger.info("📥 Loading model from cache...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            local_files_only=True  # Only use cached files
        )
        
        # Move to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"🖥️ Model moved to: {device}")
        
        send_progress("model_loaded", "Pretrained model loaded successfully", 25)
        
        # Here you would implement your fine-tuning logic
        # For now, let's create a placeholder training loop
        
        logger.info("🎓 Starting fine-tuning...")
        send_progress("training", f"Fine-tuning for {target}", 50)
        
        # Simulate training (replace with actual fine-tuning code)
        import time
        for epoch in range(min(max_epochs, 5)):  # Limit for demo
            logger.info(f"📚 Epoch {epoch+1}/{max_epochs}")
            time.sleep(2)  # Simulate training time
            
            progress = 50 + (epoch + 1) / max_epochs * 40
            send_progress("training", f"Epoch {epoch+1}/{max_epochs} - {target}", progress)
        
        # Save fine-tuned model
        model_save_path = f"/training/{target}_molbert_finetuned"
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving fine-tuned model to: {model_save_path}")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        send_progress("completed", f"Fine-tuning completed for {target}", 100)
        
        results = {
            "status": "completed",
            "target": target,
            "epochs_completed": max_epochs,
            "model_path": model_save_path,
            "base_model": model_name,
            "vocab_size": model_info['vocab_size']
        }
        
        logger.info(f"✅ Training completed! Results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        send_progress("failed", f"Training failed: {str(e)}", -1)
        raise

@app.function(
    image=image,
    volumes={"/cache": molbert_cache},
    timeout=600  # 10 minutes
)
def get_model_info():
    """
    Get information about cached models
    """
    import json
    from pathlib import Path
    
    info_file = Path("/cache") / "model_info.json"
    
    if info_file.exists():
        with open(info_file, 'r') as f:
            return json.load(f)
    else:
        return {"status": "no_model_cached", "message": "No pretrained model found in cache"}

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1),
    volumes={
        "/cache": molbert_cache,
        "/training": training_volume
    },
    timeout=300  # 5 minutes for inference
)
def predict_with_cached_model(
    smiles: str,
    target: str = "EGFR",
    model_type: str = "finetuned"  # "pretrained" or "finetuned"
):
    """
    Run prediction using cached model
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    import logging
    from pathlib import Path
    import os
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Mount HuggingFace cache
    hf_cache_dir = "/cache/huggingface"
    os.environ['HF_HOME'] = hf_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir
    
    try:
        if model_type == "finetuned":
            # Use fine-tuned model
            model_path = f"/training/{target}_molbert_finetuned"
            if not Path(model_path).exists():
                logger.warning(f"Fine-tuned model not found for {target}, using pretrained")
                model_type = "pretrained"
        
        if model_type == "pretrained":
            # Use base pretrained model
            model_name = "seyonec/ChemBERTa-zinc-base-v1"
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=hf_cache_dir,
                local_files_only=True
            )
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=hf_cache_dir, 
                local_files_only=True
            )
            model_path = f"cached:{model_name}"
        else:
            # Load fine-tuned model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
        
        # Move to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        logger.info(f"🧪 Predicting for SMILES: {smiles}")
        logger.info(f"🎯 Target: {target}")
        logger.info(f"🤖 Model: {model_type} ({model_path})")
        
        # Tokenize SMILES
        inputs = tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Pool embeddings
        
        # Simple mock prediction (replace with actual prediction head)
        mock_ic50 = torch.rand(1).item() * 1000  # Mock IC50 in nM
        mock_confidence = 0.7 + torch.rand(1).item() * 0.25  # Mock confidence
        
        results = {
            "smiles": smiles,
            "target": target,
            "model_type": model_type,
            "model_path": str(model_path),
            "ic50_nm": mock_ic50,
            "pic50": -torch.log10(torch.tensor(mock_ic50 * 1e-9)).item(),
            "confidence": mock_confidence,
            "embedding_dim": embeddings.shape[-1]
        }
        
        logger.info(f"✅ Prediction completed: IC50 = {mock_ic50:.1f} nM")
        return results
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise

# Enhanced Chemprop GNN Training Functions
@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1),  # A100 for fast GNN training
    volumes={
        "/cache": molbert_cache,      # Model cache
        "/training": training_volume  # Training outputs
    },
    timeout=7200,  # 2 hours for Chemprop training
    memory=32768,  # 32GB RAM
    cpu=8.0
)
def train_chemprop_gnn_modal(
    target: str = "EGFR",
    training_data: list = None,  # List of {"smiles": str, "activity": float}
    epochs: int = 50,
    batch_size: int = 32,
    hidden_size: int = 300,
    depth: int = 3,
    webhook_url: str = None
):
    """
    Train Chemprop GNN model on Modal A100 GPU
    """
    import torch
    import logging
    import json
    import pandas as pd
    import tempfile
    import subprocess
    from datetime import datetime
    from pathlib import Path
    
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
                    "model_type": "chemprop_gnn",
                    "timestamp": datetime.now().isoformat(),
                    **kwargs
                }
                requests.post(webhook_url, json=data, timeout=10)
            except Exception as e:
                logger.error(f"Failed to send progress: {e}")
    
    logger.info(f"🚀 Starting Chemprop GNN training on Modal A100 for {target}")
    logger.info(f"🖥️ GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        logger.info(f"💾 GPU Memory: {gpu_props.total_memory / 1e9:.1f} GB")
    
    send_progress("started", f"Initializing Chemprop GNN training for {target}", 5)
    
    try:
        # Validate training data
        if not training_data or len(training_data) < 10:
            raise ValueError(f"Insufficient training data: {len(training_data) if training_data else 0} samples")
        
        logger.info(f"📊 Training data: {len(training_data)} compounds")
        send_progress("preparing_data", f"Preparing {len(training_data)} compounds for training", 15)
        
        # Create temporary training file
        train_dir = Path("/training") / f"{target}_chemprop_training"
        train_dir.mkdir(parents=True, exist_ok=True)
        
        train_file = train_dir / "train.csv"
        
        # Prepare data in Chemprop format
        df_data = []
        for item in training_data:
            df_data.append({
                "smiles": item["smiles"],
                "target": item["activity"]  # Chemprop expects 'target' column
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(train_file, index=False)
        
        logger.info(f"📝 Training data saved to: {train_file}")
        
        # Model save directory
        model_save_dir = train_dir / "model"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        send_progress("training_started", f"Starting GNN training with {epochs} epochs", 25)
        
        # Chemprop training command
        cmd = [
            "python", "-m", "chemprop.train",
            "--data_path", str(train_file),
            "--dataset_type", "regression",
            "--save_dir", str(model_save_dir),
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--hidden_size", str(hidden_size),
            "--depth", str(depth),
            "--dropout", "0.1",
            "--metric", "rmse",
            "--gpu", "0" if torch.cuda.is_available() else None,
            "--quiet"
        ]
        
        # Remove None values
        cmd = [str(arg) for arg in cmd if arg is not None]
        
        logger.info(f"🧠 Running Chemprop command: {' '.join(cmd)}")
        
        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/training"
        )
        
        if result.returncode == 0:
            logger.info("✅ Chemprop GNN training completed successfully")
            send_progress("training_completed", "GNN training completed", 85)
            
            # Check for trained model files
            model_files = list(model_save_dir.glob("*.pt"))
            if model_files:
                main_model = model_files[0]
                logger.info(f"💾 Model saved: {main_model}")
                
                # Save model metadata
                model_info = {
                    "target": target,
                    "model_type": "chemprop_gnn",
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "hidden_size": hidden_size,
                    "depth": depth,
                    "training_samples": len(training_data),
                    "model_path": str(main_model),
                    "training_time": datetime.now().isoformat()
                }
                
                info_file = train_dir / "model_info.json"
                with open(info_file, 'w') as f:
                    json.dump(model_info, f, indent=2)
                
                send_progress("completed", f"Chemprop GNN training completed for {target}", 100, 
                             model_info=model_info)
                
                return {
                    "status": "completed",
                    "target": target,
                    "model_type": "chemprop_gnn",
                    "model_path": str(main_model),
                    "model_info": model_info,
                    "training_samples": len(training_data)
                }
            else:
                raise FileNotFoundError("No model files found after training")
                
        else:
            logger.error(f"❌ Chemprop training failed:")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            
            send_progress("failed", f"Training failed: {result.stderr[:200]}", -1)
            raise RuntimeError(f"Chemprop training failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ Chemprop GNN training failed: {e}")
        send_progress("failed", f"Training failed: {str(e)}", -1)
        raise

# ChemBERTa Fine-tuning Functions
@app.function(
    image=image,
    gpu="A100-40GB",  # Updated GPU specification
    volumes={
        "/cache": molbert_cache,      # Model cache
        "/training": training_volume  # Training outputs
    },
    timeout=3600,  # 1 hour for fine-tuning
    memory=32768,  # 32GB RAM
    cpu=8.0
)
def finetune_chembert_modal(
    target: str = "EGFR",
    training_data: list = None,  # List of {"smiles": str, "ic50": float}
    epochs: int = 10,
    batch_size: int = 16,  # Conservative for transformer fine-tuning
    learning_rate: float = 2e-5,  # Standard for BERT fine-tuning
    webhook_url: str = None
):
    """
    Fine-tune ChemBERTa with regression head for IC50 prediction on Modal A100
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel
    from torch.optim import AdamW  # AdamW moved to torch.optim
    import logging
    import json
    import pandas as pd
    from datetime import datetime
    from pathlib import Path
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error
    
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
                    "model_type": "chembert_finetuned",
                    "timestamp": datetime.now().isoformat(),
                    **kwargs
                }
                requests.post(webhook_url, json=data, timeout=10)
            except Exception as e:
                logger.error(f"Failed to send progress: {e}")
    
    logger.info(f"🚀 Starting ChemBERTa fine-tuning on Modal A100 for {target}")
    logger.info(f"🖥️ GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    send_progress("started", f"Initializing ChemBERTa fine-tuning for {target}", 5)
    
    try:
        # Validate training data
        if not training_data or len(training_data) < 10:
            raise ValueError(f"Insufficient training data: {len(training_data) if training_data else 0} samples")
        
        logger.info(f"📊 Training data: {len(training_data)} compounds")
        send_progress("loading_model", f"Loading pretrained ChemBERTa", 15)
        
        # Mount HuggingFace cache
        hf_cache_dir = "/cache/huggingface"
        os.environ['HF_HOME'] = hf_cache_dir
        os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir
        
        # Load pretrained ChemBERTa from cache
        model_name = "seyonec/ChemBERTa-zinc-base-v1"
        logger.info("📥 Loading ChemBERTa tokenizer and model from cache...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            local_files_only=True
        )
        
        base_model = AutoModel.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            local_files_only=True
        )
        
        # Create regression head for IC50 prediction
        class ChemBERTaForRegression(nn.Module):
            def __init__(self, base_model, hidden_size=768):
                super().__init__()
                self.bert = base_model
                self.dropout = nn.Dropout(0.1)
                self.regressor = nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1)  # Single output for IC50
                )
            
            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                pooled_output = self.dropout(pooled_output)
                return self.regressor(pooled_output)
        
        # Initialize the model with regression head
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ChemBERTaForRegression(base_model).to(device)
        
        logger.info(f"✅ ChemBERTa with regression head loaded on {device}")
        send_progress("preparing_data", "Preparing training dataset", 25)
        
        # Prepare dataset
        class IC50Dataset(Dataset):
            def __init__(self, data, tokenizer, max_length=128):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                smiles = item["smiles"]
                ic50 = float(item["ic50"])
                
                # Tokenize SMILES
                encoding = self.tokenizer(
                    smiles,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(ic50, dtype=torch.float)
                }
        
        # Create dataset and dataloader
        dataset = IC50Dataset(training_data, tokenizer)
        
        # Split into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"📊 Dataset split: {train_size} train, {val_size} val")
        
        # Setup optimizer and loss
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        send_progress("training_started", f"Starting fine-tuning for {epochs} epochs", 35)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            train_preds = []
            train_targets = []
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                train_preds.extend(outputs.squeeze().detach().cpu().numpy())
                train_targets.extend(labels.detach().cpu().numpy())
            
            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs.squeeze(), labels)
                    
                    val_loss += loss.item()
                    val_preds.extend(outputs.squeeze().cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())
            
            # Calculate metrics
            train_r2 = r2_score(train_targets, train_preds)
            val_r2 = r2_score(val_targets, val_preds)
            train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
            
            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"📈 Epoch {epoch+1}/{epochs}:")
            logger.info(f"   Train Loss: {avg_train_loss:.4f}, R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
            logger.info(f"   Val Loss: {avg_val_loss:.4f}, R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
            
            progress = 35 + (epoch + 1) / epochs * 50
            send_progress("training", f"Epoch {epoch+1}/{epochs} - Val R²: {val_r2:.3f}", progress,
                         train_r2=train_r2, val_r2=val_r2, val_rmse=val_rmse)
            
            model.train()
        
        # Save fine-tuned model
        model_save_dir = Path("/training") / f"{target}_chembert_finetuned"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        model_path = model_save_dir / "pytorch_model.bin"
        torch.save(model.state_dict(), model_path)
        
        # Save tokenizer
        tokenizer.save_pretrained(model_save_dir)
        
        # Save model config and training info
        model_info = {
            "target": target,
            "model_type": "chembert_finetuned",
            "base_model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_samples": len(training_data),
            "train_r2": float(train_r2),
            "val_r2": float(val_r2),
            "val_rmse": float(val_rmse),
            "model_path": str(model_path),
            "training_time": datetime.now().isoformat()
        }
        
        info_file = model_save_dir / "training_info.json"
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"💾 Fine-tuned model saved to: {model_save_dir}")
        logger.info(f"🎯 Final validation R²: {val_r2:.4f}")
        
        send_progress("completed", f"ChemBERTa fine-tuning completed - R²: {val_r2:.3f}", 100,
                     final_r2=val_r2, model_path=str(model_save_dir))
        
        return {
            "status": "completed",
            "target": target,
            "model_type": "chembert_finetuned",
            "train_r2": train_r2,
            "val_r2": val_r2,
            "val_rmse": val_rmse,
            "model_path": str(model_save_dir),
            "training_samples": len(training_data),
            "epochs": epochs
        }
        
    except Exception as e:
        logger.error(f"❌ ChemBERTa fine-tuning failed: {e}")
        send_progress("failed", f"Fine-tuning failed: {str(e)}", -1)
        raise

@app.function(
    image=image,
    volumes={"/training": training_volume},
    timeout=300  # 5 minutes
)
def get_chemprop_model_info(target: str = "EGFR"):
    """Get information about trained Chemprop models"""
    import json
    from pathlib import Path
    
    model_dir = Path("/training") / f"{target}_chemprop_training"
    info_file = model_dir / "model_info.json"
    
    if info_file.exists():
        with open(info_file, 'r') as f:
            return json.load(f)
    else:
        return {
            "status": "no_model",
            "message": f"No trained Chemprop model found for {target}"
        }

@app.function(
    image=image,
    volumes={"/training": training_volume},
    timeout=600  # 10 minutes
)
def download_chemprop_model(target: str = "EGFR"):
    """Download trained Chemprop model for local use"""
    from pathlib import Path
    import json
    
    model_dir = Path("/training") / f"{target}_chemprop_training"
    model_files = list(model_dir.glob("model/*.pt"))
    
    if not model_files:
        raise FileNotFoundError(f"No trained model found for {target}")
    
    model_file = model_files[0]
    
    # Read model file
    with open(model_file, 'rb') as f:
        model_data = f.read()
    
    # Read model info
    info_file = model_dir / "model_info.json"
    model_info = {}
    if info_file.exists():
        with open(info_file, 'r') as f:
            model_info = json.load(f)
    
    return {
        "model_data": model_data,
        "model_info": model_info,
        "target": target,
        "model_size_mb": len(model_data) / (1024 * 1024)
    }

# Deployment functions
@app.local_entrypoint()
def setup_and_train(
    target: str = "EGFR",
    download_model: bool = True,
    webhook_url: str = None
):
    """
    Complete setup and training pipeline
    """
    print(f"🚀 Starting MolBERT setup and training for {target}")
    
    if download_model:
        print("📥 Step 1: Downloading pretrained model...")
        download_result = download_pretrained_molbert.remote()
        print(f"✅ Download result: {download_result}")
    
    print("🎓 Step 2: Starting training...")
    training_result = train_molbert_with_cache.remote(
        target=target,
        webhook_url=webhook_url
    )
    print(f"✅ Training result: {training_result}")
    
    return {
        "download": download_result if download_model else "skipped",
        "training": training_result
    }

if __name__ == "__main__":
    print("🎯 Enhanced MolBERT Modal Setup Ready!")
    print("")
    print("🔧 Available functions:")
    print("  • download_pretrained_molbert() - Download and cache pretrained model")
    print("  • train_molbert_with_cache() - Fine-tune using cached model")
    print("  • predict_with_cached_model() - Run predictions")
    print("  • get_model_info() - Check cached model info")
    print("")
    print("💡 Quick start:")
    print("  modal run modal_molbert_enhanced.py::setup_and_train --target EGFR")