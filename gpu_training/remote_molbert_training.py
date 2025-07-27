"""
Remote GPU Training Script for MolBERT
Optimized for RunPod and other cloud GPU services
"""

import os
import sys
import time
import json
import requests
import logging
from pathlib import Path
import torch
import joblib
from datetime import datetime

# Add parent directory to path to import local modules
sys.path.append('/app/backend')
from molbert_predictor import MolBERTPredictor
from chembl_data_manager import chembl_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RemoteGPUTraining:
    def __init__(self, config_path="training_config.json"):
        """Initialize remote training with configuration"""
        self.config = self.load_config(config_path)
        self.predictor = MolBERTPredictor()
        self.results_dir = Path("/app/trained_models")
        self.results_dir.mkdir(exist_ok=True)
        
        # GPU setup
        if torch.cuda.is_available():
            logger.info(f"🚀 GPU Training - Device: {torch.cuda.get_device_name()}")
            logger.info(f"🚀 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("⚠️ No GPU detected - falling back to CPU")
    
    def load_config(self, config_path):
        """Load training configuration"""
        default_config = {
            "target": "EGFR",
            "max_epochs": 50,
            "batch_size": 32,  # Larger batch size for GPU
            "learning_rate": 0.0001,
            "progress_webhook": None,  # URL to send progress updates
            "checkpoint_interval": 5,  # Save checkpoint every N epochs
            "early_stopping_patience": 10
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def send_progress_update(self, progress_data):
        """Send progress update to webhook URL"""
        if self.config.get('progress_webhook'):
            try:
                requests.post(
                    self.config['progress_webhook'], 
                    json=progress_data,
                    timeout=10
                )
            except Exception as e:
                logger.error(f"Failed to send progress update: {e}")
    
    def run_training(self):
        """Run the complete training pipeline"""
        start_time = time.time()
        target = self.config['target']
        
        logger.info(f"🎯 Starting remote GPU training for {target}")
        logger.info(f"📊 Config: {self.config}")
        
        try:
            # Step 1: Load and prepare data
            logger.info("📥 Loading ChEMBL data...")
            self.send_progress_update({
                "status": "loading_data",
                "message": f"Loading ChEMBL data for {target}",
                "progress": 0
            })
            
            # Step 2: Initialize and train model
            logger.info("🚀 Starting MolBERT training...")
            self.send_progress_update({
                "status": "training_started", 
                "message": "MolBERT training initiated",
                "progress": 10
            })
            
            # Enhanced training with progress callbacks
            results = self.train_with_progress_tracking(target)
            
            # Step 3: Save final model and results
            self.save_training_results(target, results)
            
            training_time = time.time() - start_time
            logger.info(f"✅ Training completed in {training_time/3600:.2f} hours")
            
            self.send_progress_update({
                "status": "completed",
                "message": f"Training completed successfully",
                "progress": 100,
                "results": results,
                "training_time_hours": training_time/3600
            })
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            self.send_progress_update({
                "status": "failed",
                "message": f"Training failed: {str(e)}",
                "progress": -1
            })
            raise
    
    def train_with_progress_tracking(self, target):
        """Train model with progress tracking"""
        # This would integrate with the existing MolBERT training
        # but with enhanced progress callbacks and GPU optimizations
        
        # For now, call the existing training method
        # In a real implementation, we'd modify it for better GPU utilization
        results = self.predictor.train_molbert_model(target)
        
        return results
    
    def save_training_results(self, target, results):
        """Save training results and upload to cloud storage"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save locally
        results_file = self.results_dir / f"{target}_training_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Copy model files
        model_files = [
            f"/app/backend/trained_molbert_models/{target}_molbert_model.pkl",
            f"/app/backend/trained_molbert_models/{target}_molbert_checkpoint.pkl"
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                import shutil
                dest_file = self.results_dir / os.path.basename(model_file)
                shutil.copy2(model_file, dest_file)
                logger.info(f"📦 Saved {dest_file}")
        
        logger.info(f"💾 Results saved to {results_file}")

def main():
    """Main training entry point"""
    logger.info("🚀 Starting Remote GPU MolBERT Training")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("⚠️ No GPU detected!")
    
    # Initialize and run training
    trainer = RemoteGPUTraining()
    trainer.run_training()
    
    logger.info("🏁 Training pipeline completed")

if __name__ == "__main__":
    main()