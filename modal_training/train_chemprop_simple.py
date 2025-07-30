"""
Simplified Chemprop Multi-Task Training using chemprop package directly
"""

import modal
import os
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").run_commands([
    "apt-get update && apt-get install -y git"
]).pip_install([
    "chemprop",
    "pandas>=2.0.0", 
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "wandb>=0.15.0",
    "matplotlib>=3.7.0",
    "pyarrow>=12.0.0"
])

app = modal.App("chemprop-simple")
datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret")

@app.function(
    image=image,
    volumes={
        "/vol/datasets": datasets_volume,
        "/vol/models": models_volume
    },
    secrets=[wandb_secret],
    gpu="A100",
    memory=16384,
    timeout=10800  # 3 hours
)
def train_chemprop_simple():
    """
    Simplified Chemprop multi-task training using the chemprop package directly
    """
    
    import logging
    import tempfile
    from sklearn.model_selection import train_test_split
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Import chemprop after Modal setup
    try:
        from chemprop import run_training
        from chemprop.args import TrainArgs
        import wandb
    except ImportError as e:
        logger.error(f"Failed to import chemprop: {e}")
        return {"status": "error", "error": str(e)}
    
    # Initialize W&B
    wandb.init(
        project="veridica-ai-training",
        group="chemprop",
        name="chemprop-simple-multitask-14-oncoproteins",
        config={
            "model_type": "chemprop_gnn_multitask",
            "dataset": "oncoprotein_multitask_dataset",
            "targets": 14,
            "epochs": 50,
            "batch_size": 50
        },
        tags=["chemprop", "multi-task", "gnn", "oncoproteins", "simple"]
    )
    
    logger.info("🚀 Starting Simplified Chemprop Multi-Task Training")
    logger.info("🎯 Training GNN for all 14 oncoproteins simultaneously")
    
    # Load dataset
    dataset_path = Path("/vol/datasets/oncoprotein_multitask_dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    smiles_col = 'canonical_smiles'
    target_cols = [col for col in df.columns if col != smiles_col]
    
    logger.info(f"📊 Dataset: {df.shape}")
    logger.info(f"🎯 Targets: {target_cols}")
    
    # Filter compounds with some data
    df_filtered = df.dropna(subset=target_cols, how='all')
    logger.info(f"📈 Filtered dataset: {len(df_filtered)} compounds")
    
    # Split data
    train_df, test_df = train_test_split(df_filtered, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42)  # 0.1 of original
    
    logger.info(f"📊 Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Log data splits
    wandb.log({
        "train_size": len(train_df),
        "val_size": len(val_df), 
        "test_size": len(test_df),
        "total_targets": len(target_cols)
    })
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save data splits
        train_path = temp_path / "train.csv"
        val_path = temp_path / "val.csv"
        test_path = temp_path / "test.csv"
        
        # Prepare data in chemprop format (rename smiles column)
        train_data = train_df.rename(columns={smiles_col: 'smiles'})
        val_data = val_df.rename(columns={smiles_col: 'smiles'})
        test_data = test_df.rename(columns={smiles_col: 'smiles'})
        
        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        # Create output directory
        output_dir = temp_path / "chemprop_output"
        output_dir.mkdir()
        
        try:
            # Configure training arguments
            args = TrainArgs(
                data_path=str(train_path),
                separate_val_path=str(val_path),
                separate_test_path=str(test_path),
                dataset_type='regression',
                save_dir=str(output_dir),
                epochs=50,
                batch_size=50,
                hidden_size=300,
                depth=3,
                dropout=0.0,
                ffn_num_layers=2,
                init_lr=1e-4,
                max_lr=1e-3,
                final_lr=1e-4
            )
            
            logger.info("🎓 Starting Chemprop training...")
            
            # Run training
            mean_score, std_score = run_training(args)
            
            logger.info(f"✅ Training completed!")
            logger.info(f"📊 Mean score: {mean_score:.4f} ± {std_score:.4f}")
            
            # Log final results
            wandb.log({
                "final_mean_score": mean_score,
                "final_std_score": std_score,
                "training_completed": True
            })
            
            # Copy results to persistent storage
            persistent_output = Path("/vol/models/chemprop_simple_multitask")
            persistent_output.mkdir(parents=True, exist_ok=True)
            
            # Copy model files
            import shutil
            if output_dir.exists():
                shutil.copytree(str(output_dir), str(persistent_output), dirs_exist_ok=True)
                logger.info(f"✅ Model saved to: {persistent_output}")
            
            wandb.finish()
            
            return {
                "status": "success",
                "mean_score": float(mean_score),
                "std_score": float(std_score),
                "model_path": str(persistent_output),
                "targets": target_cols
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            wandb.log({"error": str(e), "training_completed": False})
            wandb.finish()
            return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    pass