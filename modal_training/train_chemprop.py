"""
Chemprop Multi-Task Training Pipeline on Modal with W&B Integration
Trains graph neural networks for molecular property prediction using official chemprop package
"""

import modal
import os
import pandas as pd
import numpy as np
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Modal setup with Chemprop and dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "chemprop>=1.7.0",
    "torch>=2.0.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0", 
    "numpy>=1.24.0",
    "rdkit>=2023.3.1",
    "wandb>=0.15.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "pyarrow>=12.0.0",
    "descriptastorus>=2.6.0",
    "hyperopt>=0.2.7"
])

app = modal.App("chemprop-training")

# Persistent volumes
datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)

# W&B secret (already configured)
wandb_secret = modal.Secret.from_name("wandb-secret")

class ChempropWandbLogger:
    """Custom W&B logger for Chemprop training"""
    
    def __init__(self, target_names: List[str]):
        self.target_names = target_names
        self.epoch_metrics = {}
    
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for a specific epoch"""
        epoch_data = {"epoch": epoch}
        epoch_data.update(metrics)
        wandb.log(epoch_data)
        
        # Store for final summary
        self.epoch_metrics[epoch] = metrics
    
    def log_final_results(self, test_metrics: Dict[str, float], model_path: str):
        """Log final test results and model artifacts"""
        
        # Log final test metrics
        final_metrics = {}
        for key, value in test_metrics.items():
            final_metrics[f"final_test_{key}"] = value
        
        wandb.log(final_metrics)
        
        # Create summary table
        if self.target_names:
            summary_data = []
            for target in self.target_names:
                r2_key = f"{target}_r2"
                mse_key = f"{target}_mse"
                mae_key = f"{target}_mae"
                
                summary_data.append([
                    target,
                    test_metrics.get(r2_key, 0.0),
                    test_metrics.get(mse_key, 0.0),
                    test_metrics.get(mae_key, 0.0)
                ])
            
            summary_table = wandb.Table(
                columns=["Target", "R²", "MSE", "MAE"],
                data=summary_data
            )
            wandb.log({"target_performance_summary": summary_table})

def prepare_chemprop_data(df: pd.DataFrame, smiles_col: str, target_cols: List[str], 
                         output_dir: Path, split_name: str) -> str:
    """
    Prepare data in Chemprop format (CSV with SMILES and targets)
    """
    
    # Create Chemprop-compatible DataFrame
    chemprop_df = df[[smiles_col] + target_cols].copy()
    chemprop_df = chemprop_df.rename(columns={smiles_col: 'smiles'})
    
    # Save to CSV
    output_path = output_dir / f"{split_name}.csv"
    chemprop_df.to_csv(output_path, index=False)
    
    return str(output_path)

def run_chemprop_training(train_path: str, val_path: str, test_path: str,
                         output_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Chemprop multi-task training using command line interface
    """
    
    # Prepare Chemprop command for multi-task learning
    cmd = [
        'chemprop_train',
        '--data_path', train_path,
        '--separate_val_path', val_path,
        '--separate_test_path', test_path,
        '--dataset_type', 'regression',  # Multi-task regression
        '--save_dir', output_dir,
        '--epochs', str(config['num_epochs']),
        '--batch_size', str(config['batch_size']),
        '--init_lr', str(config['learning_rate']),
        '--max_lr', str(config['max_learning_rate']),
        '--final_lr', str(config['final_learning_rate']),
        '--num_workers', str(config['num_workers']),
        '--hidden_size', str(config['hidden_size']),
        '--depth', str(config['depth']),
        '--dropout', str(config['dropout']),
        '--ffn_num_layers', str(config['ffn_num_layers']),
        '--ffn_hidden_size', str(config['ffn_hidden_size']),
        '--aggregation', config['aggregation'],
        '--aggregation_norm', str(config['aggregation_norm']),
        '--save_preds',  # Save predictions for analysis
        '--quiet',  # Reduce verbose output
    ]
    
    # Add multi-task specific parameters
    if config.get('multitask_scaling', True):
        cmd.append('--multitask_loss_scaling')  # Scale losses across tasks
    
    # Add optional parameters
    if config.get('early_stopping', False):
        cmd.extend(['--patience', str(config.get('patience', 10))])
    
    if config.get('use_features', False):
        cmd.append('--features_generator rdkit_2d_normalized')
    
    if config.get('atom_messages', False):
        cmd.append('--atom_messages')
    
    # Add ensemble training for better performance
    if config.get('ensemble_size', 1) > 1:
        cmd.extend(['--ensemble_size', str(config['ensemble_size'])])
    
    # Run training
    logging.info(f"🧪 Running Chemprop Multi-Task Training: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Chemprop multi-task training failed: {result.stderr}")
    
    return {"stdout": result.stdout, "stderr": result.stderr}

def parse_chemprop_results(output_dir: str) -> Dict[str, float]:
    """
    Parse Chemprop results from output files
    """
    
    results = {}
    output_path = Path(output_dir)
    
    # Look for test results
    test_scores_file = output_path / "test_scores.csv"
    if test_scores_file.exists():
        test_df = pd.read_csv(test_scores_file)
        
        # Extract metrics for each target
        for col in test_df.columns:
            if col != 'smiles':
                # Calculate metrics if we have predictions and true values
                if f"{col}_true" in test_df.columns and f"{col}_pred" in test_df.columns:
                    true_vals = test_df[f"{col}_true"].dropna()
                    pred_vals = test_df[f"{col}_pred"].dropna()
                    
                    if len(true_vals) > 0 and len(pred_vals) > 0:
                        r2 = r2_score(true_vals, pred_vals)
                        mse = mean_squared_error(true_vals, pred_vals)
                        mae = mean_absolute_error(true_vals, pred_vals)
                        
                        results[f"{col}_r2"] = r2
                        results[f"{col}_mse"] = mse
                        results[f"{col}_mae"] = mae
    
    # Look for validation results
    val_scores_file = output_path / "val_scores.csv"
    if val_scores_file.exists():
        val_df = pd.read_csv(val_scores_file)
        # Similar processing for validation metrics
        for col in val_df.columns:
            if col != 'smiles' and col.endswith('_val'):
                results[f"val_{col}"] = val_df[col].mean()
    
    return results

@app.function(
    image=image,
    volumes={
        "/vol/datasets": datasets_volume,
        "/vol/models": models_volume
    },
    secrets=[wandb_secret],
    gpu="A100",
    memory=32768,  # 32GB RAM
    timeout=14400   # 4 hours
)
def train_chemprop_multitask(
    dataset_name: str = "oncoprotein_multitask_dataset",
    num_epochs: int = 50,
    batch_size: int = 50,
    learning_rate: float = 1e-4,
    max_learning_rate: float = 1e-3,
    final_learning_rate: float = 1e-4,
    hidden_size: int = 300,
    depth: int = 3,
    dropout: float = 0.0,
    ffn_num_layers: int = 2,
    ffn_hidden_size: int = 300,
    aggregation: str = "mean",
    aggregation_norm: int = 100,
    test_size: float = 0.2,
    val_size: float = 0.1,
    num_workers: int = 8,
    use_features: bool = True,
    atom_messages: bool = False,
    early_stopping: bool = True,
    patience: int = 10,
    run_name: Optional[str] = None
):
    """
    Train Chemprop multi-task model with W&B logging
    """
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize W&B
    wandb.init(
        project="veridica-ai-training",
        group="chemprop",
        name=run_name or f"chemprop-{dataset_name}",
        config={
            "dataset_name": dataset_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_learning_rate": max_learning_rate,
            "final_learning_rate": final_learning_rate,
            "hidden_size": hidden_size,
            "depth": depth,
            "dropout": dropout,
            "ffn_num_layers": ffn_num_layers,
            "ffn_hidden_size": ffn_hidden_size,
            "aggregation": aggregation,
            "aggregation_norm": aggregation_norm,
            "test_size": test_size,
            "val_size": val_size,
            "use_features": use_features,
            "atom_messages": atom_messages,
            "early_stopping": early_stopping,
            "patience": patience
        },
        tags=["chemprop", "multi-task", "molecular-properties", "gnn"]
    )
    
    logger.info("🚀 Starting Chemprop Multi-Task Training")
    logger.info(f"📊 Dataset: {dataset_name}")
    
    # Load dataset
    dataset_path = Path(f"/vol/datasets/{dataset_name}.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    logger.info(f"📈 Loaded dataset: {df.shape}")
    
    # Prepare data
    smiles_col = 'canonical_smiles'
    target_cols = [col for col in df.columns if col != smiles_col]
    
    logger.info(f"🎯 Targets: {target_cols}")
    
    # Filter out compounds with no target data
    df_filtered = df.dropna(subset=target_cols, how='all')
    logger.info(f"📊 After filtering: {len(df_filtered)} compounds")
    
    # Split data
    train_df, temp_df = train_test_split(df_filtered, test_size=test_size + val_size, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size + val_size), random_state=42)
    
    logger.info(f"📊 Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Log dataset splits to W&B
    split_table = wandb.Table(
        columns=["split", "count", "percentage"],
        data=[
            ["train", len(train_df), f"{len(train_df)/len(df_filtered)*100:.1f}%"],
            ["validation", len(val_df), f"{len(val_df)/len(df_filtered)*100:.1f}%"],
            ["test", len(test_df), f"{len(test_df)/len(df_filtered)*100:.1f}%"]
        ]
    )
    wandb.log({"dataset_splits": split_table})
    
    # Create temporary directory for Chemprop data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Prepare Chemprop data files
        train_path = prepare_chemprop_data(train_df, smiles_col, target_cols, temp_path, "train")
        val_path = prepare_chemprop_data(val_df, smiles_col, target_cols, temp_path, "val")
        test_path = prepare_chemprop_data(test_df, smiles_col, target_cols, temp_path, "test")
        
        # Create output directory
        output_dir = f"/vol/models/chemprop_{dataset_name}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepare training configuration
        config = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_learning_rate': max_learning_rate,
            'final_learning_rate': final_learning_rate,
            'num_workers': num_workers,
            'hidden_size': hidden_size,
            'depth': depth,
            'dropout': dropout,
            'ffn_num_layers': ffn_num_layers,
            'ffn_hidden_size': ffn_hidden_size,
            'aggregation': aggregation,
            'aggregation_norm': aggregation_norm,
            'use_features': use_features,
            'atom_messages': atom_messages,
            'early_stopping': early_stopping,
            'patience': patience
        }
        
        # Initialize W&B logger
        wb_logger = ChempropWandbLogger(target_cols)
        
        try:
            # Run Chemprop training
            logger.info("🎓 Starting Chemprop training...")
            training_result = run_chemprop_training(train_path, val_path, test_path, output_dir, config)
            
            logger.info("📊 Parsing results...")
            results = parse_chemprop_results(output_dir)
            
            # Log final results
            wb_logger.log_final_results(results, output_dir)
            
            # Copy trained model to persistent volume
            model_save_path = Path(output_dir) / "model_0"  # Chemprop saves models as model_0, model_1, etc.
            
            # Save model as W&B artifact
            if model_save_path.exists():
                model_artifact = wandb.Artifact(
                    name=f"chemprop-{dataset_name}-model",
                    type="model",
                    description=f"Trained Chemprop model on {dataset_name}"
                )
                model_artifact.add_dir(str(model_save_path))
                wandb.log_artifact(model_artifact)
                
                logger.info(f"✅ Model saved to: {model_save_path}")
            
            # Create and save predictions analysis
            test_preds_path = Path(output_dir) / "test_preds.csv"
            if test_preds_path.exists():
                pred_df = pd.read_csv(test_preds_path)
                
                # Create prediction plots
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                axes = axes.ravel()
                
                plot_count = 0
                for target in target_cols[:4]:  # Plot first 4 targets
                    if f"{target}_true" in pred_df.columns and f"{target}_pred" in pred_df.columns:
                        true_vals = pred_df[f"{target}_true"].dropna()
                        pred_vals = pred_df[f"{target}_pred"].dropna()
                        
                        if len(true_vals) > 0:
                            axes[plot_count].scatter(true_vals, pred_vals, alpha=0.6)
                            axes[plot_count].plot([true_vals.min(), true_vals.max()], 
                                               [true_vals.min(), true_vals.max()], 'r--', alpha=0.8)
                            axes[plot_count].set_xlabel(f'{target} True')
                            axes[plot_count].set_ylabel(f'{target} Predicted')
                            axes[plot_count].set_title(f'{target} Predictions')
                            
                            # Add R² to plot
                            r2 = results.get(f"{target}_r2", 0.0)
                            axes[plot_count].text(0.05, 0.95, f'R² = {r2:.3f}', 
                                                transform=axes[plot_count].transAxes,
                                                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
                            plot_count += 1
                
                plt.tight_layout()
                plot_path = Path(output_dir) / "prediction_plots.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Log plot to W&B
                wandb.log({"prediction_plots": wandb.Image(str(plot_path))})
                
                # Log predictions as artifact
                pred_artifact = wandb.Artifact(
                    name=f"chemprop-{dataset_name}-predictions",
                    type="predictions",
                    description=f"Test set predictions from Chemprop model"
                )
                pred_artifact.add_file(str(test_preds_path))
                wandb.log_artifact(pred_artifact)
            
            logger.info("✅ Chemprop training completed!")
            
            # Finish W&B run
            wandb.finish()
            
            return {
                "status": "success",
                "model_path": str(model_save_path),
                "results": results,
                "wandb_run_id": wandb.run.id if wandb.run else None
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            wandb.finish()
            raise

@app.function(
    image=image,
    volumes={"/vol/models": models_volume},
    cpu=4.0,
    memory=16384,
    timeout=3600
)
def predict_chemprop(
    model_path: str,
    smiles_list: List[str],
    output_path: Optional[str] = None
):
    """
    Make predictions using trained Chemprop model
    """
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create temporary input file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Prepare input CSV
        input_df = pd.DataFrame({'smiles': smiles_list})
        input_file = temp_path / "input.csv"
        input_df.to_csv(input_file, index=False)
        
        # Prepare output file
        if output_path is None:
            output_file = temp_path / "predictions.csv"
        else:
            output_file = Path(output_path)
        
        # Run Chemprop prediction
        cmd = [
            'chemprop_predict',
            '--test_path', str(input_file),
            '--checkpoint_path', model_path,
            '--preds_path', str(output_file)
        ]
        
        logger.info(f"Running Chemprop prediction: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Chemprop prediction failed: {result.stderr}")
        
        # Load predictions
        pred_df = pd.read_csv(output_file)
        return pred_df.to_dict('records')

@app.function(
    image=image,
    volumes={
        "/vol/datasets": datasets_volume,
        "/vol/models": models_volume
    },
    secrets=[wandb_secret],
    cpu=2.0,
    memory=8192,
    timeout=3600
)
def hyperparameter_sweep_chemprop(
    dataset_name: str = "oncoprotein_multitask_dataset",
    n_trials: int = 20
):
    """
    Run hyperparameter optimization for Chemprop using W&B Sweeps
    """
    
    # Define sweep configuration
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_mean_r2',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'batch_size': {
                'values': [32, 50, 64, 100]
            },
            'hidden_size': {
                'values': [200, 300, 400, 500]
            },
            'depth': {
                'values': [2, 3, 4, 5]
            },
            'dropout': {
                'values': [0.0, 0.1, 0.2, 0.3]
            },
            'ffn_num_layers': {
                'values': [1, 2, 3]
            }
        }
    }
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project="veridica-ai-training"
    )
    
    def train_with_params():
        """Training function for sweep"""
        wandb.init()
        config = wandb.config
        
        try:
            result = train_chemprop_multitask(
                dataset_name=dataset_name,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                hidden_size=config.hidden_size,
                depth=config.depth,
                dropout=config.dropout,
                ffn_num_layers=config.ffn_num_layers,
                num_epochs=30,  # Shorter for sweep
                run_name=f"sweep-{wandb.run.name}"
            )
            
            # Log key metric for optimization
            results = result.get('results', {})
            r2_scores = [v for k, v in results.items() if k.endswith('_r2')]
            if r2_scores:
                wandb.log({'val_mean_r2': np.mean(r2_scores)})
                
        except Exception as e:
            logging.error(f"Sweep trial failed: {e}")
            wandb.log({'val_mean_r2': -1.0})  # Mark as failed
    
    # Run sweep
    wandb.agent(sweep_id, train_with_params, count=n_trials)
    
    return {"sweep_id": sweep_id, "status": "completed"}

if __name__ == "__main__":
    # Example usage
    pass