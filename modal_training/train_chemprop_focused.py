"""
Focused Chemprop Multi-Task Training
Trains Chemprop on the same 10 oncoproteins as ChemBERTa for consistency
"""

import modal
import pandas as pd
import numpy as np
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
import wandb
from datetime import datetime
import os

# Modal app setup
app = modal.App("chemprop-focused-training")

# Enhanced image with Chemprop and all required packages
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "chemprop>=2.2.0",  # Updated to latest version with new CLI
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "rdkit-pypi>=2022.9.5",
        "wandb>=0.16.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0"
    ])
)

# Shared volumes and secrets
datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret")

# FOCUSED TARGET LIST - Same 10 targets as ChemBERTa for consistency
FOCUSED_TARGETS = [
    'EGFR',     # 688 samples (13.7%) - ChemBERTa R²: 0.751
    'HER2',     # 637 samples (12.7%) - ChemBERTa R²: 0.583
    'VEGFR2',   # 775 samples (15.4%) - ChemBERTa R²: 0.555
    'BRAF',     # 601 samples (12.0%) - ChemBERTa R²: 0.595
    'MET',      # 489 samples (9.7%) - ChemBERTa R²: 0.502
    'CDK4',     # 348 samples (6.9%) - ChemBERTa R²: 0.314
    'CDK6',     # 600 samples (11.9%) - ChemBERTa R²: 0.216
    'ALK',      # 326 samples (6.5%) - ChemBERTa R²: 0.405
    'MDM2',     # 574 samples (11.4%) - ChemBERTa R²: 0.655
    'PI3KCA'    # 273 samples (5.4%) - ChemBERTa R²: 0.588
]

# EXCLUDED TARGETS (same as ChemBERTa - sparse/no data)
EXCLUDED_TARGETS = [
    'STAT3',    # 84 samples (1.7%) - too sparse
    'CTNNB1',   # 6 samples (0.1%) - too sparse  
    'RRM2',     # 0 samples (0.0%) - no data
    'MYC'       # 0 samples (0.0%) - no data
]

def prepare_chemprop_data(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """Prepare data in Chemprop format (CSV files)"""
    
    # Filter to focused targets only
    focused_df = df[['canonical_smiles'] + FOCUSED_TARGETS].copy()
    
    # Remove rows where ALL targets are NaN
    focused_df = focused_df.dropna(subset=['canonical_smiles'])
    
    # Data split (same as ChemBERTa)
    from sklearn.model_selection import train_test_split
    
    train_df, temp_df = train_test_split(focused_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.67, random_state=42)  # 20% test, 10% val
    
    print(f"   📊 Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return {
        'train': train_path,
        'val': val_path,
        'test': test_path
    }

class ChempropWandbLogger:
    """Enhanced W&B logger for Chemprop multi-task training"""
    
    def __init__(self, target_names: List[str], run_name: str):
        self.target_names = target_names
        self.run_name = run_name
        
    def log_training_config(self, config: Dict[str, Any]):
        """Log training configuration"""
        wandb.log({
            "config/model_type": "Chemprop Multi-Task GNN",
            "config/targets": self.target_names,
            "config/num_targets": len(self.target_names),
            "config/excluded_targets": EXCLUDED_TARGETS,
            **{f"config/{k}": v for k, v in config.items()}
        })
    
    def log_data_info(self, train_size: int, val_size: int, test_size: int):
        """Log dataset information"""
        wandb.log({
            "data/train_size": train_size,
            "data/val_size": val_size, 
            "data/test_size": test_size,
            "data/total_size": train_size + val_size + test_size
        })
    
    def log_training_results(self, results_dict: Dict[str, Any]):
        """Log training results and metrics"""
        
        # Log overall metrics
        for key, value in results_dict.items():
            if isinstance(value, (int, float)):
                wandb.log({f"train/{key}": value})
    
    def log_test_results(self, test_results: Dict[str, Any]):
        """Log detailed test results with per-target metrics"""
        
        # Extract per-target metrics
        target_metrics = {}
        for target in self.target_names:
            target_key = f"test_{target.lower()}"
            
            # Look for target-specific metrics in results
            for key, value in test_results.items():
                if target.lower() in key.lower() and isinstance(value, (int, float)):
                    metric_name = key.replace(target_key, '').replace('_', '').strip()
                    if metric_name:
                        target_metrics[f"test_{target}_{metric_name}"] = value
        
        # Log target metrics
        if target_metrics:
            wandb.log(target_metrics)
        
        # Log general test results
        for key, value in test_results.items():
            if isinstance(value, (int, float)) and not any(t.lower() in key.lower() for t in self.target_names):
                wandb.log({f"test/{key}": value})
        
        # Create performance summary
        r2_scores = []
        mae_scores = []
        
        for target in self.target_names:
            r2_key = f"test_{target}_r2"
            mae_key = f"test_{target}_mae"
            
            if r2_key in target_metrics:
                r2_scores.append(target_metrics[r2_key])
            if mae_key in target_metrics:
                mae_scores.append(target_metrics[mae_key])
        
        if r2_scores:
            wandb.log({
                "summary/mean_r2": np.mean(r2_scores),
                "summary/std_r2": np.std(r2_scores),
                "summary/min_r2": np.min(r2_scores),
                "summary/max_r2": np.max(r2_scores)
            })
        
        if mae_scores:
            wandb.log({
                "summary/mean_mae": np.mean(mae_scores),
                "summary/std_mae": np.std(mae_scores)
            })
        
        # Performance breakdown
        if r2_scores:
            excellent_targets = sum(1 for r2 in r2_scores if r2 > 0.6)
            good_targets = sum(1 for r2 in r2_scores if 0.4 < r2 <= 0.6)
            fair_targets = sum(1 for r2 in r2_scores if 0.2 < r2 <= 0.4)
            poor_targets = sum(1 for r2 in r2_scores if r2 <= 0.2)
            
            wandb.log({
                "performance/excellent_targets": excellent_targets,  # R² > 0.6
                "performance/good_targets": good_targets,           # 0.4 < R² ≤ 0.6
                "performance/fair_targets": fair_targets,          # 0.2 < R² ≤ 0.4
                "performance/poor_targets": poor_targets            # R² ≤ 0.2
            })

@app.function(
    image=image,
    volumes={
        "/vol/datasets": datasets_volume,
        "/vol/models": models_volume,
    },
    secrets=[wandb_secret],
    gpu="A100",
    memory=32768,
    timeout=7200  # 2 hours for full training
)
def train_focused_chemprop(
    epochs: int = 30,  # More epochs for Chemprop
    batch_size: int = 32,  # Larger batch for GNN
    learning_rate: float = 1e-3,
    hidden_size: int = 256,
    depth: int = 4,
    dropout: float = 0.1,
    ffn_num_layers: int = 3,
    test_size: float = 0.2,
    val_size: float = 0.1,
    run_name: str = None
):
    """Train focused Chemprop on 10 oncoproteins with sufficient data"""
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize W&B
    wandb.init(
        project="veridica-ai-focused-training",
        name=run_name or f"chemprop-focused-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_type": "Chemprop-focused",
            "targets": FOCUSED_TARGETS,
            "excluded_targets": EXCLUDED_TARGETS,
            "num_targets": len(FOCUSED_TARGETS),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            "depth": depth,
            "dropout": dropout,
            "ffn_num_layers": ffn_num_layers
        }
    )
    
    logger.info("🚀 FOCUSED CHEMPROP TRAINING STARTED")
    logger.info(f"📊 Training targets: {FOCUSED_TARGETS}")
    logger.info(f"🚫 Excluded targets: {EXCLUDED_TARGETS}")
    
    # Initialize W&B logger
    wandb_logger = ChempropWandbLogger(FOCUSED_TARGETS, run_name or "chemprop-focused")
    
    try:
        # Load dataset
        logger.info("📊 Loading dataset...")
        dataset_path = Path("/vol/datasets/oncoprotein_multitask_dataset.csv")
        if dataset_path.exists():
            df = pd.read_csv(dataset_path)
        else:
            dataset_path = Path("/vol/datasets/oncoprotein_multitask_dataset.parquet")
            df = pd.read_parquet(dataset_path)
        
        logger.info(f"   ✅ Dataset loaded: {df.shape}")
        
        # Data availability analysis
        for target in FOCUSED_TARGETS:
            available = df[target].notna().sum()
            percentage = (available / len(df)) * 100
            logger.info(f"   {target:10s}: {available:4d} samples ({percentage:4.1f}%)")
        
        # Prepare data for Chemprop
        logger.info("📦 Preparing Chemprop data format...")
        temp_dir = Path(tempfile.mkdtemp())
        data_paths = prepare_chemprop_data(df, temp_dir / "data")
        
        # Log data info
        train_df = pd.read_csv(data_paths['train'])
        val_df = pd.read_csv(data_paths['val'])
        test_df = pd.read_csv(data_paths['test'])
        
        wandb_logger.log_data_info(len(train_df), len(val_df), len(test_df))
        
        # Prepare output directory
        output_dir = Path(f"/vol/models/focused_chemprop_{run_name or 'default'}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🏋️ Starting Chemprop training...")
        
        # Chemprop training command - Updated for v2.2.0 CLI with correct data handling
        cmd = [
            'chemprop', 'train',
            '--data-path', str(data_paths['train']),  # Main training data
            '--task-type', 'regression',
            '--save-dir', str(output_dir),
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--init-lr', str(learning_rate),
            '--max-lr', str(learning_rate * 10),
            '--final-lr', str(learning_rate * 0.1),
            '--message-hidden-dim', str(hidden_size),
            '--depth', str(depth),
            '--dropout', str(dropout),
            '--ffn-num-layers', str(ffn_num_layers),
            '--num-workers', '4',
            '--split-sizes', '0.8', '0.1', '0.1',  # Handle splits internally
            '--save-test-preds'
        ]
        
        # Log training config
        training_config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_size': hidden_size,
            'depth': depth,
            'dropout': dropout,
            'ffn_num_layers': ffn_num_layers,
            'dataset_type': 'regression',
            'num_workers': 4
        }
        wandb_logger.log_training_config(training_config)
        
        logger.info(f"   🔧 Command: {' '.join(cmd[:8])}... (truncated)")
        
        # Execute training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=6000  # 100 minutes timeout
        )
        
        if result.returncode == 0:
            logger.info("✅ Chemprop training completed successfully!")
            logger.info("📊 TRAINING COMPLETED!")
            logger.info("=" * 60)
            
            # Parse training output for metrics
            output_lines = result.stdout.split('\n')
            training_metrics = {}
            
            for line in output_lines:
                if 'Test' in line and any(metric in line for metric in ['rmse', 'mae', 'r2']):
                    logger.info(f"   📈 {line.strip()}")
            
            # Look for test predictions file
            test_preds_path = output_dir / "test_preds.csv"
            if test_preds_path.exists():
                logger.info("📊 Evaluating test set performance...")
                
                # Load predictions and calculate detailed metrics
                test_preds = pd.read_csv(test_preds_path)
                test_df_eval = pd.read_csv(data_paths['test'])
                
                # Calculate per-target R² scores
                r2_scores = {}
                mae_scores = {}
                rmse_scores = {}
                
                for i, target in enumerate(FOCUSED_TARGETS):
                    if target in test_df_eval.columns:
                        # Get true and predicted values
                        y_true = test_df_eval[target].values
                        y_pred = test_preds.iloc[:, i+1].values  # Skip SMILES column
                        
                        # Remove NaN values
                        mask = ~np.isnan(y_true)
                        if mask.sum() > 0:
                            y_true_clean = y_true[mask]
                            y_pred_clean = y_pred[mask]
                            
                            # Calculate metrics
                            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                            
                            r2 = r2_score(y_true_clean, y_pred_clean)
                            mae = mean_absolute_error(y_true_clean, y_pred_clean)
                            rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
                            
                            r2_scores[target] = r2
                            mae_scores[target] = mae
                            rmse_scores[target] = rmse
                            
                            logger.info(f"   {target:10s}: R² = {r2:7.3f}, MAE = {mae:7.3f}, RMSE = {rmse:7.3f} (n={mask.sum()})")
                
                # Calculate summary statistics
                if r2_scores:
                    mean_r2 = np.mean(list(r2_scores.values()))
                    mean_mae = np.mean(list(mae_scores.values()))
                    mean_rmse = np.mean(list(rmse_scores.values()))
                    
                    logger.info(f"\n   📈 SUMMARY METRICS:")
                    logger.info(f"   Mean R²:  {mean_r2:.3f}")
                    logger.info(f"   Mean MAE: {mean_mae:.3f}")
                    logger.info(f"   Mean RMSE: {mean_rmse:.3f}")
                    
                    # Performance breakdown
                    excellent_targets = [t for t, r2 in r2_scores.items() if r2 > 0.6]
                    good_targets = [t for t, r2 in r2_scores.items() if 0.4 < r2 <= 0.6]
                    fair_targets = [t for t, r2 in r2_scores.items() if 0.2 < r2 <= 0.4]
                    poor_targets = [t for t, r2 in r2_scores.items() if r2 <= 0.2]
                    
                    logger.info(f"\n   🎯 PERFORMANCE BREAKDOWN:")
                    logger.info(f"   🌟 Excellent (R² > 0.6):   {len(excellent_targets)} targets {excellent_targets}")
                    logger.info(f"   ✅ Good (0.4 < R² ≤ 0.6):  {len(good_targets)} targets {good_targets}")
                    logger.info(f"   ⚠️ Fair (0.2 < R² ≤ 0.4):  {len(fair_targets)} targets {fair_targets}")
                    logger.info(f"   ❌ Poor (R² ≤ 0.2):        {len(poor_targets)} targets {poor_targets}")
                    
                    # Compare with ChemBERTa results
                    chemberta_r2 = {
                        'EGFR': 0.751, 'MDM2': 0.655, 'BRAF': 0.595, 'PI3KCA': 0.588,
                        'HER2': 0.583, 'VEGFR2': 0.555, 'MET': 0.502, 'ALK': 0.405,
                        'CDK4': 0.314, 'CDK6': 0.216
                    }
                    
                    logger.info(f"\n   📊 COMPARISON WITH ChemBERTa:")
                    chemprop_better = 0
                    chemberta_better = 0
                    
                    for target in FOCUSED_TARGETS:
                        if target in r2_scores and target in chemberta_r2:
                            chemprop_r2 = r2_scores[target]
                            bert_r2 = chemberta_r2[target]
                            diff = chemprop_r2 - bert_r2
                            
                            if diff > 0.05:
                                chemprop_better += 1
                                status = "🟢 Chemprop Better"
                            elif diff < -0.05:
                                chemberta_better += 1
                                status = "🔴 ChemBERTa Better"
                            else:
                                status = "🟡 Similar"
                            
                            logger.info(f"   {target:10s}: Chemprop={chemprop_r2:.3f}, ChemBERTa={bert_r2:.3f} ({diff:+.3f}) {status}")
                    
                    logger.info(f"\n   🏆 FINAL COMPARISON:")
                    logger.info(f"   Chemprop Better: {chemprop_better}")
                    logger.info(f"   ChemBERTa Better: {chemberta_better}")
                    logger.info(f"   Similar: {len(FOCUSED_TARGETS) - chemprop_better - chemberta_better}")
                    
                    # Prepare test results for logging
                    test_results = {}
                    for target in FOCUSED_TARGETS:
                        if target in r2_scores:
                            test_results[f"test_{target}_r2"] = r2_scores[target]
                            test_results[f"test_{target}_mae"] = mae_scores[target]
                            test_results[f"test_{target}_rmse"] = rmse_scores[target]
                    
                    test_results['mean_r2'] = mean_r2
                    test_results['mean_mae'] = mean_mae
                    test_results['mean_rmse'] = mean_rmse
                    
                    # Log to W&B
                    wandb_logger.log_test_results(test_results)
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
            # Save results summary
            results_summary = {
                "status": "success",
                "model_type": "Chemprop Multi-Task GNN",
                "training_targets": FOCUSED_TARGETS,
                "excluded_targets": EXCLUDED_TARGETS,
                "r2_scores": r2_scores if 'r2_scores' in locals() else {},
                "mae_scores": mae_scores if 'mae_scores' in locals() else {},
                "mean_r2": mean_r2 if 'mean_r2' in locals() else 0.0,
                "mean_mae": mean_mae if 'mean_mae' in locals() else 0.0,
                "model_path": str(output_dir),
                "wandb_run_id": wandb.run.id,
                "training_config": training_config,
                "performance_breakdown": {
                    "excellent": len(excellent_targets) if 'excellent_targets' in locals() else 0,
                    "good": len(good_targets) if 'good_targets' in locals() else 0,
                    "fair": len(fair_targets) if 'fair_targets' in locals() else 0,
                    "poor": len(poor_targets) if 'poor_targets' in locals() else 0
                }
            }
            
            # Save to file
            results_file = output_dir / "results_summary.json"
            with open(results_file, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            logger.info(f"   💾 Results saved to: {results_file}")
            logger.info("=" * 60)
            
            wandb.finish()
            return results_summary
            
        else:
            logger.error(f"❌ Chemprop training failed!")
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            
            wandb.finish()
            return {"status": "error", "error": f"Training failed with return code {result.returncode}", "stderr": result.stderr}
        
    except Exception as e:
        logger.error(f"❌ Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        
        wandb.finish()
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("🚀 Focused Chemprop Multi-Task Training")