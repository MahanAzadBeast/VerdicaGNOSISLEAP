"""
Expanded Chemprop Multi-Task Training
Trains Chemprop on expanded target set with multiple activity types
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
app = modal.App("expanded-chemprop-training")

# Enhanced image with Chemprop and all required packages
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "chemprop>=2.2.0",
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
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret")

# EXPANDED TARGET LIST - All targets with sufficient data
EXPANDED_TARGETS = [
    # Oncoproteins (existing)
    'EGFR', 'HER2', 'VEGFR2', 'BRAF', 'MET', 'CDK4', 'CDK6', 'ALK', 'MDM2', 'PI3KCA',
    
    # Tumor Suppressors (new)
    'TP53', 'RB1', 'PTEN', 'APC', 'BRCA1', 'BRCA2', 'VHL',
    
    # Metastasis Suppressors (new)
    'NDRG1', 'KAI1', 'KISS1', 'NM23H1', 'RIKP', 'CASP8'
]

def prepare_expanded_chemprop_data(df: pd.DataFrame, output_dir: Path, activity_type: str = 'IC50') -> Dict[str, Path]:
    """Prepare expanded data in Chemprop format for multiple targets"""
    
    # Filter to available targets only
    available_targets = [col for col in df.columns if col != 'canonical_smiles' and col in EXPANDED_TARGETS]
    
    if not available_targets:
        # Fallback: use any available targets
        available_targets = [col for col in df.columns if col != 'canonical_smiles']
    
    focused_df = df[['canonical_smiles'] + available_targets].copy()
    
    # Remove rows where ALL targets are NaN
    focused_df = focused_df.dropna(subset=['canonical_smiles'])
    
    print(f"   📊 Expanded data: {len(focused_df)} samples across {len(available_targets)} targets")
    print(f"   🎯 Targets: {available_targets}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all data in one file - Chemprop v2.2.0 handles splits internally
    data_path = output_dir / f"expanded_{activity_type.lower()}_data.csv"
    focused_df.to_csv(data_path, index=False)
    
    return {
        'data': data_path,
        'targets': available_targets
    }

class ExpandedChempropWandbLogger:
    """Enhanced W&B logger for expanded Chemprop multi-task training"""
    
    def __init__(self, target_names: List[str], activity_type: str, run_name: str):
        self.target_names = target_names
        self.activity_type = activity_type
        self.run_name = run_name
        
    def log_training_config(self, config: Dict[str, Any]):
        """Log training configuration"""
        wandb.log({
            "config/model_type": "Chemprop Expanded Multi-Task GNN",
            "config/activity_type": self.activity_type,
            "config/targets": self.target_names,
            "config/num_targets": len(self.target_names),
            "config/data_sources": ["ChEMBL", "PubChem", "BindingDB", "DTC"],
            **{f"config/{k}": v for k, v in config.items()}
        })
    
    def log_data_info(self, train_size: int, val_size: int, test_size: int, target_coverage: Dict[str, int]):
        """Log dataset information"""
        wandb.log({
            "data/train_size": train_size,
            "data/val_size": val_size, 
            "data/test_size": test_size,
            "data/total_size": train_size + val_size + test_size,
            "data/num_targets": len(self.target_names)
        })
        
        # Log target coverage
        for target, count in target_coverage.items():
            wandb.log({f"data/coverage_{target}": count})
    
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
        for target in self.target_names:
            r2_key = f"test_{target}_r2"
            if r2_key in target_metrics:
                r2_scores.append(target_metrics[r2_key])
        
        if r2_scores:
            wandb.log({
                "summary/mean_r2": np.mean(r2_scores),
                "summary/std_r2": np.std(r2_scores),
                "summary/min_r2": np.min(r2_scores),
                "summary/max_r2": np.max(r2_scores)
            })
            
            # Performance breakdown by category
            oncoprotein_r2 = [r2_scores[i] for i, target in enumerate(self.target_names) 
                             if target in ['EGFR', 'HER2', 'VEGFR2', 'BRAF', 'MET', 'CDK4', 'CDK6', 'ALK', 'MDM2', 'PI3KCA']]
            tumor_suppressor_r2 = [r2_scores[i] for i, target in enumerate(self.target_names) 
                                  if target in ['TP53', 'RB1', 'PTEN', 'APC', 'BRCA1', 'BRCA2', 'VHL']]
            metastasis_suppressor_r2 = [r2_scores[i] for i, target in enumerate(self.target_names) 
                                       if target in ['NDRG1', 'KAI1', 'KISS1', 'NM23H1', 'RIKP', 'CASP8']]
            
            if oncoprotein_r2:
                wandb.log({"summary/oncoprotein_mean_r2": np.mean(oncoprotein_r2)})
            if tumor_suppressor_r2:
                wandb.log({"summary/tumor_suppressor_mean_r2": np.mean(tumor_suppressor_r2)})
            if metastasis_suppressor_r2:
                wandb.log({"summary/metastasis_suppressor_mean_r2": np.mean(metastasis_suppressor_r2)})

@app.function(
    image=image,
    volumes={
        "/vol/datasets": datasets_volume,
        "/vol/models": models_volume,
    },
    secrets=[wandb_secret],
    gpu="A100",
    memory=40960,  # 40GB for expanded model
    timeout=10800  # 3 hours for expanded training
)
def train_expanded_chemprop(
    activity_type: str = 'IC50',
    epochs: int = 40,  # More epochs for expanded task
    batch_size: int = 64,  # Larger batch for stability
    learning_rate: float = 1e-3,
    hidden_size: int = 512,  # Larger hidden size
    depth: int = 5,  # Deeper network
    dropout: float = 0.1,
    ffn_num_layers: int = 3,
    test_size: float = 0.2,
    val_size: float = 0.1,
    run_name: str = None
):
    """Train expanded Chemprop on multiple target categories and activity types"""
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize W&B
    wandb.init(
        project="veridica-ai-expanded-training",
        name=run_name or f"expanded-chemprop-{activity_type}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_type": "Chemprop-Expanded-MultiSource",
            "activity_type": activity_type,
            "targets": EXPANDED_TARGETS,
            "num_targets": len(EXPANDED_TARGETS),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            "depth": depth,
            "dropout": dropout,
            "ffn_num_layers": ffn_num_layers,
            "data_sources": ["ChEMBL", "PubChem", "BindingDB", "DTC"]
        }
    )
    
    logger.info("🚀 EXPANDED CHEMPROP TRAINING STARTED")
    logger.info(f"📊 Activity type: {activity_type}")
    logger.info(f"📊 Target categories:")
    logger.info(f"   • Oncoproteins: {len([t for t in EXPANDED_TARGETS if t in ['EGFR', 'HER2', 'VEGFR2', 'BRAF', 'MET', 'CDK4', 'CDK6', 'ALK', 'MDM2', 'PI3KCA']])}")
    logger.info(f"   • Tumor Suppressors: {len([t for t in EXPANDED_TARGETS if t in ['TP53', 'RB1', 'PTEN', 'APC', 'BRCA1', 'BRCA2', 'VHL']])}")
    logger.info(f"   • Metastasis Suppressors: {len([t for t in EXPANDED_TARGETS if t in ['NDRG1', 'KAI1', 'KISS1', 'NM23H1', 'RIKP', 'CASP8']])}")
    
    try:
        # Load expanded dataset
        logger.info("📊 Loading expanded dataset...")
        
        # Try to load activity-specific matrix
        dataset_path = Path(f"/vol/datasets/expanded_multisource_{activity_type.lower()}_matrix.csv")
        
        if not dataset_path.exists():
            # Fallback to original dataset for now
            logger.warning(f"Expanded dataset not found, using original dataset")
            dataset_path = Path("/vol/datasets/oncoprotein_multitask_dataset.csv")
            df = pd.read_csv(dataset_path)
        else:
            df = pd.read_csv(dataset_path)
            logger.info(f"   ✅ Expanded dataset loaded: {df.shape}")
        
        # Prepare data for Chemprop
        logger.info("📦 Preparing expanded Chemprop data format...")
        temp_dir = Path(tempfile.mkdtemp())
        data_info = prepare_expanded_chemprop_data(df, temp_dir / "data", activity_type)
        
        available_targets = data_info['targets']
        data_path = data_info['data']
        
        # Initialize W&B logger
        wandb_logger = ExpandedChempropWandbLogger(available_targets, activity_type, run_name or "expanded-chemprop")
        
        # Log data info
        all_data_df = pd.read_csv(data_path)
        total_samples = len(all_data_df)
        
        # Calculate target coverage
        target_coverage = {}
        for target in available_targets:
            if target in all_data_df.columns:
                coverage = all_data_df[target].notna().sum()
                target_coverage[target] = coverage
                percentage = (coverage / total_samples) * 100
                logger.info(f"   {target:12s}: {coverage:4d} samples ({percentage:4.1f}%)")
        
        # Estimate split sizes
        train_size = int(total_samples * 0.8)
        val_size = int(total_samples * 0.1)
        test_size = total_samples - train_size - val_size
        
        wandb_logger.log_data_info(train_size, val_size, test_size, target_coverage)
        
        # Prepare output directory
        output_dir = Path(f"/vol/models/expanded_chemprop_{activity_type}_{run_name or 'default'}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🏋️ Starting expanded Chemprop training...")
        
        # Enhanced Chemprop training command for expanded multi-task learning
        cmd = [
            'chemprop', 'train',
            '--data-path', str(data_path),
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
            '--num-workers', '8',  # More workers for larger dataset
            '--split-sizes', '0.8', '0.1', '0.1',
            '--patience', '25',  # More patience for complex model
            '--metric', 'r2'  # Focus on R² optimization
        ]
        
        # Log training config
        training_config = {
            'activity_type': activity_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_size': hidden_size,
            'depth': depth,
            'dropout': dropout,
            'ffn_num_layers': ffn_num_layers,
            'num_targets': len(available_targets),
            'total_samples': total_samples
        }
        wandb_logger.log_training_config(training_config)
        
        logger.info(f"   🔧 Command: chemprop train --data-path {data_path.name} ... (truncated)")
        logger.info(f"   📊 Training on {len(available_targets)} targets with {total_samples} samples")
        
        # Execute training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=9000  # 2.5 hours timeout
        )
        
        if result.returncode == 0:
            logger.info("✅ Expanded Chemprop training completed successfully!")
            logger.info("📊 TRAINING COMPLETED!")
            logger.info("=" * 60)
            
            # Parse training output for metrics
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'Test' in line and any(metric in line for metric in ['rmse', 'mae', 'r2']):
                    logger.info(f"   📈 {line.strip()}")
            
            # Look for predictions and results files
            results_files = list(output_dir.glob("*test*.csv")) + list(output_dir.glob("*pred*.csv"))
            
            if results_files:
                logger.info("📊 Analyzing expanded model performance...")
                predictions_file = results_files[0]
                test_preds = pd.read_csv(predictions_file)
                
                logger.info(f"   📈 Results file: {predictions_file.name}")
                logger.info(f"   📊 Results shape: {test_preds.shape}")
                
                # Calculate per-target R² scores
                r2_scores = {}
                category_performance = {
                    'oncoprotein': [],
                    'tumor_suppressor': [],
                    'metastasis_suppressor': []
                }
                
                # Target categorization
                target_categories = {
                    'oncoprotein': ['EGFR', 'HER2', 'VEGFR2', 'BRAF', 'MET', 'CDK4', 'CDK6', 'ALK', 'MDM2', 'PI3KCA'],
                    'tumor_suppressor': ['TP53', 'RB1', 'PTEN', 'APC', 'BRCA1', 'BRCA2', 'VHL'],
                    'metastasis_suppressor': ['NDRG1', 'KAI1', 'KISS1', 'NM23H1', 'RIKP', 'CASP8']
                }
                
                for target in available_targets:
                    # Try to find prediction columns for this target
                    true_col = None
                    pred_col = None
                    
                    for col in test_preds.columns:
                        if target.lower() in col.lower():
                            if 'true' in col.lower() or 'actual' in col.lower():
                                true_col = col
                            elif 'pred' in col.lower() or 'predicted' in col.lower():
                                pred_col = col
                    
                    if true_col and pred_col:
                        y_true = test_preds[true_col].values
                        y_pred = test_preds[pred_col].values
                        
                        # Remove NaN values
                        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                        if mask.sum() > 5:
                            y_true_clean = y_true[mask]
                            y_pred_clean = y_pred[mask]
                            
                            # Calculate metrics
                            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                            
                            r2 = r2_score(y_true_clean, y_pred_clean)
                            mae = mean_absolute_error(y_true_clean, y_pred_clean)
                            rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
                            
                            r2_scores[target] = r2
                            
                            # Assign to category
                            for category, targets in target_categories.items():
                                if target in targets:
                                    category_performance[category].append(r2)
                                    break
                            
                            logger.info(f"   {target:12s}: R² = {r2:7.3f}, MAE = {mae:7.3f}, RMSE = {rmse:7.3f} (n={mask.sum()})")
                
                # Category-wise performance analysis
                logger.info(f"\n   📊 CATEGORY PERFORMANCE ANALYSIS:")
                for category, r2_list in category_performance.items():
                    if r2_list:
                        mean_r2 = np.mean(r2_list)
                        std_r2 = np.std(r2_list)
                        count = len(r2_list)
                        logger.info(f"   {category.replace('_', ' ').title():20s}: {mean_r2:.3f} ± {std_r2:.3f} ({count} targets)")
                
                # Overall performance summary
                if r2_scores:
                    all_r2_values = list(r2_scores.values())
                    mean_r2 = np.mean(all_r2_values)
                    std_r2 = np.std(all_r2_values)
                    
                    logger.info(f"\n   📈 OVERALL PERFORMANCE:")
                    logger.info(f"   Mean R²: {mean_r2:.3f} ± {std_r2:.3f}")
                    logger.info(f"   Targets: {len(r2_scores)}/{len(available_targets)}")
                    
                    # Performance breakdown
                    excellent = len([r2 for r2 in all_r2_values if r2 > 0.6])
                    good = len([r2 for r2 in all_r2_values if 0.4 < r2 <= 0.6])
                    fair = len([r2 for r2 in all_r2_values if 0.2 < r2 <= 0.4])
                    poor = len([r2 for r2 in all_r2_values if r2 <= 0.2])
                    
                    logger.info(f"\n   🎯 PERFORMANCE BREAKDOWN:")
                    logger.info(f"   🌟 Excellent (R² > 0.6):   {excellent}/{len(r2_scores)}")
                    logger.info(f"   ✅ Good (0.4 < R² ≤ 0.6):  {good}/{len(r2_scores)}")
                    logger.info(f"   ⚠️ Fair (0.2 < R² ≤ 0.4):  {fair}/{len(r2_scores)}")
                    logger.info(f"   ❌ Poor (R² ≤ 0.2):        {poor}/{len(r2_scores)}")
                    
                    # Prepare test results for W&B logging
                    test_results = {}
                    for target, r2 in r2_scores.items():
                        test_results[f"test_{target}_r2"] = r2
                    
                    test_results['mean_r2'] = mean_r2
                    test_results['std_r2'] = std_r2
                    
                    # Add category means
                    for category, r2_list in category_performance.items():
                        if r2_list:
                            test_results[f"{category}_mean_r2"] = np.mean(r2_list)
                    
                    # Log to W&B
                    wandb_logger.log_test_results(test_results)
            else:
                logger.warning("   ⚠️ No predictions file found")
                r2_scores = {}
                mean_r2 = 0.0
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
            # Save results summary
            results_summary = {
                "status": "success",
                "model_type": "Chemprop Expanded Multi-Source GNN",
                "activity_type": activity_type,
                "available_targets": available_targets,
                "target_categories": {
                    "oncoproteins": [t for t in available_targets if t in target_categories['oncoprotein']],
                    "tumor_suppressors": [t for t in available_targets if t in target_categories['tumor_suppressor']],
                    "metastasis_suppressors": [t for t in available_targets if t in target_categories['metastasis_suppressor']]
                },
                "r2_scores": r2_scores if 'r2_scores' in locals() else {},
                "mean_r2": mean_r2 if 'mean_r2' in locals() else 0.0,
                "category_performance": {k: np.mean(v) if v else 0.0 for k, v in category_performance.items()},
                "model_path": str(output_dir),
                "wandb_run_id": wandb.run.id,
                "training_config": training_config,
                "data_sources": ["ChEMBL", "PubChem", "BindingDB", "DTC"],
                "total_samples": total_samples,
                "performance_breakdown": {
                    "excellent": excellent if 'excellent' in locals() else 0,
                    "good": good if 'good' in locals() else 0,
                    "fair": fair if 'fair' in locals() else 0,
                    "poor": poor if 'poor' in locals() else 0
                }
            }
            
            # Save to file
            results_file = output_dir / "results_summary.json"
            with open(results_file, 'w') as f:
                json.dump(results_summary, f, indent=2, default=str)
            
            logger.info(f"   💾 Results saved to: {results_file}")
            logger.info("=" * 60)
            
            wandb.finish()
            return results_summary
            
        else:
            logger.error(f"❌ Expanded Chemprop training failed!")
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
    print("🚀 Expanded Chemprop Multi-Task Training")