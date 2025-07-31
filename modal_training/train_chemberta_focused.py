"""
Focused ChemBERTa Multi-Task Training
Trains only on targets with sufficient data for better performance
"""

import modal
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging
from pathlib import Path
from typing import Dict, List, Any
import json
import wandb
from datetime import datetime

# Modal app setup
app = modal.App("chemberta-focused-training")

# Enhanced image with all required packages
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "accelerate>=0.26.0",  # Required for Trainer
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "rdkit-pypi>=2022.9.5",
        "wandb>=0.16.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0"
    ])
)

# Shared volumes and secrets
datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret")

# FOCUSED TARGET LIST - Only targets with sufficient data
FOCUSED_TARGETS = [
    'EGFR',     # 688 samples (13.7%)
    'HER2',     # 637 samples (12.7%)  
    'VEGFR2',   # 775 samples (15.4%)
    'BRAF',     # 601 samples (12.0%)
    'MET',      # 489 samples (9.7%)
    'CDK4',     # 348 samples (6.9%)
    'CDK6',     # 600 samples (11.9%)
    'ALK',      # 326 samples (6.5%)
    'MDM2',     # 574 samples (11.4%)
    'PI3KCA'    # 273 samples (5.4%)
]

# EXCLUDED TARGETS (sparse/no data)
EXCLUDED_TARGETS = [
    'STAT3',    # 84 samples (1.7%) - too sparse
    'CTNNB1',   # 6 samples (0.1%) - too sparse  
    'RRM2',     # 0 samples (0.0%) - no data
    'MYC'       # 0 samples (0.0%) - no data
]

class FocusedMolecularDataset(Dataset):
    """Dataset for focused multi-task ChemBERTa training"""
    
    def __init__(self, smiles_list: List[str], targets_dict: Dict[str, List[float]], 
                 tokenizer, max_length: int = 512):
        self.smiles = smiles_list
        self.targets = targets_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_names = FOCUSED_TARGETS
        
    def __len__(self):
        return len(self.smiles)
        
    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        
        # Tokenize SMILES
        encoding = self.tokenizer(
            smiles,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare targets
        labels = []
        masks = []  # For handling missing values
        
        for target in self.target_names:
            if target in self.targets and idx < len(self.targets[target]):
                value = self.targets[target][idx]
                if pd.isna(value):
                    labels.append(0.0)  # Dummy value
                    masks.append(0.0)   # Mask out
                else:
                    labels.append(float(value))
                    masks.append(1.0)   # Include in loss
            else:
                labels.append(0.0)
                masks.append(0.0)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float32),
            'masks': torch.tensor(masks, dtype=torch.float32)
        }

class FocusedChemBERTaMultiTaskModel(nn.Module):
    """Focused ChemBERTa model for 10 targets with sufficient data"""
    
    def __init__(self, model_name: str, num_targets: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        # Shared feature layer
        self.shared_layer = nn.Linear(hidden_size, 512)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Regression heads for each target
        self.regression_heads = nn.ModuleList([
            nn.Linear(512, 1) for _ in range(num_targets)
        ])
        
        self.num_targets = num_targets
        
    @property
    def device(self):
        """Safe device property access"""
        return next(self.parameters()).device
        
    def forward(self, input_ids, attention_mask, labels=None, masks=None):
        # Get backbone embeddings
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        # Shared features
        shared_features = self.dropout(self.activation(self.shared_layer(pooled_output)))
        
        # Task-specific predictions
        predictions = []
        for head in self.regression_heads:
            pred = head(shared_features).squeeze(-1)
            predictions.append(pred)
        
        # Stack predictions
        logits = torch.stack(predictions, dim=1)
        
        loss = None
        if labels is not None and masks is not None:
            # Masked MSE loss
            mse_loss = nn.MSELoss(reduction='none')
            losses = mse_loss(logits, labels)
            masked_losses = losses * masks
            loss = masked_losses.sum() / (masks.sum() + 1e-8)
        
        return {
            'logits': logits,
            'loss': loss
        }

class FocusedWandbCallback:
    """W&B callback for focused training with proper NaN handling"""
    
    def __init__(self, target_names: List[str]):
        self.target_names = target_names
        
    def on_log(self, logs):
        """Log metrics to W&B"""
        if wandb.run is not None:
            wandb.log(logs)
    
    def on_evaluate(self, eval_results: Dict[str, float]):
        """Log evaluation results"""
        if wandb.run is not None:
            wandb.log(eval_results)

class FocusedChemBERTaTrainer(Trainer):
    """Custom trainer for focused ChemBERTa with enhanced metrics"""
    
    def __init__(self, target_names: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_names = target_names
        self.wandb_callback = FocusedWandbCallback(target_names)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with per-target R² scores"""
        
        # Get predictions
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        all_predictions = []
        all_labels = []
        all_masks = []
        
        self.model.eval()
        device = next(self.model.parameters()).device  # Safe device access
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                all_predictions.append(outputs['logits'].cpu().numpy())
                all_labels.append(batch['labels'].cpu().numpy())
                all_masks.append(batch['masks'].cpu().numpy())
        
        # Combine results
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)
        masks = np.vstack(all_masks)
        
        # Calculate per-target metrics
        metrics = {}
        
        for i, target in enumerate(self.target_names):
            # Get valid samples for this target
            target_mask = masks[:, i].astype(bool)
            
            if target_mask.sum() > 0:  # Only if we have valid samples
                y_true = labels[:, i][target_mask]
                y_pred = predictions[:, i][target_mask]
                
                # Calculate metrics
                r2 = r2_score(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                
                # Log metrics
                metrics[f"{metric_key_prefix}_{target}_r2"] = r2
                metrics[f"{metric_key_prefix}_{target}_mse"] = mse
                metrics[f"{metric_key_prefix}_{target}_mae"] = mae
                metrics[f"{metric_key_prefix}_{target}_samples"] = int(target_mask.sum())
        
        # Calculate overall metrics
        valid_r2_scores = [v for k, v in metrics.items() if k.endswith('_r2')]
        if valid_r2_scores:
            metrics[f"{metric_key_prefix}_mean_r2"] = np.mean(valid_r2_scores)
            metrics[f"{metric_key_prefix}_std_r2"] = np.std(valid_r2_scores)
        
        # Log to W&B
        self.wandb_callback.on_evaluate(metrics)
        
        return metrics

@app.function(
    image=image,
    volumes={
        "/vol/datasets": datasets_volume,
        "/vol/models": models_volume,
    },
    secrets=[wandb_secret],
    gpu="A100",
    memory=32768,
    timeout=10800  # 3 hours for 50-epoch training
)
def train_focused_chemberta(
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 50,  # Increased to match Chemprop training (50 epochs)
    max_length: int = 512,
    test_size: float = 0.2,
    val_size: float = 0.1,
    dropout: float = 0.1,
    warmup_steps: int = 500,
    save_steps: int = 1000,
    eval_steps: int = 500,
    early_stopping_patience: int = 3,
    run_name: str = None
):
    """Train focused ChemBERTa on targets with sufficient data"""
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize W&B
    wandb.init(
        project="veridica-ai-focused-training",
        name=run_name or f"chemberta-50epochs-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": "ChemBERTa-focused-50epochs",
            "targets": FOCUSED_TARGETS,
            "excluded_targets": EXCLUDED_TARGETS,
            "num_targets": len(FOCUSED_TARGETS),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_length": max_length,
            "dropout": dropout,
            "comparison_model": "Chemprop-50epochs",
            "training_purpose": "Equal epoch comparison with Chemprop GNN"
        }
    )
    
    logger.info("🚀 FOCUSED ChemBERTa TRAINING STARTED")
    logger.info(f"📊 Training targets: {FOCUSED_TARGETS}")
    logger.info(f"🚫 Excluded targets: {EXCLUDED_TARGETS}")
    
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
        
        # Filter to focused targets only
        focused_df = df[['canonical_smiles'] + FOCUSED_TARGETS].copy()
        
        # Remove rows where ALL focused targets are NaN
        focused_df = focused_df.dropna(subset=['canonical_smiles'])
        
        logger.info(f"   📊 Focused dataset: {focused_df.shape}")
        logger.info(f"   🎯 Training on {len(FOCUSED_TARGETS)} targets")
        
        # Data availability analysis
        for target in FOCUSED_TARGETS:
            available = focused_df[target].notna().sum()
            percentage = (available / len(focused_df)) * 100
            logger.info(f"   {target:10s}: {available:4d} samples ({percentage:4.1f}%)")
        
        # Split data
        train_df, temp_df = train_test_split(focused_df, test_size=test_size + val_size, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size + val_size), random_state=42)
        
        logger.info(f"   📊 Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Load tokenizer
        logger.info("🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        
        # Prepare datasets
        logger.info("📦 Preparing datasets...")
        
        def prepare_targets(df):
            targets = {}
            for target in FOCUSED_TARGETS:
                targets[target] = df[target].tolist()
            return targets
        
        train_dataset = FocusedMolecularDataset(
            train_df['canonical_smiles'].tolist(),
            prepare_targets(train_df),
            tokenizer,
            max_length
        )
        
        val_dataset = FocusedMolecularDataset(
            val_df['canonical_smiles'].tolist(),
            prepare_targets(val_df),
            tokenizer,
            max_length
        )
        
        test_dataset = FocusedMolecularDataset(
            test_df['canonical_smiles'].tolist(),
            prepare_targets(test_df),
            tokenizer,
            max_length
        )
        
        # Initialize model
        logger.info("🤖 Initializing model...")
        model = FocusedChemBERTaMultiTaskModel(
            model_name="seyonec/ChemBERTa-zinc-base-v1",
            num_targets=len(FOCUSED_TARGETS),
            dropout=dropout
        )
        
        # Training arguments
        output_dir = f"/vol/models/focused_chemberta_{run_name or 'default'}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=50,
            eval_steps=eval_steps,
            save_steps=save_steps,
            eval_strategy="steps",  # Updated from evaluation_strategy
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_mean_r2",
            greater_is_better=True,
            fp16=True,
            gradient_checkpointing=False,  # Disable to avoid issues
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="wandb"
        )
        
        # Initialize trainer
        logger.info("🏋️ Initializing trainer...")
        trainer = FocusedChemBERTaTrainer(
            target_names=FOCUSED_TARGETS,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer
        )
        
        # Start training
        logger.info("🚀 Starting training...")
        training_result = trainer.train()
        
        # Evaluate on test set
        logger.info("🧪 Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
        
        # Log final results
        logger.info("📊 TRAINING COMPLETED!")
        logger.info("=" * 60)
        
        # Extract R² scores
        r2_scores = {}
        for key, value in test_results.items():
            if key.endswith('_r2'):
                target = key.replace('test_', '').replace('_r2', '')
                r2_scores[target] = value
                logger.info(f"   {target:10s}: R² = {value:.3f}")
        
        # Overall performance
        mean_r2 = test_results.get('test_mean_r2', 0.0)
        logger.info(f"\n   📈 Mean R²: {mean_r2:.3f}")
        
        # Performance breakdown
        excellent = sum(1 for r2 in r2_scores.values() if r2 > 0.6)
        good = sum(1 for r2 in r2_scores.values() if 0.4 < r2 <= 0.6)
        fair = sum(1 for r2 in r2_scores.values() if 0.2 < r2 <= 0.4)
        poor = sum(1 for r2 in r2_scores.values() if r2 <= 0.2)
        
        logger.info(f"\n   🎯 Performance Breakdown:")
        logger.info(f"   🌟 Excellent (R² > 0.6):   {excellent}/{len(FOCUSED_TARGETS)}")
        logger.info(f"   ✅ Good (0.4 < R² ≤ 0.6):  {good}/{len(FOCUSED_TARGETS)}")
        logger.info(f"   ⚠️ Fair (0.2 < R² ≤ 0.4):  {fair}/{len(FOCUSED_TARGETS)}")
        logger.info(f"   ❌ Poor (R² ≤ 0.2):        {poor}/{len(FOCUSED_TARGETS)}")
        
        # Save final model
        final_model_path = Path(output_dir) / "final_model"
        trainer.save_model(str(final_model_path))
        
        # Save results summary
        results_summary = {
            "status": "success",
            "training_targets": FOCUSED_TARGETS,
            "excluded_targets": EXCLUDED_TARGETS,
            "r2_scores": r2_scores,
            "mean_r2": mean_r2,
            "test_results": test_results,
            "train_loss": training_result.training_loss,
            "model_path": str(final_model_path),
            "wandb_run_id": wandb.run.id,
            "performance_breakdown": {
                "excellent": excellent,
                "good": good,
                "fair": fair,
                "poor": poor
            }
        }
        
        # Save to file
        results_file = Path(output_dir) / "results_summary.json"
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"   💾 Results saved to: {results_file}")
        logger.info("=" * 60)
        
        wandb.finish()
        
        return results_summary
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        wandb.finish()
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("🚀 Focused ChemBERTa Multi-Task Training")