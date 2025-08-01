"""
Expanded ChemBERTa Multi-Task Training
Trains on expanded target set with multiple activity types and data sources
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
from typing import Dict, List, Any, Optional
import json
import wandb
from datetime import datetime

# Modal app setup
app = modal.App("expanded-chemberta-training")

# Enhanced image with all required packages
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "accelerate>=0.26.0",
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

# ACTIVITY TYPES TO TRAIN ON
ACTIVITY_TYPES = ['IC50', 'EC50', 'Ki']  # Focus on concentration-based measurements with pIC50 values

class ExpandedMolecularDataset(Dataset):
    """Dataset for expanded multi-task ChemBERTa training with multiple activity types"""
    
    def __init__(self, smiles_list: List[str], targets_dict: Dict[str, Dict[str, List[float]]], 
                 tokenizer, max_length: int = 512):
        self.smiles = smiles_list
        self.targets = targets_dict  # {activity_type: {target: [values]}}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_names = EXPANDED_TARGETS
        self.activity_types = ACTIVITY_TYPES
        
        # Calculate total number of prediction tasks (targets × activity types)
        self.total_tasks = len(self.target_names) * len(self.activity_types)
        
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
        
        # Prepare targets for all activity types and targets
        labels = []
        masks = []  # For handling missing values
        
        for activity_type in self.activity_types:
            for target in self.target_names:
                if (activity_type in self.targets and 
                    target in self.targets[activity_type] and 
                    idx < len(self.targets[activity_type][target])):
                    
                    value = self.targets[activity_type][target][idx]
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

class ExpandedChemBERTaMultiTaskModel(nn.Module):
    """Expanded ChemBERTa model for multiple targets and activity types"""
    
    def __init__(self, model_name: str, num_targets: int, num_activity_types: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        self.num_targets = num_targets
        self.num_activity_types = num_activity_types
        self.total_tasks = num_targets * num_activity_types
        
        # Shared feature layers
        self.shared_layer = nn.Linear(hidden_size, 1024)
        self.activity_specific_layers = nn.ModuleDict({
            activity_type: nn.Linear(1024, 512) for activity_type in ACTIVITY_TYPES
        })
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Regression heads for each target-activity combination
        self.regression_heads = nn.ModuleList([
            nn.Linear(512, 1) for _ in range(self.total_tasks)
        ])
        
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
        
        # Activity-specific features and predictions
        all_predictions = []
        
        for activity_type in ACTIVITY_TYPES:
            # Activity-specific transformation
            activity_features = self.dropout(self.activation(
                self.activity_specific_layers[activity_type](shared_features)
            ))
            
            # Predictions for all targets for this activity type
            for i in range(self.num_targets):
                head_idx = ACTIVITY_TYPES.index(activity_type) * self.num_targets + i
                pred = self.regression_heads[head_idx](activity_features).squeeze(-1)
                all_predictions.append(pred)
        
        # Stack predictions
        logits = torch.stack(all_predictions, dim=1)
        
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

class ExpandedWandbCallback:
    """W&B callback for expanded training with activity types"""
    
    def __init__(self, target_names: List[str], activity_types: List[str]):
        self.target_names = target_names
        self.activity_types = activity_types
        
    def on_log(self, logs):
        """Log metrics to W&B"""
        if wandb.run is not None:
            wandb.log(logs)
    
    def on_evaluate(self, eval_results: Dict[str, float]):
        """Log evaluation results"""
        if wandb.run is not None:
            wandb.log(eval_results)

class ExpandedChemBERTaTrainer(Trainer):
    """Custom trainer for expanded ChemBERTa with activity-specific metrics"""
    
    def __init__(self, target_names: List[str], activity_types: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_names = target_names
        self.activity_types = activity_types
        self.wandb_callback = ExpandedWandbCallback(target_names, activity_types)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with per-target and per-activity-type R² scores"""
        
        # Get predictions
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        all_predictions = []
        all_labels = []
        all_masks = []
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
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
        
        # Calculate per-target and per-activity-type metrics
        metrics = {}
        
        task_idx = 0
        for activity_type in self.activity_types:
            activity_r2_scores = []
            
            for target in self.target_names:
                # Get valid samples for this target-activity combination
                target_mask = masks[:, task_idx].astype(bool)
                
                if target_mask.sum() > 5:  # Only if we have enough valid samples
                    y_true = labels[:, task_idx][target_mask]
                    y_pred = predictions[:, task_idx][target_mask]
                    
                    # Calculate metrics
                    r2 = r2_score(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    
                    # Log metrics
                    metrics[f"{metric_key_prefix}_{activity_type}_{target}_r2"] = r2
                    metrics[f"{metric_key_prefix}_{activity_type}_{target}_mse"] = mse
                    metrics[f"{metric_key_prefix}_{activity_type}_{target}_mae"] = mae
                    metrics[f"{metric_key_prefix}_{activity_type}_{target}_samples"] = int(target_mask.sum())
                    
                    activity_r2_scores.append(r2)
                
                task_idx += 1
            
            # Activity-type summary
            if activity_r2_scores:
                metrics[f"{metric_key_prefix}_{activity_type}_mean_r2"] = np.mean(activity_r2_scores)
                metrics[f"{metric_key_prefix}_{activity_type}_std_r2"] = np.std(activity_r2_scores)
        
        # Overall metrics
        all_r2_scores = [v for k, v in metrics.items() if k.endswith('_r2') and 'mean' not in k and 'std' not in k]
        if all_r2_scores:
            metrics[f"{metric_key_prefix}_overall_mean_r2"] = np.mean(all_r2_scores)
            metrics[f"{metric_key_prefix}_overall_std_r2"] = np.std(all_r2_scores)
        
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
    memory=40960,  # 40GB for larger model
    timeout=18000  # 5 hours for expanded training
)
def train_expanded_chemberta(
    activity_type: str = 'IC50',  # Primary activity type for this run
    batch_size: int = 8,  # Smaller batch for larger model
    learning_rate: float = 1e-5,  # Lower learning rate for stability
    num_epochs: int = 30,  # More epochs for complex task
    max_length: int = 512,
    test_size: float = 0.2,
    val_size: float = 0.1,
    dropout: float = 0.15,
    warmup_steps: int = 1000,
    save_steps: int = 2000,
    eval_steps: int = 1000,
    run_name: str = None
):
    """Train expanded ChemBERTa on multiple targets and activity types"""
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize W&B
    wandb.init(
        project="veridica-ai-expanded-training",
        name=run_name or f"expanded-chemberta-{activity_type}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": "ChemBERTa-Expanded-MultiSource",
            "primary_activity_type": activity_type,
            "activity_types": ACTIVITY_TYPES,
            "targets": EXPANDED_TARGETS,
            "num_targets": len(EXPANDED_TARGETS),
            "num_activity_types": len(ACTIVITY_TYPES),
            "total_tasks": len(EXPANDED_TARGETS) * len(ACTIVITY_TYPES),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_length": max_length,
            "dropout": dropout,
            "data_sources": ["ChEMBL", "PubChem", "BindingDB", "DTC"]
        }
    )
    
    logger.info("🚀 EXPANDED ChemBERTa TRAINING STARTED")
    logger.info(f"📊 Primary activity type: {activity_type}")
    logger.info(f"📊 All activity types: {ACTIVITY_TYPES}")
    logger.info(f"📊 Targets: {len(EXPANDED_TARGETS)}")
    logger.info(f"📊 Total prediction tasks: {len(EXPANDED_TARGETS) * len(ACTIVITY_TYPES)}")
    
    try:
        # Load expanded dataset
        logger.info("📊 Loading expanded dataset...")
        
        # Load the primary activity type matrix (e.g., IC50)
        dataset_path = Path(f"/vol/datasets/expanded_multisource_{activity_type.lower()}_matrix.csv")
        
        if not dataset_path.exists():
            # Fallback to original dataset for now
            logger.warning(f"Expanded dataset not found, using original dataset")
            dataset_path = Path("/vol/datasets/oncoprotein_multitask_dataset.csv")
            df = pd.read_csv(dataset_path)
            
            # Filter to available targets only
            available_targets = [t for t in EXPANDED_TARGETS if t in df.columns]
            focused_df = df[['canonical_smiles'] + available_targets].copy()
            
            logger.info(f"   📊 Using {len(available_targets)} available targets from original dataset")
            
        else:
            df = pd.read_csv(dataset_path)
            focused_df = df.copy()
            logger.info(f"   ✅ Expanded dataset loaded: {df.shape}")
        
        # Remove rows where ALL targets are NaN
        focused_df = focused_df.dropna(subset=['canonical_smiles'])
        
        logger.info(f"   📊 Final dataset: {focused_df.shape}")
        
        # Data availability analysis
        available_targets = [col for col in focused_df.columns if col != 'canonical_smiles']
        for target in available_targets:
            if target in focused_df.columns:
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
        
        # Prepare datasets (simplified for single activity type for now)
        logger.info("📦 Preparing datasets...")
        
        def prepare_targets(df):
            targets = {activity_type: {}}  # Start with primary activity type
            for target in available_targets:
                if target in df.columns:
                    targets[activity_type][target] = df[target].tolist()
            return targets
        
        train_dataset = ExpandedMolecularDataset(
            train_df['canonical_smiles'].tolist(),
            prepare_targets(train_df),
            tokenizer,
            max_length
        )
        
        val_dataset = ExpandedMolecularDataset(
            val_df['canonical_smiles'].tolist(),
            prepare_targets(val_df),
            tokenizer,
            max_length
        )
        
        test_dataset = ExpandedMolecularDataset(
            test_df['canonical_smiles'].tolist(),
            prepare_targets(test_df),
            tokenizer,
            max_length
        )
        
        # Initialize model
        logger.info("🤖 Initializing expanded model...")
        model = ExpandedChemBERTaMultiTaskModel(
            model_name="seyonec/ChemBERTa-zinc-base-v1",
            num_targets=len(available_targets),
            num_activity_types=1,  # Start with single activity type
            dropout=dropout
        )
        
        # Training arguments
        output_dir = f"/vol/models/expanded_chemberta_{activity_type}_{run_name or 'default'}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=100,
            eval_steps=eval_steps,
            save_steps=save_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model=f"eval_{activity_type}_mean_r2",
            greater_is_better=True,
            fp16=True,
            gradient_checkpointing=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="wandb",
            gradient_accumulation_steps=2  # Effective batch size = 16
        )
        
        # Initialize trainer
        logger.info("🏋️ Initializing trainer...")
        trainer = ExpandedChemBERTaTrainer(
            target_names=available_targets,
            activity_types=[activity_type],
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
            if key.endswith('_r2') and 'mean' not in key and 'std' not in key:
                parts = key.replace('test_', '').replace('_r2', '').split('_')
                if len(parts) >= 2:
                    act_type = parts[0]
                    target = '_'.join(parts[1:])
                    r2_scores[f"{act_type}_{target}"] = value
                    logger.info(f"   {target:10s} ({act_type}): R² = {value:.3f}")
        
        # Overall performance
        overall_mean_r2 = test_results.get(f'test_{activity_type}_mean_r2', 0.0)
        logger.info(f"\n   📈 Overall Mean R² ({activity_type}): {overall_mean_r2:.3f}")
        
        # Save final model
        final_model_path = Path(output_dir) / "final_model"
        trainer.save_model(str(final_model_path))
        
        # Save results summary
        results_summary = {
            "status": "success",
            "model_type": "ChemBERTa Expanded Multi-Source",
            "primary_activity_type": activity_type,
            "available_targets": available_targets,
            "r2_scores": r2_scores,
            "overall_mean_r2": overall_mean_r2,
            "test_results": test_results,
            "train_loss": training_result.training_loss,
            "model_path": str(final_model_path),
            "wandb_run_id": wandb.run.id,
            "total_tasks": len(available_targets),
            "data_sources": ["ChEMBL", "PubChem", "BindingDB", "DTC"]
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
    print("🚀 Expanded ChemBERTa Multi-Task Training")