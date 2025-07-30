"""
ChemBERTa Multi-Task Training Pipeline on Modal with W&B Integration
Trains transformer models for molecular property prediction using Hugging Face
"""

import modal
import os
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
# Note: wandb imported inside functions due to Modal environment

# Modal setup with comprehensive ML libraries
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "torch>=2.0.0",
    "transformers[torch]>=4.30.0", 
    "accelerate>=0.26.0",
    "datasets>=2.10.0",
    "tokenizers>=0.13.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "rdkit>=2023.3.1",
    "wandb>=0.15.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "pyarrow>=12.0.0"
])

app = modal.App("chemberta-training")

# Persistent volumes
datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)

# W&B secret (already configured)
wandb_secret = modal.Secret.from_name("wandb-secret")

class MolecularDataset(Dataset):
    """Dataset for molecular SMILES and multi-target pIC50 values"""
    
    def __init__(self, smiles_list: List[str], targets: Dict[str, List[float]], tokenizer, max_length: int = 512):
        self.smiles = smiles_list
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_names = list(targets.keys())
        
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
        
        # Prepare multi-target labels
        labels = []
        mask = []  # Mask for missing values
        
        for target_name in self.target_names:
            value = self.targets[target_name][idx]
            if pd.isna(value):
                labels.append(0.0)  # Placeholder
                mask.append(0.0)    # Masked (not used in loss)
            else:
                labels.append(float(value))
                mask.append(1.0)    # Not masked (used in loss)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float),
            'label_mask': torch.tensor(mask, dtype=torch.float)
        }

class ChemBERTaMultiTaskModel(nn.Module):
    """Multi-task ChemBERTa model for molecular property prediction"""
    
    def __init__(self, model_name: str, num_targets: int, dropout: float = 0.1):
        super().__init__()
        self.num_targets = num_targets
        
        # Load ChemBERTa backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Multi-task regression heads
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.regression_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_targets)
        ])
        
        # Optional: shared feature layer
        self.shared_layer = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
    
    @property
    def device(self):
        """Return the device of the model"""
        return next(self.parameters()).device
        
    def forward(self, input_ids, attention_mask, labels=None, label_mask=None):
        # Get backbone embeddings
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Shared feature processing
        shared_features = self.activation(self.shared_layer(pooled_output))
        shared_features = self.dropout(shared_features)
        
        # Multi-task predictions
        predictions = []
        for head in self.regression_heads:
            pred = head(shared_features).squeeze(-1)  # Remove last dimension
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=1)  # [batch_size, num_targets]
        
        loss = None
        if labels is not None and label_mask is not None:
            # Masked MSE loss - only compute loss where label_mask = 1
            mse_loss = nn.MSELoss(reduction='none')
            losses = mse_loss(predictions, labels)  # [batch_size, num_targets]
            
            # Apply mask and compute mean
            masked_losses = losses * label_mask
            loss = masked_losses.sum() / (label_mask.sum() + 1e-8)  # Avoid division by zero
        
        return {
            'loss': loss,
            'logits': predictions
        }

class ChemBERTaTrainer(Trainer):
    """Custom trainer for multi-task ChemBERTa with enhanced W&B logging"""
    
    def __init__(self, target_names: List[str], **kwargs):
        super().__init__(**kwargs)
        self.target_names = target_names
        self.step_count = 0
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        label_mask = inputs.pop("label_mask")
        
        outputs = model(**inputs, labels=labels, label_mask=label_mask)
        loss = outputs["loss"]
        
        # Log per-target losses during training
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_per_target_losses(outputs['logits'], labels, label_mask)
        
        return (loss, outputs) if return_outputs else loss
    
    def _log_per_target_losses(self, predictions, labels, masks):
        """Log individual target losses to W&B"""
        import wandb
        
        # Compute per-target losses
        mse_loss = nn.MSELoss(reduction='none')
        losses = mse_loss(predictions, labels)  # [batch_size, num_targets]
        
        target_losses = {}
        for i, target_name in enumerate(self.target_names):
            target_mask = masks[:, i] == 1
            if target_mask.sum() > 0:
                target_loss = losses[target_mask, i].mean().item()
                target_losses[f"train_loss/{target_name}"] = target_loss
        
        if target_losses:
            wandb.log(target_losses, step=self.state.global_step)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """Enhanced evaluation with per-target metrics logging to W&B"""
        
        # Run standard evaluation first
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Custom per-target metrics calculation
        if eval_dataset is not None:
            import wandb
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            predictions, labels, masks = [], [], []
            
            self.model.eval()
            with torch.no_grad():
                for batch in eval_dataloader:
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    
                    predictions.append(outputs['logits'].cpu().numpy())
                    labels.append(batch['labels'].cpu().numpy())
                    masks.append(batch['label_mask'].cpu().numpy())
            
            predictions = np.vstack(predictions)
            labels = np.vstack(labels)
            masks = np.vstack(masks)
            
            # Calculate and log per-target metrics
            target_metrics = {}
            target_data_for_plots = {}
            
            for i, target_name in enumerate(self.target_names):
                target_mask = masks[:, i] == 1
                if target_mask.sum() > 5:  # Need at least 5 samples for meaningful metrics
                    target_pred = predictions[target_mask, i]
                    target_true = labels[target_mask, i]
                    
                    # Calculate metrics
                    r2 = r2_score(target_true, target_pred)
                    mse = mean_squared_error(target_true, target_pred)
                    mae = mean_absolute_error(target_true, target_pred)
                    rmse = np.sqrt(mse)
                    
                    # Add to results
                    target_metrics[f"{metric_key_prefix}_{target_name}_r2"] = r2
                    target_metrics[f"{metric_key_prefix}_{target_name}_mse"] = mse
                    target_metrics[f"{metric_key_prefix}_{target_name}_mae"] = mae
                    target_metrics[f"{metric_key_prefix}_{target_name}_rmse"] = rmse
                    target_metrics[f"{metric_key_prefix}_{target_name}_samples"] = int(target_mask.sum())
                    
                    # Store data for scatter plots
                    target_data_for_plots[target_name] = {
                        'true': target_true,
                        'pred': target_pred,
                        'r2': r2,
                        'rmse': rmse,
                        'samples': int(target_mask.sum())
                    }
            
            # Log metrics to W&B
            wandb.log(target_metrics, step=self.state.global_step)
            
            # Create and log scatter plots for top targets
            self._create_and_log_scatter_plots(target_data_for_plots, metric_key_prefix)
            
            # Create and log performance summary
            self._create_and_log_performance_summary(target_metrics, metric_key_prefix)
            
            # Update eval_results
            eval_results.update(target_metrics)
        
        return eval_results
    
    def _create_and_log_scatter_plots(self, target_data, prefix):
        """Create scatter plots for predictions vs true values"""
        import matplotlib.pyplot as plt
        import wandb
        
        # Sort targets by number of samples (descending)
        sorted_targets = sorted(target_data.keys(), 
                              key=lambda x: target_data[x]['samples'], 
                              reverse=True)
        
        # Plot top 8 targets with most data
        n_plots = min(8, len(sorted_targets))
        if n_plots == 0:
            return
            
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel() if n_plots > 1 else [axes]
        
        for i, target in enumerate(sorted_targets[:n_plots]):
            data = target_data[target]
            
            ax = axes[i] if n_plots > 1 else axes
            ax.scatter(data['true'], data['pred'], alpha=0.6, s=30)
            
            # Add perfect prediction line
            min_val, max_val = min(data['true']), max(data['true'])
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Labels and title
            ax.set_xlabel(f'{target} True pIC50')
            ax.set_ylabel(f'{target} Predicted pIC50')
            ax.set_title(f'{target}\nR² = {data["r2"]:.3f}, RMSE = {data["rmse"]:.3f}\n(n = {data["samples"]})')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Multi-Task ChemBERTa: {prefix.title()} Predictions per Target', fontsize=16)
        plt.tight_layout()
        
        # Log to W&B
        wandb.log({f"{prefix}_predictions_scatter": wandb.Image(fig)}, step=self.state.global_step)
        plt.close(fig)
    
    def _create_and_log_performance_summary(self, metrics, prefix):
        """Create performance summary plots"""
        import matplotlib.pyplot as plt
        import wandb
        
        # Extract R² scores for all targets
        r2_data = []
        rmse_data = []
        samples_data = []
        target_names = []
        
        for key, value in metrics.items():
            if key.endswith('_r2'):
                target = key.replace(f'{prefix}_', '').replace('_r2', '')
                r2_data.append(value)
                target_names.append(target)
                
                # Get corresponding RMSE and samples
                rmse_key = f"{prefix}_{target}_rmse"
                samples_key = f"{prefix}_{target}_samples"
                rmse_data.append(metrics.get(rmse_key, 0))
                samples_data.append(metrics.get(samples_key, 0))
        
        if not r2_data:
            return
        
        # Create summary plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # R² scores
        bars1 = ax1.barh(target_names, r2_data, color='skyblue')
        ax1.set_xlabel('R² Score')
        ax1.set_title(f'{prefix.title()} R² by Target')
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='R² = 0.5')
        ax1.axvline(x=0.7, color='green', linestyle='--', alpha=0.7, label='R² = 0.7')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add R² values on bars
        for i, (bar, r2) in enumerate(zip(bars1, r2_data)):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{r2:.3f}', va='center', fontsize=9)
        
        # RMSE scores
        bars2 = ax2.barh(target_names, rmse_data, color='lightcoral')
        ax2.set_xlabel('RMSE')
        ax2.set_title(f'{prefix.title()} RMSE by Target')
        ax2.grid(True, alpha=0.3)
        
        # Sample sizes
        bars3 = ax3.barh(target_names, samples_data, color='lightgreen')
        ax3.set_xlabel('Number of Samples')
        ax3.set_title(f'{prefix.title()} Sample Size by Target')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to W&B
        wandb.log({f"{prefix}_performance_summary": wandb.Image(fig)}, step=self.state.global_step)
        plt.close(fig)
        
        # Also log overall statistics
        overall_stats = {
            f"{prefix}_mean_r2": np.mean(r2_data),
            f"{prefix}_median_r2": np.median(r2_data),
            f"{prefix}_min_r2": np.min(r2_data),
            f"{prefix}_max_r2": np.max(r2_data),
            f"{prefix}_mean_rmse": np.mean(rmse_data),
            f"{prefix}_total_samples": sum(samples_data),
            f"{prefix}_targets_with_data": len(r2_data)
        }
        wandb.log(overall_stats, step=self.state.global_step)

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
def train_chemberta_multitask(
    dataset_name: str = "oncoprotein_multitask_dataset",
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1", 
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 10,
    max_length: int = 512,
    test_size: float = 0.2,
    val_size: float = 0.1,
    dropout: float = 0.1,
    warmup_steps: int = 500,
    save_steps: int = 1000,
    eval_steps: int = 500,
    early_stopping_patience: int = 3,
    run_name: Optional[str] = None
):
    """
    Train ChemBERTa multi-task model with W&B logging
    """
    
    # Import wandb inside function for Modal environment
    import wandb
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize W&B
    wandb.init(
        project="veridica-ai-training",
        group="chemberta",
        name=run_name or f"chemberta-{dataset_name}",
        config={
            "model_name": model_name,
            "dataset_name": dataset_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_length": max_length,
            "test_size": test_size,
            "val_size": val_size,
            "dropout": dropout,
            "warmup_steps": warmup_steps,
            "early_stopping_patience": early_stopping_patience
        },
        tags=["chemberta", "multi-task", "molecular-properties"]
    )
    
    logger.info("🚀 Starting ChemBERTa Multi-Task Training")
    logger.info(f"📊 Dataset: {dataset_name}")
    logger.info(f"🤖 Model: {model_name}")
    
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
    
    # Initialize tokenizer
    logger.info("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    def prepare_targets(dataframe):
        targets = {}
        for col in target_cols:
            targets[col] = dataframe[col].tolist()
        return targets
    
    train_dataset = MolecularDataset(
        train_df[smiles_col].tolist(),
        prepare_targets(train_df),
        tokenizer,
        max_length
    )
    
    val_dataset = MolecularDataset(
        val_df[smiles_col].tolist(),
        prepare_targets(val_df),
        tokenizer,
        max_length
    )
    
    test_dataset = MolecularDataset(
        test_df[smiles_col].tolist(),
        prepare_targets(test_df),
        tokenizer,
        max_length
    )
    
    # Initialize model
    logger.info("🤖 Loading ChemBERTa model...")
    model = ChemBERTaMultiTaskModel(
        model_name=model_name,
        num_targets=len(target_cols),
        dropout=dropout
    )
    
    # Training arguments compatible with newer transformers
    output_dir = f"/vol/models/chemberta_{dataset_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=100,
        eval_strategy="steps",  # Updated from evaluation_strategy
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        dataloader_num_workers=2,  # Reduced for stability
        fp16=True,  # Mixed precision for A100
        save_total_limit=2,
        remove_unused_columns=False  # Keep all columns for custom compute_loss
    )
    
    # Initialize trainer
    trainer = ChemBERTaTrainer(
        target_names=target_cols,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )
    
    # Train model
    logger.info("🎓 Starting training...")
    train_result = trainer.train()
    
    # Evaluate on test set
    logger.info("📊 Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    
    # Log final metrics
    wandb.log({
        "final_train_loss": train_result.training_loss,
        **test_results
    })
    
    # Save model
    model_save_path = Path(output_dir) / "final_model"
    trainer.save_model(model_save_path)
    
    # Save model as W&B artifact
    model_artifact = wandb.Artifact(
        name=f"chemberta-{dataset_name}-model",
        type="model",
        description=f"Trained ChemBERTa model on {dataset_name}"
    )
    model_artifact.add_dir(str(model_save_path))
    wandb.log_artifact(model_artifact)
    
    # Save predictions for analysis
    predictions = trainer.predict(test_dataset)
    pred_df = test_df.copy()
    
    for i, target in enumerate(target_cols):
        pred_df[f"{target}_predicted"] = predictions.predictions[:, i]
    
    pred_save_path = Path(output_dir) / "test_predictions.csv"
    pred_df.to_csv(pred_save_path, index=False)
    
    # Log predictions as artifact
    pred_artifact = wandb.Artifact(
        name=f"chemberta-{dataset_name}-predictions",
        type="predictions", 
        description=f"Test set predictions from ChemBERTa model"
    )
    pred_artifact.add_file(str(pred_save_path))
    wandb.log_artifact(pred_artifact)
    
    logger.info("✅ ChemBERTa training completed!")
    
    # Finish W&B run
    wandb.finish()
    
    return {
        "status": "success",
        "model_path": str(model_save_path),
        "train_loss": float(train_result.training_loss),
        "test_results": {k: float(v) for k, v in test_results.items() if isinstance(v, (int, float))},
        "wandb_run_id": wandb.run.id if wandb.run else None
    }

@app.function(
    image=image,
    volumes={"/vol/models": models_volume},
    secrets=[wandb_secret],
    cpu=2.0,
    memory=8192,
    timeout=1800
)
def load_and_predict_chemberta(
    model_path: str,
    smiles_list: List[str],
    dataset_name: str = "oncoprotein_multitask_dataset"
):
    """
    Load trained ChemBERTa model and make predictions
    """
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = ChemBERTaMultiTaskModel.from_pretrained(model_path)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for smiles in smiles_list:
            # Tokenize
            encoding = tokenizer(
                smiles,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            # Predict
            outputs = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            
            predictions.append(outputs['logits'].numpy())
    
    return np.vstack(predictions)

if __name__ == "__main__":
    # Example usage
    pass