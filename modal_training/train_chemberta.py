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
    """Custom trainer for multi-task ChemBERTa with W&B logging"""
    
    def __init__(self, target_names: List[str], **kwargs):
        super().__init__(**kwargs)
        self.target_names = target_names
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        label_mask = inputs.pop("label_mask")
        
        outputs = model(**inputs, labels=labels, label_mask=label_mask)
        loss = outputs["loss"]
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Custom metrics calculation
        if eval_dataset is not None:
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
            
            # Calculate per-target metrics
            target_metrics = {}
            for i, target_name in enumerate(self.target_names):
                target_mask = masks[:, i] == 1
                if target_mask.sum() > 0:
                    target_pred = predictions[target_mask, i]
                    target_true = labels[target_mask, i]
                    
                    r2 = r2_score(target_true, target_pred)
                    mse = mean_squared_error(target_true, target_pred)
                    mae = mean_absolute_error(target_true, target_pred)
                    
                    target_metrics[f"{metric_key_prefix}_{target_name}_r2"] = r2
                    target_metrics[f"{metric_key_prefix}_{target_name}_mse"] = mse
                    target_metrics[f"{metric_key_prefix}_{target_name}_mae"] = mae
            
            eval_results.update(target_metrics)
        
        return eval_results

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