"""
Extract R² scores from the trained ChemBERTa model (Fixed version)
This will evaluate the model on the test set and calculate R² for each target
"""

import modal
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
from pathlib import Path

# Create Modal app for evaluation
app = modal.App("chemberta-r2-extraction-fixed")

# Use the same image as training
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "rdkit-pypi>=2022.9.5",
    ])
)

# Shared volumes
datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)

@app.function(
    image=image,
    volumes={
        "/vol/datasets": datasets_volume,
        "/vol/models": models_volume,
    },
    gpu="A100",
    memory=16384,
    timeout=1800
)
def extract_chemberta_r2_scores_fixed():
    """Extract R² scores from the trained ChemBERTa model"""
    
    print("🔍 EXTRACTING R² SCORES FROM TRAINED ChemBERTa MODEL")
    print("=" * 60)
    
    try:
        # Import necessary modules
        from transformers import AutoTokenizer, AutoModel
        import sys
        sys.path.append('/vol')
        
        print("📊 Loading dataset...")
        
        # Load the dataset
        dataset_path = Path("/vol/datasets/oncoprotein_multitask_dataset.csv")
        if not dataset_path.exists():
            dataset_path = Path("/vol/datasets/oncoprotein_multitask_dataset.parquet")
            if dataset_path.exists():
                df = pd.read_parquet(dataset_path)
            else:
                return {"error": "Dataset not found"}
        else:
            df = pd.read_csv(dataset_path)
        
        print(f"   ✅ Dataset loaded: {df.shape}")
        print(f"   🎯 Columns: {list(df.columns)}")
        
        # Check for missing values
        print("📋 Checking for missing values...")
        missing_info = df.isnull().sum()
        print(f"   📊 Missing values per column:")
        for col, missing in missing_info.items():
            if missing > 0:
                print(f"     {col}: {missing} missing ({missing/len(df)*100:.1f}%)")
        
        # Identify target columns (exclude SMILES column)
        target_columns = [col for col in df.columns if col not in ['canonical_smiles', 'smiles', 'SMILES']]
        print(f"   🎯 Target columns: {target_columns}")
        
        # Use the complete dataset with NaN handling per target
        # We'll evaluate on the entire dataset like the original training
        print(f"   📊 Using complete dataset: {df.shape}")
        
        # Create the same data split as in training
        from sklearn.model_selection import train_test_split
        
        # For the split, just use rows where we have SMILES
        df_with_smiles = df[df['canonical_smiles'].notna()]
        print(f"   📊 Rows with valid SMILES: {len(df_with_smiles)}")
        
        # Split data (same splits as training)
        train_data, temp_data = train_test_split(df_with_smiles, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.67, random_state=42)  # 0.67 * 0.3 ≈ 0.2
        
        print(f"   📊 Test set size: {len(test_data)}")
        
        # Load tokenizer
        print("🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        
        # Load model
        print("🤖 Loading trained model...")
        model_path = Path("/vol/models/chemberta_oncoprotein_multitask_dataset/final_model")
        
        if not model_path.exists():
            # Try alternative paths
            alt_paths = [
                Path("/vol/models/chemberta_oncoprotein_multitask_dataset"),
                Path("/vol/models"),
            ]
            
            found_path = None
            for alt_path in alt_paths:
                if alt_path.exists():
                    print(f"   🔍 Checking: {alt_path}")
                    print(f"   📁 Contents: {list(alt_path.iterdir())}")
                    model_files = list(alt_path.rglob("*.bin"))
                    if model_files:
                        found_path = model_files[0].parent
                        break
            
            if found_path:
                model_path = found_path
                print(f"   ✅ Found model at: {model_path}")
            else:
                # List all available paths for debugging
                print("   🔍 Available model paths:")
                for path in Path("/vol/models").rglob("*"):
                    if path.is_file():
                        print(f"     {path}")
                return {"error": f"Model not found. Searched paths: {[str(p) for p in [model_path] + alt_paths]}"}
        
        print(f"   📁 Model path contents: {list(model_path.iterdir())}")
        
        # Create model architecture (same as training)
        from transformers import AutoModel
        import torch.nn as nn
        
        class ChemBERTaMultiTaskModel(nn.Module):
            def __init__(self, model_name, num_targets, dropout=0.1):
                super().__init__()
                self.backbone = AutoModel.from_pretrained(model_name)
                hidden_size = self.backbone.config.hidden_size
                
                # Shared layer
                self.shared_layer = nn.Sequential(
                    nn.Linear(hidden_size, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                
                # Task-specific heads
                self.task_heads = nn.ModuleDict({
                    target: nn.Linear(512, 1) for target in target_columns
                })
                
                self.dropout = nn.Dropout(dropout)
            
            @property
            def device(self):
                return next(self.parameters()).device
            
            def forward(self, input_ids, attention_mask, labels=None):
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                
                shared_features = self.shared_layer(pooled_output)
                
                task_outputs = {}
                for target_name, head in self.task_heads.items():
                    task_outputs[target_name] = head(shared_features).squeeze(-1)
                
                # Stack all predictions
                logits = torch.stack([task_outputs[target] for target in target_columns], dim=1)
                
                return {"logits": logits}
        
        # Initialize model
        model = ChemBERTaMultiTaskModel(
            model_name="seyonec/ChemBERTa-zinc-base-v1",
            num_targets=len(target_columns),
            dropout=0.1
        )
        
        # Load trained weights
        model_files = list(model_path.glob("*.bin"))
        if not model_files:
            model_files = list(model_path.glob("*.pth"))
        
        if model_files:
            model_file = model_files[0]
            state_dict = torch.load(model_file, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"   ✅ Model weights loaded from {model_file}")
        else:
            print(f"   ⚠️ No model weight files found in {model_path}")
            print(f"   📁 Available files: {list(model_path.iterdir())}")
            return {"error": "Model weights not found"}
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        print("🧪 Evaluating model on test set...")
        
        # Prepare test data
        test_smiles = test_data['canonical_smiles'].tolist()
        
        # Tokenize SMILES
        all_predictions = []
        batch_size = 16
        
        with torch.no_grad():
            for i in range(0, len(test_smiles), batch_size):
                batch_smiles = test_smiles[i:i+batch_size]
                
                # Tokenize batch
                encoding = tokenizer(
                    batch_smiles,
                    truncation=True,
                    padding='max_length',
                    max_length=256,
                    return_tensors='pt'
                )
                
                # Move to device
                encoding = {k: v.to(device) for k, v in encoding.items()}
                
                # Predict
                outputs = model(**encoding)
                predictions = outputs['logits'].cpu().numpy()
                all_predictions.append(predictions)
        
        # Combine all predictions
        all_predictions = np.vstack(all_predictions)
        
        print("📊 Calculating R² scores for each target...")
        
        # Calculate metrics for each target
        r2_scores = {}
        mse_scores = {}
        mae_scores = {}
        
        for i, target in enumerate(target_columns):
            # Get true values for this target
            y_true = test_data[target].values
            y_pred = all_predictions[:, i]
            
            # Remove rows where true values are NaN
            mask = ~np.isnan(y_true)
            valid_count = mask.sum()
            
            if valid_count > 0:
                y_true_clean = y_true[mask]
                y_pred_clean = y_pred[mask]
                
                r2 = r2_score(y_true_clean, y_pred_clean)
                mse = mean_squared_error(y_true_clean, y_pred_clean)
                mae = mean_absolute_error(y_true_clean, y_pred_clean)
                
                r2_scores[target] = r2
                mse_scores[target] = mse
                mae_scores[target] = mae
                
                print(f"   {target:10s}: R² = {r2:7.3f}, MSE = {mse:7.3f}, MAE = {mae:7.3f} (n={valid_count})")
            else:
                print(f"   {target:10s}: No valid data")
                r2_scores[target] = np.nan
                mse_scores[target] = np.nan
                mae_scores[target] = np.nan
        
        # Calculate average scores
        valid_r2_scores = [score for score in r2_scores.values() if not np.isnan(score)]
        valid_mse_scores = [score for score in mse_scores.values() if not np.isnan(score)]
        valid_mae_scores = [score for score in mae_scores.values() if not np.isnan(score)]
        
        avg_r2 = np.mean(valid_r2_scores) if valid_r2_scores else np.nan
        avg_mse = np.mean(valid_mse_scores) if valid_mse_scores else np.nan
        avg_mae = np.mean(valid_mae_scores) if valid_mae_scores else np.nan
        
        print("\n📈 SUMMARY:")
        print(f"   Average R²:  {avg_r2:.3f}")
        print(f"   Average MSE: {avg_mse:.3f}")
        print(f"   Average MAE: {avg_mae:.3f}")
        print(f"   Valid targets: {len(valid_r2_scores)}/{len(target_columns)}")
        
        # Count performance categories
        excellent_targets = sum(1 for r2 in valid_r2_scores if r2 > 0.7)
        good_targets = sum(1 for r2 in valid_r2_scores if 0.5 < r2 <= 0.7)
        fair_targets = sum(1 for r2 in valid_r2_scores if 0.3 < r2 <= 0.5)
        poor_targets = sum(1 for r2 in valid_r2_scores if r2 <= 0.3)
        
        print(f"\n🎯 PERFORMANCE BREAKDOWN:")
        print(f"   Excellent (R² > 0.7):    {excellent_targets} targets")
        print(f"   Good (0.5 < R² ≤ 0.7):   {good_targets} targets") 
        print(f"   Fair (0.3 < R² ≤ 0.5):   {fair_targets} targets")
        print(f"   Poor (R² ≤ 0.3):         {poor_targets} targets")
        
        # Show best and worst performing targets
        if valid_r2_scores:
            sorted_targets = sorted([(target, r2) for target, r2 in r2_scores.items() if not np.isnan(r2)], 
                                   key=lambda x: x[1], reverse=True)
            
            print(f"\n🏆 TOP PERFORMERS:")
            for target, r2 in sorted_targets[:5]:
                print(f"   {target}: R² = {r2:.3f}")
            
            print(f"\n📉 BOTTOM PERFORMERS:")
            for target, r2 in sorted_targets[-5:]:
                print(f"   {target}: R² = {r2:.3f}")
        
        return {
            "status": "success",
            "r2_scores": r2_scores,
            "mse_scores": mse_scores,
            "mae_scores": mae_scores,
            "average_r2": avg_r2,
            "average_mse": avg_mse,
            "average_mae": avg_mae,
            "test_samples": len(test_data),
            "target_columns": target_columns,
            "performance_breakdown": {
                "excellent": excellent_targets,
                "good": good_targets,
                "fair": fair_targets,
                "poor": poor_targets
            },
            "top_performers": sorted_targets[:5] if valid_r2_scores else [],
            "bottom_performers": sorted_targets[-5:] if valid_r2_scores else []
        }
        
    except Exception as e:
        print(f"❌ Error extracting R² scores: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting R² score extraction (Fixed version)...")