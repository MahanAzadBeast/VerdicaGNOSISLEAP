"""
ChemBERTa Multi-Task Inference Integration
Provides inference for the trained focused ChemBERTa model
"""

import modal
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from sklearn.metrics import r2_score
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Optional

# Modal app for inference
app = modal.App("chemberta-inference")

# Use the same image as training
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "safetensors>=0.4.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "rdkit-pypi>=2022.9.5",
    ])
)

# Shared volumes
models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)

# TRAINED TARGET LIST - Only the 10 targets we trained on
TRAINED_TARGETS = [
    'EGFR',     # 0.751 R²
    'HER2',     # 0.583 R²
    'VEGFR2',   # 0.555 R²
    'BRAF',     # 0.595 R²
    'MET',      # 0.502 R²
    'CDK4',     # 0.314 R²
    'CDK6',     # 0.216 R²
    'ALK',      # 0.405 R²
    'MDM2',     # 0.655 R²
    'PI3KCA'    # 0.588 R²
]

class FocusedChemBERTaMultiTaskModel(nn.Module):
    """Recreate the exact same model architecture as training"""
    
    def __init__(self, model_name: str, num_targets: int, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        # Shared feature layer
        self.shared_layer = nn.Linear(hidden_size, 512)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Regression heads for each target (same as training)
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
        
        return {'logits': logits}

@app.function(
    image=image,
    volumes={"/vol/models": models_volume},
    gpu="T4",  # T4 is sufficient for inference
    memory=8192,
    timeout=300
)
def load_trained_chemberta_model():
    """Load the trained focused ChemBERTa model"""
    
    print("🤖 Loading Trained ChemBERTa Model...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        
        # Find the trained model
        model_base_path = Path("/vol/models")
        model_paths = [
            model_base_path / "focused_chemberta_focused-production-accelerate-20241231" / "final_model",
            model_base_path / "focused_chemberta_focused-production-accelerate-20241231",
            model_base_path / "chemberta_focused" / "final_model",
        ]
        
        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = path
                print(f"   ✅ Found model at: {model_path}")
                break
        
        if not model_path:
            # List available models
            available = list(model_base_path.rglob("*"))
            print(f"   📁 Available paths: {available}")
            raise FileNotFoundError("Trained ChemBERTa model not found")
        
        # Load model architecture
        model = FocusedChemBERTaMultiTaskModel(
            model_name="seyonec/ChemBERTa-zinc-base-v1",
            num_targets=len(TRAINED_TARGETS),
            dropout=0.1
        )
        
        # Load trained weights
        safetensor_file = model_path / "model.safetensors"
        if safetensor_file.exists():
            from safetensors.torch import load_file
            state_dict = load_file(safetensor_file)
            model.load_state_dict(state_dict)
            print(f"   ✅ Model weights loaded from SafeTensors")
        else:
            # Try other formats
            model_files = list(model_path.glob("*.bin"))
            if model_files:
                state_dict = torch.load(model_files[0], map_location='cpu')
                model.load_state_dict(state_dict)
                print(f"   ✅ Model weights loaded from {model_files[0]}")
            else:
                raise FileNotFoundError("No model weights file found")
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        print(f"   🎯 Model loaded for {len(TRAINED_TARGETS)} targets")
        print(f"   📊 Targets: {TRAINED_TARGETS}")
        
        return {
            "status": "success",
            "model": model,
            "tokenizer": tokenizer,
            "targets": TRAINED_TARGETS,
            "device": str(device),
            "model_path": str(model_path)
        }
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }

@app.function(
    image=image,
    volumes={"/vol/models": models_volume},
    gpu="T4",
    memory=8192,
    timeout=120
)
def predict_chemberta_ic50(smiles: str):
    """Predict IC50 values for all trained targets using ChemBERTa"""
    
    print(f"🧬 Predicting IC50 for SMILES: {smiles}")
    
    try:
        # Load model
        model_data = load_trained_chemberta_model.local()
        
        if model_data["status"] != "success":
            return {"status": "error", "error": model_data.get("error", "Model loading failed")}
        
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tokenize SMILES
        encoding = tokenizer(
            smiles,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = outputs['logits'].cpu().numpy()
        
        # Convert predictions to IC50 values and create results
        results = {}
        target_info = {}
        
        for i, target in enumerate(TRAINED_TARGETS):
            # Prediction is in pIC50 space, convert to IC50 in µM
            pic50_pred = predictions[0, i]
            ic50_um = 10 ** (6 - pic50_pred)  # Convert from pIC50 to µM
            ic50_nm = ic50_um * 1000  # Convert to nM
            
            # Get performance info for this target (from training results)
            r2_scores = {
                'EGFR': 0.751, 'HER2': 0.583, 'VEGFR2': 0.555, 'BRAF': 0.595,
                'MET': 0.502, 'CDK4': 0.314, 'CDK6': 0.216, 'ALK': 0.405,
                'MDM2': 0.655, 'PI3KCA': 0.588
            }
            
            r2_score = r2_scores.get(target, 0.5)
            
            # Calculate confidence based on R² score and prediction value
            confidence = min(0.95, max(0.3, r2_score * 0.8 + 0.2))
            
            # Determine activity classification
            if ic50_um <= 0.1:
                activity_class = "Very High"
                activity_color = "#00ff00"
            elif ic50_um <= 1.0:
                activity_class = "High"
                activity_color = "#7fff00"
            elif ic50_um <= 10.0:
                activity_class = "Moderate"
                activity_color = "#ffff00"
            elif ic50_um <= 100.0:
                activity_class = "Low"
                activity_color = "#ff7f00"
            else:
                activity_class = "Very Low"
                activity_color = "#ff0000"
            
            results[target] = {
                "ic50_um": float(ic50_um),
                "ic50_nm": float(ic50_nm),
                "pic50": float(pic50_pred),
                "confidence": float(confidence),
                "r2_score": float(r2_score),
                "activity_class": activity_class,
                "activity_color": activity_color
            }
            
            # Target description
            target_descriptions = {
                'EGFR': 'Epidermal Growth Factor Receptor',
                'HER2': 'Human Epidermal Growth Factor Receptor 2',
                'VEGFR2': 'Vascular Endothelial Growth Factor Receptor 2',
                'BRAF': 'B-Raf Proto-Oncogene',
                'MET': 'MET Proto-Oncogene',
                'CDK4': 'Cyclin Dependent Kinase 4',
                'CDK6': 'Cyclin Dependent Kinase 6',
                'ALK': 'Anaplastic Lymphoma Kinase',
                'MDM2': 'MDM2 Proto-Oncogene',
                'PI3KCA': 'Phosphatidylinositol-4,5-Bisphosphate 3-Kinase Catalytic Subunit Alpha'
            }
            
            target_info[target] = {
                "name": target,
                "description": target_descriptions.get(target, f"{target} Protein"),
                "r2_score": float(r2_score)
            }
        
        # Calculate summary statistics
        ic50_values = [results[t]["ic50_um"] for t in TRAINED_TARGETS]
        best_target = min(TRAINED_TARGETS, key=lambda t: results[t]["ic50_um"])
        worst_target = max(TRAINED_TARGETS, key=lambda t: results[t]["ic50_um"])
        
        summary = {
            "smiles": smiles,
            "total_targets": len(TRAINED_TARGETS),
            "best_target": best_target,
            "best_ic50_um": results[best_target]["ic50_um"],
            "worst_target": worst_target,
            "worst_ic50_um": results[worst_target]["ic50_um"],
            "mean_ic50_um": float(np.mean(ic50_values)),
            "median_ic50_um": float(np.median(ic50_values)),
            "highly_active_targets": len([t for t in TRAINED_TARGETS if results[t]["ic50_um"] <= 1.0])
        }
        
        print(f"   ✅ Predictions completed for {len(TRAINED_TARGETS)} targets")
        print(f"   🎯 Best target: {best_target} (IC50: {results[best_target]['ic50_um']:.3f} µM)")
        
        return {
            "status": "success",
            "predictions": results,
            "target_info": target_info,
            "summary": summary,
            "model_info": {
                "model_type": "ChemBERTa Multi-Task",
                "trained_targets": TRAINED_TARGETS,
                "training_r2_mean": 0.516
            }
        }
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }

# Test function
@app.function(
    image=image,
    volumes={"/vol/models": models_volume},
    gpu="T4",
    memory=8192,
    timeout=300
)
def test_chemberta_inference():
    """Test the ChemBERTa inference with sample molecules"""
    
    test_molecules = [
        {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
        {"name": "Imatinib", "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"},
        {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"}
    ]
    
    print("🧪 Testing ChemBERTa Inference...")
    
    results = []
    for mol in test_molecules:
        print(f"\n🔍 Testing {mol['name']}: {mol['smiles']}")
        result = predict_chemberta_ic50.local(mol["smiles"])
        results.append({
            "molecule": mol["name"],
            "smiles": mol["smiles"], 
            "result": result
        })
    
    return {
        "status": "success",
        "test_results": results,
        "message": "ChemBERTa inference testing completed"
    }

if __name__ == "__main__":
    print("🚀 ChemBERTa Inference Module Ready")