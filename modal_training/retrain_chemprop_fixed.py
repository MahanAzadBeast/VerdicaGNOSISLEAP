#!/usr/bin/env python3
"""
Quick Retrain Chemprop with Proper Checkpoint Saving
Fixed training script that ensures inference-compatible model saving
"""

import modal
from pathlib import Path
import subprocess
import pandas as pd
import tempfile
from datetime import datetime

# Modal app setup
app = modal.App("chemprop-retrain-fixed")

# Enhanced image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "chemprop>=2.2.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "rdkit-pypi>=2022.9.5",
        "wandb>=0.16.0"
    ])
)

datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)

FOCUSED_TARGETS = [
    'EGFR', 'HER2', 'VEGFR2', 'BRAF', 'MET', 
    'CDK4', 'CDK6', 'ALK', 'MDM2', 'PI3KCA'
]

@app.function(
    image=image,
    volumes={
        "/vol/datasets": datasets_volume,
        "/vol/models": models_volume,
    },
    gpu="A100",
    memory=32768,
    timeout=7200  # 2 hours max
)
def retrain_chemprop_with_proper_saving(
    epochs: int = 15,  # Reduced for quick fix
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    hidden_size: int = 512,
    depth: int = 5
):
    """Retrain Chemprop with proper checkpoint saving for inference"""
    
    print("🔄 QUICK RETRAIN: Chemprop with Proper Checkpoint Saving")
    print("=" * 60)
    
    # Load and prepare data
    dataset_path = Path("/vol/datasets/oncoprotein_multitask_dataset.csv")
    if not dataset_path.exists():
        return {"status": "error", "error": "Dataset not found"}
    
    df = pd.read_csv(dataset_path)
    print(f"📊 Dataset loaded: {df.shape}")
    
    # Filter to focused targets and valid SMILES
    focused_df = df[['canonical_smiles'] + FOCUSED_TARGETS].copy()
    focused_df = focused_df.dropna(subset=['canonical_smiles'])
    
    print(f"📈 Focused dataset: {focused_df.shape}")
    
    # Create temporary directory for training
    temp_dir = Path(tempfile.mkdtemp())
    data_file = temp_dir / "training_data.csv"
    
    # Save training data
    focused_df.to_csv(data_file, index=False)
    print(f"💾 Training data saved: {data_file}")
    
    # Create model output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_output_dir = Path("/vol/models") / f"chemprop_fixed_{timestamp}"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Model output directory: {model_output_dir}")
    
    # Enhanced Chemprop training command with explicit checkpoint saving
    training_cmd = [
        'chemprop', 'train',
        '--data-path', str(data_file),
        '--task-type', 'regression',
        '--save-dir', str(model_output_dir),
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--init-lr', str(learning_rate),
        '--max-lr', str(learning_rate * 10),
        '--final-lr', str(learning_rate * 0.1),
        '--message-hidden-dim', str(hidden_size),
        '--depth', str(depth),
        '--dropout', '0.15',
        '--ffn-num-layers', '3',
        '--num-workers', '4',
        '--split-sizes', '0.8', '0.1', '0.1',
        '--patience', '10',
        '--save-smiles-splits',  # Save splits for debugging
        '--ensemble-size', '1'   # Single model for simplicity
    ]
    
    print(f"🚀 Starting training with {epochs} epochs...")
    print(f"⏰ Expected duration: {epochs * 2} minutes")
    
    try:
        # Run training
        result = subprocess.run(training_cmd, capture_output=True, text=True, timeout=7200)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            
            # List all created files
            created_files = list(model_output_dir.rglob("*"))
            print(f"📁 Created {len(created_files)} files:")
            
            model_files = []
            for f in created_files:
                if f.is_file():
                    size_mb = f.stat().st_size / 1024 / 1024
                    rel_path = f.relative_to(model_output_dir)
                    print(f"   {rel_path}: {size_mb:.2f} MB")
                    
                    if f.suffix in ['.pt', '.pth', '.ckpt']:
                        model_files.append(str(f))
            
            print(f"🧠 Model checkpoint files: {len(model_files)}")
            
            # Test inference immediately
            print(f"\n🧪 TESTING INFERENCE WITH NEW MODEL")
            test_success = test_inference_with_new_model(model_output_dir)
            
            if test_success:
                print("🎉 SUCCESS: Model training and inference both working!")
                return {
                    "status": "success",
                    "model_directory": str(model_output_dir),
                    "training_epochs": epochs,
                    "model_files": model_files,
                    "files_created": len(created_files),
                    "inference_tested": True,
                    "ready_for_production": True
                }
            else:
                print("⚠️ Training succeeded but inference failed")
                return {
                    "status": "partial_success",
                    "model_directory": str(model_output_dir),
                    "training_epochs": epochs,
                    "model_files": model_files,
                    "files_created": len(created_files),
                    "inference_tested": False,
                    "issue": "inference_still_failing"
                }
        
        else:
            print(f"❌ Training failed with return code: {result.returncode}")
            print(f"Error: {result.stderr[:500]}")
            
            return {
                "status": "training_failed",
                "error": result.stderr,
                "return_code": result.returncode
            }
    
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out")
        return {"status": "timeout", "error": "Training exceeded time limit"}
    
    except Exception as e:
        print(f"❌ Training error: {e}")
        return {"status": "error", "error": str(e)}

def test_inference_with_new_model(model_dir: Path) -> bool:
    """Test inference with the newly trained model"""
    
    print("🧪 Testing inference with new model...")
    
    try:
        # Create test input
        temp_dir = Path(tempfile.mkdtemp())
        test_input = temp_dir / "test.csv"
        test_output = temp_dir / "predictions.csv"
        
        # Test molecules
        test_df = pd.DataFrame({
            "smiles": [
                "CCO",  # Ethanol
                "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            ]
        })
        test_df.to_csv(test_input, index=False)
        
        # Try inference
        predict_cmd = [
            'chemprop', 'predict',
            '--test-path', str(test_input),
            '--checkpoint-dir', str(model_dir),
            '--preds-path', str(test_output)
        ]
        
        result = subprocess.run(predict_cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and test_output.exists():
            pred_df = pd.read_csv(test_output)
            print(f"✅ Inference successful: {pred_df.shape}")
            print(f"✅ Columns: {list(pred_df.columns)}")
            
            if len(pred_df) > 0:
                print("📊 Sample predictions:")
                print(pred_df.head().to_string())
                return True
        
        else:
            print(f"❌ Inference failed: {result.stderr[:200]}")
            return False
    
    except Exception as e:
        print(f"❌ Inference test error: {e}")
        return False

if __name__ == "__main__":
    print("🔄 CHEMPROP QUICK RETRAIN WITH INFERENCE FIX")
    print("=" * 50)
    
    with app.run():
        result = retrain_chemprop_with_proper_saving.remote(
            epochs=15,  # Quick training for testing
            batch_size=64,
            learning_rate=5e-4
        )
        
        print(f"\n📊 RETRAIN RESULTS:")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            print("🎉 SUCCESS: Model retrained and inference working!")
            print(f"📁 Model directory: {result['model_directory']}")
            print(f"🧠 Model files: {len(result['model_files'])}")
            print(f"📊 Files created: {result['files_created']}")
            print(f"✅ Inference tested: {result['inference_tested']}")
            print(f"🚀 Ready for production: {result['ready_for_production']}")
            
        elif result['status'] == 'partial_success':
            print("⚠️ Training succeeded but inference needs work")
            print(f"📁 Model directory: {result['model_directory']}")
            
        else:
            print(f"❌ Retrain failed: {result.get('error')}")
        
        print("=" * 50)