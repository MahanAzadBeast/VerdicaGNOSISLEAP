"""
End-to-End Training Pipeline Testing
Tests ChemBERTa and Chemprop fixes with actual Modal training runs
"""

import modal
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# Import the training modules
from train_chemberta import train_chemberta_multitask, app as chemberta_app
from train_chemprop_simple import train_chemprop_simple, app as chemprop_simple_app

# Test Modal app
test_app = modal.App("test-training-end-to-end")

# Shared volumes
datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret")

# Basic image for testing coordination
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas>=2.0.0", 
    "numpy>=1.24.0"
])

@test_app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    secrets=[wandb_secret],
    cpu=2.0,
    memory=4096,
    timeout=300
)
def create_test_dataset():
    """Create a minimal test dataset for validation"""
    
    print("🧪 Creating minimal test dataset...")
    
    # Create test data with valid SMILES and multiple targets
    test_data = {
        'canonical_smiles': [
            'CCO',  # ethanol
            'C',    # methane  
            'CC',   # ethane
            'CCC',  # propane
            'CCCC', # butane
            'C1=CC=CC=C1', # benzene
            'CCO',  # ethanol again
            'CC(C)C', # isobutane
            'C1=CC=CC=C1C', # toluene
            'CCN',  # ethylamine
        ],
        'EGFR': [6.5, 5.2, 7.1, 6.8, 5.9, 7.3, 6.4, 5.8, 7.0, 6.2],
        'BRAF': [7.2, 6.1, 6.9, 7.5, 6.3, 7.8, 7.0, 6.5, 7.4, 6.8],
        'HER2': [5.8, 5.9, 6.2, 6.5, 5.7, 6.9, 5.9, 6.1, 6.7, 5.8],
        'MET': [6.9, 6.3, 7.2, 7.1, 6.0, 7.5, 6.8, 6.4, 7.3, 6.6]
    }
    
    df = pd.DataFrame(test_data)
    
    # Save test dataset
    test_dataset_path = Path("/vol/datasets/test_minimal_dataset.csv")
    df.to_csv(test_dataset_path, index=False)
    
    print(f"✅ Test dataset created: {df.shape} - {test_dataset_path}")
    print("Sample data:")
    print(df.head())
    
    return {
        "status": "success",
        "dataset_path": str(test_dataset_path),
        "shape": df.shape,
        "targets": ['EGFR', 'BRAF', 'HER2', 'MET']
    }

@test_app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    timeout=60
)
def test_chemberta_training():
    """Test ChemBERTa training with minimal parameters"""
    
    print("🚀 Starting ChemBERTa training test...")
    
    try:
        # Use the actual ChemBERTa training function with minimal parameters
        result = chemberta_app.train_chemberta_multitask.remote(
            dataset_name="test_minimal_dataset",
            batch_size=2,  # Very small batch
            learning_rate=1e-4,
            num_epochs=1,  # Just 1 epoch for testing
            max_length=128,  # Shorter sequences
            test_size=0.3,
            val_size=0.2,
            warmup_steps=5,
            save_steps=100,
            eval_steps=50,
            early_stopping_patience=1,
            run_name="test-chemberta-device-fix"
        )
        
        print(f"✅ ChemBERTa training result: {result}")
        
        if result.get('status') == 'success':
            return {
                "status": "success", 
                "message": "ChemBERTa training completed without device property bug crash",
                "details": result
            }
        else:
            return {
                "status": "error",
                "message": "ChemBERTa training failed",
                "details": result
            }
            
    except Exception as e:
        print(f"❌ ChemBERTa training failed: {e}")
        return {
            "status": "error",
            "message": f"ChemBERTa training crashed: {str(e)}"
        }

@test_app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    timeout=300
)
def test_chemprop_training():
    """Test Chemprop training with CLI compatibility fix"""
    
    print("🚀 Starting Chemprop training test...")
    
    try:
        # Use the simple Chemprop training function
        result = chemprop_simple_app.train_chemprop_simple.remote()
        
        print(f"✅ Chemprop training result: {result}")
        
        if result.get('status') == 'success':
            return {
                "status": "success",
                "message": "Chemprop training completed without CLI compatibility issues", 
                "details": result
            }
        else:
            return {
                "status": "error", 
                "message": "Chemprop training failed",
                "details": result
            }
            
    except Exception as e:
        print(f"❌ Chemprop training failed: {e}")
        return {
            "status": "error",
            "message": f"Chemprop training crashed: {str(e)}"
        }

@test_app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    timeout=60
)
def summarize_results(dataset_result, chemberta_result, chemprop_result):
    """Summarize all test results"""
    
    print("\n" + "="*60)
    print("🧪 END-TO-END TRAINING PIPELINE TEST SUMMARY")
    print("="*60)
    
    # Dataset creation
    dataset_ok = dataset_result.get('status') == 'success'
    print(f"📊 Test Dataset Creation: {'✅ SUCCESS' if dataset_ok else '❌ FAILED'}")
    if dataset_ok:
        print(f"   Created dataset: {dataset_result['shape']} with targets: {dataset_result['targets']}")
    
    # ChemBERTa testing
    chemberta_ok = chemberta_result.get('status') == 'success'
    print(f"🧬 ChemBERTa Training (Device Bug Fix): {'✅ SUCCESS' if chemberta_ok else '❌ FAILED'}")
    print(f"   Message: {chemberta_result.get('message', 'No message')}")
    
    # Chemprop testing  
    chemprop_ok = chemprop_result.get('status') == 'success'
    print(f"⚗️ Chemprop Training (CLI Fix): {'✅ SUCCESS' if chemprop_ok else '❌ FAILED'}")
    print(f"   Message: {chemprop_result.get('message', 'No message')}")
    
    # Overall result
    all_passed = dataset_ok and chemberta_ok and chemprop_ok
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TRAINING PIPELINE FIXES VALIDATED SUCCESSFULLY!")
        print("✅ ChemBERTa device property bug is fixed")
        print("✅ Chemprop CLI compatibility issues are resolved")
        print("✅ Both training pipelines work end-to-end without crashing")
    else:
        print("⚠️ SOME ISSUES FOUND:")
        if not dataset_ok:
            print("❌ Test dataset creation failed")
        if not chemberta_ok:
            print("❌ ChemBERTa training still has issues")
        if not chemprop_ok:
            print("❌ Chemprop training still has issues")
    
    print("="*60)
    
    return {
        "status": "success" if all_passed else "partial",
        "dataset_creation": dataset_ok,
        "chemberta_training": chemberta_ok,
        "chemprop_training": chemprop_ok,
        "all_passed": all_passed,
        "summary": "All fixes validated" if all_passed else "Some issues remain"
    }

if __name__ == "__main__":
    print("🚀 Starting End-to-End Training Pipeline Testing...")
    print("Testing ChemBERTa device property fix and Chemprop CLI compatibility fix")