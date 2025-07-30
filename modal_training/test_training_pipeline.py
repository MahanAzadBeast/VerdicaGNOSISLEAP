"""
Test Training Pipeline
Validates that both ChemBERTa and Chemprop training pipelines work correctly
"""

import modal
import pandas as pd
import numpy as np
from pathlib import Path

# Modal setup with required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pyarrow>=12.0.0"
])

app = modal.App("test-training-pipeline")

# Import the training modules
datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    timeout=600
)
def validate_dataset():
    """Validate that our oncoprotein dataset is ready for training"""
    
    print("🧪 Validating Oncoprotein Dataset...")
    
    dataset_path = Path("/vol/datasets/oncoprotein_multitask_dataset.csv")
    
    if not dataset_path.exists():
        return {
            "status": "error",
            "message": "Dataset not found. Please run the oncoprotein extraction first."
        }
    
    # Load and validate dataset
    df = pd.read_csv(dataset_path)
    
    # Basic validation
    checks = {
        "dataset_loaded": len(df) > 0,
        "has_smiles_column": 'canonical_smiles' in df.columns,
        "has_target_columns": len([col for col in df.columns if col != 'canonical_smiles']) > 0,
        "no_empty_smiles": df['canonical_smiles'].notna().all(),
        "sufficient_data": len(df) >= 100,  # Minimum for meaningful training
    }
    
    # Target-specific validation
    target_cols = [col for col in df.columns if col != 'canonical_smiles']
    target_stats = {}
    
    for target in target_cols:
        non_null_count = df[target].notna().sum()
        target_stats[target] = {
            "compounds_with_data": non_null_count,
            "coverage_percent": (non_null_count / len(df)) * 100,
            "pIC50_range": [float(df[target].min()), float(df[target].max())] if non_null_count > 0 else [None, None]
        }
    
    # Overall validation
    total_data_points = sum(stats["compounds_with_data"] for stats in target_stats.values())
    
    validation_result = {
        "status": "success" if all(checks.values()) else "warning",
        "dataset_shape": list(df.shape),
        "total_compounds": len(df),
        "total_targets": len(target_cols),
        "total_data_points": total_data_points,
        "checks": checks,
        "target_statistics": target_stats,
        "sample_smiles": df['canonical_smiles'].head(3).tolist(),
        "ready_for_training": all(checks.values()) and total_data_points >= 1000
    }
    
    print(f"✅ Dataset validation completed:")
    print(f"   Shape: {validation_result['dataset_shape']}")
    print(f"   Compounds: {validation_result['total_compounds']}")
    print(f"   Targets: {validation_result['total_targets']}")
    print(f"   Total data points: {validation_result['total_data_points']}")
    print(f"   Ready for training: {validation_result['ready_for_training']}")
    
    return validation_result

@app.function(timeout=1800)  # 30 minutes
def test_chemberta_pipeline():
    """Test ChemBERTa training pipeline with minimal configuration"""
    
    print("🤖 Testing ChemBERTa Pipeline...")
    
    try:
        from train_chemberta import train_chemberta_multitask
        
        # Run with minimal configuration for testing
        result = train_chemberta_multitask.remote(
            dataset_name="oncoprotein_multitask_dataset",
            num_epochs=2,  # Very short for testing
            batch_size=8,  # Small batch
            learning_rate=2e-5,
            test_size=0.3,  # Larger test set for small dataset
            val_size=0.2,
            max_length=256,  # Shorter sequences for speed
            run_name="test-chemberta-pipeline"
        )
        
        print(f"✅ ChemBERTa Test Result: {result['status']}")
        return result
        
    except Exception as e:
        print(f"❌ ChemBERTa Test Failed: {e}")
        return {"status": "error", "error": str(e)}

@app.function(timeout=1800)  # 30 minutes  
def test_chemprop_pipeline():
    """Test Chemprop training pipeline with minimal configuration"""
    
    print("🧪 Testing Chemprop Pipeline...")
    
    try:
        from train_chemprop import train_chemprop_multitask
        
        # Run with minimal configuration for testing
        result = train_chemprop_multitask.remote(
            dataset_name="oncoprotein_multitask_dataset", 
            num_epochs=5,   # Very short for testing
            batch_size=32,  # Reasonable batch for Chemprop
            learning_rate=1e-3,
            test_size=0.3,  # Larger test set for small dataset
            val_size=0.2,
            hidden_size=100,  # Smaller model for speed
            depth=2,
            early_stopping=False,  # Disable for quick test
            run_name="test-chemprop-pipeline"
        )
        
        print(f"✅ Chemprop Test Result: {result['status']}")
        return result
        
    except Exception as e:
        print(f"❌ Chemprop Test Failed: {e}")
        return {"status": "error", "error": str(e)}

@app.local_entrypoint()
def run_full_test():
    """Run complete test suite for both training pipelines"""
    
    print("🧪 Running Full Training Pipeline Test Suite")
    print("=" * 60)
    
    # Step 1: Validate dataset
    print("\n📊 Step 1: Dataset Validation")
    dataset_result = validate_dataset.remote()
    
    if dataset_result["status"] == "error":
        print("❌ Dataset validation failed. Cannot proceed with training tests.")
        return dataset_result
    
    if not dataset_result["ready_for_training"]:
        print("⚠️ Dataset has issues but proceeding with limited tests...")
    
    # Step 2: Test ChemBERTa
    print("\n🤖 Step 2: ChemBERTa Pipeline Test")
    chemberta_result = test_chemberta_pipeline.remote()
    
    # Step 3: Test Chemprop  
    print("\n🧪 Step 3: Chemprop Pipeline Test")
    chemprop_result = test_chemprop_pipeline.remote()
    
    # Summary
    print("\n🎉 Test Suite Completed!")
    print("=" * 60)
    
    summary = {
        "dataset_validation": dataset_result,
        "chemberta_test": chemberta_result,
        "chemprop_test": chemprop_result
    }
    
    # Check overall success
    success_count = 0
    if dataset_result["ready_for_training"]:
        success_count += 1
    if chemberta_result.get("status") == "success":
        success_count += 1  
    if chemprop_result.get("status") == "success":
        success_count += 1
    
    print(f"✅ Success Rate: {success_count}/3 tests passed")
    
    if success_count == 3:
        print("🎉 All systems ready for production training!")
        print("🚀 You can now run:")
        print("   modal run launch_training.py --model both --epochs 50")
    elif success_count >= 2:
        print("⚠️ Most systems working. Check failed tests above.")
    else:
        print("❌ Multiple failures. Please investigate before production training.")
    
    return summary

if __name__ == "__main__":
    run_full_test()