"""
Direct Training Pipeline Fix Validation
Tests the specific fixes without complex Modal app coordination
"""

import modal
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Create test dataset first
def create_minimal_dataset():
    """Create a minimal test dataset"""
    test_data = {
        'canonical_smiles': [
            'CCO', 'C', 'CC', 'CCC', 'CCCC', 'C1=CC=CC=C1', 
            'CCO', 'CC(C)C', 'C1=CC=CC=C1C', 'CCN'
        ],
        'EGFR': [6.5, 5.2, 7.1, 6.8, 5.9, 7.3, 6.4, 5.8, 7.0, 6.2],
        'BRAF': [7.2, 6.1, 6.9, 7.5, 6.3, 7.8, 7.0, 6.5, 7.4, 6.8],
        'HER2': [5.8, 5.9, 6.2, 6.5, 5.7, 6.9, 5.9, 6.1, 6.7, 5.8],
        'MET': [6.9, 6.3, 7.2, 7.1, 6.0, 7.5, 6.8, 6.4, 7.3, 6.6]
    }
    
    df = pd.DataFrame(test_data)
    
    # Save locally first
    local_path = Path('/tmp/test_training_dataset.csv')
    df.to_csv(local_path, index=False)
    print(f"✅ Created test dataset: {df.shape} samples")
    return df, str(local_path)

# Test ChemBERTa directly
def test_chemberta_device_fix():
    """Test ChemBERTa with device property fix"""
    print("\n🧬 Testing ChemBERTa Device Property Fix...")
    
    try:
        from train_chemberta import train_chemberta_multitask, app as chemberta_app
        
        print("   ✅ ChemBERTa module imported successfully")
        
        # Test with minimal parameters on Modal
        with chemberta_app.run():
            print("   🚀 Running minimal ChemBERTa training...")
            
            result = chemberta_app.train_chemberta_multitask.remote(
                dataset_name="oncoprotein_multitask_dataset",  # Use existing dataset
                batch_size=4,  # Very small
                learning_rate=1e-4,
                num_epochs=1,  # Just 1 epoch
                max_length=128,
                test_size=0.3,
                val_size=0.2,
                warmup_steps=5,
                save_steps=100,
                eval_steps=50,
                early_stopping_patience=1,
                run_name="test-device-fix"
            )
            
            print(f"   📊 ChemBERTa result: {result}")
            
            if result.get('status') == 'success':
                print("   ✅ ChemBERTa device property bug fix VALIDATED")
                return True
            else:
                print("   ❌ ChemBERTa still has issues")
                return False
                
    except Exception as e:
        print(f"   ❌ ChemBERTa test failed: {e}")
        return False

# Test Chemprop directly  
def test_chemprop_cli_fix():
    """Test Chemprop with CLI compatibility fix"""
    print("\n⚗️ Testing Chemprop CLI Compatibility Fix...")
    
    try:
        from train_chemprop_simple import train_chemprop_simple, app as chemprop_app
        
        print("   ✅ Chemprop module imported successfully")
        
        # Test with CLI fix on Modal
        with chemprop_app.run():
            print("   🚀 Running Chemprop training...")
            
            result = chemprop_app.train_chemprop_simple.remote()
            
            print(f"   📊 Chemprop result: {result}")
            
            if result.get('status') == 'success':
                print("   ✅ Chemprop CLI compatibility fix VALIDATED")
                return True
            else:
                print("   ❌ Chemprop still has CLI issues")
                return False
                
    except Exception as e:
        print(f"   ❌ Chemprop test failed: {e}")
        return False

def main():
    """Main test execution"""
    print("🚀 VALIDATING TRAINING PIPELINE FIXES")
    print("=" * 50)
    
    # Create test dataset
    print("📊 Creating test dataset...")
    test_df, dataset_path = create_minimal_dataset()
    
    # Test ChemBERTa fix
    chemberta_ok = test_chemberta_device_fix()
    
    # Test Chemprop fix
    chemprop_ok = test_chemprop_cli_fix()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    print(f"🧬 ChemBERTa Device Bug Fix: {'✅ PASSED' if chemberta_ok else '❌ FAILED'}")
    print(f"⚗️ Chemprop CLI Compatibility: {'✅ PASSED' if chemprop_ok else '❌ FAILED'}")
    
    if chemberta_ok and chemprop_ok:
        print("\n🎉 ALL TRAINING PIPELINE FIXES VALIDATED SUCCESSFULLY!")
        print("✅ Both training pipelines work without crashing")
        print("✅ Ready for production training runs")
    else:
        print("\n⚠️ SOME ISSUES REMAIN:")
        if not chemberta_ok:
            print("❌ ChemBERTa device property issue not resolved")
        if not chemprop_ok:
            print("❌ Chemprop CLI compatibility issue not resolved")
    
    print("=" * 50)
    return chemberta_ok and chemprop_ok

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)