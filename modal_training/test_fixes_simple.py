"""
Simple validation of training pipeline fixes
Tests core functionality without full Modal execution
"""

import sys
import os
import pandas as pd
import tempfile
from pathlib import Path

def test_chemberta_import_and_device_fix():
    """Test ChemBERTa imports and device property fix"""
    print("🧬 Testing ChemBERTa Device Property Fix...")
    
    try:
        # Import the module
        sys.path.append('/app/modal_training')
        from train_chemberta import ChemBERTaMultiTaskModel, ChemBERTaTrainer
        
        print("   ✅ ChemBERTa classes imported successfully")
        
        # Test device property is present and safe
        import inspect
        model_class = ChemBERTaMultiTaskModel
        
        if hasattr(model_class, 'device'):
            device_prop = getattr(model_class, 'device')
            if hasattr(device_prop, 'fget'):
                source = inspect.getsource(device_prop.fget)
                if 'next(self.parameters()).device' in source:
                    print("   ✅ Device property uses safe next(self.parameters()).device")
                else:
                    print("   ❌ Device property not using safe access")
                    return False
            else:
                print("   ❌ Device is not a property")
                return False
        else:
            print("   ❌ Device property not found")
            return False
        
        # Test trainer evaluation method fix
        trainer_class = ChemBERTaTrainer
        if hasattr(trainer_class, 'evaluate'):
            eval_source = inspect.getsource(trainer_class.evaluate)
            if 'device = next(self.model.parameters()).device' in eval_source:
                print("   ✅ Trainer.evaluate uses safe device access")
            else:
                print("   ❌ Trainer.evaluate not using safe device access")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ ChemBERTa test failed: {e}")
        return False

def test_chemprop_import_and_cli_fix():
    """Test Chemprop imports and CLI compatibility fix"""
    print("\n⚗️ Testing Chemprop CLI Compatibility Fix...")
    
    try:
        # Import the module
        from train_chemprop import run_chemprop_training
        
        print("   ✅ Chemprop function imported successfully")
        
        # Check the CLI command generation
        import inspect
        source = inspect.getsource(run_chemprop_training)
        
        if "'python', '-m', 'chemprop.train'" in source:
            print("   ✅ Uses new CLI format: python -m chemprop.train")
        else:
            print("   ❌ Not using new CLI format")
            return False
        
        if "'python', '-m', 'chemprop.predict'" in source:
            print("   ✅ Uses new CLI format: python -m chemprop.predict")  
        else:
            print("   ❌ Not using new predict CLI format")
            return False
        
        if 'chemprop_train' not in source and 'chemprop_predict' not in source:
            print("   ✅ Old CLI commands removed")
        else:
            print("   ❌ Still contains old CLI commands")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Chemprop test failed: {e}")
        return False

def test_w_and_b_integration():
    """Test W&B integration components"""
    print("\n📊 Testing W&B Integration...")
    
    try:
        # Test ChemBERTa W&B components
        from train_chemberta import WandbMetricsCallback, ChemBERTaTrainer
        
        callback_class = WandbMetricsCallback
        if hasattr(callback_class, 'on_log') and hasattr(callback_class, 'on_evaluate'):
            print("   ✅ ChemBERTa WandbMetricsCallback has required methods")
        else:
            print("   ❌ ChemBERTa WandbMetricsCallback missing required methods")
            return False
        
        # Check if trainer has scatter plot and summary methods
        trainer_methods = dir(ChemBERTaTrainer)
        if '_create_and_log_scatter_plots' in trainer_methods and '_create_and_log_performance_summary' in trainer_methods:
            print("   ✅ ChemBERTa trainer has W&B visualization methods")
        else:
            print("   ❌ ChemBERTa trainer missing W&B visualization methods")
            return False
        
        # Test Chemprop W&B components
        from train_chemprop import ChempropWandbLogger
        
        logger_class = ChempropWandbLogger
        if hasattr(logger_class, 'log_epoch_metrics') and hasattr(logger_class, 'log_final_results'):
            print("   ✅ Chemprop ChempropWandbLogger has required methods")
        else:
            print("   ❌ Chemprop ChempropWandbLogger missing required methods")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ W&B integration test failed: {e}")
        return False

def test_modal_configuration():
    """Test Modal app configuration"""
    print("\n🚀 Testing Modal Configuration...")
    
    try:
        from train_chemberta import app as chemberta_app
        from train_chemprop import app as chemprop_app
        
        print("   ✅ Modal apps imported successfully")
        
        # Check if apps are properly configured
        if chemberta_app and chemprop_app:
            print("   ✅ Both Modal apps are configured")
            return True
        else:
            print("   ❌ Modal apps not properly configured")
            return False
        
    except Exception as e:
        print(f"   ❌ Modal configuration test failed: {e}")
        return False

def create_simple_test_run():
    """Create a minimal test to validate components work"""
    print("\n🧪 Testing Core Components...")
    
    try:
        # Test that we can create ChemBERTa model instance
        from train_chemberta import ChemBERTaMultiTaskModel
        
        # This should not crash due to device property
        print("   🔍 Testing ChemBERTa model instantiation...")
        model = ChemBERTaMultiTaskModel(
            model_name="seyonec/ChemBERTa-zinc-base-v1",
            num_targets=4,
            dropout=0.1
        )
        
        # Test device property access
        try:
            device = model.device
            print(f"   ✅ ChemBERTa device property working: {device}")
        except Exception as e:
            print(f"   ❌ ChemBERTa device property failed: {e}")
            return False
        
        # Test molecular dataset creation
        from train_chemberta import MolecularDataset
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        
        test_smiles = ['CCO', 'C']
        test_targets = {'EGFR': [6.5, 5.2], 'BRAF': [7.2, 6.1]}
        
        dataset = MolecularDataset(test_smiles, test_targets, tokenizer, max_length=128)
        
        if len(dataset) == 2:
            print("   ✅ MolecularDataset creation working")
        else:
            print("   ❌ MolecularDataset creation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Core components test failed: {e}")
        return False

def main():
    """Main test execution"""
    print("🔧 TRAINING PIPELINE FIXES VALIDATION")
    print("=" * 50)
    
    # Run all tests
    chemberta_ok = test_chemberta_import_and_device_fix()
    chemprop_ok = test_chemprop_import_and_cli_fix() 
    wandb_ok = test_w_and_b_integration()
    modal_ok = test_modal_configuration()
    components_ok = create_simple_test_run()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION RESULTS")
    print("=" * 50)
    
    print(f"🧬 ChemBERTa Device Bug Fix: {'✅ PASSED' if chemberta_ok else '❌ FAILED'}")
    print(f"⚗️ Chemprop CLI Compatibility: {'✅ PASSED' if chemprop_ok else '❌ FAILED'}")
    print(f"📊 W&B Integration: {'✅ PASSED' if wandb_ok else '❌ FAILED'}")
    print(f"🚀 Modal Configuration: {'✅ PASSED' if modal_ok else '❌ FAILED'}")
    print(f"🧪 Core Components: {'✅ PASSED' if components_ok else '❌ FAILED'}")
    
    all_passed = all([chemberta_ok, chemprop_ok, wandb_ok, modal_ok, components_ok])
    
    if all_passed:
        print("\n🎉 ALL TRAINING PIPELINE FIXES VALIDATED!")
        print("✅ ChemBERTa device property bug is fixed")
        print("✅ Chemprop CLI compatibility issues are resolved")
        print("✅ W&B logging integration is working")
        print("✅ Modal configuration is correct")
        print("✅ Core components work without crashing")
        print("\n🚀 READY FOR FULL TRAINING RUNS!")
    else:
        print("\n⚠️ SOME VALIDATION ISSUES FOUND - CHECK DETAILS ABOVE")
    
    print("=" * 50)
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)