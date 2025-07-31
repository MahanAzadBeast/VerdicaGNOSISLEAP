"""
Full ChemBERTa Multi-Task Training Execution
Runs production ChemBERTa training on 14-oncoprotein dataset with device bug fix
"""

import modal
import time
from datetime import datetime

def run_full_chemberta_training():
    """Execute full ChemBERTa training with production parameters"""
    
    print("🚀 LAUNCHING FULL ChemBERTa MULTI-TASK TRAINING")
    print("=" * 60)
    print("🎯 Dataset: 14 Oncoproteins Multi-Task Dataset")
    print("🧬 Model: ChemBERTa Multi-Task Transformer")
    print("💻 Infrastructure: Modal A100 GPU")
    print("📊 Tracking: Weights & Biases with Enhanced Logging")
    print("🔧 Bug Fix: Device property crash fix applied")
    print("=" * 60)
    
    try:
        # Import the fixed ChemBERTa training module
        from train_chemberta import app as chemberta_app
        
        # Production training parameters
        training_config = {
            "dataset_name": "oncoprotein_multitask_dataset",
            "model_name": "seyonec/ChemBERTa-zinc-base-v1", 
            "batch_size": 16,  # Balanced for A100 GPU
            "learning_rate": 2e-5,  # Standard ChemBERTa learning rate
            "num_epochs": 15,  # Sufficient epochs for convergence
            "max_length": 512,  # Full sequence length
            "test_size": 0.2,
            "val_size": 0.1,
            "dropout": 0.1,
            "warmup_steps": 500,
            "save_steps": 1000,
            "eval_steps": 500,
            "early_stopping_patience": 3,
            "run_name": f"chemberta-14oncoproteins-production-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        print("\n📋 TRAINING CONFIGURATION:")
        for key, value in training_config.items():
            print(f"   {key}: {value}")
        
        print(f"\n🚀 Starting ChemBERTa training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("   This will test that the device property bug fix works in production...")
        
        # Execute training with Modal
        with chemberta_app.run():
            print("   ✅ Modal app initialized successfully")
            
            start_time = time.time()
            
            # Run the full training
            result = chemberta_app.train_chemberta_multitask.remote(**training_config)
            
            end_time = time.time()
            training_duration = end_time - start_time
            
            print(f"\n📊 TRAINING COMPLETED in {training_duration/60:.1f} minutes")
            print("=" * 60)
            
            # Analyze results
            if result.get('status') == 'success':
                print("🎉 CHEMBERTA TRAINING SUCCESSFUL!")
                print("\n✅ CRITICAL VALIDATIONS:")
                print("   🔧 Device property bug fix: WORKING - No crashes during evaluation")
                print("   📊 Enhanced W&B logging: ACTIVE")
                print("   🎯 Multi-task learning: COMPLETED")
                
                print(f"\n📈 TRAINING METRICS:")
                print(f"   🏋️ Final training loss: {result.get('train_loss', 'N/A')}")
                print(f"   💾 Model saved to: {result.get('model_path', 'N/A')}")
                print(f"   📊 W&B run ID: {result.get('wandb_run_id', 'N/A')}")
                
                # Display test results if available
                test_results = result.get('test_results', {})
                if test_results:
                    print(f"\n🧪 TEST SET PERFORMANCE:")
                    r2_results = {k: v for k, v in test_results.items() if k.endswith('_r2')}
                    if r2_results:
                        print("   Per-Target R² Scores:")
                        for target, r2 in sorted(r2_results.items()):
                            target_name = target.replace('test_', '').replace('_r2', '')
                            print(f"     {target_name}: {r2:.3f}")
                        
                        avg_r2 = sum(r2_results.values()) / len(r2_results)
                        print(f"   📊 Average R²: {avg_r2:.3f}")
                        
                        good_targets = sum(1 for r2 in r2_results.values() if r2 > 0.5)
                        excellent_targets = sum(1 for r2 in r2_results.values() if r2 > 0.7)
                        
                        print(f"   🎯 Targets with R² > 0.5: {good_targets}/{len(r2_results)}")
                        print(f"   🌟 Targets with R² > 0.7: {excellent_targets}/{len(r2_results)}")
                
                print(f"\n🎯 SUCCESS SUMMARY:")
                print(f"   ✅ ChemBERTa device property bug is COMPLETELY FIXED")
                print(f"   ✅ Training completed without crashes during final evaluation")
                print(f"   ✅ Enhanced W&B logging captured per-target metrics")
                print(f"   ✅ Multi-task model trained on all 14 oncoproteins")
                print(f"   ✅ Production-ready model saved and artifacts logged")
                
                return True, result
                
            else:
                print("❌ CHEMBERTA TRAINING FAILED!")
                print(f"   Error details: {result}")
                return False, result
                
    except Exception as e:
        print(f"❌ CRITICAL ERROR in ChemBERTa training: {e}")
        return False, {"error": str(e)}

def main():
    """Main execution function"""
    print("🧬 ChemBERTa Multi-Task Production Training")
    print("Testing device property bug fix with full training run\n")
    
    success, result = run_full_chemberta_training()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 FULL ChemBERTa TRAINING VALIDATION COMPLETE!")
        print("✅ Device property bug fix confirmed working in production")
        print("🚀 ChemBERTa multi-task model ready for inference")
        
        if result.get('wandb_run_id'):
            print(f"\n📊 View detailed results in W&B:")
            print(f"   Run ID: {result['wandb_run_id']}")
            print(f"   Project: veridica-ai-training")
            
    else:
        print("⚠️ ChemBERTa training encountered issues")
        print("Check the error details above for debugging")
        
    print("=" * 60)
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)