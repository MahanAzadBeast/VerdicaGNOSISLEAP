#!/usr/bin/env python3
"""
Launch 50-Epoch ChemBERTa Training to Match Chemprop
"""

import sys
import os
sys.path.append('/app/modal_training')

from train_chemberta_focused import app

def launch_50_epoch_chemberta_training():
    """Launch ChemBERTa training with 50 epochs to match Chemprop"""
    
    print("🧬 LAUNCHING 50-EPOCH CHEMBERTA TRAINING")
    print("=" * 60)
    print("Training Configuration:")
    print("• Epochs: 50 (matching Chemprop)")
    print("• Batch Size: 16")
    print("• Learning Rate: 2e-5")
    print("• GPU: A100")
    print("• Max Length: 512")
    print("• Targets: 10 focused oncoproteins")
    print("• Expected Duration: ~3 hours")
    print("=" * 60)
    
    with app.run():
        print("🚀 Starting ChemBERTa 50-epoch training...")
        
        # Launch training with 50 epochs
        from train_chemberta_focused import train_focused_chemberta
        
        result = train_focused_chemberta.remote(
            batch_size=16,
            learning_rate=2e-5,
            num_epochs=50,  # Match Chemprop training
            max_length=512,
            dropout=0.1,
            warmup_steps=500,
            save_steps=1000,
            eval_steps=500
        )
        
        print("📊 Training Results:")
        print(f"Status: {result.get('status', 'Unknown')}")
        print(f"Final Performance: {result.get('final_performance', {})}")
        print(f"W&B Run ID: {result.get('wandb_run_id', 'Not available')}")
        
        if result.get('status') == 'success':
            print("✅ ChemBERTa 50-epoch training completed successfully!")
            print(f"📁 Model saved at: {result.get('model_path')}")
            print(f"📈 Training metrics logged to W&B")
            
            # Show performance summary
            performance = result.get('final_performance', {})
            if performance:
                print("\n🎯 Final Performance Summary:")
                print(f"• Mean R²: {performance.get('mean_r2', 'N/A'):.3f}")
                print(f"• Mean RMSE: {performance.get('mean_rmse', 'N/A'):.3f}")
                print(f"• Mean MAE: {performance.get('mean_mae', 'N/A'):.3f}")
                print(f"• Targets Trained: {performance.get('targets_trained', 'N/A')}")
        else:
            print("❌ Training failed. Check logs for details.")
            if result.get('error'):
                print(f"Error: {result['error']}")
        
        return result

if __name__ == "__main__":
    result = launch_50_epoch_chemberta_training()
    
    print(f"\n🏁 Training Complete!")
    print("=" * 60)
    print("Next Steps:")
    print("1. Check W&B dashboard for detailed metrics")
    print("2. Compare 50-epoch ChemBERTa vs 50-epoch Chemprop performance")
    print("3. Update backend integration with new model")
    print("4. Test model comparison in UI")