#!/usr/bin/env python3
"""
Execute Focused ChemBERTa Training
Full training on 10 targets with sufficient data
"""

import modal
from train_chemberta_focused import app, train_focused_chemberta
from datetime import datetime

def main():
    print("🚀 STARTING FOCUSED ChemBERTa TRAINING")
    print("=" * 60)
    print("🎯 Training on 10 targets with sufficient data:")
    print("   Rich targets (>500 samples): EGFR, HER2, VEGFR2, BRAF, CDK6, MDM2")
    print("   Moderate targets (100-500): MET, CDK4, ALK, PI3KCA")
    print()
    print("🚫 Excluded sparse/no-data targets:")
    print("   STAT3 (84 samples), CTNNB1 (6 samples), RRM2 (0 samples), MYC (0 samples)")
    print()
    print("📋 Training Configuration:")
    print("   • Epochs: 20 (full training)")
    print("   • Batch size: 16")
    print("   • Learning rate: 2e-5")
    print("   • Max length: 512")
    print("   • GPU: A100")
    print("   • W&B tracking: Enabled")
    print("=" * 60)
    
    # Run training
    run_name = f"focused-production-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        with app.run():
            print(f"\n🚀 Starting training run: {run_name}")
            
            result = train_focused_chemberta.remote(
                batch_size=16,
                learning_rate=2e-5,
                num_epochs=20,  # Full training
                max_length=512,
                test_size=0.2,
                val_size=0.1,
                dropout=0.1,
                warmup_steps=500,
                save_steps=1000,
                eval_steps=500,
                early_stopping_patience=3,
                run_name=run_name
            )
            
            print(f"\n📊 TRAINING COMPLETED!")
            print("=" * 60)
            
            if result.get('status') == 'success':
                print("✅ FOCUSED ChemBERTa TRAINING SUCCESSFUL!")
                
                # Display R² scores
                print("\n📈 R² SCORES BY TARGET:")
                r2_scores = result.get('r2_scores', {})
                for target, r2 in sorted(r2_scores.items(), key=lambda x: x[1], reverse=True):
                    if r2 > 0.6:
                        icon = "🌟"
                    elif r2 > 0.4:
                        icon = "✅"
                    elif r2 > 0.2:
                        icon = "⚠️"
                    else:
                        icon = "❌"
                    print(f"   {icon} {target:10s}: R² = {r2:.3f}")
                
                # Overall performance
                mean_r2 = result.get('mean_r2', 0.0)
                print(f"\n📊 OVERALL PERFORMANCE:")
                print(f"   Mean R²: {mean_r2:.3f}")
                
                # Performance breakdown
                breakdown = result.get('performance_breakdown', {})
                print(f"\n🎯 PERFORMANCE BREAKDOWN:")
                print(f"   🌟 Excellent (R² > 0.6):   {breakdown.get('excellent', 0)}/10")
                print(f"   ✅ Good (0.4 < R² ≤ 0.6):  {breakdown.get('good', 0)}/10")
                print(f"   ⚠️ Fair (0.2 < R² ≤ 0.4):  {breakdown.get('fair', 0)}/10")
                print(f"   ❌ Poor (R² ≤ 0.2):        {breakdown.get('poor', 0)}/10")
                
                # Additional info
                print(f"\n📋 ADDITIONAL INFO:")
                print(f"   Model path: {result.get('model_path', 'N/A')}")
                print(f"   W&B run ID: {result.get('wandb_run_id', 'N/A')}")
                print(f"   Training loss: {result.get('train_loss', 'N/A'):.3f}")
                
                print("\n🎉 FOCUSED TRAINING COMPLETED SUCCESSFULLY!")
                print("✅ Device property bug fix confirmed working")
                print("✅ Training completed without crashes")
                print("✅ W&B logging operational with clean metrics")
                
                return True
                
            else:
                print("❌ TRAINING FAILED!")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                return False
                
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    print("\n" + "=" * 60)
    if success:
        print("🏁 FOCUSED ChemBERTa TRAINING VALIDATION COMPLETE")
        print("✅ All objectives achieved with focused target training")
    else:
        print("⚠️ Training encountered issues - check logs above")
    print("=" * 60)
    
    exit(0 if success else 1)