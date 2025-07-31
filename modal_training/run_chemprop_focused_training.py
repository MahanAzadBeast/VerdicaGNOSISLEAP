#!/usr/bin/env python3
"""
Execute Focused Chemprop Training
Full training on the same 10 oncoproteins as ChemBERTa
"""

import modal
from train_chemprop_focused import app, train_focused_chemprop
from datetime import datetime

def main():
    print("🚀 STARTING FOCUSED CHEMPROP MULTI-TASK TRAINING")
    print("=" * 60)
    print("🎯 Training on the same 10 oncoproteins as ChemBERTa:")
    print("   Rich targets (>500 samples): EGFR, HER2, VEGFR2, BRAF, CDK6, MDM2")
    print("   Moderate targets (100-500): MET, CDK4, ALK, PI3KCA")
    print()
    print("🚫 Excluded sparse/no-data targets:")
    print("   STAT3 (84 samples), CTNNB1 (6 samples), RRM2 (0 samples), MYC (0 samples)")
    print()
    print("📋 Training Configuration:")
    print("   • Architecture: Graph Neural Network (Message Passing)")
    print("   • Epochs: 30 (more than ChemBERTa for convergence)")
    print("   • Batch size: 32 (larger for GNN)")
    print("   • Learning rate: 1e-3")
    print("   • Hidden size: 256")
    print("   • Depth: 4 layers")
    print("   • GPU: A100")
    print("   • W&B tracking: Enabled with ChemBERTa comparison")
    print("=" * 60)
    
    # Run training
    run_name = f"chemprop-focused-production-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        with app.run():
            print(f"\n🚀 Starting Chemprop training run: {run_name}")
            print("⏳ Expected duration: 60-90 minutes")
            
            result = train_focused_chemprop.remote(
                epochs=30,          # More epochs for GNN convergence
                batch_size=32,      # Larger batch size for GNN
                learning_rate=1e-3, # Standard Chemprop learning rate
                hidden_size=256,    # Good balance of capacity
                depth=4,            # 4 message passing layers
                dropout=0.1,        # Same as ChemBERTa
                ffn_num_layers=3,   # Feed-forward layers
                test_size=0.2,      # Same split as ChemBERTa
                val_size=0.1,       # Same split as ChemBERTa
                run_name=run_name
            )
            
            print(f"\n📊 CHEMPROP TRAINING COMPLETED!")
            print("=" * 60)
            
            if result.get('status') == 'success':
                print("✅ FOCUSED CHEMPROP TRAINING SUCCESSFUL!")
                
                # Display R² scores and compare with ChemBERTa
                print("\n📈 CHEMPROP R² SCORES BY TARGET:")
                r2_scores = result.get('r2_scores', {})
                
                # ChemBERTa R² scores for comparison
                chemberta_r2 = {
                    'EGFR': 0.751, 'MDM2': 0.655, 'BRAF': 0.595, 'PI3KCA': 0.588,
                    'HER2': 0.583, 'VEGFR2': 0.555, 'MET': 0.502, 'ALK': 0.405,
                    'CDK4': 0.314, 'CDK6': 0.216
                }
                
                print("   Target      | Chemprop R² | ChemBERTa R² | Difference | Better Model")
                print("   " + "-" * 70)
                
                chemprop_wins = 0
                chemberta_wins = 0
                ties = 0
                
                for target in sorted(r2_scores.keys()):
                    chemprop_r2 = r2_scores.get(target, 0.0)
                    bert_r2 = chemberta_r2.get(target, 0.0)
                    diff = chemprop_r2 - bert_r2
                    
                    if diff > 0.05:
                        better = "🟢 Chemprop"
                        chemprop_wins += 1
                    elif diff < -0.05:
                        better = "🔴 ChemBERTa"
                        chemberta_wins += 1
                    else:
                        better = "🟡 Tie"
                        ties += 1
                    
                    print(f"   {target:11s} | {chemprop_r2:11.3f} | {bert_r2:12.3f} | {diff:+10.3f} | {better}")
                
                # Overall performance
                mean_r2 = result.get('mean_r2', 0.0)
                chemberta_mean = 0.516  # From previous training
                
                print(f"\n📊 OVERALL PERFORMANCE COMPARISON:")
                print(f"   Chemprop Mean R²:  {mean_r2:.3f}")
                print(f"   ChemBERTa Mean R²: {chemberta_mean:.3f}")
                print(f"   Difference: {mean_r2 - chemberta_mean:+.3f}")
                
                # Performance breakdown
                breakdown = result.get('performance_breakdown', {})
                print(f"\n🎯 CHEMPROP PERFORMANCE BREAKDOWN:")
                print(f"   🌟 Excellent (R² > 0.6):   {breakdown.get('excellent', 0)}/10")
                print(f"   ✅ Good (0.4 < R² ≤ 0.6):  {breakdown.get('good', 0)}/10")
                print(f"   ⚠️ Fair (0.2 < R² ≤ 0.4):  {breakdown.get('fair', 0)}/10")
                print(f"   ❌ Poor (R² ≤ 0.2):        {breakdown.get('poor', 0)}/10")
                
                print(f"\n🏆 HEAD-TO-HEAD COMPARISON:")
                print(f"   Chemprop Better: {chemprop_wins}")
                print(f"   ChemBERTa Better: {chemberta_wins}")
                print(f"   Ties: {ties}")
                
                if chemprop_wins > chemberta_wins:
                    print("   🏆 Winner: Chemprop GNN")
                elif chemberta_wins > chemprop_wins:
                    print("   🏆 Winner: ChemBERTa Transformer")
                else:
                    print("   🤝 Result: Tied Performance")
                
                # Additional info
                print(f"\n📋 ADDITIONAL INFO:")
                print(f"   Model path: {result.get('model_path', 'N/A')}")
                print(f"   W&B run ID: {result.get('wandb_run_id', 'N/A')}")
                print(f"   Mean MAE: {result.get('mean_mae', 'N/A'):.3f}")
                
                print("\n🎉 FOCUSED CHEMPROP TRAINING COMPLETED SUCCESSFULLY!")
                print("✅ GNN model trained on 10 oncoproteins")
                print("✅ W&B logging operational with ChemBERTa comparison")
                print("✅ Ready for integration into Ligand Activity Predictor")
                
                return True
                
            else:
                print("❌ CHEMPROP TRAINING FAILED!")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                return False
                
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    print("\n" + "=" * 60)
    if success:
        print("🏁 FOCUSED CHEMPROP TRAINING VALIDATION COMPLETE")
        print("✅ GNN model ready for real multi-task predictions")
        print("🎯 Next: Integrate trained Chemprop model into UI")
    else:
        print("⚠️ Training encountered issues - check logs above")
    print("=" * 60)
    
    exit(0 if success else 1)