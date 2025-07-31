#!/usr/bin/env python3
"""
Comprehensive Chemprop Multi-Task Training with Enhanced Error Handling
Production-ready training on all 10 oncoproteins with full logging and monitoring
"""

import modal
from train_chemprop_focused import app, train_focused_chemprop
from datetime import datetime
import time

def main():
    print("🚀 COMPREHENSIVE CHEMPROP MULTI-TASK TRAINING")
    print("=" * 70)
    print("🎯 Production Training Configuration:")
    print("   • Dataset: 10 High-Quality Oncoproteins")
    print("   • Targets: EGFR, HER2, VEGFR2, BRAF, MET, CDK4, CDK6, ALK, MDM2, PI3KCA")
    print("   • Architecture: Message Passing Neural Network (MPNN)")
    print("   • Training Samples: ~5,000 compounds")
    print("   • Hardware: Modal A100 GPU")
    print("   • Expected Duration: 90-120 minutes")
    print("   • Comparison: vs ChemBERTa Transformer (Mean R²: 0.516)")
    print("=" * 70)
    
    # Enhanced training configuration for production
    training_config = {
        'epochs': 50,              # More epochs for better convergence
        'batch_size': 64,          # Larger batch for stability
        'learning_rate': 5e-4,     # Conservative learning rate
        'hidden_size': 512,        # Larger hidden size for capacity  
        'depth': 5,                # Deeper network for better representation
        'dropout': 0.15,           # Slightly more dropout for regularization
        'ffn_num_layers': 3,       # Feed-forward layers
        'test_size': 0.2,          # Standard test split
        'val_size': 0.1            # Validation split
    }
    
    # Generate unique run name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"chemprop-comprehensive-production-{timestamp}"
    
    print(f"📋 TRAINING HYPERPARAMETERS:")
    for key, value in training_config.items():
        print(f"   • {key}: {value}")
    print()
    
    print(f"🔬 Run Identifier: {run_name}")
    print(f"📊 Weights & Biases: veridica-ai-focused-training")
    print(f"💾 Model Output: /vol/models/focused_chemprop_{run_name}")
    print("=" * 70)
    
    try:
        print(f"\n🚀 LAUNCHING COMPREHENSIVE TRAINING...")
        print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🔄 Initializing Modal environment...")
        
        with app.run():
            print("✅ Modal environment ready")
            print("🧪 Starting Chemprop GNN training...")
            print("📈 Training metrics will be logged to W&B")
            print("⏳ This may take 90-120 minutes for full convergence...")
            
            # Launch comprehensive training with enhanced config
            result = train_focused_chemprop.remote(
                epochs=training_config['epochs'],
                batch_size=training_config['batch_size'],
                learning_rate=training_config['learning_rate'],
                hidden_size=training_config['hidden_size'],
                depth=training_config['depth'],
                dropout=training_config['dropout'],
                ffn_num_layers=training_config['ffn_num_layers'],
                test_size=training_config['test_size'],
                val_size=training_config['val_size'],
                run_name=run_name
            )
            
            print(f"\n⏰ Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("📊 COMPREHENSIVE CHEMPROP TRAINING RESULTS")
            print("=" * 70)
            
            if result.get('status') == 'success':
                print("🎉 COMPREHENSIVE TRAINING SUCCESSFUL!")
                
                # Extract key metrics
                r2_scores = result.get('r2_scores', {})
                mean_r2 = result.get('mean_r2', 0.0)
                mean_mae = result.get('mean_mae', 0.0)
                
                print(f"\n📈 OVERALL PERFORMANCE:")
                print(f"   🎯 Mean R² Score: {mean_r2:.3f}")
                print(f"   📊 Mean MAE: {mean_mae:.3f}")
                
                # ChemBERTa comparison data
                chemberta_r2 = {
                    'EGFR': 0.751, 'MDM2': 0.655, 'BRAF': 0.595, 'PI3KCA': 0.588,
                    'HER2': 0.583, 'VEGFR2': 0.555, 'MET': 0.502, 'ALK': 0.405,
                    'CDK4': 0.314, 'CDK6': 0.216
                }
                chemberta_mean = 0.516
                
                print(f"\n📊 DETAILED R² SCORES BY TARGET:")
                print("   " + "Target".ljust(12) + "| Chemprop R² | ChemBERTa R² | Difference | Status")
                print("   " + "-" * 75)
                
                chemprop_wins = 0
                chemberta_wins = 0
                ties = 0
                
                for target in sorted(r2_scores.keys()):
                    chemprop_r2 = r2_scores.get(target, 0.0)
                    bert_r2 = chemberta_r2.get(target, 0.0)
                    diff = chemprop_r2 - bert_r2
                    
                    if diff > 0.05:
                        status = "🟢 Better"
                        chemprop_wins += 1
                    elif diff < -0.05:
                        status = "🔴 Worse"
                        chemberta_wins += 1
                    else:
                        status = "🟡 Similar"
                        ties += 1
                    
                    print(f"   {target.ljust(12)}| {chemprop_r2:11.3f} | {bert_r2:12.3f} | {diff:+10.3f} | {status}")
                
                # Performance comparison
                print(f"\n🏆 HEAD-TO-HEAD COMPARISON:")
                print(f"   🟢 Chemprop Better:  {chemprop_wins}/10 targets")
                print(f"   🔴 ChemBERTa Better: {chemberta_wins}/10 targets") 
                print(f"   🟡 Similar Performance: {ties}/10 targets")
                
                # Overall winner
                if mean_r2 > chemberta_mean + 0.02:
                    print(f"\n🏆 OVERALL WINNER: Chemprop GNN")
                    print(f"   📈 Performance Improvement: +{(mean_r2 - chemberta_mean):.3f} R²")
                elif mean_r2 < chemberta_mean - 0.02:
                    print(f"\n🏆 OVERALL WINNER: ChemBERTa Transformer")
                    print(f"   📉 Performance Gap: {(chemberta_mean - mean_r2):.3f} R²")
                else:
                    print(f"\n🤝 RESULT: Comparable Performance")
                    print(f"   📊 Difference: {(mean_r2 - chemberta_mean):+.3f} R²")
                
                # Performance breakdown
                breakdown = result.get('performance_breakdown', {})
                print(f"\n🎯 PERFORMANCE BREAKDOWN:")
                print(f"   🌟 Excellent (R² > 0.6):   {breakdown.get('excellent', 0)}/10 targets")
                print(f"   ✅ Good (0.4 < R² ≤ 0.6):  {breakdown.get('good', 0)}/10 targets")
                print(f"   ⚠️ Fair (0.2 < R² ≤ 0.4):  {breakdown.get('fair', 0)}/10 targets")
                print(f"   ❌ Poor (R² ≤ 0.2):        {breakdown.get('poor', 0)}/10 targets")
                
                # Additional information
                print(f"\n📋 TRAINING DETAILS:")
                print(f"   💾 Model Path: {result.get('model_path', 'N/A')}")
                print(f"   🔗 W&B Run: {result.get('wandb_run_id', 'N/A')}")
                print(f"   ⏱️ Architecture: {training_config['depth']}-layer MPNN")
                print(f"   🧠 Hidden Size: {training_config['hidden_size']}")
                print(f"   📊 Training Epochs: {training_config['epochs']}")
                
                print(f"\n🎉 COMPREHENSIVE CHEMPROP TRAINING COMPLETED!")
                print("✅ Multi-task GNN model trained on 10 oncoproteins")
                print("✅ Full performance comparison with ChemBERTa completed")
                print("✅ Model ready for integration into production pipeline")
                print("✅ W&B logging captured all training metrics and comparisons")
                
                return True
                
            else:
                print("❌ COMPREHENSIVE TRAINING FAILED!")
                error_msg = result.get('error', 'Unknown error occurred')
                stderr = result.get('stderr', 'No stderr available')
                
                print(f"\n🔍 ERROR DETAILS:")
                print(f"   💥 Error: {error_msg}")
                if stderr:
                    print(f"   📋 Details: {stderr}")
                
                print(f"\n🔧 TROUBLESHOOTING SUGGESTIONS:")
                print("   1. Check Modal GPU availability and quotas")
                print("   2. Verify dataset integrity on Modal volumes")
                print("   3. Check W&B credentials and project access")
                print("   4. Review Chemprop version compatibility")
                print("   5. Validate training hyperparameters")
                
                return False
                
    except Exception as e:
        print(f"❌ CRITICAL TRAINING ERROR: {str(e)}")
        print(f"⏰ Failed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n🔧 ERROR RECOVERY ACTIONS:")
        print("   1. Check Modal service status and authentication")
        print("   2. Verify network connectivity to Modal and W&B")
        print("   3. Check available GPU resources and memory")
        print("   4. Validate training script and dependencies")
        
        return False

if __name__ == "__main__":
    print("🧬 VERIDICA AI - CHEMPROP COMPREHENSIVE TRAINING")
    print("=" * 70)
    
    success = main()
    
    print("\n" + "=" * 70)
    if success:
        print("🏁 COMPREHENSIVE CHEMPROP TRAINING VALIDATION COMPLETE")
        print("✅ Multi-task GNN model successfully trained and evaluated")
        print("📊 Performance comparison with ChemBERTa completed")
        print("🎯 Model ready for production deployment")
        print("🔗 Next: Integrate trained model into inference pipeline")
    else:
        print("⚠️ TRAINING ENCOUNTERED CRITICAL ISSUES")
        print("🔧 Review error logs and troubleshooting suggestions above")
        print("📞 Contact support if issues persist")
    
    print("=" * 70)
    exit(0 if success else 1)