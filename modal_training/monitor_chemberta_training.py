#!/usr/bin/env python3
"""
Monitor ChemBERTa 50-Epoch Training Progress
"""

import time
import os
from pathlib import Path

def monitor_chemberta_training():
    """Monitor the ChemBERTa 50-epoch training progress"""
    
    log_file = Path("/app/modal_training/chemberta_50_training.log")
    
    print("🔍 CHEMBERTA 50-EPOCH TRAINING MONITOR")
    print("=" * 50)
    
    if not log_file.exists():
        print("❌ Training log file not found. Training may not have started.")
        return
    
    print("📋 Current Training Status:")
    print("-" * 30)
    
    # Read the log file
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Show last 20 lines
        lines = content.strip().split('\n')
        recent_lines = lines[-20:] if len(lines) > 20 else lines
        
        for line in recent_lines:
            print(line)
            
        print("\n" + "=" * 50)
        
        # Check training status
        if "Training Complete!" in content:
            print("✅ Training completed successfully!")
            return True
        elif "❌" in content or "Error" in content:
            print("⚠️ Training may have encountered errors")
            return False
        elif "🚀 Starting" in content:
            print("⏳ Training is in progress...")
            return None
        else:
            print("📋 Training status unknown - check log file")
            return None
            
    except Exception as e:
        print(f"❌ Error reading log file: {e}")
        return False

def get_training_estimate():
    """Provide training time estimate"""
    print("\n📊 TRAINING ESTIMATES:")
    print("• Duration: ~3 hours for 50 epochs")
    print("• GPU: Modal A100 (high performance)")
    print("• Batch Size: 16")
    print("• Targets: 10 oncoproteins")
    print("• Expected Performance: Mean R² > 0.516 (current baseline)")
    print("\n💡 Comparison Purpose:")
    print("• ChemBERTa: 50 epochs (Transformer)")
    print("• Chemprop: 50 epochs (GNN)")
    print("• Fair comparison for Model Architecture Comparison feature")

if __name__ == "__main__":
    print("Starting training monitor...")
    
    # Initial check
    status = monitor_chemberta_training()
    
    if status is None:  # Training in progress
        get_training_estimate()
        print(f"\n🔄 Monitoring will continue. Check again in 30-60 minutes.")
        print("To monitor manually: tail -f /app/modal_training/chemberta_50_training.log")
    elif status is True:
        print("🎉 Training completed! Ready for model comparison testing.")
    else:
        print("⚠️ Training issues detected. Check logs for details.")