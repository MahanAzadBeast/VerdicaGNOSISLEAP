#!/usr/bin/env python3
"""
Training Progress Monitor
Continuously monitor both Model 1 and Model 2 training progress
"""

import time
import re
from datetime import datetime
from pathlib import Path

def extract_model_progress(log_file_path, model_name):
    """Extract latest training metrics from log file"""
    
    try:
        if not Path(log_file_path).exists():
            return {"status": "not_found", "model": model_name}
        
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        # Find all epoch entries
        if model_name == "Model 1":
            epoch_pattern = r'Epoch\s+(\d+): Loss=([0-9.]+) \| Test R²=([0-9.-]+) \| Test MAE=([0-9.]+)'
        else:  # Model 2
            epoch_pattern = r'Epoch\s+(\d+): Loss=([0-9.]+) \| Cancer R²=([0-9.-]+) \| Stage=([A-Za-z-]+)'
        
        matches = list(re.finditer(epoch_pattern, content))
        
        if not matches:
            return {"status": "no_epochs", "model": model_name}
        
        # Get latest match
        latest = matches[-1]
        
        if model_name == "Model 1":
            epoch, loss, r2, mae = latest.groups()
            return {
                "status": "training",
                "model": model_name,
                "current_epoch": int(epoch),
                "training_loss": float(loss),
                "test_r2": float(r2),
                "test_mae": float(mae),
                "total_epochs_found": len(matches)
            }
        else:  # Model 2
            epoch, loss, r2, stage = latest.groups()
            return {
                "status": "training", 
                "model": model_name,
                "current_epoch": int(epoch),
                "training_loss": float(loss),
                "cancer_r2": float(r2),
                "training_stage": stage,
                "total_epochs_found": len(matches)
            }
    
    except Exception as e:
        return {"status": "error", "model": model_name, "error": str(e)}

def format_progress_report(model1_progress, model2_progress):
    """Format comprehensive progress report"""
    
    report = []
    report.append("🎯 TRAINING PROGRESS TO EPOCH 50")
    report.append("=" * 50)
    report.append("")
    
    # Model 2 Progress
    report.append("📈 MODEL 2 (Cytotoxicity Prediction):")
    if model2_progress["status"] == "training":
        progress_pct = (model2_progress["current_epoch"] / 50) * 100
        stage_info = model2_progress["training_stage"]
        
        report.append(f"   📊 Current Epoch: {model2_progress['current_epoch']}/50 ({progress_pct:.1f}%)")
        report.append(f"   🎯 Training Stage: {stage_info}")
        report.append(f"   📉 Training Loss: {model2_progress['training_loss']:.4f}")
        report.append(f"   📈 Cancer R²: {model2_progress['cancer_r2']:.4f}")
        
        if model2_progress['current_epoch'] >= 25:
            report.append("   ✅ Status: STAGE 2 - Multi-task (Cancer + Normal)")
        else:
            epochs_to_stage2 = 25 - model2_progress['current_epoch']
            report.append(f"   🔄 Status: STAGE 1 - Cancer-only ({epochs_to_stage2} epochs to Stage 2)")
    else:
        report.append(f"   ❌ Status: {model2_progress['status']}")
    
    report.append("")
    
    # Model 1 Progress  
    report.append("🚀 MODEL 1 (Ligand Activity Predictor):")
    if model1_progress["status"] == "training":
        progress_pct = (model1_progress["current_epoch"] / 50) * 100
        
        report.append(f"   📊 Current Epoch: {model1_progress['current_epoch']}/50 ({progress_pct:.1f}%)")
        report.append(f"   📉 Training Loss: {model1_progress['training_loss']:.4f}")
        report.append(f"   📈 Test R²: {model1_progress['test_r2']:.4f}")
        report.append(f"   📏 Test MAE: {model1_progress['test_mae']:.4f}")
        
        if model1_progress['test_r2'] > 0.3:
            report.append("   🎉 Status: EXCELLENT - High R² achieved!")
        elif model1_progress['test_r2'] > 0.1:
            report.append("   ✅ Status: GOOD - Positive R² improving")
        else:
            report.append("   🔄 Status: Training - Building performance")
    else:
        report.append(f"   ❌ Status: {model1_progress['status']}")
    
    report.append("")
    report.append(f"⏰ Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(report)

def main():
    """Main monitoring loop"""
    
    log_files = {
        "Model 1": "/app/modal_training/model1_training_rdkit_fixed.log", 
        "Model 2": "/app/modal_training/model2_improved_training_fixed.log"
    }
    
    print("🔥 Starting Training Progress Monitor")
    print("====================================")
    print("Monitoring both Model 1 and Model 2 training to Epoch 50...")
    print("")
    
    previous_report = ""
    
    while True:
        try:
            # Extract progress from both models
            model1_progress = extract_model_progress(log_files["Model 1"], "Model 1")
            model2_progress = extract_model_progress(log_files["Model 2"], "Model 2")
            
            # Generate report
            current_report = format_progress_report(model1_progress, model2_progress)
            
            # Only print if there's been progress
            if current_report != previous_report:
                print("\n" + "="*60)
                print(current_report)
                print("="*60)
                previous_report = current_report
            
            # Check if both models completed 50 epochs
            model1_done = (model1_progress.get("current_epoch", 0) >= 50)
            model2_done = (model2_progress.get("current_epoch", 0) >= 50)
            
            if model1_done and model2_done:
                print("\n🎉 TRAINING COMPLETED!")
                print("Both models have reached Epoch 50!")
                break
            
            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n⛔ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()