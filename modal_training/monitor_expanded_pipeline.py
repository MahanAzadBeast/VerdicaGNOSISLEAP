"""
Monitor Expanded Pipeline Execution
Tracks the multi-source extraction and training pipeline progress
"""

import time
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

def check_extraction_status():
    """Check if extraction is complete and return results"""
    results_file = Path("/app/modal_training/extraction_results.json")
    log_file = Path("/app/modal_training/extraction_log.txt")
    
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            print(f"Error reading results: {e}")
            return None
    
    # Check if process is still running
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_content = f.read()
            if "EXTRACTION COMPLETED" in log_content:
                return {"status": "completed_check_results"}
            elif "EXTRACTION FAILED" in log_content:
                return {"status": "failed", "error": "Check log file"}
            else:
                return {"status": "running"}
    
    return {"status": "unknown"}

def launch_chemberta_training():
    """Launch ChemBERTa training after extraction"""
    print("🧠 LAUNCHING EXPANDED CHEMBERTA TRAINING")
    print("=" * 50)
    
    cmd = """
cd /app/modal_training && nohup python -c "
import modal
from train_expanded_chemberta import app, train_expanded_chemberta
import json
from datetime import datetime

print('🧠 STARTING EXPANDED CHEMBERTA TRAINING')
print('Targets: 23 (oncoproteins + tumor suppressors + metastasis suppressors)')
print('Activity type: IC50')
print('Epochs: 30')
print('Estimated time: 3-5 hours')

try:
    with app.run() as app_run:
        result = train_expanded_chemberta.remote(
            activity_type='IC50',
            num_epochs=30,
            run_name='expanded_multisource_chemberta'
        )
    
    with open('/app/modal_training/chemberta_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print('✅ CHEMBERTA TRAINING COMPLETED!')
    
except Exception as e:
    error_result = {
        'status': 'failed',
        'error': str(e),
        'timestamp': datetime.now().isoformat()
    }
    with open('/app/modal_training/chemberta_results.json', 'w') as f:
        json.dump(error_result, f, indent=2)
    print(f'❌ CHEMBERTA TRAINING FAILED: {e}')
    raise
" > /app/modal_training/chemberta_log.txt 2>&1 &
"""
    
    os.system(cmd)
    print("✅ ChemBERTa training launched in background")

def launch_chemprop_training():
    """Launch Chemprop training after ChemBERTa"""
    print("🕸️ LAUNCHING EXPANDED CHEMPROP TRAINING")
    print("=" * 50)
    
    cmd = """
cd /app/modal_training && nohup python -c "
import modal
from train_expanded_chemprop import app, train_expanded_chemprop
import json
from datetime import datetime

print('🕸️ STARTING EXPANDED CHEMPROP TRAINING')
print('Targets: 23 (oncoproteins + tumor suppressors + metastasis suppressors)')
print('Activity type: IC50')
print('Epochs: 40')
print('Estimated time: 2-3 hours')

try:
    with app.run() as app_run:
        result = train_expanded_chemprop.remote(
            activity_type='IC50',
            epochs=40,
            run_name='expanded_multisource_chemprop'
        )
    
    with open('/app/modal_training/chemprop_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print('✅ CHEMPROP TRAINING COMPLETED!')
    
except Exception as e:
    error_result = {
        'status': 'failed',
        'error': str(e),
        'timestamp': datetime.now().isoformat()
    }
    with open('/app/modal_training/chemprop_results.json', 'w') as f:
        json.dump(error_result, f, indent=2)
    print(f'❌ CHEMPROP TRAINING FAILED: {e}')
    raise
" > /app/modal_training/chemprop_log.txt 2>&1 &
"""
    
    os.system(cmd)
    print("✅ Chemprop training launched in background")

def monitor_pipeline():
    """Monitor the complete pipeline execution"""
    
    print("🔍 EXPANDED PIPELINE MONITORING STARTED")
    print("=" * 60)
    print(f"Started monitoring at: {datetime.now().isoformat()}")
    print()
    
    # Track pipeline stages
    extraction_completed = False
    chemberta_launched = False
    chemprop_launched = False
    
    start_time = datetime.now()
    
    while True:
        current_time = datetime.now()
        elapsed = current_time - start_time
        
        print(f"\n📊 Pipeline Status Update - {current_time.strftime('%H:%M:%S')} (Elapsed: {elapsed})")
        print("-" * 60)
        
        # Check extraction status
        if not extraction_completed:
            extraction_status = check_extraction_status()
            print(f"🔍 Data Extraction: {extraction_status.get('status', 'unknown')}")
            
            if extraction_status.get('status') == 'success':
                print(f"   ✅ Total records: {extraction_status.get('total_records', 0):,}")
                print(f"   ✅ Total targets: {extraction_status.get('total_targets', 0)}")
                print(f"   ✅ Total compounds: {extraction_status.get('total_compounds', 0):,}")
                extraction_completed = True
                
                # Launch ChemBERTa training
                if not chemberta_launched:
                    launch_chemberta_training()
                    chemberta_launched = True
                    
            elif extraction_status.get('status') == 'failed':
                print(f"   ❌ Extraction failed: {extraction_status.get('error', 'Unknown error')}")
                break
        else:
            print("🔍 Data Extraction: ✅ Completed")
        
        # Check ChemBERTa training status
        if chemberta_launched:
            chemberta_results = Path("/app/modal_training/chemberta_results.json")
            if chemberta_results.exists():
                try:
                    with open(chemberta_results, 'r') as f:
                        chemberta_data = json.load(f)
                    
                    print(f"🧠 ChemBERTa Training: {chemberta_data.get('status', 'unknown')}")
                    if chemberta_data.get('status') == 'success':
                        print(f"   ✅ Overall R²: {chemberta_data.get('overall_mean_r2', 0):.3f}")
                        
                        # Launch Chemprop training
                        if not chemprop_launched:
                            launch_chemprop_training()
                            chemprop_launched = True
                    elif chemberta_data.get('status') == 'error':
                        print(f"   ❌ ChemBERTa failed: {chemberta_data.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"🧠 ChemBERTa Training: Error reading results - {e}")
            else:
                print("🧠 ChemBERTa Training: Running...")
        
        # Check Chemprop training status
        if chemprop_launched:
            chemprop_results = Path("/app/modal_training/chemprop_results.json")
            if chemprop_results.exists():
                try:
                    with open(chemprop_results, 'r') as f:
                        chemprop_data = json.load(f)
                    
                    print(f"🕸️ Chemprop Training: {chemprop_data.get('status', 'unknown')}")
                    if chemprop_data.get('status') == 'success':
                        print(f"   ✅ Overall R²: {chemprop_data.get('mean_r2', 0):.3f}")
                        
                        # Pipeline completed!
                        print("\n🎉 EXPANDED PIPELINE COMPLETED SUCCESSFULLY!")
                        print("=" * 60)
                        
                        # Generate final summary
                        extraction_data = check_extraction_status()
                        
                        summary = {
                            "pipeline_status": "completed",
                            "completion_time": current_time.isoformat(),
                            "total_duration": str(elapsed),
                            "extraction": extraction_data,
                            "chemberta": chemberta_data,
                            "chemprop": chemprop_data
                        }
                        
                        with open('/app/modal_training/pipeline_summary.json', 'w') as f:
                            json.dump(summary, f, indent=2, default=str)
                        
                        print(f"📊 Final Summary:")
                        print(f"   • Total duration: {elapsed}")
                        print(f"   • Data extracted: {extraction_data.get('total_records', 0):,} records")
                        print(f"   • ChemBERTa R²: {chemberta_data.get('overall_mean_r2', 0):.3f}")
                        print(f"   • Chemprop R²: {chemprop_data.get('mean_r2', 0):.3f}")
                        print(f"   • Summary saved: /app/modal_training/pipeline_summary.json")
                        
                        return summary
                        
                    elif chemprop_data.get('status') == 'error':
                        print(f"   ❌ Chemprop failed: {chemprop_data.get('error', 'Unknown error')}")
                        break
                except Exception as e:
                    print(f"🕸️ Chemprop Training: Error reading results - {e}")
            else:
                print("🕸️ Chemprop Training: Running...")
        
        # Check for timeouts or failures
        if elapsed > timedelta(hours=18):  # 18 hour timeout
            print("\n⏰ PIPELINE TIMEOUT - Exceeded 18 hours")
            break
        
        # Wait before next check
        print("\n⏳ Next check in 10 minutes...")
        time.sleep(600)  # Check every 10 minutes

if __name__ == "__main__":
    try:
        result = monitor_pipeline()
        if result:
            print("\n✅ Pipeline monitoring completed successfully")
        else:
            print("\n❌ Pipeline monitoring ended with issues")
    except KeyboardInterrupt:
        print("\n🛑 Pipeline monitoring interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline monitoring error: {e}")
        import traceback
        traceback.print_exc()