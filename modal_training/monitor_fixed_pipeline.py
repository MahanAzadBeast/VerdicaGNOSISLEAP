"""
Robust Pipeline Monitor with Fixed Issues
Monitors the fixed expanded pipeline and manages training stages
"""

import time
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

def check_fixed_extraction_status():
    """Check status of the fixed extraction"""
    results_file = Path("/app/modal_training/fixed_extraction_results.json")
    log_file = Path("/app/modal_training/fixed_extraction_log.txt")
    
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            print(f"Error reading results: {e}")
            return None
    
    # Check log for progress
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_content = f.read()
            if "FIXED EXTRACTION COMPLETED" in log_content:
                return {"status": "completed_check_results"}
            elif "FIXED EXTRACTION FAILED" in log_content:
                return {"status": "failed", "error": "Check log file"}
            elif "STARTED" in log_content:
                return {"status": "running"}
    
    return {"status": "unknown"}

def launch_simple_chemberta_training(dataset_info):
    """Launch simplified ChemBERTa training using existing framework"""
    print("🧠 LAUNCHING SIMPLIFIED CHEMBERTA TRAINING")
    print("=" * 50)
    
    # Use the existing focused training which is proven to work
    cmd = f"""
cd /app/modal_training && nohup python -c "
import modal
from train_chemberta_focused import app, train_focused_chemberta
import json
from datetime import datetime

print('🧠 STARTING SIMPLIFIED CHEMBERTA TRAINING')
print('Dataset: Expanded fixed extraction')
print('Total records: {dataset_info.get('total_records', 'unknown')}')
print('Total targets: {dataset_info.get('total_targets', 'unknown')}')
print('Epochs: 30')

try:
    with app.run() as app_run:
        result = train_focused_chemberta.remote(
            epochs=30,
            batch_size=16,
            learning_rate=2e-5,
            run_name='expanded_fixed_chemberta'
        )
    
    with open('/app/modal_training/chemberta_expanded_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print('✅ CHEMBERTA TRAINING COMPLETED!')
    print(f'Results saved to: /app/modal_training/chemberta_expanded_results.json')
    
except Exception as e:
    error_result = {{
        'status': 'failed',
        'error': str(e),
        'timestamp': datetime.now().isoformat()
    }}
    with open('/app/modal_training/chemberta_expanded_results.json', 'w') as f:
        json.dump(error_result, f, indent=2)
    print(f'❌ CHEMBERTA TRAINING FAILED: {{e}}')
    import traceback
    traceback.print_exc()
" > /app/modal_training/chemberta_expanded_log.txt 2>&1 &
"""
    
    os.system(cmd)
    print("✅ Simplified ChemBERTa training launched")

def launch_simple_chemprop_training(dataset_info):
    """Launch simplified Chemprop training using existing framework"""
    print("🕸️ LAUNCHING SIMPLIFIED CHEMPROP TRAINING")
    print("=" * 50)
    
    # Use the existing focused training which is proven to work
    cmd = f"""
cd /app/modal_training && nohup python -c "
import modal
from train_chemprop_focused import app, train_focused_chemprop
import json
from datetime import datetime

print('🕸️ STARTING SIMPLIFIED CHEMPROP TRAINING')
print('Dataset: Expanded fixed extraction')
print('Total records: {dataset_info.get('total_records', 'unknown')}')
print('Total targets: {dataset_info.get('total_targets', 'unknown')}')
print('Epochs: 40')

try:
    with app.run() as app_run:
        result = train_focused_chemprop.remote(
            epochs=40,
            batch_size=64,
            learning_rate=1e-3,
            run_name='expanded_fixed_chemprop'
        )
    
    with open('/app/modal_training/chemprop_expanded_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print('✅ CHEMPROP TRAINING COMPLETED!')
    print(f'Results saved to: /app/modal_training/chemprop_expanded_results.json')
    
except Exception as e:
    error_result = {{
        'status': 'failed',
        'error': str(e),
        'timestamp': datetime.now().isoformat()
    }}
    with open('/app/modal_training/chemprop_expanded_results.json', 'w') as f:
        json.dump(error_result, f, indent=2)
    print(f'❌ CHEMPROP TRAINING FAILED: {{e}}')
    import traceback
    traceback.print_exc()
" > /app/modal_training/chemprop_expanded_log.txt 2>&1 &
"""
    
    os.system(cmd)
    print("✅ Simplified Chemprop training launched")

def monitor_fixed_pipeline():
    """Monitor the fixed pipeline execution"""
    
    print("🔍 FIXED PIPELINE MONITORING STARTED")
    print("=" * 60)
    print(f"Started monitoring at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S AEST')}")
    print()
    
    # Track pipeline stages
    extraction_completed = False
    chemberta_launched = False
    chemprop_launched = False
    
    start_time = datetime.now()
    
    while True:
        current_time = datetime.now()
        elapsed = current_time - start_time
        
        print(f"\n📊 Fixed Pipeline Status - {current_time.strftime('%H:%M:%S AEST')} (Elapsed: {elapsed})")
        print("-" * 60)
        
        # Check extraction status
        if not extraction_completed:
            extraction_status = check_fixed_extraction_status()
            print(f"🔍 Fixed Data Extraction: {extraction_status.get('status', 'unknown')}")
            
            if extraction_status.get('status') == 'success':
                print(f"   ✅ Total records: {extraction_status.get('total_records', 0):,}")
                print(f"   ✅ Total targets: {extraction_status.get('total_targets', 0)}")
                print(f"   ✅ Total compounds: {extraction_status.get('total_compounds', 0):,}")
                print(f"   ✅ Matrix shape: {extraction_status.get('matrix_shape', 'unknown')}")
                extraction_completed = True
                
                # Launch simplified training
                if not chemberta_launched:
                    launch_simple_chemberta_training(extraction_status)
                    chemberta_launched = True
                    
            elif extraction_status.get('status') == 'failed':
                print(f"   ❌ Fixed extraction failed: {extraction_status.get('error', 'Unknown error')}")
                break
        else:
            print("🔍 Fixed Data Extraction: ✅ Completed")
        
        # Check ChemBERTa training
        if chemberta_launched:
            chemberta_results = Path("/app/modal_training/chemberta_expanded_results.json")
            if chemberta_results.exists():
                try:
                    with open(chemberta_results, 'r') as f:
                        chemberta_data = json.load(f)
                    
                    print(f"🧠 ChemBERTa Training: {chemberta_data.get('status', 'unknown')}")
                    if chemberta_data.get('status') == 'success':
                        r2_scores = chemberta_data.get('r2_scores', {})
                        if r2_scores:
                            mean_r2 = sum(r2_scores.values()) / len(r2_scores)
                            print(f"   ✅ Mean R²: {mean_r2:.3f}")
                        
                        # Launch Chemprop training
                        if not chemprop_launched:
                            extraction_status = check_fixed_extraction_status()
                            launch_simple_chemprop_training(extraction_status)
                            chemprop_launched = True
                    elif chemberta_data.get('status') == 'error':
                        print(f"   ❌ ChemBERTa failed: {chemberta_data.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"🧠 ChemBERTa Training: Error reading results - {e}")
            else:
                print("🧠 ChemBERTa Training: Running...")
        
        # Check Chemprop training
        if chemprop_launched:
            chemprop_results = Path("/app/modal_training/chemprop_expanded_results.json")
            if chemprop_results.exists():
                try:
                    with open(chemprop_results, 'r') as f:
                        chemprop_data = json.load(f)
                    
                    print(f"🕸️ Chemprop Training: {chemprop_data.get('status', 'unknown')}")
                    if chemprop_data.get('status') == 'success':
                        r2_scores = chemprop_data.get('r2_scores', {})
                        if r2_scores:
                            mean_r2 = sum(r2_scores.values()) / len(r2_scores)
                            print(f"   ✅ Mean R²: {mean_r2:.3f}")
                        
                        # Pipeline completed!
                        print("\n🎉 EXPANDED FIXED PIPELINE COMPLETED!")
                        print("=" * 60)
                        
                        # Generate final summary
                        extraction_data = check_fixed_extraction_status()
                        
                        summary = {
                            "pipeline_status": "completed",
                            "completion_time": current_time.isoformat(),
                            "total_duration": str(elapsed),
                            "extraction": extraction_data,
                            "chemberta": chemberta_data,
                            "chemprop": chemprop_data
                        }
                        
                        with open('/app/modal_training/fixed_pipeline_summary.json', 'w') as f:
                            json.dump(summary, f, indent=2, default=str)
                        
                        print(f"📊 Final Summary:")
                        print(f"   • Total duration: {elapsed}")
                        print(f"   • Data extracted: {extraction_data.get('total_records', 0):,} records")
                        print(f"   • ChemBERTa training: {chemberta_data.get('status', 'unknown')}")
                        print(f"   • Chemprop training: {chemprop_data.get('status', 'unknown')}")
                        print(f"   • Summary saved: /app/modal_training/fixed_pipeline_summary.json")
                        
                        return summary
                        
                    elif chemprop_data.get('status') == 'error':
                        print(f"   ❌ Chemprop failed: {chemprop_data.get('error', 'Unknown error')}")
                        break
                except Exception as e:
                    print(f"🕸️ Chemprop Training: Error reading results - {e}")
            else:
                print("🕸️ Chemprop Training: Running...")
        
        # Check for timeouts
        if elapsed > timedelta(hours=12):
            print("\n⏰ PIPELINE TIMEOUT - Exceeded 12 hours")
            break
        
        # Wait before next check
        print("\n⏳ Next check in 5 minutes...")
        time.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    try:
        result = monitor_fixed_pipeline()
        if result:
            print("\n✅ Fixed pipeline monitoring completed successfully")
        else:
            print("\n❌ Fixed pipeline monitoring ended with issues")
    except KeyboardInterrupt:
        print("\n🛑 Pipeline monitoring interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline monitoring error: {e}")
        import traceback
        traceback.print_exc()