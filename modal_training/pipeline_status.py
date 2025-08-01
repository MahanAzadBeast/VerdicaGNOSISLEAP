"""
Pipeline Status Dashboard
Quick status check for the expanded pipeline
"""

import json
from pathlib import Path
from datetime import datetime
import subprocess

def get_pipeline_status():
    """Get comprehensive pipeline status"""
    
    print("🚀 EXPANDED PIPELINE STATUS DASHBOARD")
    print("=" * 60)
    print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S AEST')}")
    print()
    
    # Check extraction status
    extraction_results = Path("/app/modal_training/fixed_extraction_results.json")
    extraction_log = Path("/app/modal_training/fixed_extraction_log.txt")
    
    print("📊 DATA EXTRACTION STATUS:")
    if extraction_results.exists():
        try:
            with open(extraction_results, 'r') as f:
                results = json.load(f)
            
            status = results.get('status', 'unknown')
            if status == 'success':
                print(f"   ✅ Status: COMPLETED")
                print(f"   📈 Total records: {results.get('total_records', 0):,}")
                print(f"   🎯 Total targets: {results.get('total_targets', 0)}")
                print(f"   🧪 Total compounds: {results.get('total_compounds', 0):,}")
                print(f"   📋 Matrix shape: {results.get('matrix_shape', 'unknown')}")
            else:
                print(f"   ❌ Status: FAILED")
                print(f"   🔍 Error: {results.get('error', 'Unknown')}")
        except Exception as e:
            print(f"   ❌ Error reading results: {e}")
    elif extraction_log.exists():
        with open(extraction_log, 'r') as f:
            log_content = f.read()
        
        if "COMPLETED" in log_content:
            print(f"   ✅ Status: COMPLETED (check results file)")
        elif "FAILED" in log_content:
            print(f"   ❌ Status: FAILED (check log)")
        elif "STARTED" in log_content:
            print(f"   🔄 Status: RUNNING")
            # Get progress from log
            lines = log_content.split('\n')
            progress_lines = [line for line in lines if 'Processing' in line or 'records' in line]
            if progress_lines:
                print(f"   📊 Latest: {progress_lines[-1]}")
        else:
            print(f"   ❓ Status: UNKNOWN")
    else:
        print(f"   ❓ Status: NOT STARTED")
    
    print()
    
    # Check ChemBERTa training status
    chemberta_results = Path("/app/modal_training/chemberta_expanded_results.json")
    chemberta_log = Path("/app/modal_training/chemberta_expanded_log.txt")
    
    print("🧠 CHEMBERTA TRAINING STATUS:")
    if chemberta_results.exists():
        try:
            with open(chemberta_results, 'r') as f:
                results = json.load(f)
            
            status = results.get('status', 'unknown')
            print(f"   ✅ Status: {status.upper()}")
            if status == 'success':
                r2_scores = results.get('r2_scores', {})
                if r2_scores:
                    mean_r2 = sum(r2_scores.values()) / len(r2_scores)
                    print(f"   📈 Mean R²: {mean_r2:.3f}")
                    print(f"   🎯 Targets trained: {len(r2_scores)}")
        except Exception as e:
            print(f"   ❌ Error reading results: {e}")
    elif chemberta_log.exists():
        print(f"   🔄 Status: RUNNING (check log)")
    else:
        print(f"   ⏳ Status: QUEUED")
    
    print()
    
    # Check Chemprop training status
    chemprop_results = Path("/app/modal_training/chemprop_expanded_results.json")
    chemprop_log = Path("/app/modal_training/chemprop_expanded_log.txt")
    
    print("🕸️ CHEMPROP TRAINING STATUS:")
    if chemprop_results.exists():
        try:
            with open(chemprop_results, 'r') as f:
                results = json.load(f)
            
            status = results.get('status', 'unknown')
            print(f"   ✅ Status: {status.upper()}")
            if status == 'success':
                r2_scores = results.get('r2_scores', {})
                if r2_scores:
                    mean_r2 = sum(r2_scores.values()) / len(r2_scores)
                    print(f"   📈 Mean R²: {mean_r2:.3f}")
                    print(f"   🎯 Targets trained: {len(r2_scores)}")
        except Exception as e:
            print(f"   ❌ Error reading results: {e}")
    elif chemprop_log.exists():
        print(f"   🔄 Status: RUNNING (check log)")
    else:
        print(f"   ⏳ Status: QUEUED")
    
    print()
    
    # Check monitoring status
    monitor_log = Path("/app/modal_training/fixed_pipeline_monitor.log")
    print("🔍 MONITORING STATUS:")
    if monitor_log.exists():
        with open(monitor_log, 'r') as f:
            lines = f.readlines()
        
        if lines:
            latest_line = lines[-1].strip()
            if "Next check" in latest_line:
                print(f"   ✅ Monitor: ACTIVE")
                # Get the most recent status update
                status_lines = [line for line in lines if "Fixed Pipeline Status" in line]
                if status_lines:
                    latest_status = status_lines[-1]
                    print(f"   📊 Latest check: {latest_status.split(' - ')[1] if ' - ' in latest_status else 'Unknown'}")
            else:
                print(f"   ❓ Monitor: Status unclear")
        else:
            print(f"   ❓ Monitor: Log empty")
    else:
        print(f"   ❌ Monitor: NOT RUNNING")
    
    print()
    
    # Overall pipeline status
    print("🎯 OVERALL PIPELINE STATUS:")
    
    # Determine overall status
    if extraction_results.exists():
        try:
            with open(extraction_results, 'r') as f:
                ext_results = json.load(f)
            if ext_results.get('status') == 'success':
                print(f"   📊 Data Collection: ✅ COMPLETE")
                
                # Check training progress
                if chemberta_results.exists() and chemprop_results.exists():
                    print(f"   🤖 Model Training: ✅ COMPLETE")
                    print(f"   🎉 Pipeline Status: FULLY COMPLETE")
                elif chemberta_results.exists() or chemberta_log.exists():
                    print(f"   🤖 Model Training: 🔄 IN PROGRESS")
                    print(f"   ⏳ Pipeline Status: TRAINING PHASE")
                else:
                    print(f"   🤖 Model Training: ⏳ QUEUED")
                    print(f"   🔄 Pipeline Status: TRAINING PHASE")
            else:
                print(f"   📊 Data Collection: ❌ FAILED")
                print(f"   🚫 Pipeline Status: STOPPED")
        except:
            print(f"   📊 Data Collection: ❓ UNCLEAR")
    else:
        print(f"   📊 Data Collection: 🔄 IN PROGRESS")
        print(f"   ⏳ Pipeline Status: EXTRACTION PHASE")
    
    # Process information
    print()
    print("💻 PROCESS INFORMATION:")
    try:
        # Check for running processes
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout
        
        extraction_running = 'fixed_expanded_extractor' in processes
        monitor_running = 'monitor_fixed_pipeline' in processes
        
        print(f"   🔄 Extraction process: {'RUNNING' if extraction_running else 'STOPPED'}")
        print(f"   👁️ Monitor process: {'RUNNING' if monitor_running else 'STOPPED'}")
        
    except Exception as e:
        print(f"   ❌ Error checking processes: {e}")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    get_pipeline_status()