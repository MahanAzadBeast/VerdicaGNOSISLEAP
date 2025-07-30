#!/usr/bin/env python3
"""
Multi-Task ChemBERTa Pipeline Monitor
Simple monitoring script to track the progress of the oncoprotein pipeline
"""

import time
from pathlib import Path
import json
from datetime import datetime

def check_pipeline_status():
    """Check the current status of the pipeline"""
    
    status = {
        "timestamp": datetime.now().isoformat(),
        "stage": "unknown",
        "progress": "unknown",
        "status": "unknown"
    }
    
    # Check deployment status
    deploy_log = Path("/app/modal_training/oncoprotein_deploy_extraction_fix.log")
    if deploy_log.exists():
        status["deployment"] = "success"
    else:
        status["deployment"] = "unknown"
    
    # Check download progress
    download_log = Path("/app/modal_training/chembl_download_fix_test.log")
    if download_log.exists():
        try:
            with open(download_log, 'r') as f:
                content = f.read()
            
            if "Downloaded" in content and "GB" in content:
                # Extract latest download progress
                lines = content.split('\n')
                download_lines = [l for l in lines if "Downloaded" in l and "GB" in l]
                if download_lines:
                    latest = download_lines[-1]
                    # Extract GB amount
                    try:
                        gb_str = latest.split("Downloaded ")[1].split(" GB")[0]
                        current_gb = float(gb_str)
                        
                        # Look for total size
                        total_lines = [l for l in lines if "Downloading" in l and "GB" in l]
                        if total_lines:
                            total_str = total_lines[0].split("Downloading ")[1].split(" GB")[0]
                            total_gb = float(total_str)
                            progress = (current_gb / total_gb) * 100
                            
                            status["stage"] = "download"
                            status["progress"] = f"{current_gb:.1f}/{total_gb:.1f} GB ({progress:.1f}%)"
                            status["status"] = "in_progress" if progress < 100 else "completed"
                        else:
                            status["stage"] = "download"
                            status["progress"] = f"{current_gb:.1f} GB"
                            status["status"] = "in_progress"
                    except:
                        status["stage"] = "download"
                        status["progress"] = "parsing_error"
                        status["status"] = "error"
            
            # Check for completion messages
            if "✅ ChEMBL downloaded" in content:
                status["download_completed"] = True
                status["stage"] = "extraction" if status["stage"] == "download" else status["stage"]
            
            # Check for errors
            if "ERROR:" in content or "Failed" in content:
                status["status"] = "error"
                error_lines = [l for l in content.split('\n') if "ERROR:" in l]
                if error_lines:
                    status["last_error"] = error_lines[-1]
                    
        except Exception as e:
            status["download_log_error"] = str(e)
    
    # Check extraction progress (would be in a separate log)
    # Check training progress (would be in a separate log)
    
    return status

def main():
    """Main monitoring function"""
    print("🎯 Multi-Task ChemBERTa Pipeline Monitor")
    print("=" * 50)
    
    try:
        status = check_pipeline_status()
        
        print(f"⏰ Timestamp: {status['timestamp']}")
        print(f"📊 Current Stage: {status['stage']}")
        print(f"📈 Progress: {status['progress']}")
        print(f"🔄 Status: {status['status']}")
        
        if "download_completed" in status:
            print(f"✅ Download Completed: {status['download_completed']}")
        
        if "last_error" in status:
            print(f"❌ Last Error: {status['last_error']}")
        
        # Pretty print full status as JSON
        print("\n📋 Full Status:")
        print(json.dumps(status, indent=2))
        
    except Exception as e:
        print(f"❌ Monitor error: {e}")

if __name__ == "__main__":
    main()