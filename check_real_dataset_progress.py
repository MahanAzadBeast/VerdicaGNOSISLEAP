#!/usr/bin/env python3
"""
Check progress of real dataset creation
"""

import os
import time
from pathlib import Path

def check_progress():
    """Check the progress of real dataset creation"""
    
    print("🔍 Checking Real Dataset Creation Progress")
    print("=" * 50)
    
    # Check if output directory exists
    output_dir = Path("clinical_trial_dataset/data/real")
    
    if output_dir.exists():
        print(f"✅ Output directory exists: {output_dir}")
        
        # List files in the directory
        files = list(output_dir.glob("*"))
        if files:
            print(f"📁 Files found ({len(files)}):")
            for file in files:
                size = file.stat().st_size if file.is_file() else 0
                size_mb = size / (1024 * 1024)
                print(f"   - {file.name}: {size_mb:.2f} MB")
        else:
            print("📂 Directory is empty - collection still in progress")
    else:
        print("⏳ Output directory not yet created - collection starting")
    
    # Check if the main script is still running
    import subprocess
    try:
        result = subprocess.run(["pgrep", "-f", "create_real_dataset.py"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("🔄 Dataset creation script is still running")
        else:
            print("✅ Dataset creation script has completed")
    except:
        print("❓ Could not check if script is running")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_progress()