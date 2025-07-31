#!/usr/bin/env python3
"""
Extract Real Training Metrics from Chemprop Training Output
"""

import modal
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import json

# Use the same app from training
app = modal.App("chemprop-metrics-extraction")

# Same image as training
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "chemprop>=2.2.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "rdkit-pypi>=2022.9.5"
    ])
)

models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/models": models_volume},
    cpu=2.0,
    memory=8192,
    timeout=600
)
def extract_real_metrics():
    """Extract real metrics from the trained Chemprop model"""
    
    print("🔍 Extracting real metrics from Chemprop training...")
    
    # Find the latest model directory
    models_dir = Path("/vol/models")
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "focused_chemprop" in d.name]
    
    if not model_dirs:
        print("❌ No Chemprop model directories found")
        return {"status": "error", "error": "No model directories found"}
    
    # Get the most recent directory
    latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 Found model directory: {latest_dir}")
    
    # List all files in the directory
    files = list(latest_dir.glob("**/*"))
    print(f"📋 Files in model directory ({len(files)} total):")
    for f in files[:20]:  # Show first 20 files
        print(f"   {f.name} ({f.stat().st_size} bytes)")
    
    if len(files) > 20:
        print(f"   ... and {len(files) - 20} more files")
    
    # Look for different types of results files
    result_files = []
    for pattern in ["*test*.csv", "*pred*.csv", "*result*.csv", "*.csv"]:
        matches = list(latest_dir.glob(pattern))
        result_files.extend(matches)
    
    print(f"\n📊 Found {len(result_files)} potential results files:")
    for f in result_files:
        print(f"   {f.name} ({f.stat().st_size} bytes)")
    
    # Try to extract metrics from any available results file
    metrics_extracted = False
    results = {"status": "success", "metrics": {}, "files_found": [f.name for f in result_files]}
    
    for results_file in result_files:
        try:
            print(f"\n🔍 Analyzing {results_file.name}...")
            df = pd.read_csv(results_file)
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            
            if len(df) > 0:
                print(f"   Sample data:")
                print(df.head().to_string())
                
                # Try to identify if this contains predictions
                if any("pred" in col.lower() for col in df.columns):
                    print(f"   ✅ Found prediction columns in {results_file.name}")
                    results["prediction_file"] = results_file.name
                    results["prediction_data"] = df.to_dict()
                    metrics_extracted = True
                    break
                    
        except Exception as e:
            print(f"   ❌ Error reading {results_file.name}: {e}")
    
    # Look for training logs or other output files
    log_files = list(latest_dir.glob("*.log")) + list(latest_dir.glob("*.txt"))
    if log_files:
        print(f"\n📝 Found {len(log_files)} log files:")
        for f in log_files:
            print(f"   {f.name}")
            
            # Try to extract metrics from logs
            try:
                with open(f, 'r') as file:
                    content = file.read()
                    if "R²" in content or "rmse" in content.lower() or "mae" in content.lower():
                        print(f"   ✅ Found metrics in {f.name}")
                        results["metrics_log"] = f.name
                        # Extract key lines
                        lines = content.split('\n')
                        metric_lines = [line for line in lines if any(metric in line.lower() for metric in ['r²', 'r2', 'rmse', 'mae', 'test'])]
                        results["metric_lines"] = metric_lines[:10]  # First 10 metric lines
                        
            except Exception as e:
                print(f"   ❌ Error reading log {f.name}: {e}")
    
    # Check for model files
    model_files = list(latest_dir.glob("*.pt")) + list(latest_dir.glob("*.pth")) + list(latest_dir.glob("*.pkl"))
    if model_files:
        print(f"\n🧠 Found {len(model_files)} model files:")
        for f in model_files:
            print(f"   {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
        results["model_files"] = [f.name for f in model_files]
    
    # Summary
    if metrics_extracted:
        print(f"\n✅ Successfully extracted metrics from training output")
        results["status"] = "success"
    else:
        print(f"\n⚠️ Could not find clear prediction metrics, but training completed")
        results["status"] = "partial"
    
    return results

if __name__ == "__main__":
    print("🔍 CHEMPROP METRICS EXTRACTION")
    print("=" * 50)
    
    with app.run():
        result = extract_real_metrics.remote()
        
        print("\n📊 EXTRACTION RESULTS:")
        print("=" * 50)
        
        if result["status"] == "success":
            print("✅ Metrics extraction successful")
            
            if "prediction_file" in result:
                print(f"📊 Predictions found in: {result['prediction_file']}")
            
            if "metrics_log" in result:
                print(f"📝 Metrics log: {result['metrics_log']}")
                if "metric_lines" in result:
                    print("📈 Key metric lines:")
                    for line in result["metric_lines"]:
                        print(f"   {line}")
            
            if "model_files" in result:
                print(f"🧠 Model files: {result['model_files']}")
            
        else:
            print(f"⚠️ Extraction status: {result['status']}")
        
        print(f"\n📋 Files found: {result['files_found']}")
        print("=" * 50)