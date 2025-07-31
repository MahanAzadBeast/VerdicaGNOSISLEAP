#!/usr/bin/env python3
"""
Debug and Fix Chemprop CLI Issue
"""

import modal
import subprocess
from pathlib import Path
import pandas as pd
import tempfile

# Modal app for debugging
app = modal.App("chemprop-cli-debug")

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
    memory=4096,
    timeout=300
)
def debug_chemprop_model():
    """Debug the Chemprop model structure and CLI"""
    
    print("🔍 DEBUGGING CHEMPROP MODEL STRUCTURE")
    print("=" * 50)
    
    # Find model directories
    models_dir = Path("/vol/models")
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "focused_chemprop" in d.name]
    
    if not model_dirs:
        print("❌ No Chemprop models found")
        return {"error": "No models found"}
    
    # Check each model
    for model_dir in model_dirs:
        print(f"\n📁 Checking model: {model_dir.name}")
        print(f"   Path: {model_dir}")
        
        # List all files
        files = list(model_dir.rglob("*"))
        print(f"   Files found: {len(files)}")
        
        for file in files[:20]:  # Show first 20 files
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   - {file.name} ({size_mb:.2f} MB)")
        
        # Check for specific Chemprop files
        checkpoint_files = list(model_dir.glob("*.pt"))
        config_files = list(model_dir.glob("*.json"))
        csv_files = list(model_dir.glob("*.csv"))
        
        print(f"   ✅ Checkpoint files (.pt): {len(checkpoint_files)}")
        print(f"   ✅ Config files (.json): {len(config_files)}")
        print(f"   ✅ CSV files: {len(csv_files)}")
        
        # Test different CLI approaches
        latest_model_dir = model_dir
        break
    
    print(f"\n🧪 TESTING CLI APPROACHES WITH: {latest_model_dir.name}")
    print("=" * 50)
    
    # Create test input
    temp_dir = Path(tempfile.mkdtemp())
    input_file = temp_dir / "test_input.csv"
    output_file = temp_dir / "test_output.csv"
    
    test_df = pd.DataFrame({"smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"]})
    test_df.to_csv(input_file, index=False)
    
    print(f"📝 Test input created: {input_file}")
    
    # Test approaches
    approaches = [
        # Approach 1: Standard chemprop predict
        {
            "name": "Standard chemprop predict",
            "cmd": [
                'chemprop', 'predict',
                '--test-path', str(input_file),
                '--checkpoint-dir', str(latest_model_dir),
                '--preds-path', str(output_file)
            ]
        },
        # Approach 2: With specific checkpoint file
        {
            "name": "Specific checkpoint file", 
            "cmd": [
                'chemprop', 'predict',
                '--test-path', str(input_file),
                '--checkpoint-path', str(latest_model_dir / "model.pt") if (latest_model_dir / "model.pt").exists() else str(latest_model_dir),
                '--preds-path', str(output_file)
            ]
        },
        # Approach 3: Python module approach
        {
            "name": "Python module approach",
            "cmd": [
                'python', '-m', 'chemprop.predict',
                '--test-path', str(input_file),
                '--checkpoint-dir', str(latest_model_dir),
                '--preds-path', str(output_file)
            ]
        }
    ]
    
    results = {}
    
    for approach in approaches:
        print(f"\n🔧 Testing: {approach['name']}")
        print(f"   Command: {' '.join(approach['cmd'])}")
        
        try:
            result = subprocess.run(
                approach['cmd'], 
                capture_output=True, 
                text=True, 
                timeout=60,
                cwd=temp_dir
            )
            
            print(f"   Return code: {result.returncode}")
            
            if result.returncode == 0:
                print("   ✅ SUCCESS!")
                if output_file.exists():
                    pred_df = pd.read_csv(output_file)
                    print(f"   📊 Output shape: {pred_df.shape}")
                    print(f"   📋 Columns: {list(pred_df.columns)}")
                    results[approach['name']] = "SUCCESS"
                else:
                    print("   ⚠️ No output file created")
                    results[approach['name']] = "NO_OUTPUT"
            else:
                print("   ❌ FAILED")
                print(f"   STDOUT: {result.stdout[:200]}")
                print(f"   STDERR: {result.stderr[:200]}")
                results[approach['name']] = f"FAILED_{result.returncode}"
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results[approach['name']] = f"EXCEPTION_{str(e)}"
    
    return {
        "model_path": str(latest_model_dir),
        "test_results": results,
        "files_in_model": [str(f) for f in files[:10]]
    }

if __name__ == "__main__":
    print("🔍 Starting Chemprop CLI debugging...")
    
    with app.run():
        debug_result = debug_chemprop_model.remote()
        
        print("\n📊 DEBUG SUMMARY")
        print("=" * 50)
        print(f"Model Path: {debug_result.get('model_path')}")
        print("\nTest Results:")
        for approach, result in debug_result.get('test_results', {}).items():
            print(f"  {approach}: {result}")
        
        print(f"\nModel Files:")
        for file in debug_result.get('files_in_model', []):
            print(f"  - {file}")