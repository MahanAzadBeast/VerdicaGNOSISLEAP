#!/usr/bin/env python3
"""
Enhanced Chemprop CLI Debugging with Full Error Details
"""

import modal
import subprocess
from pathlib import Path
import pandas as pd
import tempfile

app = modal.App("chemprop-cli-detailed-debug")

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
def detailed_chemprop_debug():
    """Get detailed error information from Chemprop CLI"""
    
    print("🔍 DETAILED CHEMPROP CLI DEBUG")
    print("=" * 50)
    
    # Find the model
    models_dir = Path("/vol/models")
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "focused_chemprop" in d.name]
    
    if not model_dirs:
        return {"error": "No models found"}
    
    latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using model: {latest_model_dir}")
    
    # Check model structure in detail
    print(f"\n📋 Model directory contents:")
    for item in latest_model_dir.rglob("*"):
        if item.is_file():
            print(f"   {item.relative_to(latest_model_dir)} ({item.stat().st_size} bytes)")
    
    # Check for key files
    key_files = {
        "model.pt": latest_model_dir / "model.pt",
        "args.json": latest_model_dir / "args.json", 
        "config.json": latest_model_dir / "config.json",
        "scaler.pkl": latest_model_dir / "scaler.pkl"
    }
    
    print(f"\n🔑 Key files check:")
    for name, path in key_files.items():
        exists = path.exists()
        print(f"   {name}: {'✅' if exists else '❌'} ({path})")
    
    # Test Chemprop version and help
    print(f"\n📦 Chemprop version and help:")
    try:
        version_result = subprocess.run(['chemprop', '--version'], capture_output=True, text=True, timeout=10)
        print(f"   Version: {version_result.stdout.strip()}")
    except:
        print("   ❌ Version check failed")
    
    try:
        help_result = subprocess.run(['chemprop', 'predict', '--help'], capture_output=True, text=True, timeout=10)
        print(f"   Help (first 500 chars): {help_result.stdout[:500]}")
    except:
        print("   ❌ Help check failed")
    
    # Create test input
    temp_dir = Path(tempfile.mkdtemp())
    input_file = temp_dir / "test.csv"
    output_file = temp_dir / "output.csv"
    
    # Create simple test CSV
    test_df = pd.DataFrame({"smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"]})
    test_df.to_csv(input_file, index=False)
    print(f"📝 Test input: {input_file}")
    
    # Test the exact command that's failing
    cmd = [
        'chemprop', 'predict',
        '--test-path', str(input_file),
        '--checkpoint-dir', str(latest_model_dir),
        '--preds-path', str(output_file)
    ]
    
    print(f"\n🔧 Testing exact failing command:")
    print(f"   {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print(f"   Return code: {result.returncode}")
        print(f"   STDOUT: {result.stdout}")
        print(f"   STDERR: {result.stderr}")
        
        return {
            "model_path": str(latest_model_dir),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "model_files": [str(f.relative_to(latest_model_dir)) for f in latest_model_dir.rglob("*") if f.is_file()],
            "key_files_exist": {name: path.exists() for name, path in key_files.items()}
        }
        
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return {"error": str(e), "model_path": str(latest_model_dir)}

if __name__ == "__main__":
    print("🔍 Starting detailed Chemprop CLI debug...")
    
    with app.run():
        debug_result = detailed_chemprop_debug.remote()
        
        print("\n📊 DETAILED DEBUG RESULTS")
        print("=" * 50)
        
        if "error" in debug_result:
            print(f"❌ Error: {debug_result['error']}")
        else:
            print(f"Model: {debug_result.get('model_path')}")
            print(f"Return Code: {debug_result.get('return_code')}")
            print(f"\nSTDOUT:\n{debug_result.get('stdout', '')}")
            print(f"\nSTDERR:\n{debug_result.get('stderr', '')}")
            print(f"\nKey Files: {debug_result.get('key_files_exist', {})}")
            
            # Show the exact error to fix
            stderr = debug_result.get('stderr', '')
            if stderr:
                print(f"\n🎯 ROOT CAUSE:")
                if "checkpoint" in stderr.lower():
                    print("   Issue: Checkpoint file problem")
                elif "path" in stderr.lower():
                    print("   Issue: Path problem")
                elif "argument" in stderr.lower():
                    print("   Issue: CLI argument problem")
                else:
                    print("   Issue: Unknown - see stderr above")