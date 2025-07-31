#!/usr/bin/env python3
"""
Debug Chemprop Prediction Issues
Identify and fix the prediction command problems
"""

import modal
from pathlib import Path
import pandas as pd
import subprocess
import tempfile
import shutil

# Modal app setup
app = modal.App("chemprop-prediction-debug")

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
def debug_chemprop_prediction():
    """Debug Chemprop prediction issues"""
    
    print("🔍 DEBUGGING CHEMPROP PREDICTION ISSUES")
    print("=" * 50)
    
    # Find the latest model
    models_dir = Path("/vol/models")
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "focused_chemprop" in d.name]
    
    if not model_dirs:
        return {"status": "error", "error": "No model directories found"}
    
    latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using model: {latest_model_dir}")
    
    # List contents of model directory
    all_files = list(latest_model_dir.rglob("*"))
    print(f"\n📋 Model directory contents ({len(all_files)} files):")
    
    for f in all_files:
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"   {f.relative_to(latest_model_dir)}: {size_mb:.2f} MB")
    
    # Look for checkpoint files (required for chemprop predict)
    checkpoint_files = list(latest_model_dir.glob("*.pt")) + list(latest_model_dir.glob("*.pth"))
    print(f"\n🧠 Found {len(checkpoint_files)} checkpoint files:")
    for cf in checkpoint_files:
        print(f"   {cf.name}")
    
    # Check for args.json or similar config files
    config_files = list(latest_model_dir.glob("*args*")) + list(latest_model_dir.glob("*config*"))
    print(f"\n⚙️ Found {len(config_files)} config files:")
    for cf in config_files:
        print(f"   {cf.name}")
        if cf.suffix == '.json':
            try:
                import json
                with open(cf, 'r') as f:
                    config = json.load(f)
                    print(f"      Keys: {list(config.keys())}")
            except Exception as e:
                print(f"      Error reading: {e}")
    
    # Test chemprop predict help
    print(f"\n🔧 Testing chemprop predict help...")
    try:
        help_result = subprocess.run(['chemprop', 'predict', '--help'], 
                                   capture_output=True, text=True, timeout=30)
        if help_result.returncode == 0:
            print("✅ Chemprop predict help successful")
            # Look for key arguments
            help_text = help_result.stdout
            key_args = ['--test-path', '--checkpoint-dir', '--checkpoint-path', '--preds-path']
            print("   Key arguments found:")
            for arg in key_args:
                if arg in help_text:
                    print(f"     ✅ {arg}")
                else:
                    print(f"     ❌ {arg}")
        else:
            print(f"❌ Chemprop predict help failed: {help_result.stderr}")
    except Exception as e:
        print(f"❌ Error running chemprop predict help: {e}")
    
    # Try a simple prediction with corrected command
    print(f"\n🧪 Testing simple prediction...")
    try:
        # Create test input
        temp_dir = Path(tempfile.mkdtemp())
        input_file = temp_dir / "test_input.csv"
        output_file = temp_dir / "test_output.csv"
        
        # Simple test molecule
        test_df = pd.DataFrame({"smiles": ["CCO"]})  # Ethanol
        test_df.to_csv(input_file, index=False)
        
        print(f"   📝 Input file: {input_file}")
        print(f"   📊 Output file: {output_file}")
        
        # Try different command variations
        commands_to_try = [
            # Standard format
            ['chemprop', 'predict', 
             '--test-path', str(input_file),
             '--checkpoint-dir', str(latest_model_dir),
             '--preds-path', str(output_file)],
            
            # Alternative with checkpoint-path
            ['chemprop', 'predict',
             '--test-path', str(input_file), 
             '--checkpoint-path', str(checkpoint_files[0]) if checkpoint_files else str(latest_model_dir),
             '--preds-path', str(output_file)],
             
            # Minimal command
            ['chemprop', 'predict',
             '--test-path', str(input_file),
             '--checkpoint-dir', str(latest_model_dir)]
        ]
        
        for i, cmd in enumerate(commands_to_try):
            print(f"\n   🔧 Trying command {i+1}: {' '.join(cmd[:4])}...")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print(f"   ✅ Command {i+1} successful!")
                    print(f"   STDOUT preview: {result.stdout[:200]}...")
                    
                    # Check if output file was created
                    if output_file.exists():
                        output_df = pd.read_csv(output_file)
                        print(f"   📊 Output shape: {output_df.shape}")
                        print(f"   📋 Columns: {list(output_df.columns)}")
                        if len(output_df) > 0:
                            print(f"   📈 Sample output:")
                            print(output_df.head().to_string())
                        
                        # Clean up and return success
                        shutil.rmtree(temp_dir)
                        return {
                            "status": "success",
                            "working_command": cmd,
                            "output_shape": output_df.shape,
                            "output_columns": list(output_df.columns)
                        }
                    else:
                        print(f"   ⚠️ No output file generated")
                
                else:
                    print(f"   ❌ Command {i+1} failed (code {result.returncode})")
                    print(f"   STDERR: {result.stderr[:300]}...")
                    
            except subprocess.TimeoutExpired:
                print(f"   ⏰ Command {i+1} timed out")
            except Exception as e:
                print(f"   ❌ Command {i+1} error: {e}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"❌ Prediction test error: {e}")
    
    return {
        "status": "debug_complete",
        "model_dir": str(latest_model_dir),
        "checkpoint_files": [f.name for f in checkpoint_files],
        "config_files": [f.name for f in config_files]
    }

if __name__ == "__main__":
    print("🔍 CHEMPROP PREDICTION DEBUGGING")
    print("=" * 40)
    
    with app.run():
        result = debug_chemprop_prediction.remote()
        
        print(f"\n📊 DEBUG RESULTS:")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            print(f"✅ Found working command!")
            print(f"Command: {result['working_command']}")
            print(f"Output shape: {result['output_shape']}")
            print(f"Columns: {result['output_columns']}")
        else:
            print(f"⚠️ Debug completed but no working command found")
            print(f"Model dir: {result.get('model_dir')}")
            print(f"Checkpoint files: {result.get('checkpoint_files')}")
        
        print("=" * 40)