#!/usr/bin/env python3
"""
Comprehensive Chemprop Model Investigation and Fix
Identify what was saved and fix the inference pipeline
"""

import modal
from pathlib import Path
import subprocess
import tempfile
import shutil
import pandas as pd
import json
from datetime import datetime

# Modal app setup
app = modal.App("chemprop-model-fix")

# Enhanced image with Chemprop
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "chemprop>=2.2.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0"
    ])
)

models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)
datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)

ONCOPROTEIN_TARGETS = [
    'EGFR', 'HER2', 'VEGFR2', 'BRAF', 'MET', 
    'CDK4', 'CDK6', 'ALK', 'MDM2', 'PI3KCA'
]

@app.function(
    image=image,
    volumes={
        "/vol/models": models_volume,
        "/vol/datasets": datasets_volume
    },
    gpu="T4",
    memory=16384,
    timeout=1800
)
def investigate_and_fix_chemprop_model():
    """Comprehensive investigation and fix of Chemprop model deployment"""
    
    print("🔍 COMPREHENSIVE CHEMPROP MODEL INVESTIGATION & FIX")
    print("=" * 60)
    
    # Find the trained model directory
    models_dir = Path("/vol/models")
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "focused_chemprop" in d.name]
    
    if not model_dirs:
        return {"status": "error", "error": "No trained model directories found"}
    
    # Get the most recent (comprehensive training) model
    latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
    model_name = latest_model_dir.name
    
    print(f"📁 Investigating model: {model_name}")
    print(f"📅 Created: {datetime.fromtimestamp(latest_model_dir.stat().st_mtime)}")
    
    # Comprehensive file analysis
    all_files = list(latest_model_dir.rglob("*"))
    print(f"\n📊 COMPLETE MODEL DIRECTORY ANALYSIS")
    print(f"Total files found: {len(all_files)}")
    
    file_analysis = {
        'pytorch_files': [],
        'json_files': [],
        'csv_files': [],
        'log_files': [],
        'other_files': []
    }
    
    for f in all_files:
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            rel_path = f.relative_to(latest_model_dir)
            file_info = f"{rel_path}: {size_mb:.2f} MB"
            
            if f.suffix in ['.pt', '.pth', '.ckpt']:
                file_analysis['pytorch_files'].append(file_info)
            elif f.suffix == '.json':
                file_analysis['json_files'].append(file_info)
            elif f.suffix == '.csv':
                file_analysis['csv_files'].append(file_info)
            elif f.suffix in ['.log', '.txt']:
                file_analysis['log_files'].append(file_info)
            else:
                file_analysis['other_files'].append(file_info)
            
            print(f"   {file_info}")
    
    print(f"\n📋 FILE TYPE BREAKDOWN:")
    for file_type, files in file_analysis.items():
        print(f"   {file_type}: {len(files)} files")
    
    # Check what Chemprop v2.2.0 expects for prediction
    print(f"\n🔧 CHECKING CHEMPROP v2.2.0 PREDICTION REQUIREMENTS")
    
    try:
        # Get Chemprop predict help to understand requirements
        result = subprocess.run(['chemprop', 'predict', '--help'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            help_text = result.stdout
            print("✅ Chemprop predict help retrieved")
            
            # Look for key requirements
            requirements = []
            if '--checkpoint-dir' in help_text:
                requirements.append("--checkpoint-dir: Directory containing model checkpoints")
            if '--checkpoint-path' in help_text:
                requirements.append("--checkpoint-path: Path to specific checkpoint file")
            if '.pth' in help_text:
                requirements.append("Expects .pth checkpoint files")
            if '.pt' in help_text:
                requirements.append("Expects .pt checkpoint files")
                
            print("📋 Chemprop prediction requirements:")
            for req in requirements:
                print(f"   • {req}")
        else:
            print(f"❌ Failed to get Chemprop help: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error checking Chemprop requirements: {e}")
    
    # Try different approaches to make predictions work
    print(f"\n🧪 TESTING INFERENCE APPROACHES")
    
    # Create test data
    temp_dir = Path(tempfile.mkdtemp())
    test_input = temp_dir / "test.csv"
    test_output = temp_dir / "output.csv"
    
    # Simple test molecule
    test_df = pd.DataFrame({"smiles": ["CCO"]})  # Ethanol
    test_df.to_csv(test_input, index=False)
    
    approaches = [
        {
            'name': 'Checkpoint Directory Approach',
            'cmd': ['chemprop', 'predict', 
                   '--test-path', str(test_input),
                   '--checkpoint-dir', str(latest_model_dir),
                   '--preds-path', str(test_output)]
        }
    ]
    
    # Add checkpoint-path approaches if we have model files
    pytorch_files = [f for f in all_files if f.suffix in ['.pt', '.pth', '.ckpt']]
    for model_file in pytorch_files[:3]:  # Test first 3 model files
        approaches.append({
            'name': f'Checkpoint Path: {model_file.name}',
            'cmd': ['chemprop', 'predict',
                   '--test-path', str(test_input),
                   '--checkpoint-path', str(model_file),
                   '--preds-path', str(test_output)]
        })
    
    working_approach = None
    
    for i, approach in enumerate(approaches):
        print(f"\n🔬 Testing Approach {i+1}: {approach['name']}")
        
        try:
            # Clean previous output
            if test_output.exists():
                test_output.unlink()
            
            result = subprocess.run(approach['cmd'], 
                                  capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("✅ Command executed successfully!")
                
                if test_output.exists():
                    output_df = pd.read_csv(test_output)
                    print(f"✅ Output generated: {output_df.shape}")
                    print(f"✅ Columns: {list(output_df.columns)}")
                    
                    if len(output_df) > 0:
                        print("✅ Sample predictions:")
                        print(output_df.head().to_string())
                        
                        working_approach = {
                            'approach': approach,
                            'output_shape': output_df.shape,
                            'columns': list(output_df.columns),
                            'sample_data': output_df.head().to_dict()
                        }
                        print(f"🎉 WORKING APPROACH FOUND: {approach['name']}")
                        break
                else:
                    print("⚠️ Command succeeded but no output file generated")
            else:
                print(f"❌ Command failed (code {result.returncode})")
                error_msg = result.stderr[:300] if result.stderr else "No error message"
                print(f"   Error: {error_msg}")
                
        except subprocess.TimeoutExpired:
            print("⏰ Command timed out")
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    # Clean up
    shutil.rmtree(temp_dir)
    
    # If we found a working approach, create the production inference function
    if working_approach:
        print(f"\n🚀 CREATING PRODUCTION INFERENCE FUNCTION")
        
        result = {
            "status": "success",
            "model_directory": str(latest_model_dir),
            "working_approach": working_approach,
            "file_analysis": file_analysis,
            "solution": "working_inference_method_found",
            "next_steps": [
                "Deploy working inference approach to production",
                "Replace statistical fallback with real model",
                "Test with full oncoprotein prediction set"
            ]
        }
        
        return result
    
    else:
        print(f"\n❌ NO WORKING INFERENCE APPROACH FOUND")
        print(f"💡 ALTERNATIVE SOLUTIONS:")
        print(f"   1. Retrain with explicit checkpoint saving")
        print(f"   2. Use PyTorch model loading directly")
        print(f"   3. Investigate Lightning checkpoints")
        
        return {
            "status": "partial",
            "model_directory": str(latest_model_dir),
            "file_analysis": file_analysis,
            "problem": "no_working_inference_method",
            "recommendations": [
                "retrain_with_proper_checkpoints",
                "pytorch_direct_loading",
                "lightning_checkpoint_investigation"
            ]
        }

if __name__ == "__main__":
    print("🔍 CHEMPROP MODEL INVESTIGATION & FIX")
    print("=" * 50)
    
    with app.run():
        result = investigate_and_fix_chemprop_model.remote()
        
        print(f"\n📊 INVESTIGATION RESULTS")
        print("=" * 40)
        
        if result["status"] == "success":
            print("🎉 SUCCESS: Working inference method found!")
            working = result["working_approach"]
            print(f"✅ Method: {working['approach']['name']}")
            print(f"📊 Output: {working['output_shape']}")
            print(f"📋 Columns: {working['columns']}")
            
            print(f"\n🚀 NEXT: Deploying real trained model to production")
            
        elif result["status"] == "partial":
            print("⚠️ PARTIAL: Model files found but inference not working")
            print("💡 Recommendations:")
            for rec in result["recommendations"]:
                print(f"   • {rec.replace('_', ' ').title()}")
        
        else:
            print(f"❌ FAILED: {result.get('error')}")
        
        # Show file analysis
        file_analysis = result.get("file_analysis", {})
        print(f"\n📁 Model Files Summary:")
        for file_type, files in file_analysis.items():
            if files:
                print(f"   {file_type.replace('_', ' ').title()}: {len(files)}")
        
        print("=" * 50)