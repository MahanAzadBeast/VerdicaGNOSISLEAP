"""
Check for Saved Model Weights
Look for any checkpoints or trained models that were saved during training
"""

import modal
from pathlib import Path
import os

image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas")
app = modal.App("check-saved-models")

models_volume = modal.Volume.from_name("cell-line-models", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/models": models_volume}
)
def check_saved_model_weights():
    """Check what model weights were saved during training"""
    
    print("🔍 CHECKING FOR SAVED MODEL WEIGHTS")
    print("=" * 60)
    
    models_dir = Path("/models")
    
    # Check for Model 1 checkpoints
    model1_dir = models_dir / "model1_checkpoints"
    print(f"\n📁 MODEL 1 CHECKPOINTS ({model1_dir}):")
    if model1_dir.exists():
        model1_files = list(model1_dir.glob("*"))
        if model1_files:
            print(f"   ✅ Found {len(model1_files)} files:")
            for f in sorted(model1_files):
                size_mb = f.stat().st_size / (1024*1024)
                print(f"     📄 {f.name} ({size_mb:.1f} MB)")
        else:
            print("   ❌ Directory exists but no files found")
    else:
        print("   ❌ No Model 1 checkpoint directory found")
    
    # Check for Model 2 checkpoints
    model2_dir = models_dir / "model2_cancer_checkpoints"
    print(f"\n📁 MODEL 2 CHECKPOINTS ({model2_dir}):")
    if model2_dir.exists():
        model2_files = list(model2_dir.glob("*"))
        if model2_files:
            print(f"   ✅ Found {len(model2_files)} files:")
            for f in sorted(model2_files):
                size_mb = f.stat().st_size / (1024*1024)
                print(f"     📄 {f.name} ({size_mb:.1f} MB)")
        else:
            print("   ❌ Directory exists but no files found")
    else:
        print("   ❌ No Model 2 checkpoint directory found")
    
    # Check for any other model files
    print(f"\n📁 ALL MODEL FILES:")
    all_model_files = []
    for pattern in ["*.pt", "*.pth", "*.pkl", "*.json"]:
        all_model_files.extend(models_dir.rglob(pattern))
    
    if all_model_files:
        print(f"   ✅ Found {len(all_model_files)} total model-related files:")
        for f in sorted(all_model_files):
            size_mb = f.stat().st_size / (1024*1024)
            rel_path = f.relative_to(models_dir)
            print(f"     📄 {rel_path} ({size_mb:.1f} MB)")
    else:
        print("   ❌ No model files found in the entire volume")
    
    # Check root directory files
    print(f"\n📁 ROOT MODEL DIRECTORY CONTENTS:")
    root_files = [f for f in models_dir.iterdir() if f.is_file()]
    if root_files:
        for f in root_files:
            size_mb = f.stat().st_size / (1024*1024)
            print(f"     📄 {f.name} ({size_mb:.1f} MB)")
    
    # Summary
    total_checkpoints = len([f for f in all_model_files if f.suffix in ['.pt', '.pth', '.pkl']])
    print(f"\n📊 SUMMARY:")
    print(f"   • Total model weight files: {total_checkpoints}")
    print(f"   • Model 1 checkpoints: {'✅ Available' if any('model1' in str(f) for f in all_model_files) else '❌ None'}")
    print(f"   • Model 2 checkpoints: {'✅ Available' if any('model2' in str(f) for f in all_model_files) else '❌ None'}")
    
    return {
        'total_files': len(all_model_files),
        'model1_available': any('model1' in str(f) for f in all_model_files),
        'model2_available': any('model2' in str(f) for f in all_model_files),
        'checkpoint_files': total_checkpoints
    }

if __name__ == "__main__":
    with app.run():
        result = check_saved_model_weights.remote()
        print("Model weight check completed:", result)