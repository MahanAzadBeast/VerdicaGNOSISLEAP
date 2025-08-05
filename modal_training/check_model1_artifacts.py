import modal
import os
import glob

app = modal.App("check-model1-artifacts")

@app.function(volumes={"/data": modal.Volume.from_name("trained-models", create_if_missing=False)})
def check_trained_models():
    print("=== CHECKING TRAINED-MODELS VOLUME ===")
    
    # Look for any model1 related files
    patterns = [
        "/data/*model1*",
        "/data/*gnosis*model1*", 
        "/data/*simplified*",
        "/data/best_*.pt",
        "/data/*.pt"
    ]
    
    found_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        for f in files:
            if os.path.isfile(f):
                size = os.path.getsize(f) / (1024*1024)  # MB
                mtime = os.path.getmtime(f)
                found_files.append((f, size, mtime))
                
    if found_files:
        print(f"✅ Found {len(found_files)} files:")
        for path, size, mtime in found_files:
            print(f"  📁 {path} ({size:.1f} MB)")
    else:
        print("❌ No model files found")
        
    return found_files

@app.function(volumes={"/data": modal.Volume.from_name("chemberta-models", create_if_missing=False)})
def check_chemberta_models():
    print("=== CHECKING CHEMBERTA-MODELS VOLUME ===")
    
    patterns = [
        "/data/*model1*",
        "/data/*simplified*",
        "/data/*.pt"
    ]
    
    found_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        for f in files:
            if os.path.isfile(f):
                size = os.path.getsize(f) / (1024*1024)
                found_files.append((f, size))
                
    if found_files:
        print(f"✅ Found {len(found_files)} files:")
        for path, size in found_files:
            print(f"  📁 {path} ({size:.1f} MB)")
    else:
        print("❌ No model files found")
        
    return found_files

if __name__ == "__main__":
    with app.run():
        trained_results = check_trained_models.remote()
        chemberta_results = check_chemberta_models.remote()
        
        print("\n=== SUMMARY ===")
        total = len(trained_results) + len(chemberta_results)
        print(f"Total Model 1 artifacts found: {total}")
        
        if total > 0:
            print("✅ Model 1 training weights ARE available!")
        else:
            print("❌ No Model 1 checkpoints found - need to restart training")