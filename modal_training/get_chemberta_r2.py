#!/usr/bin/env python3
"""
Get ChemBERTa 50-Epoch R² Performance Scores
"""

import modal
import sys
import os

# Modal app setup
app = modal.App("chemberta-performance-check")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0"
    ])
)

models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/models": models_volume},
    timeout=300
)
def extract_chemberta_r2_scores():
    """Extract R² scores from the completed ChemBERTa 50-epoch training"""
    
    import json
    import pickle
    from pathlib import Path
    
    print("🔍 EXTRACTING CHEMBERTA 50-EPOCH R² SCORES")
    print("=" * 50)
    
    # Check model directory
    model_dir = Path("/vol/models/focused_chemberta_default")
    
    if not model_dir.exists():
        return {"error": "Model directory not found", "path": str(model_dir)}
    
    print(f"✅ Model directory found: {model_dir}")
    
    # List all files in the directory
    all_files = []
    for item in model_dir.rglob("*"):
        if item.is_file():
            all_files.append(str(item.relative_to(model_dir)))
    
    print(f"📁 Found {len(all_files)} files in model directory")
    
    results = {
        "model_directory": str(model_dir),
        "files_found": all_files,
        "r2_scores": {},
        "performance_summary": {}
    }
    
    # Look for performance files
    performance_files = [
        "final_performance.json",
        "training_results.json", 
        "performance.json",
        "metrics.json",
        "results.json"
    ]
    
    for perf_file in performance_files:
        file_path = model_dir / perf_file
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    results[f"data_from_{perf_file}"] = data
                    print(f"✅ Loaded: {perf_file}")
                    
                    # Extract R² scores if available
                    if 'r2_scores' in data:
                        results['r2_scores'] = data['r2_scores']
                    elif 'target_performance' in data:
                        results['target_performance'] = data['target_performance']
                    elif 'final_metrics' in data:
                        results['final_metrics'] = data['final_metrics']
                        
            except Exception as e:
                print(f"⚠️ Failed to read {perf_file}: {e}")
    
    # Look for pickle files that might contain performance data
    pickle_files = list(model_dir.rglob("*.pkl"))
    for pkl_file in pickle_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    results[f"pickle_data_{pkl_file.name}"] = data
                    print(f"✅ Loaded pickle: {pkl_file.name}")
        except Exception as e:
            print(f"⚠️ Failed to read pickle {pkl_file.name}: {e}")
    
    # Look for trainer state files
    trainer_files = list(model_dir.rglob("trainer_state.json"))
    for trainer_file in trainer_files:
        try:
            with open(trainer_file, 'r') as f:
                trainer_data = json.load(f)
                results["trainer_state"] = trainer_data
                print(f"✅ Loaded trainer state")
                
                # Extract final metrics from trainer logs
                if 'log_history' in trainer_data:
                    logs = trainer_data['log_history']
                    if logs:
                        final_log = logs[-1]
                        results["final_training_log"] = final_log
                        
        except Exception as e:
            print(f"⚠️ Failed to read trainer state: {e}")
    
    return results

if __name__ == "__main__":
    print("🧬 Extracting ChemBERTa 50-epoch R² scores...")
    
    with app.run():
        performance_data = extract_chemberta_r2_scores.remote()
        
        print("\n📊 PERFORMANCE EXTRACTION RESULTS")
        print("=" * 50)
        
        if "error" in performance_data:
            print(f"❌ Error: {performance_data['error']}")
        else:
            print(f"📁 Model directory: {performance_data['model_directory']}")
            print(f"📂 Files found: {len(performance_data['files_found'])}")
            
            # Display R² scores if found
            if 'r2_scores' in performance_data and performance_data['r2_scores']:
                print("\n🎯 R² SCORES BY TARGET:")
                for target, r2 in performance_data['r2_scores'].items():
                    print(f"  {target}: {r2:.3f}")
                    
                # Calculate mean R²
                r2_values = list(performance_data['r2_scores'].values())
                mean_r2 = sum(r2_values) / len(r2_values)
                print(f"\n📊 MEAN R²: {mean_r2:.3f}")
                
            elif 'target_performance' in performance_data:
                print("\n🎯 TARGET PERFORMANCE:")
                perf = performance_data['target_performance']
                print(perf)
                
            elif 'final_metrics' in performance_data:
                print("\n🎯 FINAL METRICS:")
                metrics = performance_data['final_metrics']
                print(metrics)
                
            elif 'final_training_log' in performance_data:
                print("\n🎯 FINAL TRAINING LOG:")
                log = performance_data['final_training_log']
                print(f"Training step: {log.get('step', 'N/A')}")
                print(f"Training loss: {log.get('train_loss', 'N/A')}")
                if 'eval_loss' in log:
                    print(f"Eval loss: {log.get('eval_loss', 'N/A')}")
                
            else:
                print("\n⚠️ No R² scores found in standard locations")
                print("📋 Available data keys:")
                for key in performance_data.keys():
                    if key not in ['model_directory', 'files_found']:
                        print(f"  • {key}")
                
        print(f"\n🔗 W&B Run ID: 6v1be0pf")
        print("📈 Check W&B dashboard for detailed training curves")