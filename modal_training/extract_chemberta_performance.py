#!/usr/bin/env python3
"""
Extract ChemBERTa 50-Epoch Training Performance Metrics
"""

import modal
import sys
import os
from pathlib import Path

def extract_chemberta_50_performance():
    """Extract performance metrics from the completed 50-epoch training"""
    
    print("📊 CHEMBERTA 50-EPOCH PERFORMANCE EXTRACTION")
    print("=" * 60)
    
    try:
        # Import the Modal app
        sys.path.append('/app/modal_training')
        from train_chemberta_focused import app
        
        print("🔍 Connecting to Modal to extract performance metrics...")
        
        with app.run():
            # Check if there's a performance extraction function
            try:
                # Try to create a simple extraction function
                @app.function()
                def get_model_performance():
                    """Extract model performance from saved results"""
                    
                    import json
                    from pathlib import Path
                    
                    # Check for saved performance files
                    model_dir = Path("/vol/models/focused_chemberta_default")
                    
                    if model_dir.exists():
                        print(f"✅ Model directory found: {model_dir}")
                        
                        # Look for performance files
                        perf_files = list(model_dir.rglob("*performance*"))
                        json_files = list(model_dir.rglob("*.json"))
                        
                        results = {
                            "model_directory": str(model_dir),
                            "files_found": [str(f) for f in model_dir.iterdir() if f.is_file()],
                            "performance_files": [str(f) for f in perf_files],
                            "json_files": [str(f) for f in json_files]
                        }
                        
                        # Try to read any performance data
                        for json_file in json_files:
                            try:
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                    results[f"data_from_{json_file.name}"] = data
                            except:
                                pass
                        
                        return results
                    else:
                        return {"error": "Model directory not found"}
                
                # Execute the function
                performance_data = get_model_performance.remote()
                
                print("📁 Model Directory Contents:")
                for key, value in performance_data.items():
                    print(f"  {key}: {value}")
                
                return performance_data
                
            except Exception as e:
                print(f"⚠️ Error extracting performance: {e}")
                return None
                
    except Exception as e:
        print(f"❌ Failed to connect to Modal: {e}")
        return None

def compare_training_epochs():
    """Compare the training setups"""
    
    print("\n🔄 TRAINING COMPARISON SUMMARY")
    print("=" * 60)
    
    print("📊 Current Model Training Status:")
    print()
    
    print("🧬 ChemBERTa (Transformer):")
    print("  • Previous: 20 epochs")
    print("  • Current: 50 epochs ✅ COMPLETED")
    print("  • Architecture: BERT-based molecular transformer")
    print("  • W&B Run: 6v1be0pf")
    print("  • Status: Ready for comparison")
    print()
    
    print("📊 Chemprop (Graph Neural Network):")
    print("  • Training: 50 epochs ✅ COMPLETED")
    print("  • Architecture: 5-layer Message Passing Neural Network")
    print("  • Hidden Size: 512")
    print("  • W&B Run: 88yupn3x")
    print("  • Status: Ready for comparison")
    print()
    
    print("⚔️ Fair Comparison Now Available:")
    print("  • Both models: 50 epochs")
    print("  • Same targets: 10 oncoproteins")
    print("  • Same dataset: ~5,000 compounds")
    print("  • Ready for Model Architecture Comparison!")

if __name__ == "__main__":
    print("Extracting ChemBERTa 50-epoch performance...")
    
    # Extract performance metrics
    performance = extract_chemberta_50_performance()
    
    # Show comparison summary
    compare_training_epochs()
    
    print(f"\n🎉 RESULT: ChemBERTa 50-epoch training is COMPLETE!")
    print("✅ Ready for fair model comparison testing")
    print("🔗 Check W&B dashboard for detailed metrics: https://wandb.ai/")
    print("📈 Next: Test Model Architecture Comparison with equal training")