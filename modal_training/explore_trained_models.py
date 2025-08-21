"""
Explore Trained Models in Modal
Detailed analysis of the 10.7 GB of trained models
"""
import modal
from pathlib import Path

image = modal.Image.debian_slim().pip_install(["pandas"])
app = modal.App("explore-trained-models", image=image)

trained_models = modal.Volume.from_name("trained-models")

@app.function(
    volumes={"/models": trained_models}
)
def explore_trained_models_detailed():
    """Detailed exploration of trained models"""
    
    print("🔍 DETAILED TRAINED MODELS ANALYSIS")
    print("=" * 60)
    
    models_path = Path("/models")
    
    if not models_path.exists():
        print("❌ Models path not found")
        return
    
    # Categorize files by type and size
    file_analysis = {
        'model_files': [],      # .pt, .pth, .pkl
        'checkpoints': [],      # .ckpt 
        'huggingface': [],      # .bin, .safetensors
        'metadata': [],         # .json, .yaml
        'reports': [],          # evaluation files
        'other': []
    }
    
    total_size_gb = 0
    
    print("\n📊 Scanning all files...")
    
    for item in models_path.rglob('*'):
        if item.is_file():
            try:
                size_mb = item.stat().st_size / (1024 * 1024)
                total_size_gb += size_mb / 1024
                
                # Categorize by extension and path
                suffix = item.suffix.lower()
                relative_path = str(item.relative_to(models_path))
                
                file_info = {
                    'path': relative_path,
                    'size_mb': size_mb,
                    'type': suffix
                }
                
                if suffix in ['.pt', '.pth', '.pkl']:
                    file_analysis['model_files'].append(file_info)
                elif suffix == '.ckpt':
                    file_analysis['checkpoints'].append(file_info)
                elif suffix in ['.bin', '.safetensors']:
                    file_analysis['huggingface'].append(file_info)
                elif suffix in ['.json', '.yaml', '.yml']:
                    file_analysis['metadata'].append(file_info)
                elif 'eval' in relative_path or 'report' in relative_path or suffix in ['.csv', '.txt']:
                    file_analysis['reports'].append(file_info)
                else:
                    file_analysis['other'].append(file_info)
                    
            except Exception as e:
                print(f"   ⚠️ Error processing {item}: {e}")
    
    print(f"\n📈 TOTAL SIZE: {total_size_gb:.1f} GB")
    
    # Show breakdown by category
    for category, files in file_analysis.items():
        if files:
            total_size = sum(f['size_mb'] for f in files)
            print(f"\n📦 {category.upper()} ({len(files)} files, {total_size:.1f} MB):")
            
            # Sort by size, show largest files
            files_sorted = sorted(files, key=lambda x: x['size_mb'], reverse=True)
            
            for file_info in files_sorted[:10]:  # Top 10 largest
                print(f"   {file_info['size_mb']:8.1f} MB - {file_info['path']}")
            
            if len(files) > 10:
                print(f"   ... and {len(files)-10} more files")
    
    # Identify potential production models
    print(f"\n🎯 POTENTIAL PRODUCTION MODELS:")
    
    production_candidates = []
    for file_info in file_analysis['model_files']:
        path = file_info['path'].lower()
        size_mb = file_info['size_mb']
        
        # Look for production indicators
        if any(keyword in path for keyword in ['best', 'final', 'production', 'v1', 'gnosis']):
            production_candidates.append(file_info)
        elif size_mb > 100:  # Large models likely to be important
            production_candidates.append(file_info)
    
    production_candidates = sorted(production_candidates, key=lambda x: x['size_mb'], reverse=True)
    
    for candidate in production_candidates[:15]:  # Top 15 candidates
        print(f"   🌟 {candidate['size_mb']:8.1f} MB - {candidate['path']}")
    
    return file_analysis

@app.local_entrypoint()
def main():
    """Run detailed model exploration"""
    results = explore_trained_models_detailed.remote()
    print("\n🎯 Exploration complete!")
    return results

if __name__ == "__main__":
    main()