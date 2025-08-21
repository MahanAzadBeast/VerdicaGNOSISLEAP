"""
Explore Modal Datasets
Check what datasets are available in Modal volumes
"""
import modal

# Modal setup
image = modal.Image.debian_slim().pip_install(["pandas", "numpy"])
app = modal.App("explore-datasets", image=image)

# Volume references  
gnosis_datasets = modal.Volume.from_name("gnosis-ai-datasets")
trained_models = modal.Volume.from_name("trained-models")
chembl_db = modal.Volume.from_name("chembl-database")
expanded_datasets = modal.Volume.from_name("expanded-datasets")

@app.function(
    volumes={
        "/gnosis": gnosis_datasets,
        "/models": trained_models,
        "/chembl": chembl_db,
        "/expanded": expanded_datasets
    }
)
def explore_available_datasets():
    """Explore what datasets are available"""
    import os
    from pathlib import Path
    
    print("🔍 EXPLORING MODAL DATASETS")
    print("=" * 50)
    
    volumes = {
        "gnosis-ai-datasets": "/gnosis",
        "trained-models": "/models", 
        "chembl-database": "/chembl",
        "expanded-datasets": "/expanded"
    }
    
    for volume_name, mount_path in volumes.items():
        print(f"\n📦 Volume: {volume_name}")
        print(f"   Mount: {mount_path}")
        
        volume_path = Path(mount_path)
        if not volume_path.exists():
            print(f"   ❌ Volume not mounted")
            continue
        
        # Count files by type
        file_counts = {}
        total_size_mb = 0
        sample_files = []
        
        try:
            for item in volume_path.rglob('*'):
                if item.is_file():
                    suffix = item.suffix.lower() or 'no_ext'
                    file_counts[suffix] = file_counts.get(suffix, 0) + 1
                    
                    # Calculate size
                    try:
                        size_bytes = item.stat().st_size
                        total_size_mb += size_bytes / (1024 * 1024)
                    except:
                        pass
                    
                    # Collect sample files
                    if len(sample_files) < 10 and suffix in ['.csv', '.json', '.parquet', '.pkl', '.pt', '.pth']:
                        sample_files.append({
                            'path': str(item.relative_to(volume_path)),
                            'size_mb': size_bytes / (1024 * 1024) if 'size_bytes' in locals() else 0,
                            'type': suffix
                        })
            
            print(f"   📊 Total size: {total_size_mb:.1f} MB")
            print(f"   📁 File types: {dict(sorted(file_counts.items()))}")
            
            if sample_files:
                print(f"   📄 Sample files:")
                for sample in sample_files[:5]:  # Show first 5
                    print(f"      {sample['type']} - {sample['path']} ({sample['size_mb']:.1f} MB)")
            
        except Exception as e:
            print(f"   ❌ Error exploring volume: {e}")
    
    return file_counts

@app.local_entrypoint()  
def main():
    """Run dataset exploration"""
    print("🚀 Exploring Available Datasets on Modal")
    results = explore_available_datasets.remote()
    print(f"\n🎯 Exploration complete!")
    return results