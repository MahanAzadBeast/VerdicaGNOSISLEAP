"""
Verify Modal Cleanup Results
Check what remains on Modal and what's secured in S3
"""
import modal
from pathlib import Path

image = modal.Image.debian_slim()
app = modal.App("verify-cleanup", image=image)

trained_models = modal.Volume.from_name("trained-models")

@app.function(volumes={"/models": trained_models})
def verify_modal_after_cleanup():
    """Verify what remains on Modal after cleanup"""
    
    print("🔍 MODAL VOLUME VERIFICATION AFTER CLEANUP")
    print("=" * 50)
    
    models_path = Path("/models")
    
    if not models_path.exists():
        print("❌ Models path not found")
        return
    
    # Scan remaining files
    remaining_files = []
    total_size_gb = 0
    
    for item in models_path.rglob('*'):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            total_size_gb += size_mb / 1024
            
            relative_path = str(item.relative_to(models_path))
            remaining_files.append({
                'path': relative_path,
                'size_mb': size_mb,
                'type': item.suffix.lower()
            })
    
    print(f"📊 REMAINING ON MODAL:")
    print(f"   Total files: {len(remaining_files)}")
    print(f"   Total size: {total_size_gb:.2f} GB")
    
    # Categorize remaining files
    categories = {}
    for file_info in remaining_files:
        file_type = file_info['type'] or 'no_extension'
        if file_type not in categories:
            categories[file_type] = []
        categories[file_type].append(file_info)
    
    # Show breakdown
    for file_type, files in sorted(categories.items(), key=lambda x: sum(f['size_mb'] for f in x[1]), reverse=True):
        total_size = sum(f['size_mb'] for f in files)
        print(f"\n   📁 {file_type}: {len(files)} files, {total_size:.1f} MB")
        
        # Show largest files of this type
        largest_files = sorted(files, key=lambda x: x['size_mb'], reverse=True)[:5]
        for file_info in largest_files:
            if file_info['size_mb'] > 1:  # Only show files > 1 MB
                print(f"      {file_info['size_mb']:6.1f} MB - {file_info['path']}")
    
    # Check if cleanup was successful
    success_criteria = {
        'under_2gb': total_size_gb < 2,
        'no_optimizer_states': not any('optimizer.pt' in f['path'] for f in remaining_files),
        'minimal_checkpoints': len([f for f in remaining_files if 'checkpoint-' in f['path']]) < 10
    }
    
    print(f"\n✅ CLEANUP SUCCESS CRITERIA:")
    for criterion, passed in success_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {criterion}: {status}")
    
    overall_success = all(success_criteria.values())
    print(f"\n🎯 OVERALL CLEANUP: {'✅ SUCCESS' if overall_success else '❌ NEEDS MORE WORK'}")
    
    return {
        'remaining_files': len(remaining_files),
        'remaining_size_gb': total_size_gb,
        'categories': {k: len(v) for k, v in categories.items()},
        'success_criteria': success_criteria,
        'overall_success': overall_success
    }

@app.local_entrypoint()
def main():
    """Run verification"""
    result = verify_modal_after_cleanup.remote()
    
    if result['overall_success']:
        print(f"\n🎉 Cleanup was successful!")
        print(f"   Modal volume reduced to {result['remaining_size_gb']:.2f} GB")
    else:
        print(f"\n⚠️ Cleanup needs improvement")
        print(f"   Modal volume still at {result['remaining_size_gb']:.2f} GB")
    
    return result

if __name__ == "__main__":
    main()