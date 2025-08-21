"""
Selective Modal Cleanup & Essential Asset Transfer
Keep only: best checkpoints, final checkpoints, compressed training logs
Remove: optimizer states, intermediate checkpoints, clutter
"""
import modal
import boto3
import json
import shutil
import gzip
from pathlib import Path
from datetime import datetime

image = modal.Image.debian_slim().pip_install(["boto3"])
app = modal.App("selective-cleanup", image=image)

trained_models = modal.Volume.from_name("trained-models")

@app.function(
    volumes={"/models": trained_models},
    timeout=3600
)
def selective_cleanup_and_transfer():
    """Clean up Modal volume and transfer only essential assets"""
    
    print("🧹 SELECTIVE MODAL CLEANUP & ESSENTIAL TRANSFER")
    print("=" * 60)
    
    # Initialize S3
    s3_client = boto3.client(
        's3',
        aws_access_key_id="AKIAQUTRBTW6SBJTMMN5",
        aws_secret_access_key="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",
        region_name="us-east-1"
    )
    bucket_name = "veridicabatabase"
    
    models_path = Path("/models")
    
    # Assets to keep and transfer
    essential_assets = {
        'best_checkpoints': [],     # Best performing models
        'final_checkpoints': [],    # End-of-run models
        'training_logs': [],        # Compressed logs
        'essential_configs': []     # Critical config files
    }
    
    # Assets to remove
    removal_targets = {
        'optimizer_states': [],     # ~6.8 GB of optimizer.pt files
        'intermediate_checkpoints': [], # checkpoint-1000, checkpoint-2000, etc.
        'cache_files': [],          # Various cache and temp files
        'duplicate_configs': []     # Redundant config files
    }
    
    cleanup_stats = {
        'files_scanned': 0,
        'kept_size_mb': 0,
        'removed_size_mb': 0,
        'transferred_mb': 0
    }
    
    print(f"\n🔍 Phase 1: Scanning and categorizing assets...")
    
    for item in models_path.rglob('*'):
        if item.is_file():
            cleanup_stats['files_scanned'] += 1
            relative_path = str(item.relative_to(models_path))
            size_mb = item.stat().st_size / (1024 * 1024)
            
            # Categorize files
            file_lower = relative_path.lower()
            
            # ESSENTIAL ASSETS TO KEEP
            if ('final_model' in file_lower or 
                'best' in file_lower and 'epoch' in file_lower or
                file_lower.endswith('.pth') and 'production' in file_lower):
                essential_assets['best_checkpoints'].append((item, relative_path, size_mb))
                cleanup_stats['kept_size_mb'] += size_mb
                
            elif (file_lower.endswith('.ckpt') and 'best' in file_lower):
                essential_assets['final_checkpoints'].append((item, relative_path, size_mb))
                cleanup_stats['kept_size_mb'] += size_mb
                
            elif ('events.out.tfevents' in file_lower or 
                  'metrics.csv' in file_lower or
                  'trainer_logs' in file_lower):
                essential_assets['training_logs'].append((item, relative_path, size_mb))
                cleanup_stats['kept_size_mb'] += size_mb
                
            elif (file_lower.endswith('.json') and 
                  ('config' in file_lower or 'trainer_state' in file_lower) and
                  'final_model' in relative_path):
                essential_assets['essential_configs'].append((item, relative_path, size_mb))
                cleanup_stats['kept_size_mb'] += size_mb
            
            # REMOVAL TARGETS
            elif 'optimizer.pt' in file_lower:
                removal_targets['optimizer_states'].append((item, relative_path, size_mb))
                cleanup_stats['removed_size_mb'] += size_mb
                
            elif ('checkpoint-' in file_lower and 
                  not ('best' in file_lower or 'final' in file_lower)):
                removal_targets['intermediate_checkpoints'].append((item, relative_path, size_mb))
                cleanup_stats['removed_size_mb'] += size_mb
                
            elif (file_lower.endswith('.cache') or 
                  'temp' in file_lower or
                  '__pycache__' in file_lower):
                removal_targets['cache_files'].append((item, relative_path, size_mb))
                cleanup_stats['removed_size_mb'] += size_mb
                
            elif (file_lower.endswith('.json') and 
                  len([f for f in item.parent.glob('*.json')]) > 3):  # Too many configs
                removal_targets['duplicate_configs'].append((item, relative_path, size_mb))
                cleanup_stats['removed_size_mb'] += size_mb
    
    # Show analysis
    print(f"\n📊 CLEANUP ANALYSIS:")
    print(f"   Files scanned: {cleanup_stats['files_scanned']}")
    print(f"   Keep: {cleanup_stats['kept_size_mb']:.1f} MB")
    print(f"   Remove: {cleanup_stats['removed_size_mb']:.1f} MB")
    print(f"   Space savings: {cleanup_stats['removed_size_mb'] / (cleanup_stats['kept_size_mb'] + cleanup_stats['removed_size_mb']) * 100:.1f}%")
    
    # Phase 2: Transfer essential assets to S3
    print(f"\n📤 Phase 2: Transferring essential assets to S3...")
    
    transferred_assets = []
    
    for category, assets in essential_assets.items():
        if assets:
            print(f"\n   📦 {category}: {len(assets)} files")
            
            for item, relative_path, size_mb in assets:
                # Determine S3 key
                s3_key = f"models/essential/{category}/{item.name}"
                
                try:
                    # Special handling for training logs - compress them
                    if category == 'training_logs' and size_mb > 1:
                        compressed_path = f"/tmp/{item.name}.gz"
                        with open(item, 'rb') as f_in:
                            with gzip.open(compressed_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        # Upload compressed version
                        s3_client.upload_file(compressed_path, bucket_name, s3_key + '.gz')
                        compressed_size = Path(compressed_path).stat().st_size / (1024 * 1024)
                        cleanup_stats['transferred_mb'] += compressed_size
                        
                        print(f"      ✅ {item.name} → {compressed_size:.1f} MB (compressed from {size_mb:.1f} MB)")
                        
                    else:
                        # Upload as-is
                        s3_client.upload_file(str(item), bucket_name, s3_key)
                        cleanup_stats['transferred_mb'] += size_mb
                        print(f"      ✅ {item.name} → {size_mb:.1f} MB")
                    
                    transferred_assets.append({
                        'category': category,
                        'filename': item.name,
                        'original_path': relative_path,
                        's3_key': s3_key,
                        'size_mb': size_mb,
                        'compressed': category == 'training_logs' and size_mb > 1
                    })
                    
                except Exception as e:
                    print(f"      ❌ Failed to transfer {item.name}: {e}")
    
    # Phase 3: Remove clutter from Modal
    print(f"\n🗑️  Phase 3: Removing clutter from Modal...")
    
    removed_count = 0
    removed_size_mb = 0
    
    for category, targets in removal_targets.items():
        if targets:
            print(f"\n   🗑️  Removing {category}: {len(targets)} files")
            
            for item, relative_path, size_mb in targets:
                try:
                    item.unlink()  # Delete the file
                    removed_count += 1
                    removed_size_mb += size_mb
                    
                    if removed_count % 50 == 0:  # Progress indicator
                        print(f"      Removed {removed_count} files, {removed_size_mb:.1f} MB freed")
                        
                except Exception as e:
                    print(f"      ⚠️ Failed to remove {relative_path}: {e}")
    
    # Clean up empty directories
    print(f"\n🧹 Phase 4: Cleaning up empty directories...")
    empty_dirs_removed = 0
    
    for item in sorted(models_path.rglob('*'), key=lambda p: str(p), reverse=True):
        if item.is_dir() and not any(item.iterdir()):
            try:
                item.rmdir()
                empty_dirs_removed += 1
            except:
                pass
    
    # Final summary
    cleanup_report = {
        'cleanup_timestamp': datetime.utcnow().isoformat(),
        'stats': cleanup_stats,
        'transferred_assets': transferred_assets,
        'removed_files': removed_count,
        'removed_size_mb': removed_size_mb,
        'empty_dirs_removed': empty_dirs_removed,
        'final_modal_size_estimate_gb': (cleanup_stats['kept_size_mb'] - cleanup_stats['transferred_mb']) / 1024
    }
    
    # Upload cleanup report
    report_key = f"models/_cleanup_reports/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_modal_cleanup.json"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=report_key,
        Body=json.dumps(cleanup_report, indent=2),
        ContentType='application/json'
    )
    
    print(f"\n🎉 CLEANUP COMPLETE!")
    print(f"   ✅ Transferred: {cleanup_stats['transferred_mb']:.1f} MB to S3")
    print(f"   🗑️  Removed: {removed_size_mb:.1f} MB from Modal")
    print(f"   📁 Empty dirs removed: {empty_dirs_removed}")
    print(f"   📊 Estimated Modal size after cleanup: {cleanup_report['final_modal_size_estimate_gb']:.1f} GB")
    print(f"   📄 Cleanup report: s3://{bucket_name}/{report_key}")
    
    return cleanup_report

@app.local_entrypoint()
def main():
    """Run selective cleanup"""
    print("🚀 Starting Selective Modal Cleanup")
    print("🎯 Strategy: Keep best/final checkpoints + logs, remove optimizer states + intermediate checkpoints")
    
    result = selective_cleanup_and_transfer.remote()
    
    print(f"\n📈 CLEANUP RESULTS:")
    print(f"   Modal storage freed: {result['removed_size_mb']:.1f} MB")
    print(f"   Essential assets secured in S3: {result['stats']['transferred_mb']:.1f} MB")
    print(f"   Estimated Modal size after cleanup: {result['final_modal_size_estimate_gb']:.1f} GB")
    
    if result['final_modal_size_estimate_gb'] < 1:
        print("   🎉 Modal volume now under 1 GB - mission accomplished!")
    
    return result

if __name__ == "__main__":
    main()