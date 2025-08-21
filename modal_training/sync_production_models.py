"""
Sync Production Models from Modal to S3
Transfer only the final production models, not training artifacts
"""
import modal
import boto3
import hashlib
import json
from pathlib import Path
from datetime import datetime

image = modal.Image.debian_slim().pip_install(["boto3"])
app = modal.App("sync-production-models", image=image)

trained_models = modal.Volume.from_name("trained-models")

@app.function(
    volumes={"/models": trained_models},
    timeout=3600
)
def sync_production_models_only():
    """Sync only production-ready models, skip training artifacts"""
    
    print("🎯 PRODUCTION MODEL SYNC: Modal → S3")
    print("=" * 50)
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id="AKIAQUTRBTW6SBJTMMN5",
        aws_secret_access_key="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",
        region_name="us-east-1"
    )
    bucket_name = "veridicabatabase"
    
    models_path = Path("/models")
    
    # Define production model criteria
    production_criteria = {
        'include_patterns': [
            'final_model',           # Final trained models
            'best',                  # Best checkpoints  
            'production',            # Production models
            '.pth',                  # PyTorch model files
            '.pkl'                   # Pickled models
        ],
        'exclude_patterns': [
            'optimizer.pt',          # Skip optimizer states (huge files)
            'checkpoint-',           # Skip intermediate checkpoints
            'trainer_state',         # Skip training state
            'runs/',                 # Skip tensorboard logs
            'cache/'                 # Skip cache files
        ],
        'min_size_mb': 0.1,         # Skip tiny files
        'max_size_mb': 200          # Skip huge optimizer files
    }
    
    synced_models = []
    total_size_mb = 0
    
    print(f"\n🔍 Scanning for production models...")
    
    for item in models_path.rglob('*'):
        if item.is_file():
            relative_path = str(item.relative_to(models_path))
            size_mb = item.stat().st_size / (1024 * 1024)
            
            # Apply production criteria
            is_production = False
            
            # Check include patterns
            for pattern in production_criteria['include_patterns']:
                if pattern in relative_path.lower():
                    is_production = True
                    break
            
            # Check exclude patterns
            for pattern in production_criteria['exclude_patterns']:
                if pattern in relative_path.lower():
                    is_production = False
                    break
            
            # Size filters
            if size_mb < production_criteria['min_size_mb'] or size_mb > production_criteria['max_size_mb']:
                is_production = False
            
            if is_production:
                print(f"   ✅ Production model: {relative_path} ({size_mb:.1f} MB)")
                
                # Calculate checksum
                checksum = calculate_file_checksum(item)
                
                # Determine model category and S3 path
                s3_key = determine_s3_path(relative_path)
                
                try:
                    # Upload to S3
                    s3_client.upload_file(str(item), bucket_name, s3_key)
                    
                    model_info = {
                        'filename': item.name,
                        'original_path': relative_path,
                        's3_key': s3_key,
                        's3_uri': f"s3://{bucket_name}/{s3_key}",
                        'size_mb': size_mb,
                        'sha256': checksum,
                        'model_type': classify_model_type(relative_path)
                    }
                    
                    synced_models.append(model_info)
                    total_size_mb += size_mb
                    
                    print(f"      📤 Uploaded: s3://{bucket_name}/{s3_key}")
                    
                except Exception as e:
                    print(f"      ❌ Upload failed: {e}")
    
    # Create sync report
    sync_report = {
        'sync_timestamp': datetime.utcnow().isoformat(),
        'sync_type': 'production_models_only',
        'total_models': len(synced_models),
        'total_size_mb': total_size_mb,
        'models': synced_models,
        'criteria': production_criteria
    }
    
    # Upload sync report
    report_key = f"models/_sync_reports/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_production_models_sync.json"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=report_key,
        Body=json.dumps(sync_report, indent=2),
        ContentType='application/json'
    )
    
    print(f"\n📊 PRODUCTION MODEL SYNC SUMMARY:")
    print(f"   ✅ Models synced: {len(synced_models)}")
    print(f"   ✅ Total size: {total_size_mb:.1f} MB")
    print(f"   📄 Sync report: s3://{bucket_name}/{report_key}")
    
    # Show breakdown by model type
    model_types = {}
    for model in synced_models:
        model_type = model['model_type']
        model_types[model_type] = model_types.get(model_type, [])
        model_types[model_type].append(model)
    
    print(f"\n📦 Models by type:")
    for model_type, models in model_types.items():
        total_size = sum(m['size_mb'] for m in models)
        print(f"   {model_type}: {len(models)} models, {total_size:.1f} MB")
    
    return sync_report

def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum"""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def determine_s3_path(relative_path: str) -> str:
    """Determine appropriate S3 path based on model type"""
    path_lower = relative_path.lower()
    
    if 'chemberta' in path_lower and 'oncoprotein' in path_lower:
        return f"models/gnosis-i/alternatives/chemberta_oncoprotein/{Path(relative_path).name}"
    elif 'chemberta' in path_lower:
        return f"models/gnosis-i/alternatives/chemberta_focused/{Path(relative_path).name}"
    elif 'chemprop' in path_lower:
        return f"models/gnosis-i/alternatives/chemprop/{Path(relative_path).name}"
    elif 'model2' in path_lower or 'cytotox' in path_lower:
        return f"models/gnosis-ii/alternatives/{Path(relative_path).name}"
    else:
        return f"models/alternatives/uncategorized/{Path(relative_path).name}"

def classify_model_type(relative_path: str) -> str:
    """Classify model type from path"""
    path_lower = relative_path.lower()
    
    if 'chemberta' in path_lower and 'oncoprotein' in path_lower:
        return "gnosis-i-chemberta-oncoprotein"
    elif 'chemberta' in path_lower:
        return "gnosis-i-chemberta-focused"
    elif 'chemprop' in path_lower:
        return "gnosis-i-chemprop"
    elif 'model2' in path_lower or 'cytotox' in path_lower:
        return "gnosis-ii-cytotoxicity"
    else:
        return "uncategorized"

@app.local_entrypoint()
def main():
    """Run production model sync"""
    print("🚀 Starting Production Model Sync")
    print("🎯 Transferring only production-ready models (skipping training artifacts)")
    
    result = sync_production_models_only.remote()
    
    print(f"\n🎉 Production model sync completed!")
    print(f"   Transferred: {result['total_models']} models")
    print(f"   Size: {result['total_size_mb']:.1f} MB")
    
    return result

if __name__ == "__main__":
    main()