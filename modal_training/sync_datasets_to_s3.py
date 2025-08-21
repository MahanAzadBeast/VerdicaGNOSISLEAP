"""
Modal → S3 Dataset Sync Job
Transfer curated datasets, splits, and eval reports with checksums
"""
import modal
import os
import hashlib
import json
import boto3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Modal setup
image = modal.Image.debian_slim().pip_install([
    "boto3",
    "pandas",
    "numpy"
])

app = modal.App("gnosis-dataset-sync", image=image)

# Volume references (using actual Modal volume names)
datasets_volume = modal.Volume.from_name("gnosis-ai-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)

@app.function(
    volumes={
        "/datasets": datasets_volume,
        "/models": models_volume
    },
    timeout=3600  # 1 hour timeout
)
def sync_curated_datasets_to_s3():
    """Sync curated datasets from Modal volumes to S3"""
    
    print("🔄 GNOSIS Dataset Sync: Modal → S3")
    print("=" * 50)
    
    # Initialize S3 client with hardcoded credentials (for demo)
    s3_client = boto3.client(
        's3',
        aws_access_key_id="AKIAQUTRBTW6SBJTMMN5",
        aws_secret_access_key="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",
        region_name="us-east-1"
    )
    
    bucket_name = "veridicabatabase"
    
    # Define curated datasets to sync
    dataset_configs = [
        {
            'name': 'chembl_bioactivity',
            'version': 'v32',
            'modal_path': '/datasets/chembl/processed/',
            's3_prefix': 'datasets/chembl/v32/',
            'description': 'Curated ChEMBL bioactivity data for Gnosis I training',
            'model': 'gnosis-i'
        },
        {
            'name': 'bindingdb_bioactivity', 
            'version': 'v2024',
            'modal_path': '/datasets/bindingdb/processed/',
            's3_prefix': 'datasets/bindingdb/v2024/',
            'description': 'Curated BindingDB bioactivity data for Gnosis I training',
            'model': 'gnosis-i'
        },
        {
            'name': 'gdsc_cytotoxicity',
            'version': 'v1.0',
            'modal_path': '/datasets/gdsc/processed/',
            's3_prefix': 'datasets/gdsc/v1.0/',
            'description': 'Curated GDSC cytotoxicity data for Gnosis II training',
            'model': 'gnosis-ii'
        }
    ]
    
    sync_results = []
    
    for config in dataset_configs:
        print(f"\n📦 Processing dataset: {config['name']}")
        
        modal_path = Path(config['modal_path'])
        s3_prefix = config['s3_prefix']
        
        # Check if Modal path exists
        if not modal_path.exists():
            print(f"   ⚠️  Modal path not found: {modal_path}")
            print(f"   🔍 Available paths in /datasets:")
            
            # List available directories
            datasets_root = Path('/datasets')
            if datasets_root.exists():
                for item in datasets_root.rglob('*'):
                    if item.is_file() and item.suffix in ['.csv', '.json', '.parquet']:
                        print(f"      📄 {item}")
            continue
        
        dataset_result = {
            'name': config['name'],
            'version': config['version'],
            'files': [],
            'total_size_bytes': 0,
            'sync_timestamp': datetime.utcnow().isoformat(),
            'model': config['model']
        }
        
        # Find and sync curated files
        file_patterns = ['*.csv', '*.json', '*.parquet', '*.pkl']
        synced_files = []
        
        for pattern in file_patterns:
            for file_path in modal_path.rglob(pattern):
                if file_path.is_file():
                    print(f"   📄 Found: {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")
                    
                    # Calculate checksum
                    checksum = calculate_file_checksum(file_path)
                    file_size = file_path.stat().st_size
                    
                    # Determine S3 key
                    relative_path = file_path.relative_to(modal_path)
                    s3_key = f"{s3_prefix}{relative_path}"
                    
                    # Upload to S3
                    try:
                        s3_client.upload_file(str(file_path), bucket_name, s3_key)
                        
                        file_info = {
                            'filename': file_path.name,
                            'relative_path': str(relative_path),
                            's3_key': s3_key,
                            's3_uri': f"s3://{bucket_name}/{s3_key}",
                            'size_bytes': file_size,
                            'sha256': checksum,
                            'file_type': file_path.suffix[1:] if file_path.suffix else 'unknown'
                        }
                        
                        synced_files.append(file_info)
                        dataset_result['total_size_bytes'] += file_size
                        
                        print(f"      ✅ Uploaded to S3: {s3_key}")
                        
                    except Exception as e:
                        print(f"      ❌ Upload failed: {e}")
        
        dataset_result['files'] = synced_files
        dataset_result['file_count'] = len(synced_files)
        
        if synced_files:
            # Create dataset metadata file
            metadata = {
                'dataset': config,
                'sync_result': dataset_result,
                'provenance': {
                    'source': 'Modal volumes',
                    'sync_job': 'gnosis-dataset-sync',
                    'sync_timestamp': dataset_result['sync_timestamp']
                }
            }
            
            # Upload metadata
            metadata_key = f"{s3_prefix}metadata.json"
            s3_client.put_object(
                Bucket=bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2),
                ContentType='application/json'
            )
            
            print(f"   📋 Metadata uploaded: {metadata_key}")
        
        sync_results.append(dataset_result)
    
    # Create overall sync report
    sync_report = {
        'sync_timestamp': datetime.utcnow().isoformat(),
        'total_datasets': len(sync_results),
        'total_files': sum(r['file_count'] for r in sync_results),
        'total_size_mb': sum(r['total_size_bytes'] for r in sync_results) / (1024 * 1024),
        'datasets': sync_results
    }
    
    # Upload sync report
    report_key = f"datasets/_sync_reports/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_sync_report.json"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=report_key,
        Body=json.dumps(sync_report, indent=2),
        ContentType='application/json'
    )
    
    print(f"\n📊 SYNC SUMMARY:")
    print(f"   Datasets synced: {sync_report['total_datasets']}")
    print(f"   Files synced: {sync_report['total_files']}")
    print(f"   Total size: {sync_report['total_size_mb']:.1f} MB")
    print(f"   Sync report: s3://{bucket_name}/{report_key}")
    
    return sync_report

def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum for a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

@app.function(
    volumes={
        "/models": models_volume
    }
)
def sync_eval_reports_to_s3():
    """Sync model evaluation reports to S3"""
    
    print("📊 Syncing evaluation reports...")
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id="AKIAQUTRBTW6SBJTMMN5",
        aws_secret_access_key="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",
        region_name="us-east-1"
    )
    
    bucket_name = "veridicabatabase"
    
    # Look for evaluation reports
    models_path = Path('/models')
    eval_files = []
    
    if models_path.exists():
        # Find evaluation reports
        for eval_file in models_path.rglob('*eval*'):
            if eval_file.is_file() and eval_file.suffix in ['.json', '.csv', '.txt']:
                eval_files.append(eval_file)
        
        for eval_file in models_path.rglob('*report*'):
            if eval_file.is_file() and eval_file.suffix in ['.json', '.csv', '.txt']:
                eval_files.append(eval_file)
    
    synced_reports = []
    
    for eval_file in eval_files:
        # Determine which model this report belongs to
        model_slug = 'unknown'
        if 'gnosis-i' in str(eval_file) or 'model1' in str(eval_file):
            model_slug = 'gnosis-i'
        elif 'gnosis-ii' in str(eval_file) or 'model2' in str(eval_file) or 'cytotox' in str(eval_file):
            model_slug = 'gnosis-ii'
        
        # Upload to S3
        s3_key = f"models/{model_slug}/1.0.0/reports/{eval_file.name}"
        
        try:
            s3_client.upload_file(str(eval_file), bucket_name, s3_key)
            
            synced_reports.append({
                'filename': eval_file.name,
                's3_uri': f"s3://{bucket_name}/{s3_key}",
                'model': model_slug,
                'size_bytes': eval_file.stat().st_size
            })
            
            print(f"   ✅ {eval_file.name} → {model_slug}")
            
        except Exception as e:
            print(f"   ❌ Failed to sync {eval_file.name}: {e}")
    
    return synced_reports

@app.local_entrypoint()
def main():
    """Run the complete dataset sync"""
    print("🚀 Starting GNOSIS Dataset Sync Job")
    print("=" * 60)
    
    # Sync curated datasets
    dataset_results = sync_curated_datasets_to_s3.remote()
    
    # Sync evaluation reports
    eval_results = sync_eval_reports_to_s3.remote()
    
    print("\n🎉 Sync job completed!")
    print(f"   Dataset sync: {dataset_results}")
    print(f"   Eval reports: {eval_results}")
    
    return {
        'datasets': dataset_results,
        'eval_reports': eval_results
    }

if __name__ == "__main__":
    main()