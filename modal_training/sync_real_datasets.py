"""
Real Modal → S3 Dataset Sync
Transfer actual curated datasets found in Modal volumes
"""
import modal
import os
import hashlib
import json
import boto3
from pathlib import Path
from datetime import datetime
import pandas as pd

# Modal setup
image = modal.Image.debian_slim().pip_install([
    "boto3",
    "pandas",
    "numpy"
])

app = modal.App("gnosis-real-dataset-sync", image=image)

# Volume references (based on exploration results)
expanded_datasets = modal.Volume.from_name("expanded-datasets")
trained_models = modal.Volume.from_name("trained-models")
gnosis_datasets = modal.Volume.from_name("gnosis-ai-datasets")

@app.function(
    volumes={
        "/expanded": expanded_datasets,
        "/models": trained_models,
        "/gnosis": gnosis_datasets
    },
    timeout=3600
)
def sync_real_curated_datasets():
    """Sync actual curated datasets to S3"""
    
    print("🔄 REAL GNOSIS Dataset Sync: Modal → S3")
    print("=" * 50)
    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id="AKIAQUTRBTW6SBJTMMN5",
        aws_secret_access_key="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",
        region_name="us-east-1"
    )
    
    bucket_name = "veridicabatabase"
    
    # Define datasets to sync based on exploration
    sync_configs = [
        {
            'name': 'bindingdb_curated',
            'version': 'v2024',
            'modal_path': '/expanded',
            'pattern': 'bindingdb_*.csv',
            's3_prefix': 'datasets/bindingdb/v2024/',
            'description': 'Curated BindingDB data for Gnosis I',
            'model': 'gnosis-i'
        },
        {
            'name': 'chembl_curated', 
            'version': 'v32',
            'modal_path': '/expanded',
            'pattern': 'chembl_*.csv',
            's3_prefix': 'datasets/chembl/v32/',
            'description': 'Curated ChEMBL data for Gnosis I',
            'model': 'gnosis-i'
        },
        {
            'name': 'gdsc_curated',
            'version': 'v1.0',
            'modal_path': '/expanded',
            'pattern': 'gdsc_*.csv',
            's3_prefix': 'datasets/gdsc/v1.0/',
            'description': 'Curated GDSC data for Gnosis II',
            'model': 'gnosis-ii'
        },
        {
            'name': 'cell_lines_data',
            'version': 'v1.0',
            'modal_path': '/gnosis',
            'pattern': 'cell_lines_*.csv',
            's3_prefix': 'datasets/cell_lines/v1.0/',
            'description': 'Cell line genomic data for Gnosis II',
            'model': 'gnosis-ii'
        }
    ]
    
    sync_results = []
    
    for config in sync_configs:
        print(f"\n📦 Syncing: {config['name']}")
        
        modal_path = Path(config['modal_path'])
        pattern = config['pattern']
        s3_prefix = config['s3_prefix']
        
        # Find matching files
        matching_files = list(modal_path.rglob(pattern))
        
        if not matching_files:
            print(f"   ⚠️  No files found matching pattern: {pattern}")
            continue
        
        dataset_result = {
            'name': config['name'],
            'version': config['version'],
            'files': [],
            'total_size_bytes': 0,
            'sync_timestamp': datetime.utcnow().isoformat(),
            'model': config['model']
        }
        
        synced_files = []
        
        for file_path in matching_files:
            if file_path.is_file():
                print(f"   📄 Processing: {file_path.name}")
                
                # Get file info
                file_size = file_path.stat().st_size
                print(f"      Size: {file_size / 1024 / 1024:.1f} MB")
                
                # Calculate checksum
                checksum = calculate_file_checksum(file_path)
                
                # Read file to get record count (for CSV files)
                record_count = None
                if file_path.suffix.lower() == '.csv':
                    try:
                        df = pd.read_csv(file_path)
                        record_count = len(df)
                        print(f"      Records: {record_count:,}")
                    except Exception as e:
                        print(f"      ⚠️  Could not read CSV: {e}")
                
                # Determine S3 key
                s3_key = f"{s3_prefix}{file_path.name}"
                
                # Upload to S3
                try:
                    s3_client.upload_file(str(file_path), bucket_name, s3_key)
                    
                    file_info = {
                        'filename': file_path.name,
                        's3_key': s3_key,
                        's3_uri': f"s3://{bucket_name}/{s3_key}",
                        'size_bytes': file_size,
                        'sha256': checksum,
                        'record_count': record_count,
                        'file_type': file_path.suffix[1:] if file_path.suffix else 'unknown'
                    }
                    
                    synced_files.append(file_info)
                    dataset_result['total_size_bytes'] += file_size
                    
                    print(f"      ✅ Uploaded: s3://{bucket_name}/{s3_key}")
                    
                except Exception as e:
                    print(f"      ❌ Upload failed: {e}")
        
        if synced_files:
            dataset_result['files'] = synced_files
            dataset_result['file_count'] = len(synced_files)
            
            # Create dataset metadata
            metadata = {
                'dataset': config,
                'sync_result': dataset_result,
                'provenance': {
                    'source': f'Modal volume: {config["modal_path"]}',
                    'sync_job': 'gnosis-real-dataset-sync',
                    'sync_timestamp': dataset_result['sync_timestamp'],
                    'total_records': sum(f.get('record_count', 0) for f in synced_files if f.get('record_count'))
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
            
            print(f"   📋 Metadata: s3://{bucket_name}/{metadata_key}")
            sync_results.append(dataset_result)
        else:
            print(f"   ⚠️  No files synced for {config['name']}")
    
    # Create comprehensive sync report
    sync_report = {
        'sync_timestamp': datetime.utcnow().isoformat(),
        'sync_job': 'gnosis-real-dataset-sync',
        'total_datasets': len(sync_results),
        'total_files': sum(r['file_count'] for r in sync_results),
        'total_size_mb': sum(r['total_size_bytes'] for r in sync_results) / (1024 * 1024),
        'total_records': sum(
            sum(f.get('record_count', 0) for f in r['files'] if f.get('record_count')) 
            for r in sync_results
        ),
        'datasets': sync_results
    }
    
    # Upload sync report
    report_key = f"datasets/_sync_reports/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_real_sync_report.json"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=report_key,
        Body=json.dumps(sync_report, indent=2),
        ContentType='application/json'
    )
    
    print(f"\n📊 FINAL SYNC SUMMARY:")
    print(f"   ✅ Datasets synced: {sync_report['total_datasets']}")
    print(f"   ✅ Files synced: {sync_report['total_files']}")
    print(f"   ✅ Total size: {sync_report['total_size_mb']:.1f} MB")
    print(f"   ✅ Total records: {sync_report['total_records']:,}")
    print(f"   📄 Sync report: s3://{bucket_name}/{report_key}")
    
    return sync_report

def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum for a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

@app.local_entrypoint()
def main():
    """Run the real dataset sync"""
    print("🚀 Starting Real GNOSIS Dataset Sync")
    print("=" * 60)
    
    result = sync_real_curated_datasets.remote()
    
    print("\n🎉 Real dataset sync completed!")
    print(f"📊 Summary: {result['total_files']} files, {result['total_size_mb']:.1f} MB")
    
    return result

if __name__ == "__main__":
    main()