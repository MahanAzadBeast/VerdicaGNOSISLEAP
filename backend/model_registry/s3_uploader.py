#!/usr/bin/env python3
"""
S3 Model Uploader
Upload existing model files to S3 and update registry
"""
import os
import sys
import asyncio
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / '.env')

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from model_registry.database import get_db_manager
from model_registry.s3_manager import get_s3_manager
from model_registry.schemas import ArtifactDB

async def upload_model_files():
    """Upload existing model files to S3 and update registry"""
    print("🚀 Starting model file upload to S3...")
    
    # Initialize managers
    db = get_db_manager()
    await db.connect()
    s3 = get_s3_manager()
    
    # Model files to upload
    model_files = [
        {
            'local_path': '/app/backend/models/gnosis_model1_best.pt',
            'model_slug': 'gnosis-i',
            'version': '1.0.0',
            's3_key': 'models/gnosis-i/1.0.0/gnosis_model1_best.pt',
            'artifact_kind': 'model'
        },
        {
            'local_path': '/app/models/real_gdsc_chemberta_cytotox_v1.pth',
            'model_slug': 'cytotoxicity-predictor', 
            'version': '1.0.0',
            's3_key': 'models/cytotoxicity-predictor/1.0.0/real_gdsc_chemberta_cytotox_v1.pth',
            'artifact_kind': 'model'
        }
    ]
    
    uploaded_files = []
    
    for model_file in model_files:
        local_path = model_file['local_path']
        s3_key = model_file['s3_key']
        model_slug = model_file['model_slug']
        
        print(f"\n📁 Processing: {model_slug}")
        
        # Check if local file exists
        if not os.path.exists(local_path):
            print(f"⚠️  File not found: {local_path}")
            continue
        
        # Calculate file size and hash
        file_size = os.path.getsize(local_path)
        print(f"   File size: {file_size / (1024*1024):.1f} MB")
        
        # Calculate SHA256 hash
        print("   Calculating SHA256 hash...")
        sha256_hash = hashlib.sha256()
        with open(local_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256_hash.update(chunk)
        file_hash = sha256_hash.hexdigest()
        print(f"   SHA256: {file_hash[:16]}...")
        
        # Check if file already exists in S3
        if s3.object_exists(s3_key):
            print(f"   ✅ File already exists in S3: {s3_key}")
        else:
            # Upload to S3
            print(f"   ⬆️  Uploading to S3: {s3_key}")
            try:
                s3_uri, uploaded_hash, uploaded_size = s3.upload_file(local_path, s3_key)
                
                if uploaded_hash != file_hash:
                    print(f"   ❌ Hash mismatch! Expected: {file_hash}, Got: {uploaded_hash}")
                    continue
                
                print(f"   ✅ Upload successful: {s3_uri}")
                
            except Exception as e:
                print(f"   ❌ Upload failed: {e}")
                continue
        
        # Update artifact record in database with real hash
        try:
            # Find the artifact record
            collection = db.db[db.artifacts_collection_name]
            
            # Get model and version IDs
            model = await db.get_model_by_slug(model_slug)
            if not model:
                print(f"   ❌ Model '{model_slug}' not found in database")
                continue
                
            version = await db.get_latest_version(model.id)
            if not version:
                print(f"   ❌ No version found for model '{model_slug}'")
                continue
            
            # Update artifact with real hash and size
            update_result = await collection.update_one(
                {"model_version_id": version.id, "kind": "model"},
                {
                    "$set": {
                        "sha256": file_hash,
                        "size_bytes": file_size,
                        "s3_uri": f"s3://{s3.bucket_name}/{s3_key}"
                    }
                }
            )
            
            if update_result.modified_count > 0:
                print(f"   ✅ Database updated with real hash and size")
            else:
                print(f"   ⚠️  No database record updated")
                
        except Exception as e:
            print(f"   ❌ Database update failed: {e}")
            continue
        
        uploaded_files.append({
            'model_slug': model_slug,
            's3_uri': f"s3://{s3.bucket_name}/{s3_key}",
            'size_mb': file_size / (1024*1024),
            'hash': file_hash
        })
    
    # Generate upload summary
    print(f"\n📊 Upload Summary:")
    print(f"   Successfully uploaded: {len(uploaded_files)} files")
    
    total_size_mb = sum(f['size_mb'] for f in uploaded_files)
    print(f"   Total size: {total_size_mb:.1f} MB")
    print(f"   Estimated S3 cost: ${total_size_mb * 0.023 / 1024:.2f}/month")
    
    for file_info in uploaded_files:
        print(f"   ✅ {file_info['model_slug']}: {file_info['size_mb']:.1f} MB")
    
    await db.disconnect()
    print("\n🎉 Model upload complete!")

async def verify_uploads():
    """Verify that all uploads are accessible"""
    print("\n🔍 Verifying S3 uploads...")
    
    s3 = get_s3_manager()
    db = get_db_manager()
    await db.connect()
    
    # Get all artifacts
    collection = db.db[db.artifacts_collection_name]
    artifacts = []
    async for doc in collection.find({}):
        doc["_id"] = str(doc["_id"])
        artifacts.append(ArtifactDB(**doc))
    
    print(f"Found {len(artifacts)} artifacts to verify:")
    
    for artifact in artifacts:
        s3_key = s3.get_s3_key_from_uri(artifact.s3_uri)
        
        # Check if object exists
        exists = s3.object_exists(s3_key)
        
        # Generate presigned URL
        download_url = s3.generate_presigned_url(s3_key, expiration=300) if exists else None
        
        status = "✅ ACCESSIBLE" if (exists and download_url) else "❌ MISSING"
        print(f"   {artifact.kind}: {status}")
        print(f"      S3 URI: {artifact.s3_uri}")
        print(f"      Size: {artifact.size_bytes / (1024*1024):.1f} MB")
        if download_url:
            print(f"      Download URL: {download_url[:50]}...")
        print()
    
    await db.disconnect()

async def main():
    """Main function"""
    print("🏗️  Model Registry S3 Upload Tool")
    print("=" * 50)
    
    # Test S3 connection first
    s3 = get_s3_manager()
    if not s3.test_connection():
        print("❌ S3 connection failed! Check your credentials.")
        return
    
    # Upload model files
    await upload_model_files()
    
    # Verify uploads
    await verify_uploads()
    
    print("🎯 Next steps:")
    print("   1. Test model download via API")
    print("   2. Create YAML manifests")
    print("   3. Set up Git repo for manifests")

if __name__ == "__main__":
    asyncio.run(main())