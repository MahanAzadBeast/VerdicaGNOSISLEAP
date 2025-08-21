#!/usr/bin/env python3
"""
Rename Cytotoxicity Predictor to Gnosis II
Update registry and S3 paths for consistent naming
"""
import os
import sys
import asyncio
import boto3
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / '.env')

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from model_registry.database import get_db_manager
from model_registry.s3_manager import get_s3_manager

async def rename_cytotoxicity_to_gnosis_ii():
    """Rename cytotoxicity-predictor to gnosis-ii"""
    print("🏷️  Renaming Cytotoxicity Predictor → Gnosis II")
    print("=" * 50)
    
    # Initialize managers
    db = get_db_manager()
    await db.connect()
    s3 = get_s3_manager()
    
    old_slug = "cytotoxicity-predictor"
    new_slug = "gnosis-ii"
    
    # Step 1: Update MongoDB records
    print(f"\n1️⃣ Updating MongoDB records...")
    
    # Update model record
    models_collection = db.db[db.models_collection_name]
    model_update = await models_collection.update_one(
        {"slug": old_slug},
        {
            "$set": {
                "slug": new_slug,
                "name": "Gnosis II - Cancer Cell Line Cytotoxicity Predictor",
                "description": "ChemBERTa + genomics model for IC50 prediction on cancer cell lines (part of GNOSIS AI suite)"
            }
        }
    )
    
    if model_update.modified_count > 0:
        print(f"   ✅ Model record updated: {old_slug} → {new_slug}")
    else:
        print(f"   ⚠️  No model record found for {old_slug}")
        return
    
    # Get updated model info
    model = await db.get_model_by_slug(new_slug)
    if not model:
        print(f"   ❌ Failed to get updated model")
        return
    
    # Step 2: Update S3 artifacts (copy to new path)
    print(f"\n2️⃣ Updating S3 artifacts...")
    
    old_s3_key = "models/cytotoxicity-predictor/1.0.0/real_gdsc_chemberta_cytotox_v1.pth"
    new_s3_key = "models/gnosis-ii/1.0.0/real_gdsc_chemberta_cytotox_v1.pth"
    
    # Copy object to new location
    try:
        copy_source = {
            'Bucket': s3.bucket_name,
            'Key': old_s3_key
        }
        
        s3.s3_client.copy_object(
            CopySource=copy_source,
            Bucket=s3.bucket_name,
            Key=new_s3_key
        )
        
        print(f"   ✅ S3 artifact copied:")
        print(f"      From: s3://{s3.bucket_name}/{old_s3_key}")
        print(f"      To:   s3://{s3.bucket_name}/{new_s3_key}")
        
        # Delete old object
        s3.s3_client.delete_object(Bucket=s3.bucket_name, Key=old_s3_key)
        print(f"   🗑️  Old S3 object deleted")
        
    except Exception as e:
        print(f"   ❌ S3 copy failed: {e}")
        return
    
    # Step 3: Update artifact records in MongoDB
    print(f"\n3️⃣ Updating artifact records...")
    
    artifacts_collection = db.db[db.artifacts_collection_name]
    artifact_update = await artifacts_collection.update_many(
        {"model_version_id": {"$regex": ".*"}},  # Find all artifacts for this model
        {
            "$set": {
                "s3_uri": f"s3://{s3.bucket_name}/{new_s3_key}"
            }
        }
    )
    
    print(f"   ✅ Updated {artifact_update.modified_count} artifact records")
    
    # Step 4: Update manifest files
    print(f"\n4️⃣ Updating manifest files...")
    
    manifests_dir = Path('/app/model_registry_manifests')
    
    # Rename directory
    old_manifest_dir = manifests_dir / old_slug
    new_manifest_dir = manifests_dir / new_slug
    
    if old_manifest_dir.exists():
        old_manifest_dir.rename(new_manifest_dir)
        print(f"   ✅ Manifest directory renamed: {old_slug} → {new_slug}")
        
        # Update manifest content
        for manifest_file in new_manifest_dir.glob("*.yaml"):
            with open(manifest_file, 'r') as f:
                content = f.read()
            
            # Replace old references with new ones
            content = content.replace(old_slug, new_slug)
            content = content.replace(
                "Gnosis Model 2 - Cancer Cell Line Cytotoxicity Predictor",
                "Gnosis II - Cancer Cell Line Cytotoxicity Predictor"
            )
            content = content.replace(
                old_s3_key,
                new_s3_key
            )
            
            with open(manifest_file, 'w') as f:
                f.write(content)
            
            print(f"   ✅ Updated manifest: {manifest_file.name}")
    
    await db.disconnect()
    
    print(f"\n🎉 Renaming complete!")
    print(f"   Model: {old_slug} → {new_slug}")
    print(f"   Name: Gnosis II - Cancer Cell Line Cytotoxicity Predictor")
    print(f"   S3 Path: s3://{s3.bucket_name}/{new_s3_key}")

async def verify_rename():
    """Verify the rename was successful"""
    print(f"\n🔍 Verifying rename...")
    
    db = get_db_manager()
    await db.connect()
    
    # Check model exists
    gnosis_ii = await db.get_model_by_slug("gnosis-ii")
    if gnosis_ii:
        print(f"   ✅ Gnosis II found: {gnosis_ii.name}")
    else:
        print(f"   ❌ Gnosis II not found")
    
    # Check old model is gone
    old_model = await db.get_model_by_slug("cytotoxicity-predictor")
    if old_model:
        print(f"   ⚠️  Old model still exists")
    else:
        print(f"   ✅ Old model removed")
    
    await db.disconnect()

async def main():
    """Main function"""
    await rename_cytotoxicity_to_gnosis_ii()
    await verify_rename()
    
    print(f"\n📋 Updated GNOSIS Suite:")
    print(f"   🧬 Gnosis I  - Ligand Activity Predictor (62 targets)")
    print(f"   🦠 Gnosis II - Cancer Cell Line Cytotoxicity Predictor (36 cell lines)")
    print(f"   🔮 Gnosis III - Toxicity & Safety Module (future)")
    print(f"   🔮 Gnosis IV - Clinical Trials Predictor (future)")
    print(f"   🔮 Gnosis V  - Generative Molecule Optimizer (future)")

if __name__ == "__main__":
    asyncio.run(main())