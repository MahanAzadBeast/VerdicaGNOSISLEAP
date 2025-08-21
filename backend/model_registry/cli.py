#!/usr/bin/env python3
"""
Model Registry CLI Tool
Command-line interface for managing the model registry
"""
import os
import sys
import argparse
import asyncio
import json
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from model_registry.database import get_db_manager
from model_registry.s3_manager import get_s3_manager
from model_registry.manifest_manager import get_manifest_manager
from model_registry.schemas import (
    ModelDB, ModelVersionDB, ArtifactDB, MetricDB,
    ModelStage, ArtifactKind, ArtifactFlavor
)

async def register_existing_models():
    """Register existing GNOSIS models in the registry"""
    print("🚀 Registering existing GNOSIS models...")
    
    # Initialize managers
    db = get_db_manager()
    await db.connect()
    
    s3 = get_s3_manager()
    manifest = get_manifest_manager()
    
    # Register Model 1 (Gnosis I - Ligand Activity Predictor)
    print("\n📋 Registering Model 1: Gnosis I...")
    
    # Create Model 1
    model1_data = ModelDB(
        slug="gnosis-i",
        name="Gnosis I - Ligand Activity Predictor",
        description="Fine-tuned ChemBERTa model for IC50/Ki/EC50 prediction on 62 oncology targets",
        owner="veridica-ai",
        category="ligand-activity"
    )
    
    try:
        model1_id = await db.create_model(model1_data)
        print(f"✅ Model 1 created with ID: {model1_id}")
        
        # Create Model 1 Version
        version1_data = ModelVersionDB(
            model_id=model1_id,
            semver="1.0.0",
            stage=ModelStage.PRODUCTION,
            hyperparameters={
                "model_architecture": "ChemBERTa",
                "base_model": "seyonec/ChemBERTa-zinc-base-v1",
                "num_targets": 62,
                "r2_score": 0.6281,
                "training_samples": "ChEMBL + BindingDB"
            },
            notes="Production model trained on ChEMBL and BindingDB bioactivity data"
        )
        
        version1_id = await db.create_model_version(model1_id, version1_data)
        print(f"✅ Model 1 version created with ID: {version1_id}")
        
        # Check if model file exists and upload to S3 (placeholder)
        model1_path = "/app/backend/models/gnosis_model1_best.pt"
        if os.path.exists(model1_path):
            print(f"📁 Model 1 file found: {model1_path}")
            # For now, just record the local path - actual S3 upload would happen here
            
            artifact1_data = ArtifactDB(
                model_version_id=version1_id,
                kind=ArtifactKind.MODEL,
                flavor=ArtifactFlavor.PYTORCH,
                s3_uri=f"s3://veridicabatabase/models/gnosis-i/1.0.0/gnosis_model1_best.pt",
                sha256="placeholder_hash_1",  # Would calculate actual hash
                size_bytes=os.path.getsize(model1_path)
            )
            
            artifact1_id = await db.create_artifact(artifact1_data)
            print(f"✅ Model 1 artifact registered with ID: {artifact1_id}")
        
        # Add Model 1 metrics
        metrics1 = [
            MetricDB(model_version_id=version1_id, name="r2_score", value=0.6281, split="test"),
            MetricDB(model_version_id=version1_id, name="num_targets", value=62, split="all"),
            MetricDB(model_version_id=version1_id, name="training_samples", value=15000, split="train")  # Approximate
        ]
        
        for metric in metrics1:
            await db.create_metric(metric)
        print(f"✅ Model 1 metrics added: {len(metrics1)} metrics")
        
    except Exception as e:
        print(f"❌ Failed to register Model 1: {e}")
    
    # Register Model 2 (Cytotoxicity Prediction Model)
    print("\n📋 Registering Model 2: Cytotoxicity Predictor...")
    
    model2_data = ModelDB(
        slug="cytotoxicity-predictor",
        name="Gnosis Model 2 - Cancer Cell Line Cytotoxicity Predictor",
        description="ChemBERTa + genomics model for IC50 prediction on cancer cell lines using GDSC data",
        owner="veridica-ai",
        category="cytotoxicity"
    )
    
    try:
        model2_id = await db.create_model(model2_data)
        print(f"✅ Model 2 created with ID: {model2_id}")
        
        # Create Model 2 Version
        version2_data = ModelVersionDB(
            model_id=model2_id,
            semver="1.0.0",
            stage=ModelStage.PRODUCTION,
            hyperparameters={
                "model_architecture": "ChemBERTa + Genomics MLP",
                "chemberta_dim": 768,
                "genomics_dim": 30,
                "num_cell_lines": 36,
                "training_dataset": "GDSC real data",
                "training_samples": 9603
            },
            notes="Production model trained on real GDSC cytotoxicity data with genomic features"
        )
        
        version2_id = await db.create_model_version(model2_id, version2_data)
        print(f"✅ Model 2 version created with ID: {version2_id}")
        
        # Check for Model 2 files
        model2_paths = [
            "/app/models/real_gdsc_chemberta_cytotox_v1.pth",
            "/app/models/model2_enhanced_v1.pth"
        ]
        
        for model_path in model2_paths:
            if os.path.exists(model_path):
                print(f"📁 Model 2 file found: {model_path}")
                
                artifact2_data = ArtifactDB(
                    model_version_id=version2_id,
                    kind=ArtifactKind.MODEL,
                    flavor=ArtifactFlavor.PYTORCH,
                    s3_uri=f"s3://veridicabatabase/models/cytotoxicity-predictor/1.0.0/{os.path.basename(model_path)}",
                    sha256="placeholder_hash_2",  # Would calculate actual hash
                    size_bytes=os.path.getsize(model_path)
                )
                
                artifact2_id = await db.create_artifact(artifact2_data)
                print(f"✅ Model 2 artifact registered: {os.path.basename(model_path)}")
                break
        
        # Add Model 2 metrics (estimated based on previous testing)
        metrics2 = [
            MetricDB(model_version_id=version2_id, name="training_samples", value=9603, split="train"),
            MetricDB(model_version_id=version2_id, name="num_cell_lines", value=36, split="all"),
            MetricDB(model_version_id=version2_id, name="num_compounds", value=25, split="all")
        ]
        
        for metric in metrics2:
            await db.create_metric(metric)
        print(f"✅ Model 2 metrics added: {len(metrics2)} metrics")
        
    except Exception as e:
        print(f"❌ Failed to register Model 2: {e}")
    
    # Show registry stats
    print("\n📊 Registry Statistics:")
    stats = await db.get_registry_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    await db.disconnect()
    print("\n🎉 Model registration complete!")

async def list_models():
    """List all models in the registry"""
    db = get_db_manager()
    await db.connect()
    
    models = await db.list_models()
    
    print(f"\n📚 Registry Models ({len(models)} total):")
    print("-" * 80)
    
    for model in models:
        latest_version = await db.get_latest_version(model.id, ModelStage.PRODUCTION)
        latest_semver = latest_version.semver if latest_version else "N/A"
        
        print(f"🔹 {model.slug}")
        print(f"   Name: {model.name}")
        print(f"   Category: {model.category}")
        print(f"   Owner: {model.owner}")  
        print(f"   Latest Version: {latest_semver}")
        print(f"   Created: {model.created_at}")
        print()
    
    await db.disconnect()

async def create_manifest_template():
    """Create a manifest template for a new model"""
    manifest = get_manifest_manager()
    
    template = manifest.create_manifest_template(
        model_slug="example-model",
        model_name="Example Model",
        version="1.0.0",
        description="Example model for demonstration",
        owner="veridica-ai",
        category="example"
    )
    
    print("📄 YAML Manifest Template:")
    print("-" * 50)
    print(template)

async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Model Registry CLI")
    parser.add_argument("command", choices=["register", "list", "template", "health"])
    parser.add_argument("--category", help="Filter models by category")
    parser.add_argument("--owner", help="Filter models by owner")
    
    args = parser.parse_args()
    
    if args.command == "register":
        await register_existing_models()
    elif args.command == "list":
        await list_models()
    elif args.command == "template":
        await create_manifest_template()
    elif args.command == "health":
        # Test connections
        print("🔍 Testing Model Registry Health...")
        
        try:
            # Test S3
            s3 = get_s3_manager()
            s3_ok = s3.test_connection()
            print(f"S3 Connection: {'✅ OK' if s3_ok else '❌ FAILED'}")
            
            # Test MongoDB
            db = get_db_manager()
            await db.connect()
            mongo_ok = await db.test_connection()
            print(f"MongoDB Connection: {'✅ OK' if mongo_ok else '❌ FAILED'}")
            
            # Show stats
            if mongo_ok:
                stats = await db.get_registry_stats()
                print(f"\nRegistry Stats: {stats}")
            
            await db.disconnect()
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())