#!/usr/bin/env python3
"""
Generate YAML Manifests for Existing Models
Creates production-ready manifests for the separate Git repo
"""
import os
import sys
import asyncio
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / '.env')

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from model_registry.database import get_db_manager
from model_registry.schemas import ArtifactDB

async def generate_manifests():
    """Generate YAML manifests for all registered models"""
    print("📄 Generating YAML manifests for registered models...")
    
    # Initialize database
    db = get_db_manager()
    await db.connect()
    
    # Create manifests directory
    manifests_dir = Path('/app/model_registry_manifests')
    manifests_dir.mkdir(exist_ok=True)
    
    # Get all models
    models = await db.list_models()
    
    for model in models:
        print(f"\n🔹 Processing model: {model.slug}")
        
        # Get latest production version
        version = await db.get_latest_version(model.id)
        if not version:
            print(f"   ⚠️  No version found for {model.slug}")
            continue
        
        # Get artifacts and metrics
        artifacts = await db.get_version_artifacts(version.id)
        metrics = await db.get_version_metrics(version.id)
        
        # Build manifest data
        manifest_data = {
            'model': {
                'slug': model.slug,
                'name': model.name,
                'version': version.semver,
                'description': model.description,
                'owner': model.owner,
                'category': model.category,
                'created_at': version.created_at.isoformat() + 'Z',
                'git_commit': version.git_commit or 'production-release',
                'training_script': version.training_script_path or 'training/train_model.py'
            },
            'artifacts': [],
            'datasets': [],
            'metrics': [],
            'hyperparameters': version.hyperparameters or {},
            'requirements': [],
            'tags': [model.category, 'production', 'chemberta'],
            'notes': version.notes or f"Production {model.category} model"
        }
        
        # Add artifacts
        for artifact in artifacts:
            manifest_data['artifacts'].append({
                'name': artifact.s3_uri.split('/')[-1],
                'kind': artifact.kind.value,
                'flavor': artifact.flavor.value,
                's3_uri': artifact.s3_uri,
                'sha256': artifact.sha256,
                'size_bytes': artifact.size_bytes
            })
        
        # Add metrics
        for metric in metrics:
            manifest_data['metrics'].append({
                'name': metric.name,
                'value': metric.value,
                'split': metric.split
            })
        
        # Add model-specific data
        if model.slug == 'gnosis-i':
            manifest_data['datasets'] = [
                {
                    'name': 'chembl_bioactivity',
                    'version': 'v32',
                    'role': 'train',
                    's3_prefix': 's3://veridicabatabase/datasets/chembl/v32/',
                    'records_count': 12000
                },
                {
                    'name': 'bindingdb_bioactivity', 
                    'version': 'v2024',
                    'role': 'train',
                    's3_prefix': 's3://veridicabatabase/datasets/bindingdb/v2024/',
                    'records_count': 3000
                }
            ]
            manifest_data['requirements'] = [
                'torch>=1.9.0',
                'transformers>=4.33.0',
                'rdkit-pypi>=2022.9.5',
                'scikit-learn>=1.0.0'
            ]
            
        elif model.slug == 'cytotoxicity-predictor':
            manifest_data['datasets'] = [
                {
                    'name': 'gdsc_cytotoxicity',
                    'version': 'v1.0',
                    'role': 'train',
                    's3_prefix': 's3://veridicabatabase/datasets/gdsc/v1.0/',
                    'records_count': 9603
                }
            ]
            manifest_data['requirements'] = [
                'torch>=1.9.0',
                'transformers>=4.33.0',
                'rdkit-pypi>=2022.9.5',
                'pandas>=1.3.0',
                'numpy>=1.21.0'
            ]
        
        # Create model directory
        model_dir = manifests_dir / model.slug
        model_dir.mkdir(exist_ok=True)
        
        # Write manifest file
        manifest_file = model_dir / f"{version.semver}.yaml"
        
        with open(manifest_file, 'w') as f:
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False, indent=2)
        
        print(f"   ✅ Manifest created: {manifest_file}")
        
        # Also create a latest.yaml symlink equivalent
        latest_file = model_dir / "latest.yaml"
        with open(latest_file, 'w') as f:
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False, indent=2)
        
        print(f"   ✅ Latest manifest: {latest_file}")
    
    await db.disconnect()
    
    # Create repository structure documentation
    readme_content = f"""# Veridica Model Registry

This repository contains YAML manifests for all Veridica AI models.

## Repository Structure

```
manifests/
├── gnosis-i/
│   ├── 1.0.0.yaml
│   └── latest.yaml
├── cytotoxicity-predictor/
│   ├── 1.0.0.yaml
│   └── latest.yaml
└── schemas/
    └── manifest.schema.json
```

## Usage

### For Agents/Applications
```python
# Get model info
response = requests.get("https://api.veridica.ai/registry/models/gnosis-i/latest")
model_info = response.json()

# Download model
artifact_url = model_info['artifacts'][0]['download_url']
```

### For CI/CD
```bash
# Validate manifest
python ci/validate_manifest.py manifests/gnosis-i/1.0.0.yaml

# Deploy to production
python ci/promote_to_prod.py gnosis-i 1.0.0
```

## Generated on: {datetime.utcnow().isoformat()}Z
"""
    
    readme_file = manifests_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"\n📚 Repository documentation: {readme_file}")
    
    # Show directory structure
    print(f"\n📁 Generated manifest structure:")
    for item in sorted(manifests_dir.rglob('*')):
        if item.is_file():
            relative_path = item.relative_to(manifests_dir)
            size = item.stat().st_size
            print(f"   {relative_path} ({size} bytes)")

async def main():
    """Main function"""
    print("📋 Model Registry Manifest Generator")
    print("=" * 50)
    
    await generate_manifests()
    
    print("\n🎯 Next Steps:")
    print("   1. Review generated manifests in /app/model_registry_manifests/")
    print("   2. Create separate Git repository: veridica-model-registry")
    print("   3. Set up CI/CD for manifest validation")
    print("   4. Configure webhook for production deployments")

if __name__ == "__main__":
    asyncio.run(main())