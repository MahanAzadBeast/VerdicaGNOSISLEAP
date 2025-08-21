# Veridica Model Registry

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

## Generated on: 2025-08-21T05:53:33.999788Z
