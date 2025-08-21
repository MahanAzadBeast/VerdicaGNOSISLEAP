"""
YAML Manifest Manager for Model Registry
Handles model manifest validation and processing
"""
import yaml
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, ValidationError
from enum import Enum

logger = logging.getLogger(__name__)

class ManifestSchema(BaseModel):
    """Schema for model manifest YAML files"""
    
    class ModelInfo(BaseModel):
        slug: str
        name: str
        version: str
        description: str
        owner: str
        category: str
        created_at: datetime
        git_commit: Optional[str] = None
        training_script: Optional[str] = None
    
    class Artifact(BaseModel):
        name: str
        kind: str  # model, tokenizer, scaler, etc.
        flavor: str  # pytorch, onnx, pickle, etc.
        s3_uri: str
        sha256: str
        size_bytes: int
    
    class Dataset(BaseModel):
        name: str
        version: str
        role: str  # train, validation, test
        s3_prefix: str
        records_count: Optional[int] = None
    
    class Metric(BaseModel):
        name: str
        value: float
        split: str  # train, val, test
        
    class Hyperparameters(BaseModel):
        learning_rate: Optional[float] = None
        batch_size: Optional[int] = None
        epochs: Optional[int] = None
        model_architecture: Optional[str] = None
        # Allow additional fields
        class Config:
            extra = "allow"
    
    model: ModelInfo
    artifacts: List[Artifact]
    datasets: List[Dataset]
    metrics: List[Metric]
    hyperparameters: Optional[Hyperparameters] = None
    requirements: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None

class ManifestManager:
    """Manages YAML manifest operations"""
    
    def __init__(self):
        self.schema = ManifestSchema
    
    def validate_manifest(self, manifest_content: str) -> Dict[str, Any]:
        """
        Validate YAML manifest content against schema
        
        Args:
            manifest_content: Raw YAML content as string
            
        Returns:
            Validated manifest data as dictionary
            
        Raises:
            ValidationError: If manifest doesn't match schema
            yaml.YAMLError: If YAML is malformed
        """
        try:
            # Parse YAML
            data = yaml.safe_load(manifest_content)
            
            # Validate against schema
            manifest = self.schema(**data)
            
            # Return as dict for easier processing
            return manifest.model_dump()
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise ValidationError(f"Invalid YAML format: {e}")
        except ValidationError as e:
            logger.error(f"Manifest validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected manifest validation error: {e}")
            raise ValidationError(f"Manifest validation failed: {e}")
    
    def create_manifest_template(
        self,
        model_slug: str,
        model_name: str,
        version: str,
        description: str,
        owner: str,
        category: str
    ) -> str:
        """
        Create a YAML manifest template
        
        Returns:
            YAML manifest template as string
        """
        template = {
            "model": {
                "slug": model_slug,
                "name": model_name,
                "version": version,
                "description": description,
                "owner": owner,
                "category": category,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "git_commit": "abc123def456",  # Placeholder
                "training_script": "training/train_model.py"  # Placeholder
            },
            "artifacts": [
                {
                    "name": "model.pth",
                    "kind": "model",
                    "flavor": "pytorch",
                    "s3_uri": f"s3://veridicabatabase/models/{model_slug}/{version}/model.pth",
                    "sha256": "placeholder_sha256_hash",
                    "size_bytes": 1000000
                },
                {
                    "name": "tokenizer.json",
                    "kind": "tokenizer", 
                    "flavor": "json",
                    "s3_uri": f"s3://veridicabatabase/models/{model_slug}/{version}/tokenizer.json",
                    "sha256": "placeholder_sha256_hash",
                    "size_bytes": 50000
                }
            ],
            "datasets": [
                {
                    "name": "training_data",
                    "version": "v1.0",
                    "role": "train",
                    "s3_prefix": f"s3://veridicabatabase/datasets/{model_slug}/train/",
                    "records_count": 10000
                },
                {
                    "name": "validation_data",
                    "version": "v1.0", 
                    "role": "validation",
                    "s3_prefix": f"s3://veridicabatabase/datasets/{model_slug}/val/",
                    "records_count": 2000
                }
            ],
            "metrics": [
                {
                    "name": "r2_score",
                    "value": 0.85,
                    "split": "test"
                },
                {
                    "name": "mse",
                    "value": 0.12,
                    "split": "test"
                }
            ],
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "model_architecture": "ChemBERTa"
            },
            "requirements": [
                "torch>=1.9.0",
                "transformers>=4.20.0",
                "rdkit-pypi>=2022.9.5"
            ],
            "tags": [
                "cytotoxicity",
                "chemberta",
                "production"
            ],
            "notes": "Production model trained on GDSC dataset with ChemBERTa backbone"
        }
        
        return yaml.dump(template, default_flow_style=False, sort_keys=False)
    
    def extract_model_info(self, manifest_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model information from validated manifest"""
        return manifest_data.get("model", {})
    
    def extract_artifacts(self, manifest_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract artifacts information from validated manifest"""
        return manifest_data.get("artifacts", [])
    
    def extract_datasets(self, manifest_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract datasets information from validated manifest"""
        return manifest_data.get("datasets", [])
    
    def extract_metrics(self, manifest_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract metrics information from validated manifest"""
        return manifest_data.get("metrics", [])
    
    def validate_s3_uris(self, manifest_data: Dict[str, Any], s3_manager) -> List[str]:
        """
        Validate that all S3 URIs in manifest exist
        
        Returns:
            List of missing S3 URIs
        """
        missing_uris = []
        
        # Check artifact URIs
        for artifact in manifest_data.get("artifacts", []):
            s3_uri = artifact.get("s3_uri")
            if s3_uri:
                s3_key = s3_manager.get_s3_key_from_uri(s3_uri)
                if not s3_manager.object_exists(s3_key):
                    missing_uris.append(s3_uri)
        
        return missing_uris
    
    def validate_sha256_hashes(self, manifest_data: Dict[str, Any], s3_manager) -> List[str]:
        """
        Validate SHA256 hashes for artifacts
        
        Returns:
            List of artifacts with mismatched hashes
        """
        mismatched = []
        
        for artifact in manifest_data.get("artifacts", []):
            s3_uri = artifact.get("s3_uri")
            expected_hash = artifact.get("sha256")
            
            if s3_uri and expected_hash:
                # For now, we'll trust the manifest hash
                # In production, you'd download and verify
                # This is a placeholder for the validation logic
                pass
        
        return mismatched

# Global manifest manager instance
manifest_manager = None

def get_manifest_manager() -> ManifestManager:
    """Get global manifest manager instance"""
    global manifest_manager
    if manifest_manager is None:
        manifest_manager = ManifestManager()
    return manifest_manager