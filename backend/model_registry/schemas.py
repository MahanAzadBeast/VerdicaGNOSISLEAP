"""
Model Registry Pydantic Schemas
Defines the data models for the Model Registry system
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum

class ModelStage(str, Enum):
    """Model deployment stages"""
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    ARCHIVED = "archived"

class ArtifactKind(str, Enum):
    """Types of model artifacts"""
    MODEL = "model"  # .pt, .pth, .pkl
    TOKENIZER = "tokenizer"
    SCALER = "scaler"
    ENCODER = "encoder"
    CONFIG = "config"
    MANIFEST = "manifest"
    EVALUATION_REPORT = "evaluation_report"

class ArtifactFlavor(str, Enum):
    """Model serialization formats"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    PICKLE = "pickle"
    JSON = "json"
    YAML = "yaml"
    TORCHSCRIPT = "torchscript"

# Database Models
class ModelDB(BaseModel):
    """Model metadata stored in MongoDB"""
    id: Optional[str] = Field(None, alias="_id")
    slug: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable model name")
    description: str = Field(..., description="Model description")
    owner: str = Field(..., description="Model owner/team")
    category: str = Field(..., description="Model category (e.g., 'cytotoxicity', 'admet')")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True

class ModelVersionDB(BaseModel):
    """Model version metadata stored in MongoDB"""
    id: Optional[str] = Field(None, alias="_id")
    model_id: str = Field(..., description="Reference to parent model")
    semver: str = Field(..., description="Semantic version (e.g., '1.2.3')")
    stage: ModelStage = Field(default=ModelStage.DEVELOPMENT)
    git_commit: Optional[str] = Field(None, description="Git commit hash")
    training_script_path: Optional[str] = Field(None, description="Path to training script")
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    training_dataset_hash: Optional[str] = Field(None, description="Training dataset hash")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = Field(None, description="Version notes")
    
    class Config:
        populate_by_name = True

class ArtifactDB(BaseModel):
    """Model artifact metadata stored in MongoDB"""
    id: Optional[str] = Field(None, alias="_id")
    model_version_id: str = Field(..., description="Reference to model version")
    kind: ArtifactKind = Field(..., description="Type of artifact")
    flavor: ArtifactFlavor = Field(..., description="Serialization format")
    s3_uri: str = Field(..., description="S3 URI for artifact")
    sha256: str = Field(..., description="SHA256 hash for integrity")
    size_bytes: int = Field(..., description="File size in bytes")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True

class DatasetDB(BaseModel):
    """Dataset metadata stored in MongoDB"""
    id: Optional[str] = Field(None, alias="_id")
    name: str = Field(..., description="Dataset name")
    version: str = Field(..., description="Dataset version")
    s3_prefix: str = Field(..., description="S3 prefix for dataset files")
    schema_fingerprint: Optional[str] = Field(None, description="Dataset schema hash")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True

class MetricDB(BaseModel):
    """Model metrics stored in MongoDB"""
    id: Optional[str] = Field(None, alias="_id")
    model_version_id: str = Field(..., description="Reference to model version")
    name: str = Field(..., description="Metric name (e.g., 'r2_score', 'accuracy')")
    value: float = Field(..., description="Metric value")
    split: str = Field(..., description="Data split (train/val/test)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True

# API Request/Response Models
class ModelCreateRequest(BaseModel):
    """Request to create a new model"""
    slug: str
    name: str
    description: str
    owner: str
    category: str

class ModelVersionCreateRequest(BaseModel):
    """Request to create a new model version"""
    semver: str
    stage: ModelStage = ModelStage.DEVELOPMENT
    git_commit: Optional[str] = None
    training_script_path: Optional[str] = None  
    hyperparameters: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

class ArtifactUploadRequest(BaseModel):
    """Request to register an artifact"""
    kind: ArtifactKind
    flavor: ArtifactFlavor
    filename: str

class ModelResponse(BaseModel):
    """API response for model information"""
    slug: str
    name: str
    description: str
    owner: str
    category: str
    latest_version: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class ModelVersionResponse(BaseModel):
    """API response for model version information"""
    model_slug: str
    semver: str
    stage: ModelStage
    git_commit: Optional[str]
    artifacts: List[Dict[str, Any]] = []
    metrics: List[Dict[str, Any]] = []
    created_at: datetime

class RegistryHealthResponse(BaseModel):
    """Health check response for model registry"""
    status: str
    models_count: int
    versions_count: int
    artifacts_count: int
    s3_accessible: bool
    mongodb_accessible: bool