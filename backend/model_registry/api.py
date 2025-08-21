"""
Model Registry API Routes
FastAPI endpoints for model registry operations
"""
import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Query
from fastapi.responses import JSONResponse

from .schemas import (
    ModelCreateRequest, ModelVersionCreateRequest, ModelResponse, 
    ModelVersionResponse, RegistryHealthResponse, ArtifactUploadRequest,
    ModelDB, ModelVersionDB, ArtifactDB, MetricDB,
    ModelStage, ArtifactKind, ArtifactFlavor
)
from .database import get_db_manager, DatabaseManager
from .s3_manager import get_s3_manager, S3Manager
from .manifest_manager import get_manifest_manager, ManifestManager
from .agent_discovery import get_discovery_agent, ModelDiscoveryAgent

logger = logging.getLogger(__name__)

# Create API router
registry_router = APIRouter(prefix="/registry", tags=["Model Registry"])

# Dependency functions
async def get_db() -> DatabaseManager:
    """Dependency to get database manager"""
    db = get_db_manager()
    if not db.client:
        await db.connect()
    return db

def get_s3() -> S3Manager:
    """Dependency to get S3 manager"""
    return get_s3_manager()

def get_discovery() -> ModelDiscoveryAgent:
    """Dependency to get discovery agent"""
    return get_discovery_agent()

@registry_router.get("/health", response_model=RegistryHealthResponse)
async def registry_health(
    db: DatabaseManager = Depends(get_db),
    s3: S3Manager = Depends(get_s3)
):
    """Health check for model registry"""
    try:
        # Test database connection
        mongodb_accessible = await db.test_connection()
        
        # Test S3 connection
        s3_accessible = s3.test_connection()
        
        # Get registry stats
        stats = await db.get_registry_stats() if mongodb_accessible else {}
        
        return RegistryHealthResponse(
            status="healthy" if (mongodb_accessible and s3_accessible) else "degraded",
            models_count=stats.get("models_count", 0),
            versions_count=stats.get("versions_count", 0),
            artifacts_count=stats.get("artifacts_count", 0),
            s3_accessible=s3_accessible,
            mongodb_accessible=mongodb_accessible
        )
    except Exception as e:
        logger.error(f"Registry health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@registry_router.get("/models", response_model=List[ModelResponse])
async def list_models(
    category: Optional[str] = Query(None, description="Filter by model category"),
    owner: Optional[str] = Query(None, description="Filter by model owner"),
    db: DatabaseManager = Depends(get_db)
):
    """List all models with optional filtering"""
    try:
        models = await db.list_models(category=category, owner=owner)
        
        # Convert to response format and get latest versions
        response_models = []
        for model in models:
            latest_version = await db.get_latest_version(model.id, ModelStage.PRODUCTION)
            latest_semver = latest_version.semver if latest_version else None
            
            response_models.append(ModelResponse(
                slug=model.slug,
                name=model.name,
                description=model.description,
                owner=model.owner,
                category=model.category,
                latest_version=latest_semver,
                created_at=model.created_at,
                updated_at=model.updated_at
            ))
        
        return response_models
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@registry_router.post("/models", response_model=Dict[str, str])
async def create_model(
    model_request: ModelCreateRequest,
    db: DatabaseManager = Depends(get_db)
):
    """Create a new model"""
    try:
        # Check if model with slug already exists
        existing = await db.get_model_by_slug(model_request.slug)
        if existing:
            raise HTTPException(status_code=409, detail=f"Model with slug '{model_request.slug}' already exists")
        
        # Create model
        model_data = ModelDB(**model_request.model_dump())
        model_id = await db.create_model(model_data)
        
        return {"model_id": model_id, "slug": model_request.slug}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")

@registry_router.get("/models/{slug}", response_model=ModelResponse)
async def get_model(
    slug: str,
    db: DatabaseManager = Depends(get_db)
):
    """Get model by slug"""
    try:
        model = await db.get_model_by_slug(slug)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{slug}' not found")
        
        # Get latest production version
        latest_version = await db.get_latest_version(model.id, ModelStage.PRODUCTION)
        latest_semver = latest_version.semver if latest_version else None
        
        return ModelResponse(
            slug=model.slug,
            name=model.name,
            description=model.description,
            owner=model.owner,
            category=model.category,
            latest_version=latest_semver,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model {slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model: {str(e)}")

@registry_router.get("/models/{slug}/versions", response_model=List[ModelVersionResponse])
async def list_model_versions(
    slug: str,
    stage: Optional[ModelStage] = Query(None, description="Filter by stage"),
    db: DatabaseManager = Depends(get_db)
):
    """List all versions for a model"""
    try:
        model = await db.get_model_by_slug(slug)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{slug}' not found")
        
        versions = await db.get_model_versions(model.id)
        
        # Filter by stage if specified
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        # Build response with artifacts and metrics
        response_versions = []
        for version in versions:
            artifacts = await db.get_version_artifacts(version.id)
            metrics = await db.get_version_metrics(version.id)
            
            response_versions.append(ModelVersionResponse(
                model_slug=slug,
                semver=version.semver,
                stage=version.stage,
                git_commit=version.git_commit,
                artifacts=[{
                    "kind": artifact.kind,
                    "flavor": artifact.flavor,
                    "s3_uri": artifact.s3_uri,
                    "size_bytes": artifact.size_bytes
                } for artifact in artifacts],
                metrics=[{
                    "name": metric.name,
                    "value": metric.value,
                    "split": metric.split
                } for metric in metrics],
                created_at=version.created_at
            ))
        
        return response_versions
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list versions for {slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list versions: {str(e)}")

@registry_router.get("/models/{slug}/latest", response_model=ModelVersionResponse)
async def get_latest_model_version(
    slug: str,
    stage: ModelStage = Query(ModelStage.PRODUCTION, description="Stage to get latest from"),
    db: DatabaseManager = Depends(get_db)
):
    """Get latest version of a model for a specific stage"""
    try:
        model = await db.get_model_by_slug(slug)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{slug}' not found")
        
        version = await db.get_latest_version(model.id, stage)
        if not version:
            raise HTTPException(status_code=404, detail=f"No {stage.value} version found for '{slug}'")
        
        # Get artifacts and metrics
        artifacts = await db.get_version_artifacts(version.id)
        metrics = await db.get_version_metrics(version.id)
        
        return ModelVersionResponse(
            model_slug=slug,
            semver=version.semver,
            stage=version.stage,
            git_commit=version.git_commit,
            artifacts=[{
                "kind": artifact.kind,
                "flavor": artifact.flavor,
                "s3_uri": artifact.s3_uri,
                "size_bytes": artifact.size_bytes
            } for artifact in artifacts],
            metrics=[{
                "name": metric.name,
                "value": metric.value,
                "split": metric.split
            } for metric in metrics],
            created_at=version.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest version for {slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get latest version: {str(e)}")

@registry_router.post("/models/{slug}/versions", response_model=Dict[str, str])
async def create_model_version(
    slug: str,
    version_request: ModelVersionCreateRequest,
    db: DatabaseManager = Depends(get_db)
):
    """Create a new version of a model"""
    try:
        model = await db.get_model_by_slug(slug)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{slug}' not found")
        
        # Create version
        version_data = ModelVersionDB(**version_request.model_dump())
        version_id = await db.create_model_version(model.id, version_data)
        
        return {"version_id": version_id, "semver": version_request.semver}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create version for {slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create version: {str(e)}")

@registry_router.get("/artifacts/{artifact_id}/download")
async def download_artifact(
    artifact_id: str,
    db: DatabaseManager = Depends(get_db),
    s3: S3Manager = Depends(get_s3)
):
    """Generate presigned URL for artifact download"""
    try:
        # Get artifact from database
        collection = db.db[db.artifacts_collection_name]
        artifact_doc = await collection.find_one({"_id": artifact_id})
        
        if not artifact_doc:
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        artifact = ArtifactDB(**artifact_doc)
        
        # Extract S3 key from URI
        s3_key = s3.get_s3_key_from_uri(artifact.s3_uri)
        
        # Generate presigned URL (valid for 1 hour)
        presigned_url = s3.generate_presigned_url(s3_key, expiration=3600)
        
        if not presigned_url:
            raise HTTPException(status_code=500, detail="Failed to generate download URL")
        
        return {"download_url": presigned_url, "filename": s3_key.split("/")[-1]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate download URL for artifact {artifact_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@registry_router.get("/stats")
async def get_registry_stats(db: DatabaseManager = Depends(get_db)):
    """Get registry statistics"""
    try:
        stats = await db.get_registry_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get registry stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")