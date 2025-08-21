"""
MongoDB Database Manager for Model Registry
Handles database operations for model metadata
"""
import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import IndexModel, ASCENDING, DESCENDING
from .schemas import (
    ModelDB, ModelVersionDB, ArtifactDB, DatasetDB, MetricDB,
    ModelStage, ArtifactKind
)

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages MongoDB operations for model registry"""
    
    def __init__(self):
        self.mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
        self.db_name = os.getenv("DB_NAME", "test_database")
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        
        # Collection names
        self.models_collection_name = "model_registry_models"
        self.versions_collection_name = "model_registry_versions"  
        self.artifacts_collection_name = "model_registry_artifacts"
        self.datasets_collection_name = "model_registry_datasets"
        self.metrics_collection_name = "model_registry_metrics"
    
    async def connect(self):
        """Connect to MongoDB and setup collections"""
        try:
            self.client = AsyncIOMotorClient(self.mongo_url)
            self.db = self.client[self.db_name]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"✅ Connected to MongoDB: {self.db_name}")
            
            # Setup indexes
            await self._setup_indexes()
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("✅ Disconnected from MongoDB")
    
    async def test_connection(self) -> bool:
        """Test MongoDB connection"""
        try:
            if not self.client:
                await self.connect()
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"❌ MongoDB connection test failed: {e}")
            return False
    
    async def _setup_indexes(self):
        """Setup database indexes for optimal performance"""
        collections_indexes = {
            self.models_collection_name: [
                IndexModel([("slug", ASCENDING)], unique=True),
                IndexModel([("category", ASCENDING)]),
                IndexModel([("owner", ASCENDING)])
            ],
            self.versions_collection_name: [
                IndexModel([("model_id", ASCENDING), ("semver", ASCENDING)], unique=True),
                IndexModel([("model_id", ASCENDING), ("stage", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)])
            ],
            self.artifacts_collection_name: [
                IndexModel([("model_version_id", ASCENDING)]),
                IndexModel([("sha256", ASCENDING)], unique=True),
                IndexModel([("kind", ASCENDING)])
            ],
            self.datasets_collection_name: [
                IndexModel([("name", ASCENDING), ("version", ASCENDING)], unique=True)
            ],
            self.metrics_collection_name: [
                IndexModel([("model_version_id", ASCENDING)]),
                IndexModel([("name", ASCENDING)])
            ]
        }
        
        for collection_name, indexes in collections_indexes.items():
            collection = self.db[collection_name]
            try:
                await collection.create_indexes(indexes)
                logger.info(f"✅ Indexes created for {collection_name}")
            except Exception as e:
                logger.warning(f"⚠️ Index creation warning for {collection_name}: {e}")
    
    # Model Operations
    async def create_model(self, model_data: ModelDB) -> str:
        """Create a new model"""
        collection = self.db[self.models_collection_name]
        result = await collection.insert_one(model_data.model_dump(exclude={"id"}, by_alias=True))
        logger.info(f"✅ Created model: {model_data.slug}")
        return str(result.inserted_id)
    
    async def get_model_by_slug(self, slug: str) -> Optional[ModelDB]:
        """Get model by slug"""
        collection = self.db[self.models_collection_name]
        doc = await collection.find_one({"slug": slug})
        if doc:
            doc["_id"] = str(doc["_id"])
            return ModelDB(**doc)
        return None
    
    async def list_models(self, category: Optional[str] = None, owner: Optional[str] = None) -> List[ModelDB]:
        """List models with optional filtering"""
        collection = self.db[self.models_collection_name]
        query = {}
        if category:
            query["category"] = category
        if owner:
            query["owner"] = owner
        
        cursor = collection.find(query).sort("created_at", DESCENDING)
        models = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            models.append(ModelDB(**doc))
        return models
    
    # Model Version Operations
    async def create_model_version(self, model_id: str, version_data: ModelVersionDB) -> str:
        """Create a new model version"""
        version_data.model_id = model_id
        collection = self.db[self.versions_collection_name]
        result = await collection.insert_one(version_data.model_dump(exclude={"id"}, by_alias=True))
        logger.info(f"✅ Created model version: {version_data.semver}")
        return str(result.inserted_id)
    
    async def get_model_versions(self, model_id: str) -> List[ModelVersionDB]:
        """Get all versions for a model"""
        collection = self.db[self.versions_collection_name]
        cursor = collection.find({"model_id": model_id}).sort("created_at", DESCENDING)
        versions = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            versions.append(ModelVersionDB(**doc))
        return versions
    
    async def get_latest_version(self, model_id: str, stage: Optional[ModelStage] = None) -> Optional[ModelVersionDB]:
        """Get latest version for a model, optionally filtered by stage"""
        collection = self.db[self.versions_collection_name]
        query = {"model_id": model_id}
        if stage:
            query["stage"] = stage.value
        
        doc = await collection.find_one(query, sort=[("created_at", DESCENDING)])
        if doc:
            doc["_id"] = str(doc["_id"])
            return ModelVersionDB(**doc)
        return None
    
    # Artifact Operations  
    async def create_artifact(self, artifact_data: ArtifactDB) -> str:
        """Create a new artifact"""
        collection = self.db[self.artifacts_collection_name]
        result = await collection.insert_one(artifact_data.model_dump(exclude={"id"}, by_alias=True))
        logger.info(f"✅ Created artifact: {artifact_data.kind} - {artifact_data.s3_uri}")
        return str(result.inserted_id)
    
    async def get_version_artifacts(self, version_id: str) -> List[ArtifactDB]:
        """Get all artifacts for a model version"""
        collection = self.db[self.artifacts_collection_name]
        cursor = collection.find({"model_version_id": version_id})
        artifacts = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            artifacts.append(ArtifactDB(**doc))
        return artifacts
    
    # Metrics Operations
    async def create_metric(self, metric_data: MetricDB) -> str:
        """Create a new metric"""
        collection = self.db[self.metrics_collection_name]
        result = await collection.insert_one(metric_data.model_dump(exclude={"id"}, by_alias=True))
        return str(result.inserted_id)
    
    async def get_version_metrics(self, version_id: str) -> List[MetricDB]:
        """Get all metrics for a model version"""
        collection = self.db[self.metrics_collection_name]
        cursor = collection.find({"model_version_id": version_id})
        metrics = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            metrics.append(MetricDB(**doc))
        return metrics
    
    # Statistics
    async def get_registry_stats(self) -> Dict[str, int]:
        """Get registry statistics"""
        stats = {}
        stats["models_count"] = await self.db[self.models_collection_name].count_documents({})
        stats["versions_count"] = await self.db[self.versions_collection_name].count_documents({})
        stats["artifacts_count"] = await self.db[self.artifacts_collection_name].count_documents({})
        stats["datasets_count"] = await self.db[self.datasets_collection_name].count_documents({})
        stats["metrics_count"] = await self.db[self.metrics_collection_name].count_documents({})
        return stats

# Global database manager instance
db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager