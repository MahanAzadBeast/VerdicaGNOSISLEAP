"""
Agent Discovery System for Model Registry
Allows AI agents to dynamically discover and load models from the registry
"""
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import json

from .database import get_db_manager
from .s3_manager import get_s3_manager
from .schemas import ModelStage

logger = logging.getLogger(__name__)

class ModelDiscoveryAgent:
    """Agent that discovers and manages available models from the registry"""
    
    def __init__(self, cache_duration_minutes: int = 30):
        self.db = get_db_manager()
        self.s3 = get_s3_manager()
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.last_update = None
        self.cached_models = {}
        self.cached_capabilities = {}
    
    async def refresh_model_cache(self):
        """Refresh the cached model information"""
        try:
            if not self.db.client:
                await self.db.connect()
            
            # Get all production models
            models = await self.db.list_models()
            self.cached_models = {}
            self.cached_capabilities = {}
            
            for model in models:
                # Get latest production version
                latest_version = await self.db.get_latest_version(model.id, ModelStage.PRODUCTION)
                
                if latest_version:
                    # Get artifacts and metrics
                    artifacts = await self.db.get_version_artifacts(latest_version.id)
                    metrics = await self.db.get_version_metrics(latest_version.id)
                    
                    model_info = {
                        'slug': model.slug,
                        'name': model.name,
                        'description': model.description,
                        'category': model.category,
                        'owner': model.owner,
                        'version': latest_version.semver,
                        'stage': latest_version.stage.value,
                        'hyperparameters': latest_version.hyperparameters or {},
                        'artifacts': [
                            {
                                'kind': artifact.kind.value,
                                'flavor': artifact.flavor.value,
                                's3_uri': artifact.s3_uri,
                                'size_bytes': artifact.size_bytes
                            } for artifact in artifacts
                        ],
                        'metrics': [
                            {
                                'name': metric.name,
                                'value': metric.value,
                                'split': metric.split
                            } for metric in metrics
                        ],
                        'created_at': latest_version.created_at.isoformat(),
                        'capabilities': self._extract_capabilities(model, latest_version, metrics)
                    }
                    
                    self.cached_models[model.slug] = model_info
                    self.cached_capabilities[model.category] = self.cached_capabilities.get(model.category, [])
                    self.cached_capabilities[model.category].append(model.slug)
            
            self.last_update = datetime.utcnow()
            logger.info(f"✅ Model cache refreshed: {len(self.cached_models)} models discovered")
            
        except Exception as e:
            logger.error(f"❌ Failed to refresh model cache: {e}")
    
    def _extract_capabilities(self, model, version, metrics) -> Dict[str, Any]:
        """Extract model capabilities from metadata"""
        capabilities = {
            'model_type': model.category,
            'architecture': version.hyperparameters.get('model_architecture', 'unknown'),
            'input_types': ['smiles'],  # Default for chemistry models
            'output_types': [],
            'performance': {}
        }
        
        # Extract performance metrics
        for metric in metrics:
            capabilities['performance'][metric.name] = metric.value
        
        # Category-specific capabilities
        if model.category == 'ligand-activity':
            capabilities['output_types'] = ['ic50', 'ki', 'ec50']
            capabilities['num_targets'] = version.hyperparameters.get('num_targets', 0)
            capabilities['assay_types'] = ['binding_ic50', 'functional_ic50', 'ki', 'ec50']
            
        elif model.category == 'cytotoxicity':
            capabilities['output_types'] = ['ic50']
            capabilities['num_cell_lines'] = version.hyperparameters.get('num_cell_lines', 0)
            capabilities['input_types'] = ['smiles', 'genomics']
            
        return capabilities
    
    async def get_available_models(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get all available models, optionally filtered by category"""
        # Refresh cache if needed
        if (self.last_update is None or 
            datetime.utcnow() - self.last_update > self.cache_duration):
            await self.refresh_model_cache()
        
        if category:
            return {slug: info for slug, info in self.cached_models.items() 
                   if info['category'] == category}
        
        return self.cached_models
    
    async def get_model_info(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model"""
        models = await self.get_available_models()
        return models.get(slug)
    
    async def find_models_by_capability(self, capability_type: str, capability_value: Any = None) -> List[str]:
        """Find models that have a specific capability"""
        models = await self.get_available_models()
        matching_models = []
        
        for slug, info in models.items():
            capabilities = info.get('capabilities', {})
            
            if capability_type in capabilities:
                if capability_value is None:
                    matching_models.append(slug)
                elif capabilities[capability_type] == capability_value:
                    matching_models.append(slug)
                elif isinstance(capabilities[capability_type], list) and capability_value in capabilities[capability_type]:
                    matching_models.append(slug)
        
        return matching_models
    
    async def get_best_model_for_task(self, task_type: str, metric_name: str = 'r2_score') -> Optional[str]:
        """Get the best model for a specific task based on performance metrics"""
        models = await self.get_available_models(category=task_type)
        
        best_model = None
        best_score = -float('inf')
        
        for slug, info in models.items():
            performance = info.get('capabilities', {}).get('performance', {})
            score = performance.get(metric_name, -float('inf'))
            
            if score > best_score:
                best_score = score
                best_model = slug
        
        return best_model
    
    async def generate_model_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of available models"""
        models = await self.get_available_models()
        
        summary = {
            'total_models': len(models),
            'categories': {},
            'capabilities_overview': {},
            'top_performers': {},
            'models': []
        }
        
        # Categorize models
        for slug, info in models.items():
            category = info['category']
            summary['categories'][category] = summary['categories'].get(category, 0) + 1
            
            # Track capabilities
            capabilities = info.get('capabilities', {})
            for cap_type, cap_value in capabilities.items():
                if cap_type not in summary['capabilities_overview']:
                    summary['capabilities_overview'][cap_type] = set()
                
                if isinstance(cap_value, list):
                    summary['capabilities_overview'][cap_type].update(cap_value)
                else:
                    summary['capabilities_overview'][cap_type].add(str(cap_value))
            
            # Add to models list
            summary['models'].append({
                'slug': slug,
                'name': info['name'],
                'category': category,
                'version': info['version'],
                'performance': capabilities.get('performance', {})
            })
        
        # Convert sets to lists for JSON serialization
        for cap_type in summary['capabilities_overview']:
            summary['capabilities_overview'][cap_type] = list(summary['capabilities_overview'][cap_type])
        
        # Find top performers
        categories = list(summary['categories'].keys())
        for category in categories:
            best_model = await self.get_best_model_for_task(category)
            if best_model:
                summary['top_performers'][category] = best_model
        
        return summary
    
    async def get_model_download_url(self, slug: str, artifact_kind: str = 'model') -> Optional[str]:
        """Get download URL for a model artifact"""
        model_info = await self.get_model_info(slug)
        if not model_info:
            return None
        
        # Find the requested artifact
        for artifact in model_info.get('artifacts', []):
            if artifact['kind'] == artifact_kind:
                s3_uri = artifact['s3_uri']
                s3_key = self.s3.get_s3_key_from_uri(s3_uri)
                return self.s3.generate_presigned_url(s3_key, expiration=3600)
        
        return None

# Global discovery agent instance
discovery_agent = None

def get_discovery_agent() -> ModelDiscoveryAgent:
    """Get global model discovery agent instance"""
    global discovery_agent
    if discovery_agent is None:
        discovery_agent = ModelDiscoveryAgent()
    return discovery_agent

# Agent API functions for easy integration
async def discover_models(category: Optional[str] = None) -> Dict[str, Any]:
    """Discover available models"""
    agent = get_discovery_agent()
    return await agent.get_available_models(category)

async def get_model_capabilities(slug: str) -> Optional[Dict[str, Any]]:
    """Get model capabilities"""
    agent = get_discovery_agent()
    model_info = await agent.get_model_info(slug)
    return model_info.get('capabilities') if model_info else None

async def find_best_model(task_type: str) -> Optional[str]:
    """Find the best model for a task"""
    agent = get_discovery_agent()
    return await agent.get_best_model_for_task(task_type)