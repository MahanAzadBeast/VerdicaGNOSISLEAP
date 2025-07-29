"""
Modal API Integration for Backend
Add this to your FastAPI backend to deploy Modal training via API calls
"""

from fastapi import APIRouter
from pydantic import BaseModel
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Modal API Models
class ModalDeploymentRequest(BaseModel):
    modal_token_id: str
    modal_token_secret: str  
    target: Optional[str] = "EGFR"
    all_targets: Optional[bool] = False
    gpu_type: Optional[str] = "A100"

class ModalDeploymentResponse(BaseModel):
    status: str
    message: str
    training_type: Optional[str] = None
    target: Optional[str] = None
    webhook_url: Optional[str] = None
    deployment_time: str
    error: Optional[str] = None

# Modal deployment router
modal_router = APIRouter(prefix="/api/modal")

@modal_router.post("/deploy-training", response_model=ModalDeploymentResponse)
async def deploy_modal_training(request: ModalDeploymentRequest):
    """Deploy MolBERT training to Modal via API"""
    
    try:
        logger.info(f"🚀 Received Modal deployment request")
        
        # Import the deployment function
        from modal_training.modal_api_deployment import deploy_with_credentials
        
        # Set webhook URL to this app's progress endpoint
        webhook_url = f"{os.environ.get('BACKEND_URL', 'https://fe6fb94e-8195-4707-8bf5-159ba572116c.preview.emergentagent.com')}/api/gpu/training-progress"
        
        # Deploy to Modal
        result = deploy_with_credentials(
            modal_token_id=request.modal_token_id,
            modal_token_secret=request.modal_token_secret,
            target=request.target,
            all_targets=request.all_targets,
            webhook_url=webhook_url
        )
        
        logger.info(f"📊 Modal deployment result: {result['status']}")
        
        return ModalDeploymentResponse(
            status=result["status"],
            message=result.get("message", ""),
            training_type=result.get("training_type"),
            target=result.get("target"),
            webhook_url=result.get("webhook_url"),
            deployment_time=result.get("deployment_time", datetime.now().isoformat()),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"❌ Modal deployment failed: {e}")
        return ModalDeploymentResponse(
            status="failed",
            message="Deployment failed",
            deployment_time=datetime.now().isoformat(),
            error=str(e)
        )

@modal_router.get("/deployment-status")
async def get_modal_deployment_status():
    """Get status of Modal deployments"""
    
    try:
        from modal_training.modal_api_deployment import ModalAPIDeployment
        
        # This would check actual Modal deployment status
        # For now, return the training progress we're already tracking
        return {
            "status": "check_training_progress",
            "message": "Check /api/gpu/training-progress for real-time updates",
            "modal_dashboard": "https://modal.com/apps"
        }
        
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }