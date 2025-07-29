"""
Modal API-based MolBERT Training Deployment
Deploy MolBERT training programmatically using Modal's API
"""

import modal
import os
import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModalAPIDeployment:
    """Handle Modal deployment via API"""
    
    def __init__(self, token_id=None, token_secret=None):
        """Initialize with Modal API credentials"""
        
        # Set Modal credentials from environment or parameters
        if token_id and token_secret:
            os.environ['MODAL_TOKEN_ID'] = token_id
            os.environ['MODAL_TOKEN_SECRET'] = token_secret
            logger.info("✅ Modal credentials set from parameters")
        elif 'MODAL_TOKEN_ID' in os.environ and 'MODAL_TOKEN_SECRET' in os.environ:
            logger.info("✅ Modal credentials found in environment")
        else:
            logger.error("❌ Modal credentials required!")
            raise ValueError("Modal API credentials required. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET")
    
    def deploy_molbert_training(self, 
                               target="EGFR", 
                               all_targets=False,
                               webhook_url=None,
                               gpu_type="A100"):
        """Deploy MolBERT training via Modal API"""
        
        try:
            logger.info(f"🚀 Deploying MolBERT training via Modal API")
            logger.info(f"🎯 Target: {'All targets' if all_targets else target}")
            logger.info(f"🖥️ GPU: {gpu_type}")
            
            # Import and configure the Modal app
            from modal_molbert import app, train_molbert_gpu, train_all_targets
            
            # Deploy the app first
            logger.info("📦 Deploying Modal app...")
            with modal.enable_output():
                app.deploy()
            
            logger.info("✅ App deployed successfully!")
            
            # Run the training function
            if all_targets:
                logger.info("🎯 Starting multi-target training...")
                result = train_all_targets.remote(
                    targets=["EGFR", "BRAF", "CDK2", "PARP1", "BCL2", "VEGFR2"],
                    max_epochs=50,
                    batch_size=64,
                    webhook_url=webhook_url
                )
            else:
                logger.info(f"🎯 Starting single target training: {target}")
                result = train_molbert_gpu.remote(
                    target=target,
                    max_epochs=50,
                    batch_size=64,
                    webhook_url=webhook_url
                )
            
            logger.info("✅ Training deployment successful!")
            logger.info("📊 Training is now running on Modal GPU")
            logger.info("🔗 Monitor at: https://modal.com/apps")
            
            if webhook_url:
                logger.info(f"📡 Progress updates will be sent to: {webhook_url}")
            
            return {
                "status": "deployed",
                "message": "MolBERT training deployed successfully",
                "training_type": "all_targets" if all_targets else "single_target",
                "target": target if not all_targets else "all",
                "webhook_url": webhook_url,
                "deployment_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "deployment_time": datetime.now().isoformat()
            }
    
    def check_deployment_status(self):
        """Check status of Modal deployments"""
        try:
            # This would typically use Modal's API to check running functions
            logger.info("📊 Checking Modal deployment status...")
            return {"status": "checking", "message": "Check Modal dashboard for real-time status"}
        except Exception as e:
            logger.error(f"❌ Status check failed: {e}")
            return {"status": "error", "error": str(e)}

def deploy_with_credentials(modal_token_id, modal_token_secret, 
                          target="EGFR", all_targets=False, 
                          webhook_url=None):
    """
    Main deployment function - call this with your Modal credentials
    """
    
    try:
        # Initialize deployment handler
        deployer = ModalAPIDeployment(
            token_id=modal_token_id,
            token_secret=modal_token_secret
        )
        
        # Deploy training
        result = deployer.deploy_molbert_training(
            target=target,
            all_targets=all_targets,
            webhook_url=webhook_url
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Overall deployment failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "message": "Deployment failed - check credentials and try again"
        }

if __name__ == "__main__":
    print("🚀 Modal API Deployment System Ready!")
    print("")
    print("📋 To deploy, call:")
    print("deploy_with_credentials(")
    print("    modal_token_id='your-token-id',")
    print("    modal_token_secret='your-token-secret',")
    print("    all_targets=True,")  
    print("    webhook_url='your-webhook-url'")
    print(")")
    print("")
    print("🔑 Get your Modal API credentials at: https://modal.com/settings/tokens")