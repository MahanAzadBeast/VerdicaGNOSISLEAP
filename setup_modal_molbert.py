"""
Complete Modal MolBERT Setup Helper
Handles end-to-end setup of Modal infrastructure with proper volume management
"""

import os
import subprocess
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModalMolBERTSetup:
    """Helper class for Modal MolBERT setup"""
    
    def __init__(self):
        self.modal_dir = Path("/app/modal_training")
        self.setup_script = self.modal_dir / "deploy_enhanced.sh"
        self.modal_script = self.modal_dir / "modal_molbert_enhanced.py"
    
    def check_modal_cli(self):
        """Check if Modal CLI is installed and authenticated"""
        try:
            # Check if modal is installed
            result = subprocess.run(['modal', '--version'], 
                                    capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("❌ Modal CLI not found!")
                logger.error("💡 Install with: pip install modal-client")
                return False
            
            logger.info(f"✅ Modal CLI found: {result.stdout.strip()}")
            
            # Check if authenticated
            result = subprocess.run(['modal', 'token', 'verify'], 
                                    capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("❌ Modal not authenticated!")
                logger.error("💡 Run: modal token new")
                logger.error("🔗 Get token at: https://modal.com/settings/tokens")
                return False
            
            logger.info("✅ Modal authenticated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking Modal CLI: {e}")
            return False
    
    def install_modal_cli(self):
        """Install Modal CLI if not present"""
        try:
            logger.info("📦 Installing Modal CLI...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "modal-client"
            ])
            logger.info("✅ Modal CLI installed successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to install Modal CLI: {e}")
            return False
    
    def deploy_modal_app(self):
        """Deploy the Modal app"""
        try:
            logger.info("📦 Deploying Enhanced Modal MolBERT app...")
            
            # Change to modal_training directory
            os.chdir(self.modal_dir)
            
            # Deploy the app
            result = subprocess.run([
                'modal', 'deploy', 'modal_molbert_enhanced.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ App deployed successfully!")
                logger.info(result.stdout)
                return True
            else:
                logger.error("❌ App deployment failed!")
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"❌ Deployment error: {e}")
            return False
    
    def setup_pretrained_model(self):
        """Download and cache pretrained model"""
        try:
            logger.info("📥 Setting up pretrained MolBERT model...")
            
            os.chdir(self.modal_dir)
            
            # Run model download
            result = subprocess.run([
                'modal', 'run', 'modal_molbert_enhanced.py::download_pretrained_molbert'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Pretrained model cached successfully!")
                logger.info(result.stdout)
                return True
            else:
                logger.error("❌ Model caching failed!")
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"❌ Model setup error: {e}")
            return False
    
    def test_setup(self):
        """Test the Modal setup"""
        try:
            logger.info("🧪 Testing Modal setup...")
            
            os.chdir(self.modal_dir)
            
            # Test model info
            result = subprocess.run([
                'modal', 'run', 'modal_molbert_enhanced.py::get_model_info'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Setup test passed!")
                logger.info(result.stdout)
                return True
            else:
                logger.error("❌ Setup test failed!")
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
            return False
    
    def run_complete_setup(self):
        """Run complete Modal MolBERT setup"""
        logger.info("🚀 Starting Complete Modal MolBERT Setup")
        logger.info("=" * 50)
        
        # Step 1: Check/install Modal CLI
        if not self.check_modal_cli():
            logger.info("📦 Installing Modal CLI...")
            if not self.install_modal_cli():
                logger.error("❌ Setup failed at CLI installation")
                return False
            
            # Recheck after installation
            if not self.check_modal_cli():
                logger.error("❌ Modal CLI still not working after installation")
                logger.error("💡 Please run 'modal token new' to authenticate")
                return False
        
        # Step 2: Deploy app
        logger.info("📦 Step 2: Deploying Modal app...")
        if not self.deploy_modal_app():
            logger.error("❌ Setup failed at app deployment")
            return False
        
        # Step 3: Setup model cache
        logger.info("📥 Step 3: Setting up model cache...")
        if not self.setup_pretrained_model():
            logger.error("❌ Setup failed at model caching")
            return False
        
        # Step 4: Test setup
        logger.info("🧪 Step 4: Testing setup...")
        if not self.test_setup():
            logger.error("❌ Setup test failed")
            return False
        
        # Success!
        logger.info("")
        logger.info("🎉 Complete Modal MolBERT Setup Successful!")
        logger.info("=" * 50)
        logger.info("")
        logger.info("📋 What was created:")
        logger.info("  • Modal app: molbert-enhanced")
        logger.info("  • Volume: molbert-cache (model storage + HF cache)")
        logger.info("  • Volume: molbert-training (fine-tuned models)")
        logger.info("  • Cached model: seyonec/ChemBERTa-zinc-base-v1")
        logger.info("")
        logger.info("🔗 Monitor at: https://modal.com/apps")
        logger.info("")
        logger.info("🎯 Next steps:")
        logger.info("  1. Set Modal credentials in backend/.env:")
        logger.info("     MODAL_TOKEN_ID=your_token_id")
        logger.info("     MODAL_TOKEN_SECRET=your_token_secret")
        logger.info("")
        logger.info("  2. Test via API:")
        logger.info("     GET /api/modal/molbert/status")
        logger.info("     POST /api/modal/molbert/predict")
        logger.info("")
        
        return True

def main():
    """Main setup function"""
    setup = ModalMolBERTSetup()
    success = setup.run_complete_setup()
    
    if success:
        print("✅ Setup completed successfully!")
        return 0
    else:
        print("❌ Setup failed!")
        return 1

if __name__ == "__main__":
    exit(main())