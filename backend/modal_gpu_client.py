"""
Modal GPU Client for Gnosis I Predictions
Calls Modal GPU inference instead of running locally
"""

import logging
import os
from typing import Dict, List, Any, Optional
import modal

logger = logging.getLogger(__name__)

class GnosisIModalClient:
    """Client to call Gnosis I predictions on Modal GPU servers"""
    
    def __init__(self):
        self.app_name = "gnosis-i-inference"
        self.modal_available = self._check_modal_credentials()
    
    def _check_modal_credentials(self) -> bool:
        """Check if Modal credentials are available"""
        try:
            # Check for Modal token in environment
            token_id = os.getenv('MODAL_TOKEN_ID')
            token_secret = os.getenv('MODAL_TOKEN_SECRET')
            
            if token_id and token_secret:
                logger.info("✅ Modal credentials found")
                return True
            else:
                logger.warning("⚠️ Modal credentials not found - will fall back to local inference")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Error checking Modal credentials: {e}")
            return False
    
    async def predict_gpu(self, smiles: str, targets: List[str], assay_types: List[str]) -> Optional[Dict[str, Any]]:
        """
        Call Gnosis I prediction on Modal GPU servers
        Much faster than local CPU inference
        """
        if not self.modal_available:
            logger.warning("⚠️ Modal not available - cannot use GPU inference")
            return None
            
        try:
            logger.info(f"🚀 Calling Modal GPU inference for {len(targets)} targets")
            
            # Get reference to Modal function
            predict_fn = modal.Function.lookup(self.app_name, "predict_gnosis_i_gpu")
            
            # Call Modal GPU inference
            result = predict_fn.remote(
                smiles=smiles,
                targets=targets, 
                assay_types=assay_types
            )
            
            logger.info("✅ Modal GPU prediction completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Modal GPU prediction failed: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Modal GPU service health"""
        if not self.modal_available:
            return {"status": "unavailable", "reason": "No Modal credentials"}
            
        try:
            health_fn = modal.Function.lookup(self.app_name, "health_check")
            result = health_fn.remote()
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Global client instance
_modal_client = None

def get_modal_client() -> GnosisIModalClient:
    """Get global Modal client instance"""
    global _modal_client
    if _modal_client is None:
        _modal_client = GnosisIModalClient()
    return _modal_client