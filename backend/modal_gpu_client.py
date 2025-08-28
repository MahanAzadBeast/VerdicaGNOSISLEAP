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
        self.app_name = "gnosis-i-real-inference"  # Real model app
        self.modal_available = True  # Modal access available as confirmed
    
    def _check_modal_credentials(self) -> bool:
        """Modal access available"""
        return True
    
    async def predict_gpu(self, smiles: str, targets: List[str], assay_types: List[str]) -> Optional[Dict[str, Any]]:
        """
        Call real Gnosis I trained ChemBERTa model on Modal GPU servers
        Uses the actual trained transformer model for accurate predictions
        """
        if not self.modal_available:
            logger.warning("⚠️ Modal not available - cannot use real GPU inference")
            return None
            
        try:
            logger.info(f"🚀 Calling real Gnosis I ChemBERTa on Modal GPU for {len(targets)} targets")
            
            # Get reference to Modal function for real model
            predict_fn = modal.Function.lookup(self.app_name, "predict_gnosis_i_real_gpu")
            
            # Call Modal GPU inference with real trained model
            result = predict_fn.remote(
                smiles=smiles,
                targets=targets, 
                assay_types=assay_types
            )
            
            logger.info("✅ Real Gnosis I ChemBERTa GPU prediction completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Real Gnosis I GPU prediction failed: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check real Gnosis I GPU service health"""
        if not self.modal_available:
            return {"status": "unavailable", "reason": "Modal not configured"}
            
        try:
            health_fn = modal.Function.lookup(self.app_name, "health_check_real")
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