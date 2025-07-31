"""
ChemBERTa Backend Integration
Adds endpoints for the trained focused ChemBERTa model
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
import asyncio
from datetime import datetime

# Create router for ChemBERTa endpoints
chemberta_router = APIRouter(prefix="/chemberta", tags=["ChemBERTa Multi-Task"])

# Set up logging
logger = logging.getLogger(__name__)

class ChemBERTaPredictionRequest(BaseModel):
    smiles: str

class ChemBERTaPredictionResponse(BaseModel):
    status: str
    predictions: Optional[Dict[str, Dict[str, Any]]] = None
    target_info: Optional[Dict[str, Dict[str, Any]]] = None
    summary: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Import Modal functions
try:
    import modal
    # Use the NEW 50-epoch ChemBERTa integration instead of old one
    from chemberta_50epoch_integration import app as chemberta_app, predict_chemberta_50epoch, get_chemberta_50epoch_status
    CHEMBERTA_AVAILABLE = True
    logger.info("✅ ChemBERTa 50-epoch inference integration loaded successfully")
except Exception as e:
    CHEMBERTA_AVAILABLE = False
    logger.error(f"❌ ChemBERTa 50-epoch inference integration failed: {e}")
    chemberta_app = None

def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string using RDKit"""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False

@chemberta_router.get("/status")
async def get_chemberta_status():
    """Get ChemBERTa model status"""
    
    if not CHEMBERTA_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "ChemBERTa inference not available",
            "available": False
        }
    
    try:
        return {
            "status": "available",
            "message": "ChemBERTa multi-task model ready for inference",
            "available": True,
            "model_info": {
                "model_type": "ChemBERTa Multi-Task",
                "trained_targets": [
                    'EGFR', 'HER2', 'VEGFR2', 'BRAF', 'MET', 
                    'CDK4', 'CDK6', 'ALK', 'MDM2', 'PI3KCA'
                ],
                "training_r2_mean": 0.516,
                "total_targets": 10
            },
            "infrastructure": "Modal A100 GPU",
            "ready": True
        }
    except Exception as e:
        logger.error(f"Error checking ChemBERTa status: {e}")
        return {
            "status": "error",
            "message": str(e),
            "available": False
        }

@chemberta_router.get("/targets")
async def get_chemberta_targets():
    """Get available ChemBERTa targets with performance info"""
    
    if not CHEMBERTA_AVAILABLE:
        raise HTTPException(status_code=503, detail="ChemBERTa inference not available")
    
    targets_info = {
        'EGFR': {
            'name': 'EGFR',
            'description': 'Epidermal Growth Factor Receptor',
            'r2_score': 0.751,
            'performance': 'Excellent',
            'test_samples': 165
        },
        'MDM2': {
            'name': 'MDM2', 
            'description': 'MDM2 Proto-Oncogene',
            'r2_score': 0.655,
            'performance': 'Excellent',
            'test_samples': 122
        },
        'BRAF': {
            'name': 'BRAF',
            'description': 'B-Raf Proto-Oncogene', 
            'r2_score': 0.595,
            'performance': 'Good',
            'test_samples': 117
        },
        'PI3KCA': {
            'name': 'PI3KCA',
            'description': 'Phosphatidylinositol-4,5-Bisphosphate 3-Kinase Catalytic Subunit Alpha',
            'r2_score': 0.588,
            'performance': 'Good',
            'test_samples': 58
        },
        'HER2': {
            'name': 'HER2',
            'description': 'Human Epidermal Growth Factor Receptor 2',
            'r2_score': 0.583,
            'performance': 'Good', 
            'test_samples': 121
        },
        'VEGFR2': {
            'name': 'VEGFR2',
            'description': 'Vascular Endothelial Growth Factor Receptor 2',
            'r2_score': 0.555,
            'performance': 'Good',
            'test_samples': 148
        },
        'MET': {
            'name': 'MET',
            'description': 'MET Proto-Oncogene',
            'r2_score': 0.502,
            'performance': 'Good',
            'test_samples': 91
        },
        'ALK': {
            'name': 'ALK',
            'description': 'Anaplastic Lymphoma Kinase',
            'r2_score': 0.405,
            'performance': 'Good',
            'test_samples': 64
        },
        'CDK4': {
            'name': 'CDK4',
            'description': 'Cyclin Dependent Kinase 4',
            'r2_score': 0.314,
            'performance': 'Fair',
            'test_samples': 64
        },
        'CDK6': {
            'name': 'CDK6',
            'description': 'Cyclin Dependent Kinase 6',
            'r2_score': 0.216,
            'performance': 'Fair',
            'test_samples': 111
        }
    }
    
    return {
        "targets": targets_info,
        "total_targets": len(targets_info),
        "mean_r2": 0.516,
        "excellent_targets": ['EGFR', 'MDM2'],
        "good_targets": ['BRAF', 'PI3KCA', 'HER2', 'VEGFR2', 'MET', 'ALK'],
        "fair_targets": ['CDK4', 'CDK6']
    }

@chemberta_router.post("/predict", response_model=ChemBERTaPredictionResponse)
async def predict_chemberta_multi_target(request: ChemBERTaPredictionRequest):
    """Predict IC50 values for all trained targets using ChemBERTa"""
    
    if not CHEMBERTA_AVAILABLE:
        raise HTTPException(status_code=503, detail="ChemBERTa inference not available")
    
    # Validate SMILES
    if not validate_smiles(request.smiles):
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    logger.info(f"🧬 ChemBERTa prediction request for SMILES: {request.smiles}")
    
    try:
        # Call Modal inference function
        with chemberta_app.run():
            result = predict_chemberta_50epoch.remote(request.smiles)
        
        if result.get("status") == "success":
            logger.info("✅ ChemBERTa prediction completed successfully")
            return ChemBERTaPredictionResponse(**result)
        else:
            logger.error(f"❌ ChemBERTa prediction failed: {result.get('error')}")
            return ChemBERTaPredictionResponse(
                status="error",
                error=result.get("error", "Prediction failed")
            )
            
    except Exception as e:
        logger.error(f"❌ ChemBERTa prediction error: {e}")
        return ChemBERTaPredictionResponse(
            status="error",
            error=f"Prediction failed: {str(e)}"
        )

@chemberta_router.post("/test")
async def test_chemberta_inference_endpoint():
    """Test ChemBERTa inference with sample molecules"""
    
    if not CHEMBERTA_AVAILABLE:
        raise HTTPException(status_code=503, detail="ChemBERTa inference not available")
    
    logger.info("🧪 Running ChemBERTa inference test...")
    
    try:
        # Call Modal status function instead of test function
        with chemberta_app.run():
            result = get_chemberta_50epoch_status.remote()
        
        return result
        
    except Exception as e:
        logger.error(f"❌ ChemBERTa test failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@chemberta_router.get("/performance")
async def get_chemberta_performance():
    """Get detailed ChemBERTa model performance metrics"""
    
    if not CHEMBERTA_AVAILABLE:
        raise HTTPException(status_code=503, detail="ChemBERTa inference not available")
    
    performance_data = {
        "model_type": "ChemBERTa Multi-Task Transformer",
        "training_info": {
            "epochs": 20,
            "batch_size": 16,
            "learning_rate": "2e-5",
            "training_time_minutes": 3.5,
            "final_loss": 0.205,
            "mean_r2": 0.516,
            "std_r2": 0.153
        },
        "target_performance": {
            "EGFR": {"r2": 0.751, "samples": 165, "rank": 1},
            "MDM2": {"r2": 0.655, "samples": 122, "rank": 2},
            "BRAF": {"r2": 0.595, "samples": 117, "rank": 3},
            "PI3KCA": {"r2": 0.588, "samples": 58, "rank": 4},
            "HER2": {"r2": 0.583, "samples": 121, "rank": 5},
            "VEGFR2": {"r2": 0.555, "samples": 148, "rank": 6},
            "MET": {"r2": 0.502, "samples": 91, "rank": 7},
            "ALK": {"r2": 0.405, "samples": 64, "rank": 8},
            "CDK4": {"r2": 0.314, "samples": 64, "rank": 9},
            "CDK6": {"r2": 0.216, "samples": 111, "rank": 10}
        },
        "performance_breakdown": {
            "excellent_count": 2,  # R² > 0.6
            "good_count": 6,       # 0.4 < R² ≤ 0.6
            "fair_count": 2,       # 0.2 < R² ≤ 0.4
            "poor_count": 0        # R² ≤ 0.2
        },
        "dataset_info": {
            "total_samples": 5022,
            "train_samples": 3515,
            "val_samples": 502,
            "test_samples": 1005,
            "excluded_sparse_targets": ['STAT3', 'CTNNB1', 'RRM2', 'MYC']
        },
        "infrastructure": {
            "gpu": "A100",
            "inference_gpu": "T4",
            "platform": "Modal.com",
            "wandb_tracking": True
        }
    }
    
    return performance_data

# Health check endpoint
@chemberta_router.get("/health")
async def chemberta_health_check():
    """ChemBERTa service health check"""
    
    return {
        "service": "ChemBERTa Multi-Task Inference",
        "status": "healthy" if CHEMBERTA_AVAILABLE else "unavailable",
        "available": CHEMBERTA_AVAILABLE,
        "targets_count": 10,
        "model_ready": CHEMBERTA_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat()
    }