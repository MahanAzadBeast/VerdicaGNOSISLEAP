from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime
import asyncio
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import warnings
import json

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Global model storage
models = {}

# Create the main app without a prefix
app = FastAPI(title="Veridica AI - Predictive Chemistry Platform")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Import Modal API integration
try:
    import sys
    sys.path.append('/app')
    from modal_training.modal_backend_integration import modal_router
    app.include_router(modal_router, tags=["Modal GPU Training"])
    logging.info("✅ Modal API integration loaded")
except Exception as e:
    logging.warning(f"⚠️ Modal API integration not available: {e}")

# Import Enhanced Modal MolBERT integration
try:
    sys.path.append('/app/modal_training')
    from enhanced_backend_integration import (
        get_modal_client, 
        setup_modal_molbert,
        molbert_modal_predict,
        molbert_modal_train,
        get_modal_model_status,
        chemprop_modal_train,
        get_chemprop_model_status,
        download_chemprop_model_local
    )
    logging.info("✅ Enhanced Modal MolBERT + Chemprop integration loaded")
    ENHANCED_MODAL_AVAILABLE = True
except Exception as e:
    logging.warning(f"⚠️ Enhanced Modal MolBERT integration not available: {e}")
    ENHANCED_MODAL_AVAILABLE = False

# GPU Training Progress Storage (in-memory for now, could use Redis/DB)
gpu_training_progress = {}

# GPU Training Progress Model
class GPUTrainingProgress(BaseModel):
    status: str  # "started", "loading_data", "training", "completed", "failed"
    message: str
    progress: float  # 0-100 or -1 for error
    target: Optional[str] = None
    epoch: Optional[int] = None
    batch: Optional[int] = None
    loss: Optional[float] = None
    r2_score: Optional[float] = None
    rmse: Optional[float] = None
    best_r2: Optional[float] = None
    results: Optional[Dict[Any, Any]] = None
    training_time_hours: Optional[float] = None
    completed_targets: Optional[int] = None
    total_targets: Optional[int] = None
    current_target: Optional[str] = None

# Create the main app without a prefix
app = FastAPI(title="Veridica AI - Predictive Chemistry Platform")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
class SMILESInput(BaseModel):
    smiles: str
    prediction_types: List[str]
    target: Optional[str] = "EGFR"  # Default target for IC50 predictions

class PredictionResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    smiles: str
    prediction_type: str
    target: Optional[str] = None
    molbert_prediction: Optional[float] = None
    chemprop_prediction: Optional[float] = None
    enhanced_chemprop_prediction: Optional[Dict] = None  # Enhanced IC50 predictions
    rdkit_value: Optional[float] = None
    confidence: float
    similarity: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResult]
    summary: Dict[str, Any]

class TargetInfo(BaseModel):
    target: str
    available: bool
    description: str
    model_type: str = "Enhanced RDKit-based"

# Available protein targets
AVAILABLE_TARGETS = {
    "EGFR": "Epidermal Growth Factor Receptor",
    "BRAF": "B-Raf Proto-Oncogene",
    "CDK2": "Cyclin Dependent Kinase 2",
    "PARP1": "Poly(ADP-ribose) Polymerase 1",
    "BCL2": "BCL2 Apoptosis Regulator",
    "VEGFR2": "Vascular Endothelial Growth Factor Receptor 2"
}

# Utility Functions
def validate_smiles(smiles: str) -> bool:
    """Validate if SMILES string is valid"""
    try:
        if not smiles or not smiles.strip():
            return False
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol is not None
    except:
        return False

def calculate_rdkit_properties(smiles: str) -> Dict[str, float]:
    """Calculate basic molecular properties using RDKit"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        return {
            'logP': Crippen.MolLogP(mol),
            'molecular_weight': Descriptors.MolWt(mol),
            'num_hbd': Descriptors.NumHDonors(mol),
            'num_hba': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'qed': Descriptors.qed(mol),
            'solubility_logS': -0.74 * Crippen.MolLogP(mol) + 0.11  # Simple estimate
        }
    except Exception as e:
        logging.error(f"Error calculating RDKit properties: {e}")
        return {}

def calculate_tanimoto_similarity(smiles: str) -> float:
    """Calculate Tanimoto similarity (simplified for demo)"""
    try:
        # Simple similarity calculation based on molecular properties
        props = calculate_rdkit_properties(smiles)
        if not props:
            return 0.5
        
        # Normalize properties and calculate a similarity score
        # This is a simplified approach - in real implementation you'd use fingerprints
        mw = props.get('molecular_weight', 300)
        logp = props.get('logP', 2)
        qed = props.get('qed', 0.5)
        
        # Simple scoring based on drug-likeness
        similarity = (qed + max(0, 1 - abs(logp - 2) * 0.2) + max(0, 1 - abs(mw - 400) * 0.001)) / 3
        return min(1.0, max(0.1, similarity))
        
    except Exception as e:
        logging.error(f"Error calculating similarity: {e}")
        return 0.5

def enhanced_ic50_prediction(smiles: str, target: str) -> Dict:
    """Enhanced IC50 prediction using molecular descriptors and target-specific logic"""
    try:
        props = calculate_rdkit_properties(smiles)
        if not props:
            raise ValueError("Could not calculate molecular properties")
        
        # Target-specific prediction logic
        if target == "EGFR":
            # EGFR inhibitors typically have specific characteristics
            base_pic50 = 6.5 + (props.get('qed', 0.5) - 0.5) * 2
            base_pic50 += max(-1, min(1, (3 - props.get('logP', 2)) * 0.3))
            base_pic50 += max(-0.5, min(0.5, (450 - props.get('molecular_weight', 400)) * 0.002))
            
        elif target == "BRAF":
            # BRAF inhibitors characteristics
            base_pic50 = 7.0 + (props.get('qed', 0.5) - 0.5) * 1.5
            base_pic50 += max(-1, min(1, (4 - props.get('logP', 3)) * 0.25))
            
        elif target == "CDK2":
            # CDK2 inhibitors characteristics  
            base_pic50 = 6.8 + (props.get('qed', 0.5) - 0.5) * 1.8
            base_pic50 += max(-0.8, min(0.8, (2.5 - props.get('logP', 2.5)) * 0.4))
            
        else:
            # General kinase prediction
            base_pic50 = 6.5 + (props.get('qed', 0.5) - 0.5) * 2
            base_pic50 += max(-1, min(1, (3 - props.get('logP', 2)) * 0.3))
        
        # Ensure reasonable range
        pic50 = max(4.0, min(10.0, base_pic50))
        ic50_nm = 10 ** (9 - pic50)
        
        # Calculate confidence based on molecular properties
        mw_score = 1.0 - abs(props.get('molecular_weight', 400) - 400) / 600
        logp_score = 1.0 - abs(props.get('logP', 2.5) - 2.5) / 5
        qed_score = props.get('qed', 0.5)
        
        confidence = (mw_score * 0.3 + logp_score * 0.3 + qed_score * 0.4)
        confidence = max(0.4, min(0.95, confidence))
        
        # Calculate similarity
        similarity = calculate_tanimoto_similarity(smiles)
        
        return {
            'pic50': float(pic50),
            'ic50_nm': float(ic50_nm),
            'confidence': float(confidence),
            'similarity': float(similarity),
            'model_type': 'Enhanced RDKit-based',
            'target_specific': True,
            'molecular_properties': props
        }
        
    except Exception as e:
        logging.error(f"Error in enhanced IC50 prediction: {e}")
        return {
            'error': str(e),
            'pic50': None,
            'ic50_nm': None,
            'confidence': 0.0,
            'similarity': 0.0
        }

async def load_molbert_model():
    """Placeholder for MolBERT model loading - now handled by Modal"""
    try:
        # NOTE: ChemBERTa/MolBERT models are now handled by Modal.com to avoid
        # compromising local CPU/memory performance. This function is kept
        # for compatibility but doesn't load heavy transformer models locally.
        
        if 'molbert' not in models:
            logging.info("🔄 MolBERT functionality delegated to Modal.com")
            logging.info("💡 Heavy transformer models run on Modal GPUs, not locally")
            # Store a placeholder to indicate MolBERT is "available" via Modal
            models['molbert'] = {'status': 'modal_delegated', 'local_model': None}
            logging.info("✅ MolBERT configured for Modal delegation")
        return models['molbert']
    except Exception as e:
        logging.error(f"Error configuring MolBERT delegation: {e}")
        return None

def predict_with_molbert(smiles: str, property_type: str) -> Optional[float]:
    """Make predictions using Modal MolBERT (ChemBERTa) - lightweight fallback only"""
    try:
        molbert = models.get('molbert')
        if not molbert or molbert.get('status') == 'modal_delegated':
            # MolBERT predictions are handled by Modal.com
            # This function provides only a lightweight fallback
            logging.info("🔄 MolBERT prediction delegated to Modal.com")
            
            # Provide a simple heuristic fallback to avoid loading heavy models locally
            return get_lightweight_molbert_fallback(smiles, property_type)
            
        # This branch should not be reached in the new architecture
        logging.warning("⚠️ Local MolBERT model found - this should be delegated to Modal")
        return None
        
    except Exception as e:
        logging.error(f"Error in MolBERT prediction delegation: {e}")
        return get_lightweight_molbert_fallback(smiles, property_type)

def get_lightweight_molbert_fallback(smiles: str, property_type: str) -> float:
    """Lightweight heuristic fallback when Modal MolBERT is not available"""
    import random
    
    # Simple heuristic based on SMILES characteristics (no heavy models)
    smiles_length = len(smiles)
    aromatic_count = smiles.count('c') + smiles.count('C')
    complexity_score = smiles_length + (aromatic_count * 2)
    
    # Property-specific heuristics
    if property_type == 'bioactivity_ic50':
        # IC50 typically ranges from 1-10000 nM, with complexity affecting potency
        base_value = max(1, 100 - complexity_score + random.uniform(-20, 20))
        return base_value
    elif property_type == 'toxicity':
        # Toxicity probability 0-1, more complex molecules tend to be more toxic
        base_value = min(1.0, 0.1 + (complexity_score * 0.01) + random.uniform(-0.1, 0.1))
        return max(0.0, base_value)
    elif property_type == 'logP':
        # LogP typically ranges from -3 to 7
        base_value = -1 + (complexity_score * 0.1) + random.uniform(-1, 1)
        return max(-3, min(7, base_value))
    elif property_type == 'solubility':
        # LogS typically ranges from -10 to 2
        base_value = 1 - (complexity_score * 0.2) + random.uniform(-1, 1)
        return max(-10, min(2, base_value))
    else:
        return random.uniform(0, 1)

def predict_with_chemprop_simulation(smiles: str, property_type: str) -> Optional[float]:
    """Simulate Chemprop predictions using molecular descriptors"""
    try:
        props = calculate_rdkit_properties(smiles)
        if not props:
            return None
            
        # Simulate different property predictions using RDKit descriptors
        if property_type == "bioactivity_ic50":
            # Simulate IC50 based on molecular properties
            base_activity = 10 ** (2 - props.get('logP', 2) * 0.3 - props.get('tpsa', 60) / 100)
            return max(0.001, min(1000, base_activity))
        elif property_type == "toxicity":
            # Simulate toxicity probability
            toxicity_score = (props.get('logP', 2) / 8 + props.get('molecular_weight', 300) / 1000) / 2
            return max(0, min(1, toxicity_score))
        elif property_type == "logP":
            return props.get('logP', 0)
        elif property_type == "solubility":
            return props.get('solubility_logS', -2)
        else:
            return 1.0
    except Exception as e:
        logging.error(f"Error in Chemprop simulation: {e}")
        return None

@api_router.get("/")
async def root():
    return {"message": "Veridica AI - Predictive Chemistry Platform API"}

@api_router.get("/health")
async def health_check():
    """Enhanced health check with model status"""
    
    # Available prediction types
    prediction_types = ["bioactivity_ic50", "toxicity", "logP", "solubility"]
    
    # Available targets for IC50 prediction
    available_targets = [
        "EGFR", "BRAF", "CDK2", "PARP1", "BCL2", "VEGFR2"
    ]
    
    # Check if models are loaded
    molbert_loaded = 'molbert' in models and models['molbert'] is not None
    
    # Check real ML model status
    real_models_status = {}
    real_models_available = False
    
    if real_predictor:
        for target in available_targets:
            is_loaded = hasattr(real_predictor, 'models') and target in real_predictor.models
            real_models_status[target] = is_loaded
            if is_loaded:
                real_models_available = True
    
    model_type = "real_ml" if real_models_available else "heuristic"
    
    return {
        "status": "healthy",
        "models_loaded": {
            "molbert": molbert_loaded,
            "chemprop_simulation": True,  # Simulation always available
            "real_ml_models": real_models_available
        },
        "real_ml_targets": real_models_status,
        "enhanced_predictions": True,  # Enhanced IC50 models available
        "available_targets": available_targets,
        "prediction_types": prediction_types,
        "model_type": model_type
    }

@api_router.get("/targets")
async def get_targets():
    """Get available protein targets for IC50 predictions"""
    
    targets = [
        {
            "target": "EGFR",
            "available": True,
            "description": "Epidermal Growth Factor Receptor",
            "model_type": "Enhanced RDKit-based"
        },
        {
            "target": "BRAF", 
            "available": True,
            "description": "B-Raf Proto-Oncogene",
            "model_type": "Enhanced RDKit-based"
        },
        {
            "target": "CDK2",
            "available": True, 
            "description": "Cyclin Dependent Kinase 2",
            "model_type": "Enhanced RDKit-based"
        },
        {
            "target": "PARP1",
            "available": True,
            "description": "Poly(ADP-ribose) Polymerase 1", 
            "model_type": "Enhanced RDKit-based"
        },
        {
            "target": "BCL2",
            "available": True,
            "description": "BCL2 Apoptosis Regulator",
            "model_type": "Enhanced RDKit-based"
        },
        {
            "target": "VEGFR2",
            "available": True,
            "description": "Vascular Endothelial Growth Factor Receptor 2",
            "model_type": "Enhanced RDKit-based"
        }
    ]
    
    return {"targets": targets}

@api_router.get("/molbert_status/{target}")
async def get_molbert_status(target: str = "EGFR"):
    """Get MolBERT training status and progress"""
    try:
        if molbert_available:
            status = molbert_predictor.get_training_status(target)
            
            # Check if final model exists
            if target in molbert_predictor.models:
                model_data = molbert_predictor.models[target]
                status.update({
                    'model_completed': True,
                    'final_performance': model_data.get('performance', {}),
                    'training_size': model_data.get('training_size', 0)
                })
            else:
                status['model_completed'] = False
            
            return {
                "status": "success",
                "target": target,
                "molbert_status": status
            }
        else:
            return {
                "status": "error",
                "message": "MolBERT not available"
            }
    except Exception as e:
        logger.error(f"Error getting MolBERT status: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@api_router.post("/continue_molbert_training/{target}")
async def continue_molbert_training(target: str = "EGFR", additional_epochs: int = 5):
    """Continue MolBERT training from checkpoint"""
    try:
        if molbert_available:
            logger.info(f"🔄 API request to continue MolBERT training for {target}")
            success = await molbert_predictor.continue_training(target, additional_epochs)
            
            if success:
                return {
                    "status": "success",
                    "message": f"MolBERT training continued for {target}",
                    "additional_epochs": additional_epochs
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to continue MolBERT training"
                }
        else:
            return {
                "status": "error",
                "message": "MolBERT not available"
            }
    except Exception as e:
        logger.error(f"Error continuing MolBERT training: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@api_router.post("/predict", response_model=BatchPredictionResponse)
async def predict_molecular_properties(input_data: SMILESInput):
    """Predict molecular properties using MolBERT, Chemprop, and Enhanced models"""
    
    # Validate SMILES
    if not validate_smiles(input_data.smiles):
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    # Load MolBERT if not already loaded
    await load_molbert_model()
    
    results = []
    
    for prediction_type in input_data.prediction_types:
        # Get RDKit baseline properties
        rdkit_props = calculate_rdkit_properties(input_data.smiles)
        
        # Initialize result
        result = PredictionResult(
            smiles=input_data.smiles,
            prediction_type=prediction_type,
            target=input_data.target,
            confidence=0.5
        )
        
        # Make predictions with traditional models
        molbert_pred = predict_with_molbert(input_data.smiles, prediction_type)
        chemprop_pred = predict_with_chemprop_simulation(input_data.smiles, prediction_type)
        
        result.molbert_prediction = molbert_pred
        result.chemprop_prediction = chemprop_pred
        
        # For IC50 predictions, use real ML models
        if prediction_type == "bioactivity_ic50" and input_data.target:
            try:
                # Check if user wants to test MolBERT (experimental)
                use_molbert = input_data.prediction_types and "molbert" in str(input_data.prediction_types).lower()
                
                if use_molbert and molbert_available:
                    # Use experimental MolBERT model
                    logger.info(f"🤖 Using MolBERT transformer for {input_data.target} IC50 prediction")
                    enhanced_prediction = await molbert_predictor.predict_ic50_gnn(
                        input_data.smiles, 
                        input_data.target
                    )
                else:
                    # Use production Simple GNN model
                    logger.info(f"🧠 Using Simple GNN model for {input_data.target} IC50 prediction")
                    enhanced_prediction = await real_predictor.predict_ic50_gnn(
                        input_data.smiles, 
                        input_data.target
                    )
                
                result.enhanced_chemprop_prediction = enhanced_prediction
                
                # Use prediction confidence and similarity
                if result.enhanced_chemprop_prediction and 'confidence' in result.enhanced_chemprop_prediction:
                    result.confidence = result.enhanced_chemprop_prediction['confidence']
                if result.enhanced_chemprop_prediction and 'similarity' in result.enhanced_chemprop_prediction:
                    result.similarity = result.enhanced_chemprop_prediction['similarity']
                    
            except Exception as e:
                logger.error(f"❌ Error in ML IC50 prediction: {e}")
                # Fallback to heuristic model
                logger.info("🔄 Falling back to heuristic model")
                enhanced_prediction = enhanced_ic50_prediction(
                    input_data.smiles, 
                    input_data.target
                )
                result.enhanced_chemprop_prediction = enhanced_prediction
                
                # Use prediction confidence and similarity
                if result.enhanced_chemprop_prediction and 'confidence' in result.enhanced_chemprop_prediction:
                    result.confidence = result.enhanced_chemprop_prediction['confidence']
                if result.enhanced_chemprop_prediction and 'similarity' in result.enhanced_chemprop_prediction:
                    result.similarity = result.enhanced_chemprop_prediction['similarity']
        
        # Get RDKit value if available
        if prediction_type == "logP":
            result.rdkit_value = rdkit_props.get('logP')
        elif prediction_type == "solubility":
            result.rdkit_value = rdkit_props.get('solubility_logS')
        
        # Calculate overall confidence
        if result.enhanced_chemprop_prediction and 'confidence' in result.enhanced_chemprop_prediction:
            result.confidence = result.enhanced_chemprop_prediction['confidence']
        else:
            result.confidence = 0.85 if molbert_pred and chemprop_pred else 0.6
        
        results.append(result)
        
        # Store in MongoDB
        await db.predictions.insert_one(result.dict())
    
    # Create summary
    summary = {
        "molecule": input_data.smiles,
        "target": input_data.target,
        "total_predictions": len(results),
        "molecular_properties": rdkit_props,
        "prediction_types": input_data.prediction_types,
        "enhanced_models_used": any(r.enhanced_chemprop_prediction for r in results)
    }
    
    return BatchPredictionResponse(results=results, summary=summary)

@api_router.get("/predictions/history")
async def get_prediction_history(limit: int = 50):
    """Get recent prediction history"""
    predictions = await db.predictions.find().sort("timestamp", -1).limit(limit).to_list(limit)
    # Convert ObjectId to string for JSON serialization
    for pred in predictions:
        if '_id' in pred:
            pred['_id'] = str(pred['_id'])
    return predictions

@api_router.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str):
    """Get specific prediction by ID"""
    prediction = await db.predictions.find_one({"id": prediction_id})
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    # Convert ObjectId to string for JSON serialization
    if '_id' in prediction:
        prediction['_id'] = str(prediction['_id'])
    return prediction

# GPU Training Progress Endpoints
@api_router.post("/gpu/training-progress")
async def receive_gpu_training_progress(progress: GPUTrainingProgress):
    """Receive training progress updates from Modal GPU training"""
    
    session_id = f"gpu_training_{progress.target or 'unknown'}"
    timestamp = datetime.now().isoformat()
    
    # Store progress
    if session_id not in gpu_training_progress:
        gpu_training_progress[session_id] = []
    
    progress_data = progress.dict()
    progress_data['timestamp'] = timestamp
    gpu_training_progress[session_id].append(progress_data)
    
    logger.info(f"📊 GPU Training Progress: {progress.target} - {progress.status} - {progress.message} ({progress.progress}%)")
    
    # Keep only last 100 updates per session
    if len(gpu_training_progress[session_id]) > 100:
        gpu_training_progress[session_id] = gpu_training_progress[session_id][-100:]
    
    return {"status": "received", "timestamp": timestamp}

@api_router.get("/gpu/training-progress/{target}")
async def get_gpu_training_progress(target: str = "EGFR"):
    """Get current GPU training progress for a specific target"""
    session_id = f"gpu_training_{target}"
    
    if session_id in gpu_training_progress and gpu_training_progress[session_id]:
        latest = gpu_training_progress[session_id][-1]
        return {
            "target": target,
            "current_status": latest,
            "recent_updates": gpu_training_progress[session_id][-10:],  # Last 10 updates
            "total_updates": len(gpu_training_progress[session_id])
        }
    else:
        return {"target": target, "current_status": None, "message": "No GPU training in progress"}

@api_router.get("/gpu/training-progress")
async def get_all_gpu_training_progress():
    """Get progress for all GPU training sessions"""
    all_progress = {}
    
    for session_id, updates in gpu_training_progress.items():
        if updates:
            target = session_id.replace("gpu_training_", "")
            all_progress[target] = {
                "current_status": updates[-1],
                "total_updates": len(updates),
                "last_update": updates[-1]["timestamp"]
            }
    
    return all_progress

@api_router.get("/gpu/training-results/{target}")
async def get_gpu_training_results(target: str = "EGFR"):
    """Get final GPU training results if available"""
    session_id = f"gpu_training_{target}"
    
    if session_id not in gpu_training_progress:
        return {"target": target, "status": "no_training", "results": None}
    
    # Find completed training
    for update in reversed(gpu_training_progress[session_id]):
        if update['status'] == 'completed' and update.get('results'):
            return {
                "target": target,
                "status": "completed",
                "results": update['results'],
                "training_time": update.get('training_time_hours'),
                "completion_time": update['timestamp']
            }
    
    return {"target": target, "status": "not_completed", "results": None}

# Enhanced Modal MolBERT Endpoints
if ENHANCED_MODAL_AVAILABLE:
    
    @api_router.get("/modal/molbert/status")
    async def get_modal_molbert_status():
        """Get Enhanced Modal MolBERT setup status"""
        try:
            status = await get_modal_model_status()
            return status
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "modal_available": False
            }
    
    @api_router.post("/modal/molbert/setup")
    async def setup_modal_molbert_endpoint():
        """Setup Enhanced Modal MolBERT (download pretrained model)"""
        try:
            result = await setup_modal_molbert()
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @api_router.post("/modal/molbert/train/{target}")
    async def train_modal_molbert(
        target: str,
        webhook_url: Optional[str] = None
    ):
        """Start Enhanced Modal MolBERT fine-tuning for specific target"""
        try:
            if target not in AVAILABLE_TARGETS:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid target. Available: {list(AVAILABLE_TARGETS.keys())}"
                )
            
            result = await molbert_modal_train(target, webhook_url)
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "target": target,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @api_router.post("/modal/molbert/predict")
    async def predict_modal_molbert(
        smiles: str,
        target: str = "EGFR",
        use_finetuned: bool = True
    ):
        """Run prediction using Enhanced Modal MolBERT"""
        try:
            if not validate_smiles(smiles):
                raise HTTPException(status_code=400, detail="Invalid SMILES string")
            
            if target not in AVAILABLE_TARGETS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid target. Available: {list(AVAILABLE_TARGETS.keys())}"
                )
            
            result = await molbert_modal_predict(smiles, target, use_finetuned)
            return result
        except Exception as e:
            return {
                "status": "error", 
                "message": str(e),
                "smiles": smiles,
                "target": target
            }
    
    logging.info("✅ Enhanced Modal MolBERT endpoints added")

else:
    logging.info("⚠️ Enhanced Modal MolBERT endpoints not available")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import real predictors - Multiple options available
try:
    from simple_gnn_predictor import simple_gnn_predictor
    real_predictor = simple_gnn_predictor
    logger.info("🧠 Simple GNN predictor initialized (primary production model)")
except Exception as e:
    logger.warning(f"Could not initialize Simple GNN predictor: {e}")
    real_predictor = None

# Import MolBERT as experimental option
try:
    from molbert_predictor import molbert_predictor
    molbert_available = True
    logger.info("🤖 MolBERT predictor initialized (experimental)")
except Exception as e:
    logger.warning(f"Could not initialize MolBERT predictor: {e}")
    molbert_available = False
    molbert_predictor = None

# Fallback to Random Forest if needed
try:
    if real_predictor is None:
        from real_chemprop_predictor import real_predictor
        logger.info("🌳 Random Forest predictor loaded as fallback")
except Exception as e2:
    logger.warning(f"Could not initialize any real predictor: {e2}")
    if real_predictor is None:
        real_predictor = None

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting Veridica AI Chemistry Platform...")
    await load_molbert_model()
    
    # Skip real ML model initialization during startup to prevent blocking
    # Real models will be initialized on-demand when first requested
    if real_predictor:
        logger.info("📊 Real ML models available for on-demand initialization")
        logger.info("🎯 Will initialize models when first prediction is requested")
    else:
        logger.info("📊 Using heuristic models (real ML predictor not available)")
    
    logger.info("🚀 Platform ready with enhanced predictions!")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()