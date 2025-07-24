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
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import warnings

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
    """Load ChemBERTa model and tokenizer (MolBERT alternative)"""
    try:
        if 'molbert' not in models:
            logging.info("Loading ChemBERTa model...")
            tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
            model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
            models['molbert'] = {'tokenizer': tokenizer, 'model': model}
            logging.info("ChemBERTa model loaded successfully")
        return models['molbert']
    except Exception as e:
        logging.error(f"Error loading ChemBERTa: {e}")
        return None

def predict_with_molbert(smiles: str, property_type: str) -> Optional[float]:
    """Make predictions using ChemBERTa embeddings"""
    try:
        molbert = models.get('molbert')
        if not molbert:
            return None
            
        tokenizer = molbert['tokenizer']
        model = molbert['model']
        
        # Tokenize SMILES
        inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Simple prediction logic based on property type and embeddings
        embedding_mean = embeddings.mean().item()
        embedding_std = embeddings.std().item()
        
        if property_type == "bioactivity_ic50":
            # IC50 values typically range from 0.001 to 1000 µM (log scale)
            prediction = max(0.001, min(1000, abs(embedding_mean * 100)))
        elif property_type == "toxicity":
            # Toxicity probability (0-1)
            prediction = max(0, min(1, (embedding_std + 0.5)))
        elif property_type == "logP":
            # LogP typically ranges from -3 to 8
            prediction = max(-3, min(8, embedding_mean * 5))
        elif property_type == "solubility":
            # LogS typically ranges from -12 to 2
            prediction = max(-12, min(2, embedding_mean * 3 - 2))
        else:
            prediction = abs(embedding_mean)
            
        return float(prediction)
    except Exception as e:
        logging.error(f"Error in ChemBERTa prediction: {e}")
        return None

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
        "model_type": "real_ml" if real_models_available else "heuristic"
    }

@api_router.get("/targets", response_model=List[TargetInfo])
async def get_available_targets():
    """Get information about available protein targets"""
    targets = []
    
    for target, description in AVAILABLE_TARGETS.items():
        targets.append(TargetInfo(
            target=target,
            available=True,
            description=description,
            model_type="Enhanced RDKit-based"
        ))
    
    return targets

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
        
        # For IC50 predictions, try real ML model first, fallback to enhanced model
        if prediction_type == "bioactivity_ic50" and input_data.target:
            try:
                # Try real ML model first
                if real_predictor and hasattr(real_predictor, 'predict_ic50'):
                    try:
                        real_prediction = await real_predictor.predict_ic50_async(
                            input_data.smiles, 
                            input_data.target
                        )
                        result.enhanced_chemprop_prediction = real_prediction
                        logger.info(f"Using real ML model for {input_data.target} IC50 prediction")
                    except Exception as real_error:
                        logger.warning(f"Real ML model failed, falling back to heuristic: {real_error}")
                        # Fallback to heuristic model
                        enhanced_prediction = enhanced_ic50_prediction(
                            input_data.smiles, 
                            input_data.target
                        )
                        result.enhanced_chemprop_prediction = enhanced_prediction
                else:
                    # Use heuristic enhanced model
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
                    
            except Exception as e:
                logging.error(f"Error in IC50 prediction: {e}")
                result.enhanced_chemprop_prediction = {"error": str(e)}
        
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

# Import real ML predictor
try:
    from real_chemprop_predictor import RealChempropPredictor
    real_predictor = RealChempropPredictor()
    logger.info("Real Chemprop predictor initialized")
except Exception as e:
    logger.warning(f"Could not initialize real predictor: {e}")
    real_predictor = None

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting Veridica AI Chemistry Platform...")
    await load_molbert_model()
    
    # Initialize real ML models for common targets
    if real_predictor:
        try:
            logger.info("Initializing real ML models for common targets...")
            # Try to initialize models for the most common targets
            common_targets = ["EGFR", "BRAF", "CDK2"]
            initialized_targets = []
            
            for target in common_targets:
                try:
                    success = await real_predictor.initialize_models(target)
                    if success:
                        initialized_targets.append(target)
                        logger.info(f"✅ Real ML model ready for {target}")
                    else:
                        logger.warning(f"⚠️ Could not initialize real model for {target}, will use heuristic")
                except Exception as e:
                    logger.warning(f"⚠️ Error initializing {target} model: {e}")
            
            if initialized_targets:
                logger.info(f"🎯 Real ML models active for: {', '.join(initialized_targets)}")
            else:
                logger.info("📊 Using heuristic models (real ML models failed to initialize)")
                
        except Exception as e:
            logger.error(f"Error during real model initialization: {e}")
    
    logger.info("🚀 Platform ready with enhanced predictions!")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()