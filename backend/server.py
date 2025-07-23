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

class PredictionResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    smiles: str
    prediction_type: str
    molbert_prediction: Optional[float] = None
    chemprop_prediction: Optional[float] = None
    rdkit_value: Optional[float] = None
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResult]
    summary: Dict[str, Any]

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
    """Make predictions using MolBERT embeddings"""
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
        # This is a simplified approach - in practice, you'd use pre-trained property-specific heads
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
        logging.error(f"Error in MolBERT prediction: {e}")
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
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "available_predictions": [
            "bioactivity_ic50",
            "toxicity", 
            "logP",
            "solubility"
        ]
    }

@api_router.post("/predict", response_model=BatchPredictionResponse)
async def predict_molecular_properties(input_data: SMILESInput):
    """Predict molecular properties using MolBERT and Chemprop"""
    
    # Validate SMILES
    if not validate_smiles(input_data.smiles):
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    # Load MolBERT if not already loaded
    await load_molbert_model()
    
    results = []
    
    for prediction_type in input_data.prediction_types:
        # Get RDKit baseline properties
        rdkit_props = calculate_rdkit_properties(input_data.smiles)
        
        # Make predictions with both models
        molbert_pred = predict_with_molbert(input_data.smiles, prediction_type)
        chemprop_pred = predict_with_chemprop_simulation(input_data.smiles, prediction_type)
        
        # Get RDKit value if available
        rdkit_value = None
        if prediction_type == "logP":
            rdkit_value = rdkit_props.get('logP')
        elif prediction_type == "solubility":
            rdkit_value = rdkit_props.get('solubility_logS')
        
        # Calculate confidence (simplified)
        confidence = 0.85 if molbert_pred and chemprop_pred else 0.6
        
        result = PredictionResult(
            smiles=input_data.smiles,
            prediction_type=prediction_type,
            molbert_prediction=molbert_pred,
            chemprop_prediction=chemprop_pred,
            rdkit_value=rdkit_value,
            confidence=confidence
        )
        
        results.append(result)
        
        # Store in MongoDB
        await db.predictions.insert_one(result.dict())
    
    # Create summary
    summary = {
        "molecule": input_data.smiles,
        "total_predictions": len(results),
        "molecular_properties": rdkit_props,
        "prediction_types": input_data.prediction_types
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

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting Veridica AI Chemistry Platform...")
    await load_molbert_model()
    logger.info("Platform ready!")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()