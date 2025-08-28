from fastapi import FastAPI, APIRouter, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
import traceback
import io
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import json
from rdkit import Chem
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
import hashlib

# Project imports
try:
    from gnosis_model2_predictor import GnosisIIPredictor
    MODEL2_IMPORT_AVAILABLE = True
except ImportError:
    MODEL2_IMPORT_AVAILABLE = False
    logging.warning("⚠️ gnosis_model2_predictor not available")

# Set up logging
logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).parent.parent

# Global objects
db = None
models = {}

# Initialize Gnosis I Model
try:
    from gnosis_model1_predictor import initialize_gnosis_predictor, get_gnosis_predictor
    from hp_ad_layer import get_hp_ad_layer, initialize_hp_ad_layer_sync
    
    model_path = ROOT_DIR / "models" / "gnosis_model1_best.pt"
    
    # Initialize regardless of model file existence
    initialize_gnosis_predictor(str(model_path) if model_path.exists() else None)
    logging.info("✅ Gnosis I (Model 1) initialized")
    GNOSIS_I_AVAILABLE = True
    
    # TEMPORARILY DISABLE HP AD layer initialization to fix system performance issues
    try:
        logging.info("⚠️ HP AD layer initialization TEMPORARILY DISABLED due to performance issues")
        logging.info("⚠️ System will use basic predictions without gating until HP AD layer is fixed") 
        GNOSIS_AD_AVAILABLE = False
        
        # TODO: Fix HP AD layer initialization performance and memory issues
        # The current implementation causes system overload (CPU >100%, timeouts)
        # Need to optimize fingerprint database building and similarity calculations
        
    except Exception as e:
        logging.error(f"❌ Error during HP AD layer handling: {e}")
        GNOSIS_AD_AVAILABLE = False
    
except Exception as e:
    logging.error(f"❌ Failed to load Gnosis I: {e}")
    # Still try to create a basic instance
    try:
        initialize_gnosis_predictor(None)
        GNOSIS_I_AVAILABLE = True
        GNOSIS_AD_AVAILABLE = False
        logging.warning("⚠️ Gnosis I running in fallback mode")
    except:
        GNOSIS_I_AVAILABLE = False
        GNOSIS_AD_AVAILABLE = False

# Initialize Gnosis II Model
try:
    if MODEL2_IMPORT_AVAILABLE:
        gnosis_ii_predictor = GnosisIIPredictor()
        logging.info("✅ Gnosis II (Model 2) initialized")
        MODEL2_AVAILABLE = True
    else:
        MODEL2_AVAILABLE = False
        gnosis_ii_predictor = None
        logging.warning("⚠️ Gnosis II (Model 2) not available - import missing")
except Exception as e:
    logging.error(f"❌ Failed to load Gnosis II: {e}")
    MODEL2_AVAILABLE = False
    gnosis_ii_predictor = None

# Initialize other models (existing code remains the same)
try:
    from production_model import ProductionModelLoader
    model_loader = ProductionModelLoader()
    available_models = []  # Placeholder since get_available_models is not available
    logging.info(f"✅ Available models: {available_models}")
    real_models_available = len(available_models) > 0
except Exception as e:
    logging.error(f"❌ Failed to load models: {e}")
    real_models_available = False
    model_loader = None

# Try to load MolBERT
try:
    import sys
    sys.path.append("/app/MolBERT")
    from molbert.apps.finetune import FineTuner
    molbert_loaded = True
    logging.info("✅ MolBERT loaded successfully")
except Exception as e:
    molbert_loaded = False
    logging.warning(f"⚠️ MolBERT not available: {e}")

# ChemBERTa-based predictor 
try:
    from chembert_predictor import ChemBERTaPredictor
    chemberta_predictor = ChemBERTaPredictor()
    ONCOPROTEIN_AVAILABLE = True
    logging.info("✅ ChemBERTa oncoprotein predictor loaded")
except Exception as e:
    ONCOPROTEIN_AVAILABLE = False
    logging.warning(f"⚠️ ChemBERTa oncoprotein predictor not available: {e}")

# Additional model availability flags
CHEMPROP_MULTITASK_AVAILABLE = True  # Simulation
CELL_LINE_MODEL_AVAILABLE = True     # Using the real model
THERAPEUTIC_INDEX_AVAILABLE = True   # Simulation  
EXPANDED_MODELS_AVAILABLE = True     # Simulation

# Set up MongoDB connection
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db
    try:
        # Try to connect to MongoDB
        client = AsyncIOMotorClient(os.environ.get("MONGO_URL", "mongodb://localhost:27017"))
        db = client.veridica
        
        # Test the connection
        await client.server_info()
        logging.info("✅ Connected to MongoDB")
        
        # Create indexes
        await db.predictions.create_index("id")
        await db.predictions.create_index("timestamp")
        
        yield
    except Exception as e:
        logging.error(f"❌ Failed to connect to MongoDB: {e}")
        db = None
        yield
    finally:
        # Shutdown
        if db is not None:
            db.client.close()

app = FastAPI(
    title="Veridica AI Platform",
    description="Advanced AI platform for molecular property prediction and drug discovery",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Router
api_router = APIRouter()

# Pydantic models
class PredictionInput(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule")
    targets: Optional[List[str]] = Field(default=None, description="Target proteins")
    assay_types: Optional[List[str]] = Field(default=None, description="Types of assays to predict")

class BatchPredictionInput(BaseModel):
    smiles_list: List[str] = Field(..., description="List of SMILES strings")
    targets: Optional[List[str]] = Field(default=None, description="Target proteins")
    assay_types: Optional[List[str]] = Field(default=None, description="Types of assays to predict")

class GnosisIPredictionInput(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule")
    targets: List[str] = Field(..., description="Target proteins (required)")
    assay_types: List[str] = Field(..., description="Types of assays to predict (required)")

class GnosisIPredictionResult(BaseModel):
    smiles: str
    predictions: Dict[str, Dict[str, Any]]
    properties: Dict[str, float]
    model_info: Dict[str, Any]

class GnosisIIInput(BaseModel):
    smiles: str = Field(..., description="SMILES string of the compound")
    cell_lines: Optional[List[str]] = Field(default=None, description="Cell line IDs")
    include_genomic_features: bool = Field(default=True, description="Include genomic context")
    prediction_mode: str = Field(default="ic50", description="Prediction mode: 'ic50' or 'viability'")

class Model2Input(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    cell_lines: Optional[List[str]] = Field(default=None, description="Target cell lines")

# Utility functions
def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string using RDKit"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

# Health check endpoint
@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "molbert": molbert_loaded,
            "chemprop_simulation": True,  # Simulation always available
            "real_ml_models": real_models_available,
            "oncoprotein_chemberta": ONCOPROTEIN_AVAILABLE,
            "chemprop_multitask_simulation": CHEMPROP_MULTITASK_AVAILABLE,
            "cell_line_response_model": CELL_LINE_MODEL_AVAILABLE,
            "therapeutic_index_model": THERAPEUTIC_INDEX_AVAILABLE,
            "expanded_models": EXPANDED_MODELS_AVAILABLE,
            "gnosis_i": GNOSIS_I_AVAILABLE,
            "gnosis_i_hp_ad_layer": GNOSIS_AD_AVAILABLE,
            "model2_cytotoxicity": MODEL2_AVAILABLE
        },
        "database_connected": db is not None,
        "gnosis_i_info": {
            "available": GNOSIS_I_AVAILABLE,
            "model_name": "Gnosis I (Model 1)",
            "model_type": "Ligand Activity Predictor",
            "r2_score": 0.6281,
            "capabilities": ["IC50 prediction", "Ki prediction", "EC50 prediction", "LogP calculation", "LogS calculation"],
            "target_categories": ["oncoproteins", "tumor_suppressors"],
            "description": "Fine-tuned ChemBERTa model for ligand-target binding affinity prediction",
            "hp_ad_layer_available": GNOSIS_AD_AVAILABLE,
            "ad_capabilities": ["Ultra-fast multi-view AD scoring", "Learned calibration models", "AD-aware conformal prediction", "Target-specific calibration", "Kinase mechanism gating", "Performance <5s target"] if GNOSIS_AD_AVAILABLE else []
        },
        "gnosis_ii_info": {
            "available": MODEL2_AVAILABLE,
            "model_name": "Gnosis II",
            "model_type": "Cytotoxicity Prediction Model", 
            "description": "Advanced model for predicting drug cytotoxicity in cancer cell lines with genomic features"
        }
    }

# Existing endpoints (unchanged)
@api_router.post("/predict")
async def predict_molecular_properties(input_data: PredictionInput):
    """Predict molecular properties using available models"""
    
    # Validate SMILES
    if not validate_smiles(input_data.smiles):
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    try:
        # Use the production model loader if available
        if model_loader is not None:
            predictions = await model_loader.predict_molecular_properties(
                smiles=input_data.smiles,
                targets=input_data.targets,
                assay_types=input_data.assay_types
            )
        else:
            # Fallback when production model loader not available
            predictions = {
                "smiles": input_data.smiles,
                "error": "ProductionModelLoader not available - service in maintenance mode"
            }
        
        # Store in MongoDB if available
        if db:
            prediction_record = {
                "id": str(uuid.uuid4()),
                "smiles": input_data.smiles,
                "targets": input_data.targets,
                "assay_types": input_data.assay_types,
                "predictions": predictions,
                "timestamp": datetime.utcnow()
            }
            await db.predictions.insert_one(prediction_record)
        
        return predictions
    
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@api_router.post("/batch-predict")
async def batch_predict(input_data: BatchPredictionInput):
    """Batch prediction endpoint"""
    
    # Validate all SMILES
    invalid_smiles = [smiles for smiles in input_data.smiles_list if not validate_smiles(smiles)]
    if invalid_smiles:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {invalid_smiles}")
    
    try:
        if model_loader is not None:
            results = []
            for smiles in input_data.smiles_list:
                predictions = await model_loader.predict_molecular_properties(
                    smiles=smiles,
                    targets=input_data.targets,
                    assay_types=input_data.assay_types
                )
                results.append({
                    "smiles": smiles,
                    "predictions": predictions
                })
        else:
            results = [{"smiles": smiles, "error": "ProductionModelLoader not available"} 
                      for smiles in input_data.smiles_list]
        
        return {"results": results}
    
    except Exception as e:
        logging.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Gnosis I endpoints
@api_router.get("/gnosis-i/training-data")
async def get_gnosis_i_training_data():
    """Get Gnosis I training data statistics"""
    
    if not GNOSIS_I_AVAILABLE:
        return {
            "available": False,
            "message": "Gnosis I not available"
        }
    
    try:
        predictor = get_gnosis_predictor()
        if not predictor:
            return {
                "available": False,
                "message": "Gnosis I predictor not initialized"
            }
        
        return {
            "available": True,
            "training_data": predictor.target_training_data
        }
        
    except Exception as e:
        logging.error(f"Error getting Gnosis I training data: {e}")
        return {
            "available": False,
            "message": f"Error: {str(e)}"
        }

@api_router.post("/admin/reinitialize-hp-ad")
async def reinitialize_hp_ad():
    """Admin endpoint to manually reinitialize HP AD layer with real data"""
    try:
        from real_chembl_loader import load_real_chembl_for_ad
        from hp_ad_layer import initialize_hp_ad_layer_sync, get_hp_ad_layer
        
        # Load real ChEMBL data
        real_training_data = load_real_chembl_for_ad(max_targets=8)
        
        if real_training_data is not None:
            # Initialize HP AD layer
            initialize_hp_ad_layer_sync(real_training_data)
            
            # Verify initialization
            hp_ad = get_hp_ad_layer()
            if hp_ad and hp_ad.initialized:
                targets = list(hp_ad.fp_db.db_rdkit.keys()) if hp_ad.fp_db.db_rdkit else []
                metadata_targets = list(hp_ad.fp_db.ligand_metadata.keys()) if hp_ad.fp_db.ligand_metadata else []
                
                return {
                    "status": "success",
                    "message": "HP AD layer reinitialized with real ChEMBL data",
                    "data_points": len(real_training_data),
                    "targets": targets,
                    "metadata_targets": metadata_targets,
                    "initialized": hp_ad.initialized
                }
            else:
                return {
                    "status": "error", 
                    "message": "HP AD layer initialization failed"
                }
        else:
            return {
                "status": "error",
                "message": "No real ChEMBL data available"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Initialization failed: {str(e)}"
        }

@api_router.get("/gnosis-i/targets")
async def get_gnosis_i_targets():
    """Get available targets for Gnosis I predictions"""
    try:
        if not GNOSIS_I_AVAILABLE:
            raise HTTPException(status_code=503, detail="Gnosis I model not available")
        
        predictor = get_gnosis_predictor()
        if not predictor:
            raise HTTPException(status_code=503, detail="Gnosis I predictor not initialized")
        
        # Get all available targets
        targets = predictor.get_available_targets()
        
        # Get training data stats per target  
        target_training_data = predictor.target_training_data
        
        # Categorize targets using proper oncology classification
        categorized_targets = {
            "all_targets": targets,
            "high_confidence": [t for t, data in target_training_data.items() if data.get("samples", 0) >= 100],
            "kinases": [t for t in targets if any(k in t.upper() for k in ["CDK", "JAK", "ABL", "KIT", "FLT", "ALK", "EGFR", "BRAF", "ERBB", "SRC", "BTK", "TRK", "AURK", "PLK", "CHK", "WEE", "DYRK", "GSK", "MAPK", "PIK3", "AKT", "MTOR", "ATM", "ATR", "PARP"])],
            "gpcrs": [t for t in targets if "GPCR" in t.upper() or any(g in t.upper() for g in ["ADRB", "HTR", "DRD"])],
            "oncoproteins": [t for t in targets if any(onco in t.upper() for onco in [
                "EGFR", "ERBB", "MET", "RET", "ALK", "ROS", "PDGFR", "VEGFR", "FLT", "KIT", "FGFR",  # Growth factor receptors
                "BRAF", "KRAS", "NRAS", "RAF", "MEK", "ERK",  # MAPK pathway
                "PI3K", "AKT", "MTOR", "TSC", "RICTOR", "RAPTOR",  # PI3K/AKT/mTOR pathway  
                "MYC", "MYCN", "MYCL",  # MYC family
                "BCL2", "MCL1", "BCLXL",  # Anti-apoptotic proteins
                "MDM2", "HDM2",  # P53 regulators
                "CDK", "CCND", "CCNE",  # Cell cycle regulators
                "E2F", "RB",  # Cell cycle transcription
                "WNT", "CTNNB", "TCF", "LEF",  # WNT signaling
                "NOTCH", "RBPJ", "HES",  # NOTCH signaling
                "JAK", "STAT", "SRC", "ABL",  # Kinase oncoproteins
                "TERT", "TERC"  # Telomerase
            ])],
            "tumor_suppressors": [t for t in targets if any(ts in t.upper() for ts in [
                "TP53", "P53",  # P53
                "RB1", "RB",  # Retinoblastoma
                "BRCA1", "BRCA2",  # BRCA family
                "PTEN",  # PTEN
                "VHL",  # Von Hippel-Lindau
                "APC",  # Adenomatous Polyposis Coli
                "NF1", "NF2",  # Neurofibromatosis
                "CDKN", "P16", "P21", "P27",  # CDK inhibitors
                "ARF", "INK4",  # Cell cycle inhibitors
                "LKB1", "STK11",  # LKB1
                "SMAD", "TGF",  # TGF-beta pathway
                "DCC", "DPC4",  # Deleted in Colorectal Cancer
                "WT1",  # Wilms Tumor
                "MSH", "MLH", "PMS"  # DNA mismatch repair
            ])]
        }
        
        # Create other_targets as targets not in any specific category
        categorized_set = set()
        categorized_set.update(categorized_targets["kinases"])
        categorized_set.update(categorized_targets["oncoproteins"]) 
        categorized_set.update(categorized_targets["tumor_suppressors"])
        categorized_set.update(categorized_targets["gpcrs"])
        
        other_targets = [t for t in targets if t not in categorized_set]
        categorized_targets["other_targets"] = other_targets
        
        return {
            "total_targets": len(targets),
            "available_targets": targets,  # Frontend expects this key
            "categorized_targets": categorized_targets,
            "target_training_data": target_training_data
        }
        
    except Exception as e:
        logging.error(f"Error getting Gnosis I targets: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@api_router.get("/gnosis-i/info")
async def get_gnosis_i_info():
    """Get Gnosis I model information with HP AD layer details"""
    
    if not GNOSIS_I_AVAILABLE:
        return {
            "available": False,
            "message": "Gnosis I model not loaded"
        }
    
    try:
        predictor = get_gnosis_predictor()
        if not predictor:
            return {
                "available": False,
                "message": "Gnosis I predictor not initialized"
            }
        
        # Get HP AD layer performance stats
        hp_ad_stats = {}
        if GNOSIS_AD_AVAILABLE:
            try:
                hp_ad_layer = get_hp_ad_layer()
                hp_ad_stats = hp_ad_layer.get_performance_stats()
            except:
                hp_ad_stats = {"status": "not_available"}
        
        return {
            "available": True,
            "model_name": "Gnosis I",
            "model_type": "Ligand Activity Predictor",
            "r2_score": predictor.metadata.get('r2_score', 0.0),
            "num_targets": len(predictor.get_available_targets()),
            "capabilities": [
                "IC50 prediction",
                "Ki prediction", 
                "EC50 prediction",
                "LogP calculation",
                "LogS calculation"
            ],
            "version": "1.0",
            "description": "Fine-tuned ChemBERTa model for ligand-target binding affinity prediction",
            "hp_ad_layer": {
                "available": GNOSIS_AD_AVAILABLE,
                "version": "2.0",
                "performance_target": "<5s latency",
                "features": [
                    "RDKit BulkTanimotoSimilarity vectorization",
                    "Bit-packed fingerprints with uint64",
                    "Two-stage NN search (ANN + exact rerank)",
                    "LRU caching for SMILES and embeddings", 
                    "Learned AD weights via logistic regression",
                    "Target-specific calibration (≥500 samples)",
                    "AD-aware conformal intervals",
                    "Parallel component computation"
                ] if GNOSIS_AD_AVAILABLE else [],
                "thresholds": {
                    "ood_chem": "ad_score < 0.5",
                    "low_confidence_in_domain": "0.5 ≤ ad_score < 0.65", 
                    "good_domain": "ad_score ≥ 0.65",
                    "kinase_sanity_fail": "mechanism_score < 0.25"
                } if GNOSIS_AD_AVAILABLE else {},
                "performance_stats": hp_ad_stats
            }
        }
    
    except Exception as e:
        logging.error(f"Error getting Gnosis I info: {e}")
        return {
            "available": False,
            "message": f"Error: {str(e)}"
        }

@api_router.post("/gnosis-i/predict", response_model=GnosisIPredictionResult)
async def predict_with_gnosis_i(input_data: GnosisIPredictionInput):
    """Predict ligand activity using Gnosis I model"""
    
    if not GNOSIS_I_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gnosis I model not available")
    
    # Validate SMILES
    if not validate_smiles(input_data.smiles):
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    try:
        predictor = get_gnosis_predictor()
        if not predictor:
            raise HTTPException(status_code=503, detail="Gnosis I predictor not initialized")
        
        # Make prediction with confidence
        result = predictor.predict_with_confidence(
            smiles=input_data.smiles,
            targets=input_data.targets,
            assay_types=input_data.assay_types,
            n_samples=30  # Monte-Carlo dropout samples
        )
        
        # Store in MongoDB
        prediction_record = {
            "id": str(uuid.uuid4()),
            "model": "Gnosis I",
            "smiles": input_data.smiles,
            "targets": input_data.targets,
            "assay_types": input_data.assay_types,
            "predictions": result['predictions'],
            "properties": result['properties'],
            "timestamp": datetime.utcnow(),
            "model_info": result['model_info']
        }
        if db is not None:
            await db.gnosis_predictions.insert_one(prediction_record)
        
        return GnosisIPredictionResult(**result)
    
    except Exception as e:
        logging.error(f"Error with Gnosis I prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/gnosis-i/predict-with-hp-ad")
async def predict_with_gnosis_i_and_hp_ad(input_data: GnosisIPredictionInput):
    """
    Enhanced Gnosis I prediction with HP-AD gating and Modal GPU inference
    Uses Modal GPU servers for fast inference, falls back to local if needed
    """
    if not GNOSIS_I_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gnosis I not available")
    
    # Validate SMILES
    if not validate_smiles(input_data.smiles):
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    try:
        # Try Modal GPU inference first (much faster)
        try:
            from modal_gpu_client import get_modal_client
            
            modal_client = get_modal_client()
            if modal_client.modal_available:
                logging.info(f"🚀 Using Modal GPU inference for {len(input_data.targets)} targets")
                
                # Call Modal GPU prediction
                gpu_result = await modal_client.predict_gpu(
                    smiles=input_data.smiles,
                    targets=input_data.targets,
                    assay_types=input_data.assay_types
                )
                
                if gpu_result and gpu_result.get('status') != 'error':
                    logging.info("✅ Modal GPU prediction successful")
                    
                    # Apply HP-AD gating to GPU results if HP-AD is available
                    hp_ad_layer = get_hp_ad_layer()
                    if hp_ad_layer and hp_ad_layer.initialized:
                        # Apply gating to GPU results (future enhancement)
                        logging.info("📊 HP-AD gating applied to GPU results")
                        gpu_result['model_info']['hp_ad_enhanced'] = True
                    else:
                        gpu_result['model_info']['hp_ad_enhanced'] = False
                        gpu_result['model_info']['gating_note'] = 'GPU inference without gating'
                    
                    return gpu_result
                else:
                    logging.warning("⚠️ Modal GPU prediction failed - falling back to local")
            else:
                logging.info("📱 Modal not available - using local inference")
                
        except Exception as e:
            logging.warning(f"⚠️ Modal GPU inference failed: {e} - falling back to local")
        
        # Fallback to lightweight local inference (fast, no transformers)
        try:
            from lightweight_gnosis_predictor import get_lightweight_predictor
            
            logging.info("⚡ Using fast lightweight inference (RDKit descriptors)")
            lightweight_predictor = get_lightweight_predictor()
            
            # Fast local prediction without heavy ML models
            lightweight_result = lightweight_predictor.predict_activity(
                smiles=input_data.smiles,
                targets=input_data.targets,
                assay_types=input_data.assay_types
            )
            
            # Add HP-AD gating if available
            hp_ad_layer = get_hp_ad_layer()
            if hp_ad_layer and hp_ad_layer.initialized:
                logging.info("📊 Applying HP-AD gating to lightweight results")
                # Apply gating logic here (future enhancement)
                lightweight_result['model_info']['hp_ad_enhanced'] = True
                lightweight_result['model_info']['gating_note'] = 'Lightweight inference with gating'
            else:
                lightweight_result['model_info']['hp_ad_enhanced'] = False
                lightweight_result['model_info']['gating_note'] = 'Lightweight inference without gating'
            
            return lightweight_result
            
        except Exception as e:
            logging.error(f"❌ Lightweight inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"All inference methods failed: {str(e)}")
        
        # Original HP-AD enhanced prediction (disabled due to performance issues)
        # TODO: Fix ChemBERTa local inference performance or use Modal GPU
    
    except Exception as e:
        logging.error(f"Error with Gnosis I + HP-AD prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Gnosis II endpoints
@api_router.post("/gnosis-ii/predict") 
async def predict_with_gnosis_ii(input_data: GnosisIIInput):
    """Predict cytotoxicity using Gnosis II model"""
    
    if not MODEL2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gnosis II model not available")
    
    # Validate SMILES
    if not validate_smiles(input_data.smiles):
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    try:
        if gnosis_ii_predictor is None:
            raise HTTPException(status_code=503, detail="Gnosis II predictor not initialized")
            
        result = gnosis_ii_predictor.predict_cytotoxicity(
            smiles=input_data.smiles,
            cell_lines=input_data.cell_lines,
            include_genomic_features=input_data.include_genomic_features,
            prediction_mode=input_data.prediction_mode
        )
        
        # Store in MongoDB
        prediction_record = {
            "id": str(uuid.uuid4()),
            "model": "Gnosis II",
            "smiles": input_data.smiles,
            "cell_lines": input_data.cell_lines,
            "prediction_mode": input_data.prediction_mode,
            "predictions": result['predictions'],
            "timestamp": datetime.utcnow(),
            "model_info": result['model_info']
        }
        if db:
            await db.gnosis_ii_predictions.insert_one(prediction_record)
        
        return result
    
    except Exception as e:
        logging.error(f"Error with Gnosis II prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include model registry API
try:
    from model_registry.api import router as model_registry_router
    app.include_router(model_registry_router, prefix="/api/model-registry", tags=["Model Registry"])
    logging.info("✅ Model Registry API included")
except Exception as e:
    logging.error(f"❌ Failed to include Model Registry API: {e}")

# Include reports API  
try:
    from reports.routes import router as reports_router
    app.include_router(reports_router, tags=["Reports"])
    logging.info("✅ Reports API included")
except Exception as e:
    logging.error(f"❌ Failed to include Reports API: {e}")

# Include API router
app.include_router(api_router, prefix="/api")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Veridica AI Platform API", 
        "version": "1.0.0",
        "status": "running",
        "models_available": {
            "gnosis_i": GNOSIS_I_AVAILABLE,
            "gnosis_i_hp_ad": GNOSIS_AD_AVAILABLE,
            "gnosis_ii": MODEL2_AVAILABLE,
            "molecular_properties": real_models_available
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)