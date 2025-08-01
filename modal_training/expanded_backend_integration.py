"""
Expanded Backend Integration for Multi-Source Trained Models
Integrates the expanded ChemBERTa and Chemprop models with multiple targets and activity types
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import json
from pathlib import Path
import modal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/api/expanded", tags=["Expanded Models"])

# Expanded target information
EXPANDED_TARGETS = {
    # Oncoproteins
    "EGFR": {"category": "oncoprotein", "full_name": "Epidermal Growth Factor Receptor"},
    "HER2": {"category": "oncoprotein", "full_name": "Human Epidermal Growth Factor Receptor 2"},
    "VEGFR2": {"category": "oncoprotein", "full_name": "Vascular Endothelial Growth Factor Receptor 2"},
    "BRAF": {"category": "oncoprotein", "full_name": "B-Raf Proto-Oncogene"},
    "MET": {"category": "oncoprotein", "full_name": "MET Proto-Oncogene"},
    "CDK4": {"category": "oncoprotein", "full_name": "Cyclin Dependent Kinase 4"},
    "CDK6": {"category": "oncoprotein", "full_name": "Cyclin Dependent Kinase 6"},
    "ALK": {"category": "oncoprotein", "full_name": "Anaplastic Lymphoma Kinase"},
    "MDM2": {"category": "oncoprotein", "full_name": "MDM2 Proto-Oncogene"},
    "PI3KCA": {"category": "oncoprotein", "full_name": "Phosphatidylinositol-4,5-Bisphosphate 3-Kinase Catalytic Subunit Alpha"},
    
    # Tumor Suppressors
    "TP53": {"category": "tumor_suppressor", "full_name": "Tumor Protein P53"},
    "RB1": {"category": "tumor_suppressor", "full_name": "RB Transcriptional Corepressor 1"},
    "PTEN": {"category": "tumor_suppressor", "full_name": "Phosphatase And Tensin Homolog"},
    "APC": {"category": "tumor_suppressor", "full_name": "APC Regulator Of WNT Signaling Pathway"},
    "BRCA1": {"category": "tumor_suppressor", "full_name": "BRCA1 DNA Repair Associated"},
    "BRCA2": {"category": "tumor_suppressor", "full_name": "BRCA2 DNA Repair Associated"},
    "VHL": {"category": "tumor_suppressor", "full_name": "Von Hippel-Lindau Tumor Suppressor"},
    
    # Metastasis Suppressors
    "NDRG1": {"category": "metastasis_suppressor", "full_name": "N-Myc Downstream Regulated 1"},
    "KAI1": {"category": "metastasis_suppressor", "full_name": "CD82 Molecule"},
    "KISS1": {"category": "metastasis_suppressor", "full_name": "KiSS-1 Metastasis Suppressor"},
    "NM23H1": {"category": "metastasis_suppressor", "full_name": "NME/NM23 Nucleoside Diphosphate Kinase 1"},
    "RIKP": {"category": "metastasis_suppressor", "full_name": "Raf Kinase Inhibitor Protein"},
    "CASP8": {"category": "metastasis_suppressor", "full_name": "Caspase 8"}
}

ACTIVITY_TYPES = ["IC50", "EC50", "Ki", "Inhibition", "Activity"]

# Request/Response models
class ExpandedPredictionRequest(BaseModel):
    smiles: str
    targets: Optional[List[str]] = None  # If None, predict all available targets
    activity_types: Optional[List[str]] = None  # If None, predict all available activity types

class ExpandedTargetPrediction(BaseModel):
    target: str
    category: str
    full_name: str
    predictions: Dict[str, Any]  # activity_type -> prediction values

class ExpandedModelResponse(BaseModel):
    model_name: str
    model_type: str
    smiles: str
    predictions: List[ExpandedTargetPrediction]
    metadata: Dict[str, Any]

class ExpandedComparisonResponse(BaseModel):
    smiles: str
    models: Dict[str, ExpandedModelResponse]
    summary: Dict[str, Any]

# Modal function references
def get_expanded_chemberta_function():
    """Get reference to expanded ChemBERTa prediction function"""
    try:
        # This would be the deployed expanded ChemBERTa function
        chemberta_app = modal.App.lookup("expanded-chemberta-inference", create_if_missing=False)
        return chemberta_app.predict_expanded_chemberta
    except Exception as e:
        logger.warning(f"Expanded ChemBERTa function not available: {e}")
        return None

def get_expanded_chemprop_function():
    """Get reference to expanded Chemprop prediction function"""
    try:
        # This would be the deployed expanded Chemprop function
        chemprop_app = modal.App.lookup("expanded-chemprop-inference", create_if_missing=False)
        return chemprop_app.predict_expanded_chemprop
    except Exception as e:
        logger.warning(f"Expanded Chemprop function not available: {e}")
        return None

# API Endpoints

@router.get("/health")
async def expanded_models_health():
    """Health check for expanded models"""
    
    chemberta_func = get_expanded_chemberta_function()
    chemprop_func = get_expanded_chemprop_function()
    
    return {
        "status": "healthy",
        "expanded_models": {
            "chemberta_available": chemberta_func is not None,
            "chemprop_available": chemprop_func is not None
        },
        "target_categories": {
            "oncoproteins": len([t for t, info in EXPANDED_TARGETS.items() if info["category"] == "oncoprotein"]),
            "tumor_suppressors": len([t for t, info in EXPANDED_TARGETS.items() if info["category"] == "tumor_suppressor"]),
            "metastasis_suppressors": len([t for t, info in EXPANDED_TARGETS.items() if info["category"] == "metastasis_suppressor"])
        },
        "total_targets": len(EXPANDED_TARGETS),
        "activity_types": ACTIVITY_TYPES,
        "data_sources": ["ChEMBL", "PubChem", "BindingDB", "DTC"]
    }

@router.get("/targets")
async def get_expanded_targets():
    """Get information about all expanded targets"""
    
    targets_info = []
    for target, info in EXPANDED_TARGETS.items():
        targets_info.append({
            "target": target,
            "category": info["category"],
            "full_name": info["full_name"],
            "available_chemberta": True,  # Would check model availability
            "available_chemprop": True    # Would check model availability
        })
    
    # Group by category
    by_category = {}
    for target_info in targets_info:
        category = target_info["category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(target_info)
    
    return {
        "targets": targets_info,
        "by_category": by_category,
        "total_targets": len(targets_info),
        "activity_types": ACTIVITY_TYPES
    }

@router.post("/predict/chemberta")
async def predict_expanded_chemberta(request: ExpandedPredictionRequest):
    """Predict using expanded ChemBERTa model"""
    
    chemberta_func = get_expanded_chemberta_function()
    if not chemberta_func:
        raise HTTPException(status_code=503, detail="Expanded ChemBERTa model not available")
    
    try:
        # Prepare targets and activity types
        targets = request.targets or list(EXPANDED_TARGETS.keys())
        activity_types = request.activity_types or ["IC50", "EC50", "Ki"]  # Focus on concentration-based
        
        # Call Modal function
        result = chemberta_func.remote(
            smiles=request.smiles,
            targets=targets,
            activity_types=activity_types
        )
        
        # Process results
        predictions = []
        for target in targets:
            if target in EXPANDED_TARGETS and target in result.get("predictions", {}):
                target_predictions = {}
                for activity_type in activity_types:
                    if activity_type in result["predictions"][target]:
                        pred_data = result["predictions"][target][activity_type]
                        target_predictions[activity_type] = {
                            "pIC50": pred_data.get("pic50"),
                            "IC50_nM": pred_data.get("ic50_nm"),
                            "confidence": pred_data.get("confidence"),
                            "activity_class": pred_data.get("activity_class")
                        }
                
                predictions.append(ExpandedTargetPrediction(
                    target=target,
                    category=EXPANDED_TARGETS[target]["category"],
                    full_name=EXPANDED_TARGETS[target]["full_name"],
                    predictions=target_predictions
                ))
        
        return ExpandedModelResponse(
            model_name="Expanded ChemBERTa Multi-Source",
            model_type="transformer",
            smiles=request.smiles,
            predictions=predictions,
            metadata={
                "data_sources": ["ChEMBL", "PubChem", "BindingDB", "DTC"],
                "training_targets": len(targets),
                "activity_types": activity_types,
                "model_version": "expanded_v1"
            }
        )
        
    except Exception as e:
        logger.error(f"Expanded ChemBERTa prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/chemprop")
async def predict_expanded_chemprop(request: ExpandedPredictionRequest):
    """Predict using expanded Chemprop model"""
    
    chemprop_func = get_expanded_chemprop_function()
    if not chemprop_func:
        raise HTTPException(status_code=503, detail="Expanded Chemprop model not available")
    
    try:
        # Prepare targets and activity types
        targets = request.targets or list(EXPANDED_TARGETS.keys())
        activity_types = request.activity_types or ["IC50", "EC50", "Ki"]  # Focus on concentration-based
        
        # Call Modal function
        result = chemprop_func.remote(
            smiles=request.smiles,
            targets=targets,
            activity_types=activity_types
        )
        
        # Process results
        predictions = []
        for target in targets:
            if target in EXPANDED_TARGETS and target in result.get("predictions", {}):
                target_predictions = {}
                for activity_type in activity_types:
                    if activity_type in result["predictions"][target]:
                        pred_data = result["predictions"][target][activity_type]
                        target_predictions[activity_type] = {
                            "pIC50": pred_data.get("pic50"),
                            "IC50_nM": pred_data.get("ic50_nm"),
                            "confidence": pred_data.get("confidence"),
                            "activity_class": pred_data.get("activity_class")
                        }
                
                predictions.append(ExpandedTargetPrediction(
                    target=target,
                    category=EXPANDED_TARGETS[target]["category"],
                    full_name=EXPANDED_TARGETS[target]["full_name"],
                    predictions=target_predictions
                ))
        
        return ExpandedModelResponse(
            model_name="Expanded Chemprop Multi-Source GNN",
            model_type="graph_neural_network", 
            smiles=request.smiles,
            predictions=predictions,
            metadata={
                "data_sources": ["ChEMBL", "PubChem", "BindingDB", "DTC"],
                "training_targets": len(targets),
                "activity_types": activity_types,
                "model_version": "expanded_v1"
            }
        )
        
    except Exception as e:
        logger.error(f"Expanded Chemprop prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/compare")
async def compare_expanded_models(request: ExpandedPredictionRequest):
    """Compare predictions from both expanded models"""
    
    try:
        # Get predictions from both models
        models = {}
        
        # ChemBERTa predictions
        try:
            chemberta_response = await predict_expanded_chemberta(request)
            models["chemberta"] = chemberta_response
        except HTTPException as e:
            logger.warning(f"ChemBERTa prediction failed: {e.detail}")
            models["chemberta"] = None
        
        # Chemprop predictions
        try:
            chemprop_response = await predict_expanded_chemprop(request)
            models["chemprop"] = chemprop_response
        except HTTPException as e:
            logger.warning(f"Chemprop prediction failed: {e.detail}")
            models["chemprop"] = None
        
        # Calculate summary statistics
        summary = {
            "models_available": len([m for m in models.values() if m is not None]),
            "total_targets": len(request.targets or EXPANDED_TARGETS.keys()),
            "activity_types": request.activity_types or ["IC50", "EC50", "Ki"]
        }
        
        # Category-wise analysis
        if models["chemberta"] and models["chemprop"]:
            category_comparison = {}
            for category in ["oncoprotein", "tumor_suppressor", "metastasis_suppressor"]:
                category_targets = [t for t, info in EXPANDED_TARGETS.items() if info["category"] == category]
                
                chemberta_preds = []
                chemprop_preds = []
                
                for pred in models["chemberta"].predictions:
                    if pred.target in category_targets and "IC50" in pred.predictions:
                        chemberta_preds.append(pred.predictions["IC50"].get("pIC50"))
                
                for pred in models["chemprop"].predictions:
                    if pred.target in category_targets and "IC50" in pred.predictions:
                        chemprop_preds.append(pred.predictions["IC50"].get("pIC50"))
                
                if chemberta_preds and chemprop_preds:
                    category_comparison[category] = {
                        "chemberta_mean": sum(p for p in chemberta_preds if p) / len([p for p in chemberta_preds if p]),
                        "chemprop_mean": sum(p for p in chemprop_preds if p) / len([p for p in chemprop_preds if p]),
                        "targets_compared": min(len(chemberta_preds), len(chemprop_preds))
                    }
            
            summary["category_comparison"] = category_comparison
        
        return ExpandedComparisonResponse(
            smiles=request.smiles,
            models=models,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@router.get("/stats/performance")
async def get_expanded_model_stats():
    """Get performance statistics for expanded models"""
    
    # This would load actual model performance metrics from training results
    # For now, return placeholder structure
    
    return {
        "chemberta": {
            "overall_r2": 0.65,  # Placeholder
            "category_performance": {
                "oncoprotein": 0.72,
                "tumor_suppressor": 0.58,
                "metastasis_suppressor": 0.51
            },
            "training_info": {
                "epochs": 30,
                "total_compounds": 25000,  # Placeholder
                "data_sources": ["ChEMBL", "PubChem", "BindingDB", "DTC"]
            }
        },
        "chemprop": {
            "overall_r2": 0.61,  # Placeholder
            "category_performance": {
                "oncoprotein": 0.69,
                "tumor_suppressor": 0.54,
                "metastasis_suppressor": 0.48
            },
            "training_info": {
                "epochs": 40,
                "total_compounds": 25000,  # Placeholder
                "data_sources": ["ChEMBL", "PubChem", "BindingDB", "DTC"]
            }
        },
        "comparison": {
            "chemberta_better_targets": 12,
            "chemprop_better_targets": 11,
            "similar_targets": 0
        }
    }

# Export router
expanded_router = router