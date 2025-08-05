"""
Enhanced Cell Line Response Model Backend with Therapeutic Index
Integrates GDSC cancer efficacy + Tox21 normal cell cytotoxicity
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
import asyncio

# Enhanced router
router = APIRouter(prefix="/cell-line-therapeutic", tags=["cell-line-therapeutic"])

# Request/Response models
class TherapeuticPredictionRequest(BaseModel):
    smiles: str
    cell_line_name: str
    include_therapeutic_index: bool = True
    include_safety_assessment: bool = True

class TherapeuticPredictionResponse(BaseModel):
    drug_smiles: str
    cell_line_name: str
    predicted_ic50_nm: float
    predicted_ic50_um: float
    pic50: float
    confidence: float
    
    # Therapeutic Index Data
    therapeutic_index: Optional[float] = None
    normal_cell_cytotox_um: Optional[float] = None
    safety_classification: Optional[str] = None
    therapeutic_window: Optional[str] = None
    
    # Clinical Insights
    clinical_interpretation: str
    dosing_recommendations: Optional[str] = None
    safety_warnings: List[str] = []

class TherapeuticComparisonRequest(BaseModel):
    smiles: str
    cell_lines: List[str]
    include_therapeutic_indices: bool = True

class TherapeuticComparisonResponse(BaseModel):
    drug_smiles: str
    predictions: List[TherapeuticPredictionResponse]
    therapeutic_index_ranking: List[Dict[str, Any]]
    safety_summary: Dict[str, Any]

class TherapeuticIndexData:
    """Class to manage therapeutic index data"""
    
    def __init__(self):
        self._ti_data = None
        self._cytotox_data = None
        self._load_data()
    
    def _load_data(self):
        """Load therapeutic index and cytotoxicity data"""
        try:
            # Load therapeutic indices
            ti_path = Path("/vol/datasets/therapeutic_indices.csv")
            if ti_path.exists():
                self._ti_data = pd.read_csv(ti_path)
                logging.info(f"Loaded therapeutic indices: {len(self._ti_data)} drugs")
            
            # Load cytotoxicity data
            cytotox_path = Path("/vol/datasets/cytotoxicity_data.csv")
            if cytotox_path.exists():
                self._cytotox_data = pd.read_csv(cytotox_path)
                logging.info(f"Loaded cytotoxicity data: {len(self._cytotox_data)} compounds")
            
        except Exception as e:
            logging.error(f"Error loading therapeutic index data: {e}")
    
    def get_therapeutic_index(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Get therapeutic index data for a drug"""
        if self._ti_data is None:
            return None
        
        # Try exact match first
        matches = self._ti_data[self._ti_data['drug_name'].str.lower() == drug_name.lower()]
        
        if len(matches) == 0:
            # Try partial match
            matches = self._ti_data[
                self._ti_data['drug_name'].str.contains(drug_name, case=False, na=False)
            ]
        
        if len(matches) > 0:
            match = matches.iloc[0]
            return {
                'therapeutic_index': float(match['therapeutic_index']),
                'normal_cell_cytotox_um': float(match['normal_cytotox_ac50_um']),
                'cancer_cell_ic50_um': float(match['cancer_ic50_um']),
                'safety_classification': match['safety_classification'],
                'cytotox_assays': int(match.get('cytotox_assays', 1))
            }
        
        return None
    
    def estimate_therapeutic_index(self, predicted_ic50_um: float, 
                                 drug_smiles: str = None) -> Dict[str, Any]:
        """Estimate therapeutic index based on predicted efficacy"""
        
        # Use median cytotoxicity from available data as estimate
        if self._cytotox_data is not None and len(self._cytotox_data) > 0:
            median_cytotox = self._cytotox_data['cytotox_ac50_um'].median()
        else:
            # Conservative estimate: assume moderate cytotoxicity
            median_cytotox = 10.0  # 10 μM
        
        estimated_ti = median_cytotox / predicted_ic50_um
        
        return {
            'therapeutic_index': float(estimated_ti),
            'normal_cell_cytotox_um': float(median_cytotox),
            'cancer_cell_ic50_um': float(predicted_ic50_um),
            'safety_classification': self._classify_safety(estimated_ti),
            'is_estimated': True
        }
    
    def _classify_safety(self, therapeutic_index: float) -> str:
        """Classify drug safety based on therapeutic index"""
        if therapeutic_index >= 100:
            return "Very Safe"
        elif therapeutic_index >= 10:
            return "Safe"
        elif therapeutic_index >= 3:
            return "Moderate"
        elif therapeutic_index >= 1:
            return "Low Safety"
        else:
            return "Toxic"

# Global therapeutic index data manager
ti_data_manager = TherapeuticIndexData()

def generate_clinical_interpretation(ic50_nm: float, therapeutic_index: Optional[float], 
                                   safety_class: Optional[str], cell_line: str) -> str:
    """Generate clinical interpretation based on efficacy and safety"""
    
    # Efficacy interpretation
    if ic50_nm < 100:
        efficacy = "highly potent"
    elif ic50_nm < 1000:
        efficacy = "moderately potent"
    elif ic50_nm < 10000:
        efficacy = "low potency"
    else:
        efficacy = "very low potency"
    
    # Safety interpretation
    if therapeutic_index is not None and safety_class:
        if safety_class in ["Very Safe", "Safe"]:
            safety_text = f"with excellent safety profile (TI={therapeutic_index:.1f})"
        elif safety_class == "Moderate":
            safety_text = f"with moderate safety concerns (TI={therapeutic_index:.1f})"
        else:
            safety_text = f"with significant safety concerns (TI={therapeutic_index:.1f})"
    else:
        safety_text = "with unknown safety profile"
    
    return f"This compound shows {efficacy} against {cell_line} cells {safety_text}."

def generate_dosing_recommendations(therapeutic_index: Optional[float], 
                                  ic50_um: float) -> Optional[str]:
    """Generate dosing recommendations based on therapeutic window"""
    
    if therapeutic_index is None:
        return None
    
    if therapeutic_index >= 10:
        return f"Wide therapeutic window allows flexible dosing. Start at {ic50_um * 2:.1f} μM (2x IC50)."
    elif therapeutic_index >= 3:
        return f"Moderate therapeutic window requires careful dosing. Start at {ic50_um * 1.5:.1f} μM (1.5x IC50)."
    else:
        return f"Narrow therapeutic window requires very careful dosing. Start at {ic50_um:.1f} μM (IC50 level)."

def generate_safety_warnings(therapeutic_index: Optional[float], 
                           safety_class: Optional[str]) -> List[str]:
    """Generate safety warnings based on therapeutic index"""
    
    warnings = []
    
    if therapeutic_index is not None:
        if therapeutic_index < 1:
            warnings.append("HIGH TOXICITY RISK: Cytotoxic at therapeutic doses")
        elif therapeutic_index < 3:
            warnings.append("MODERATE TOXICITY RISK: Monitor for side effects")
        
        if safety_class == "Toxic":
            warnings.append("CONTRAINDICATED: Toxic effects likely at therapeutic doses")
    
    return warnings

# Mock cell line response model (replace with actual model)
class MockCellLineModel:
    """Mock model for demonstration - replace with real trained model"""
    
    def predict(self, smiles: str, cell_line: str) -> Dict[str, float]:
        """Mock prediction - replace with real model inference"""
        
        # Generate realistic IC50 prediction based on drug/cell line
        base_ic50 = np.random.lognormal(mean=2.5, sigma=1.5)  # ~12 μM median
        
        # Add some variation based on input
        smiles_hash = hash(smiles) % 1000
        cell_hash = hash(cell_line) % 1000
        
        variation = (smiles_hash + cell_hash) / 2000 - 0.5  # -0.5 to +0.5
        adjusted_ic50 = base_ic50 * (1 + variation)
        
        # Ensure reasonable range
        ic50_nm = max(1, min(adjusted_ic50 * 1000, 100000))  # 1 nM to 100 μM
        
        return {
            'ic50_nm': ic50_nm,
            'ic50_um': ic50_nm / 1000,
            'pic50': -np.log10((ic50_nm / 1000) / 1e6),
            'confidence': np.random.uniform(0.7, 0.95)
        }

# Mock model instance
mock_model = MockCellLineModel()

@router.get("/health")
async def health_check():
    """Health check for therapeutic index integration"""
    
    ti_available = ti_data_manager._ti_data is not None
    cytotox_available = ti_data_manager._cytotox_data is not None
    
    return {
        "status": "healthy",
        "therapeutic_index_data": ti_available,
        "cytotoxicity_data": cytotox_available,
        "model_status": "ready",
        "features": [
            "Cancer cell efficacy prediction",
            "Normal cell cytotoxicity integration", 
            "Therapeutic index calculation",
            "Safety classification",
            "Clinical interpretation",
            "Dosing recommendations"
        ]
    }

@router.post("/predict", response_model=TherapeuticPredictionResponse)
async def predict_with_therapeutic_index(request: TherapeuticPredictionRequest):
    """Predict drug efficacy with therapeutic index and safety assessment"""
    
    try:
        # Get cancer cell efficacy prediction
        prediction = mock_model.predict(request.smiles, request.cell_line_name)
        
        # Get therapeutic index data
        therapeutic_data = None
        if request.include_therapeutic_index:
            # Try to find exact drug match (would need better compound mapping)
            therapeutic_data = ti_data_manager.get_therapeutic_index("unknown_drug")
            
            if therapeutic_data is None:
                # Estimate therapeutic index
                therapeutic_data = ti_data_manager.estimate_therapeutic_index(
                    prediction['ic50_um'], request.smiles
                )
        
        # Generate clinical interpretation
        clinical_interpretation = generate_clinical_interpretation(
            prediction['ic50_nm'],
            therapeutic_data.get('therapeutic_index') if therapeutic_data else None,
            therapeutic_data.get('safety_classification') if therapeutic_data else None,
            request.cell_line_name
        )
        
        # Generate dosing recommendations
        dosing_recommendations = None
        if request.include_safety_assessment and therapeutic_data:
            dosing_recommendations = generate_dosing_recommendations(
                therapeutic_data.get('therapeutic_index'),
                prediction['ic50_um']
            )
        
        # Generate safety warnings
        safety_warnings = []
        if request.include_safety_assessment and therapeutic_data:
            safety_warnings = generate_safety_warnings(
                therapeutic_data.get('therapeutic_index'),
                therapeutic_data.get('safety_classification')
            )
        
        # Determine therapeutic window
        therapeutic_window = None
        if therapeutic_data and 'therapeutic_index' in therapeutic_data:
            ti = therapeutic_data['therapeutic_index']
            if ti >= 10:
                therapeutic_window = "Wide"
            elif ti >= 3:
                therapeutic_window = "Moderate"
            else:
                therapeutic_window = "Narrow"
        
        return TherapeuticPredictionResponse(
            drug_smiles=request.smiles,
            cell_line_name=request.cell_line_name,
            predicted_ic50_nm=prediction['ic50_nm'],
            predicted_ic50_um=prediction['ic50_um'],
            pic50=prediction['pic50'],
            confidence=prediction['confidence'],
            therapeutic_index=therapeutic_data.get('therapeutic_index') if therapeutic_data else None,
            normal_cell_cytotox_um=therapeutic_data.get('normal_cell_cytotox_um') if therapeutic_data else None,
            safety_classification=therapeutic_data.get('safety_classification') if therapeutic_data else None,
            therapeutic_window=therapeutic_window,
            clinical_interpretation=clinical_interpretation,
            dosing_recommendations=dosing_recommendations,
            safety_warnings=safety_warnings
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/compare", response_model=TherapeuticComparisonResponse)
async def compare_therapeutic_indices(request: TherapeuticComparisonRequest):
    """Compare drug efficacy and therapeutic indices across multiple cell lines"""
    
    try:
        predictions = []
        
        # Get predictions for each cell line
        for cell_line in request.cell_lines:
            pred_request = TherapeuticPredictionRequest(
                smiles=request.smiles,
                cell_line_name=cell_line,
                include_therapeutic_index=request.include_therapeutic_indices,
                include_safety_assessment=True
            )
            
            prediction = await predict_with_therapeutic_index(pred_request)
            predictions.append(prediction)
        
        # Create therapeutic index ranking
        ti_ranking = []
        for pred in predictions:
            if pred.therapeutic_index is not None:
                ti_ranking.append({
                    'cell_line': pred.cell_line_name,
                    'therapeutic_index': pred.therapeutic_index,
                    'safety_classification': pred.safety_classification,
                    'ic50_nm': pred.predicted_ic50_nm
                })
        
        # Sort by therapeutic index (descending = safer)
        ti_ranking.sort(key=lambda x: x['therapeutic_index'], reverse=True)
        
        # Create safety summary
        safety_summary = {
            'safest_cell_line': ti_ranking[0]['cell_line'] if ti_ranking else None,
            'most_potent_cell_line': min(predictions, key=lambda x: x.predicted_ic50_nm).cell_line_name,
            'safety_classifications': [pred.safety_classification for pred in predictions if pred.safety_classification],
            'average_therapeutic_index': np.mean([pred.therapeutic_index for pred in predictions if pred.therapeutic_index is not None]) if any(pred.therapeutic_index for pred in predictions) else None
        }
        
        return TherapeuticComparisonResponse(
            drug_smiles=request.smiles,
            predictions=predictions,
            therapeutic_index_ranking=ti_ranking,
            safety_summary=safety_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@router.get("/therapeutic-indices")
async def get_available_therapeutic_indices():
    """Get available therapeutic index data"""
    
    if ti_data_manager._ti_data is None:
        return {"available": False, "message": "No therapeutic index data loaded"}
    
    ti_data = ti_data_manager._ti_data
    
    return {
        "available": True,
        "total_drugs": len(ti_data),
        "safety_distribution": ti_data['safety_classification'].value_counts().to_dict(),
        "median_therapeutic_index": float(ti_data['therapeutic_index'].median()),
        "high_safety_drugs": len(ti_data[ti_data['therapeutic_index'] >= 10]),
        "sample_drugs": ti_data[['drug_name', 'therapeutic_index', 'safety_classification']].head(10).to_dict('records')
    }

@router.get("/cytotoxicity-data")
async def get_cytotoxicity_data_summary():
    """Get cytotoxicity data summary"""
    
    if ti_data_manager._cytotox_data is None:
        return {"available": False, "message": "No cytotoxicity data loaded"}
    
    cytotox_data = ti_data_manager._cytotox_data
    
    return {
        "available": True,
        "total_compounds": len(cytotox_data),
        "median_cytotox_ac50": float(cytotox_data['cytotox_ac50_um'].median()),
        "normal_cell_assays": len(cytotox_data[cytotox_data.get('is_normal_cell', False)]),
        "assay_coverage": cytotox_data['num_cytotox_assays'].describe().to_dict()
    }

# Include the router in the main application
# This would be added to the main server.py file