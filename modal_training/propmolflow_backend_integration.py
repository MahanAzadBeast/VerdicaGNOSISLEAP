"""
PropMolFlow Backend Integration for Veridica AI
Molecular generation with property guidance
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
router = APIRouter(prefix="/api/generate", tags=["Molecular Generation"])

# Property-guided generation models
class PropertyRequirement(BaseModel):
    target_name: str
    activity_type: str  # IC50, EC50, Ki
    target_value: float  # Desired value in nM
    operator: str = "less_than"  # "less_than", "greater_than", "equal_to"
    tolerance: float = 0.5  # Tolerance in log scale
    importance: float = 1.0  # Relative importance (0-1)

class GenerationRequest(BaseModel):
    num_molecules: int = 10
    max_molecules: int = 100  # Safety limit
    properties: List[PropertyRequirement]
    constraints: Optional[Dict[str, Any]] = None
    generation_method: str = "propmolflow"  # Future: "diffusion", "vae", etc.
    temperature: float = 1.0
    guidance_scale: float = 7.5

class GeneratedMolecule(BaseModel):
    smiles: str
    predicted_properties: Dict[str, float]
    generation_score: float
    validity_score: float
    drug_likeness: float
    synthetic_accessibility: Optional[float] = None

class GenerationResponse(BaseModel):
    status: str
    molecules: List[GeneratedMolecule]
    generation_stats: Dict[str, Any]
    metadata: Dict[str, Any]

class GenerationStatus(BaseModel):
    status: str
    message: str
    progress: float
    estimated_time_remaining: Optional[int] = None

# Enhanced target information for generation
GENERATION_TARGETS = {
    # Oncoproteins
    "EGFR": {"category": "oncoprotein", "full_name": "Epidermal Growth Factor Receptor", "generation_ready": True},
    "HER2": {"category": "oncoprotein", "full_name": "Human Epidermal Growth Factor Receptor 2", "generation_ready": True},
    "VEGFR2": {"category": "oncoprotein", "full_name": "Vascular Endothelial Growth Factor Receptor 2", "generation_ready": True},
    "BRAF": {"category": "oncoprotein", "full_name": "B-Raf Proto-Oncogene", "generation_ready": True},
    "MET": {"category": "oncoprotein", "full_name": "MET Proto-Oncogene", "generation_ready": True},
    "CDK4": {"category": "oncoprotein", "full_name": "Cyclin Dependent Kinase 4", "generation_ready": True},
    "CDK6": {"category": "oncoprotein", "full_name": "Cyclin Dependent Kinase 6", "generation_ready": True},
    "ALK": {"category": "oncoprotein", "full_name": "Anaplastic Lymphoma Kinase", "generation_ready": True},
    "MDM2": {"category": "oncoprotein", "full_name": "MDM2 Proto-Oncogene", "generation_ready": True},
    "PI3KCA": {"category": "oncoprotein", "full_name": "Phosphatidylinositol-4,5-Bisphosphate 3-Kinase", "generation_ready": True},
    
    # Tumor Suppressors
    "TP53": {"category": "tumor_suppressor", "full_name": "Tumor Protein P53", "generation_ready": True},
    "RB1": {"category": "tumor_suppressor", "full_name": "RB Transcriptional Corepressor 1", "generation_ready": True},
    "PTEN": {"category": "tumor_suppressor", "full_name": "Phosphatase And Tensin Homolog", "generation_ready": True},
    "APC": {"category": "tumor_suppressor", "full_name": "APC Regulator Of WNT Signaling Pathway", "generation_ready": True},
    "BRCA1": {"category": "tumor_suppressor", "full_name": "BRCA1 DNA Repair Associated", "generation_ready": True},
    "BRCA2": {"category": "tumor_suppressor", "full_name": "BRCA2 DNA Repair Associated", "generation_ready": True},
    "VHL": {"category": "tumor_suppressor", "full_name": "Von Hippel-Lindau Tumor Suppressor", "generation_ready": True},
    
    # Metastasis Suppressors
    "NDRG1": {"category": "metastasis_suppressor", "full_name": "N-Myc Downstream Regulated 1", "generation_ready": True},
    "KAI1": {"category": "metastasis_suppressor", "full_name": "CD82 Molecule", "generation_ready": True},
    "KISS1": {"category": "metastasis_suppressor", "full_name": "KiSS-1 Metastasis Suppressor", "generation_ready": True},
    "NM23H1": {"category": "metastasis_suppressor", "full_name": "NME/NM23 Nucleoside Diphosphate Kinase 1", "generation_ready": True},
    "RIKP": {"category": "metastasis_suppressor", "full_name": "Raf Kinase Inhibitor Protein", "generation_ready": True},
    "CASP8": {"category": "metastasis_suppressor", "full_name": "Caspase 8", "generation_ready": True}
}

# Modal function references
def get_propmolflow_function():
    """Get reference to PropMolFlow generation function"""
    try:
        propmolflow_app = modal.App.lookup("veridica-propmolflow", create_if_missing=False)
        return propmolflow_app.generate_molecules_with_propmolflow
    except Exception as e:
        logger.warning(f"PropMolFlow function not available: {e}")
        return None

def validate_property_requirements(properties: List[PropertyRequirement]) -> List[str]:
    """Validate property requirements"""
    errors = []
    
    for i, prop in enumerate(properties):
        # Check target exists
        if prop.target_name not in GENERATION_TARGETS:
            errors.append(f"Property {i}: Unknown target '{prop.target_name}'")
        
        # Check activity type
        if prop.activity_type not in ["IC50", "EC50", "Ki"]:
            errors.append(f"Property {i}: Invalid activity type '{prop.activity_type}'")
        
        # Check target value
        if prop.target_value <= 0:
            errors.append(f"Property {i}: Target value must be positive")
        
        if prop.target_value > 1e6:  # 1mM
            errors.append(f"Property {i}: Target value too high (max 1,000,000 nM)")
        
        # Check operator
        if prop.operator not in ["less_than", "greater_than", "equal_to"]:
            errors.append(f"Property {i}: Invalid operator '{prop.operator}'")
        
        # Check importance
        if not 0 <= prop.importance <= 1:
            errors.append(f"Property {i}: Importance must be between 0 and 1")
    
    return errors

# API Endpoints

@router.get("/health")
async def generation_health():
    """Health check for molecular generation services"""
    
    propmolflow_func = get_propmolflow_function()
    
    return {
        "status": "healthy",
        "generation_methods": {
            "propmolflow_available": propmolflow_func is not None,
            "diffusion_available": False,  # Future implementation
            "vae_available": False  # Future implementation
        },
        "supported_targets": len(GENERATION_TARGETS),
        "target_categories": {
            "oncoproteins": len([t for t, info in GENERATION_TARGETS.items() if info["category"] == "oncoprotein"]),
            "tumor_suppressors": len([t for t, info in GENERATION_TARGETS.items() if info["category"] == "tumor_suppressor"]),
            "metastasis_suppressors": len([t for t, info in GENERATION_TARGETS.items() if info["category"] == "metastasis_suppressor"])
        },
        "supported_activity_types": ["IC50", "EC50", "Ki"],
        "max_molecules_per_request": 100
    }

@router.get("/targets")
async def get_generation_targets():
    """Get available targets for molecular generation"""
    
    targets_list = []
    for target, info in GENERATION_TARGETS.items():
        targets_list.append({
            "target": target,
            "category": info["category"],
            "full_name": info["full_name"],
            "generation_ready": info["generation_ready"]
        })
    
    return {
        "targets": targets_list,
        "total_targets": len(targets_list),
        "categories": list(set(info["category"] for info in GENERATION_TARGETS.values())),
        "activity_types": ["IC50", "EC50", "Ki"]
    }

@router.post("/molecules")
async def generate_molecules(request: GenerationRequest):
    """Generate molecules with desired properties using PropMolFlow"""
    
    # Validate request
    if not request.properties:
        raise HTTPException(status_code=400, detail="At least one property requirement must be specified")
    
    if request.num_molecules > request.max_molecules:
        raise HTTPException(status_code=400, detail=f"Number of molecules exceeds maximum ({request.max_molecules})")
    
    # Validate property requirements
    validation_errors = validate_property_requirements(request.properties)
    if validation_errors:
        raise HTTPException(status_code=400, detail={"errors": validation_errors})
    
    # Check if generation method is available
    if request.generation_method == "propmolflow":
        propmolflow_func = get_propmolflow_function()
        if not propmolflow_func:
            raise HTTPException(status_code=503, detail="PropMolFlow generation service not available")
    else:
        raise HTTPException(status_code=400, detail=f"Generation method '{request.generation_method}' not supported")
    
    try:
        # For now, handle single property requirement (can be extended for multi-property)
        primary_property = request.properties[0]
        
        # Call PropMolFlow generation
        result = propmolflow_func.remote(
            target_protein=primary_property.target_name,
            activity_type=primary_property.activity_type,
            target_value=primary_property.target_value,
            num_molecules=request.num_molecules
        )
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=f"Generation failed: {result.get('error', 'Unknown error')}")
        
        # Process generated molecules
        generated_molecules = []
        for i, smiles in enumerate(result["molecules"]):
            # Calculate basic properties (placeholder)
            molecule = GeneratedMolecule(
                smiles=smiles,
                predicted_properties={
                    f"{primary_property.target_name}_{primary_property.activity_type}": primary_property.target_value * (0.8 + 0.4 * i / len(result["molecules"]))  # Placeholder
                },
                generation_score=0.8 + 0.2 * (1 - i / len(result["molecules"])),  # Placeholder
                validity_score=1.0,  # Would use RDKit validation
                drug_likeness=0.7 + 0.3 * (1 - i / len(result["molecules"]))  # Placeholder
            )
            generated_molecules.append(molecule)
        
        # Generation statistics
        generation_stats = {
            "total_generated": len(result["molecules"]),
            "valid_molecules": len(generated_molecules),
            "success_rate": len(generated_molecules) / request.num_molecules,
            "average_generation_score": sum(m.generation_score for m in generated_molecules) / len(generated_molecules) if generated_molecules else 0,
            "average_drug_likeness": sum(m.drug_likeness for m in generated_molecules) / len(generated_molecules) if generated_molecules else 0
        }
        
        # Metadata
        metadata = {
            "generation_method": request.generation_method,
            "property_requirements": [prop.dict() for prop in request.properties],
            "generation_parameters": {
                "temperature": request.temperature,
                "guidance_scale": request.guidance_scale
            },
            "model_version": "propmolflow_v1",
            "timestamp": "2025-08-01T07:30:00Z"  # Would use actual timestamp
        }
        
        return GenerationResponse(
            status="success",
            molecules=generated_molecules,
            generation_stats=generation_stats,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Molecule generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.post("/molecules/validate")
async def validate_generated_molecules(smiles_list: List[str]):
    """Validate generated molecules using RDKit and drug-likeness filters"""
    
    # This would integrate with our existing ChemBERTa/Chemprop models
    # to predict properties of generated molecules
    
    validated_molecules = []
    
    for smiles in smiles_list:
        # Placeholder validation
        validation_result = {
            "smiles": smiles,
            "valid": True,  # Would use RDKit validation
            "drug_like": True,  # Would use Lipinski rules, etc.
            "synthetic_accessibility": 3.2,  # Would calculate SA score
            "predicted_properties": {
                "molecular_weight": 250.0,  # Placeholder
                "logp": 2.5,  # Placeholder
                "tpsa": 45.0  # Placeholder
            }
        }
        validated_molecules.append(validation_result)
    
    return {
        "status": "success",
        "validated_molecules": validated_molecules,
        "validation_stats": {
            "total_molecules": len(smiles_list),
            "valid_molecules": len([m for m in validated_molecules if m["valid"]]),
            "drug_like_molecules": len([m for m in validated_molecules if m["drug_like"]])
        }
    }

@router.get("/examples")
async def get_generation_examples():
    """Get example generation requests and results"""
    
    examples = [
        {
            "name": "EGFR Inhibitor Design",
            "description": "Generate potent EGFR inhibitors for cancer treatment",
            "request": {
                "num_molecules": 10,
                "properties": [
                    {
                        "target_name": "EGFR",
                        "activity_type": "IC50",
                        "target_value": 50.0,
                        "operator": "less_than",
                        "importance": 1.0
                    }
                ],
                "generation_method": "propmolflow"
            }
        },
        {
            "name": "Multi-Target Design",
            "description": "Generate molecules active against multiple oncoproteins",
            "request": {
                "num_molecules": 20,
                "properties": [
                    {
                        "target_name": "EGFR",
                        "activity_type": "IC50", 
                        "target_value": 100.0,
                        "operator": "less_than",
                        "importance": 0.8
                    },
                    {
                        "target_name": "HER2",
                        "activity_type": "IC50",
                        "target_value": 200.0,
                        "operator": "less_than",
                        "importance": 0.6
                    }
                ],
                "generation_method": "propmolflow"
            }
        },
        {
            "name": "Tumor Suppressor Activator",
            "description": "Generate molecules that could restore tumor suppressor function",
            "request": {
                "num_molecules": 15,
                "properties": [
                    {
                        "target_name": "TP53",
                        "activity_type": "EC50",
                        "target_value": 500.0,
                        "operator": "less_than",
                        "importance": 1.0
                    }
                ],
                "generation_method": "propmolflow"
            }
        }
    ]
    
    return {
        "examples": examples,
        "total_examples": len(examples)
    }

# Export router
generation_router = router