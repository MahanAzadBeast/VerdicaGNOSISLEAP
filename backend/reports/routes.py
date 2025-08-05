"""
FastAPI routes for PDF report generation
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from typing import Dict, Any
import logging
from datetime import datetime
import re

from .pdf_generator import generate_prediction_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reports", tags=["reports"])

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for download"""
    # Remove non-alphanumeric characters except dots, hyphens, underscores
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    # Limit length
    return sanitized[:50]

@router.post("/export-pdf")
async def export_prediction_pdf(predictions_data: Dict[str, Any]):
    """
    Export prediction results as a branded PDF report
    
    Expects the same data structure as the frontend receives from 
    /api/gnosis-i/predict endpoint
    """
    try:
        # Generate PDF
        pdf_bytes = generate_prediction_pdf(predictions_data)
        
        # Create filename
        smiles = predictions_data.get('smiles', 'compound')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        compound_name = f"compound_{smiles[:8]}..." if len(smiles) > 8 else smiles
        filename = sanitize_filename(f"veridica_report_{compound_name}_{timestamp}.pdf")
        
        # Return PDF as response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=\"{filename}\"",
                "Cache-Control": "no-cache",
                "X-Report-Size": str(len(pdf_bytes))
            }
        )
        
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"PDF generation failed: {str(e)}"
        )

@router.get("/health")
async def reports_health():
    """Health check for reports service"""
    try:
        # Test PDF generation with minimal data
        test_data = {
            "smiles": "CCO",
            "properties": {"LogP": 1.2, "LogS": -0.3},
            "predictions": {
                "TEST_TARGET": {
                    "IC50": {"pActivity": 5.5, "activity_uM": 3.16, "confidence": 0.85},
                    "selectivity_ratio": 12.4
                }
            },
            "model_info": {"name": "Gnosis I", "r2_score": 0.628}
        }
        
        pdf_bytes = generate_prediction_pdf(test_data)
        
        return {
            "status": "healthy",
            "pdf_generation": "available",
            "test_pdf_size": len(pdf_bytes)
        }
    except Exception as e:
        return {
            "status": "error",
            "pdf_generation": "failed",
            "error": str(e)
        }