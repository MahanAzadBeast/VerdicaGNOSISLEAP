"""
Pydantic schemas for PDF report generation
"""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class PredictionRow(BaseModel):
    protein: str
    display_ic50: str  # e.g., "12.3 μM" or "> 100 μM"
    pic50: str         # e.g., "5.91" or "—"
    heat_bg: str       # HSL color string
    opacity: float     # 0.4-1.0
    selectivity_label: str  # e.g., "12.4×" or "Panel not available"
    selectivity_class: str  # CSS class: selective, moderate, non-selective, panel-na
    selectivity_icon: str   # ✔︎, 🟡, ❌, —
    confidence_percent: int # 0-100
    assay: str         # IC50, Ki, EC50
    assay_bg: str      # Background color for assay type
    assay_color: str   # Text color for assay type

class PredictionBatch(BaseModel):
    compound_name: str
    smiles: str
    timestamp: str
    model_name: str
    r2: float
    total_predictions: int
    logp: float
    logs: float
    avg_confidence: int
    highest_potency_target: str
    highest_potency_value: str
    predictions: List[PredictionRow]
    logo_path: str = "veridica_logo.png"
    report_id: str = None
    
    def __init__(self, **data):
        if 'report_id' not in data or not data['report_id']:
            data['report_id'] = str(uuid.uuid4())[:8].upper()
        super().__init__(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Jinja2 template"""
        return self.dict()

def format_predictions_for_pdf(predictions_data: Dict[str, Any]) -> PredictionBatch:
    """Convert API response to PDF-ready format"""
    from datetime import datetime
    import math
    
    # Extract data from API response
    smiles = predictions_data.get('smiles', '')
    properties = predictions_data.get('properties', {})
    predictions = predictions_data.get('predictions', {})
    model_info = predictions_data.get('model_info', {})
    
    # Calculate compound name from SMILES (simplified)
    compound_name = f"Compound_{smiles[:10]}..." if len(smiles) > 10 else smiles
    
    # Helper functions
    def get_heat_color(p_value, confidence):
        """Calculate HSL heat-map color"""
        clipped_p = max(0, p_value)
        
        def hue(p):
            if p >= 9: return 220
            if p >= 7: return 140 + (220-140)*(p-7)/2
            if p >= 5: return 50 + (140-50)*(p-5)/2  
            if p >= 3: return 25 + (50-25)*(p-3)/2
            return 0
        
        lightness = 45 + 10 * confidence
        return f"hsl({hue(clipped_p)}, 70%, {lightness}%)"
    
    def get_selectivity_display(ratio):
        """Get selectivity display information"""
        if ratio is None:
            return "Panel not available", "panel-na", "—"
        elif ratio >= 10:
            return f"{ratio:.1f}×", "selective", "✔︎"
        elif ratio >= 3:
            return f"{ratio:.1f}×", "moderate", "🟡"
        else:
            return f"{ratio:.1f}×", "non-selective", "❌"
    
    def get_assay_colors(assay_type):
        """Get assay type styling"""
        colors = {
            'IC50': ('#7c3aed', '#ffffff'),  # purple
            'Ki': ('#1d4ed8', '#ffffff'),    # blue  
            'EC50': ('#059669', '#ffffff')   # green
        }
        bg, text = colors.get(assay_type, ('#6b7280', '#ffffff'))
        return bg, text
    
    # Process predictions into rows
    pdf_rows = []
    confidences = []
    potencies = []
    
    # Only include OK status predictions in potency ranking (exclude gated predictions)
    ok_predictions = {}
    for target, target_data in predictions.items():
        ok_target_predictions = {}
        for assay_type in ['IC50', 'Ki', 'EC50']:
            pred = target_data.get(assay_type)
            if pred and pred.get('status', 'OK') == 'OK':  # Only OK status
                ok_target_predictions[assay_type] = pred
        if ok_target_predictions:
            ok_predictions[target] = ok_target_predictions
    
    for target, target_data in predictions.items():
        selectivity_ratio = target_data.get('selectivity_ratio')
        sel_label, sel_class, sel_icon = get_selectivity_display(selectivity_ratio)
        
        for assay_type in ['IC50', 'Ki', 'EC50']:
            pred = target_data.get(assay_type)
            if not pred:
                continue
            
            # Check if prediction was gated (numeric potency suppressed)
            pred_status = pred.get('status', 'OK')
            
            if pred_status == 'HYPOTHESIS_ONLY':
                # Gated prediction - show as hypothesis only
                display_ic50 = "Hypothesis only"
                pic50_display = "Out of domain"
                confidence = 0.0  # No confidence for gated predictions
                
                # Use gray styling for gated predictions
                heat_bg = 'hsl(0, 0%, 25%)'  # Dark gray
                opacity = 0.8
                
            else:
                # Normal prediction processing
                p_value = pred.get('pActivity', 0)
                activity_um = pred.get('activity_uM', 0)
                confidence = pred.get('confidence', 0.8)
                
                # Format display values with quality handling
                quality_flag = pred.get('quality_flag', 'good')
                
                if quality_flag == 'not_trained':
                    display_ic50 = "Not trained"
                    pic50_display = f"({assay_type} unavailable)"
                    heat_bg = 'hsl(0, 0%, 25%)'  # Dark gray
                    opacity = 0.8
                elif activity_um > 100:
                    display_ic50 = "> 100 μM"
                    pic50_display = "—"
                    heat_bg = get_heat_color(p_value, confidence)
                    opacity = 0.4 + 0.6 * confidence
                else:
                    display_ic50 = f"{activity_um:.1f} μM"
                    pic50_display = f"{max(0, p_value):.2f}"
                    heat_bg = get_heat_color(p_value, confidence)
                    opacity = 0.4 + 0.6 * confidence
                
                # Only include in potency ranking if not gated and activity < 100 μM
                if pred_status == 'OK' and quality_flag != 'not_trained' and activity_um <= 100:
                    potencies.append((target, activity_um, assay_type))
            
            assay_bg, assay_color = get_assay_colors(assay_type)
            
            pdf_rows.append(PredictionRow(
                protein=target,
                display_ic50=display_ic50,
                pic50=pic50_display,
                heat_bg=heat_bg,
                opacity=opacity,
                selectivity_label=sel_label,
                selectivity_class=sel_class,
                selectivity_icon=sel_icon,
                confidence_percent=int(confidence * 100),
                assay=assay_type,
                assay_bg=assay_bg,
                assay_color=assay_color
            ))
            
            # Only include confidence in average if not gated
            if pred_status == 'OK':
                confidences.append(confidence)
    
    # Calculate summary statistics
    avg_conf = int(sum(confidences) / len(confidences) * 100) if confidences else 0
    
    # Find highest potency (lowest IC50)
    if potencies:
        highest_pot = min(potencies, key=lambda x: x[1])
        highest_potency_target = highest_pot[0]
        highest_potency_value = f"{highest_pot[1]:.1f} μM ({highest_pot[2]})"
    else:
        highest_potency_target = "N/A"
        highest_potency_value = "N/A"
    
    return PredictionBatch(
        compound_name=compound_name,
        smiles=smiles,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        model_name=model_info.get('name', 'Gnosis I'),
        r2=model_info.get('r2_score', 0.0),
        total_predictions=len(pdf_rows),
        logp=properties.get('LogP', 0.0),
        logs=properties.get('LogS', 0.0),
        avg_confidence=avg_conf,
        highest_potency_target=highest_potency_target,
        highest_potency_value=highest_potency_value,
        predictions=pdf_rows
    )