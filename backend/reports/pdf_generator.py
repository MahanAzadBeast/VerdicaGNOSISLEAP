"""
PDF generation using WeasyPrint with fallback to ReportLab
"""
import os
import io
from pathlib import Path
from typing import Dict, Any
from .templating import env, get_asset_path
from .schemas import PredictionBatch, format_predictions_for_pdf

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

def create_veridica_logo_svg() -> str:
    """Create a simple SVG logo for Veridica AI"""
    return '''<svg width="120" height="32" viewBox="0 0 120 32" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#8b5cf6;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#06b6d4;stop-opacity:1" />
            </linearGradient>
        </defs>
        <circle cx="16" cy="16" r="12" fill="url(#grad1)"/>
        <text x="35" y="12" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#1e293b">VERIDICA</text>
        <text x="35" y="24" font-family="Arial, sans-serif" font-size="8" fill="#64748b">AI Platform</text>
    </svg>'''

def generate_pdf_weasyprint(batch: PredictionBatch) -> bytes:
    """Generate PDF using WeasyPrint"""
    template = env.get_template("prediction_report.html")
    
    # Add logo as embedded SVG
    logo_svg = create_veridica_logo_svg()
    batch_dict = batch.to_dict()
    batch_dict['logo_path'] = f"data:image/svg+xml;base64,{io.base64encode(logo_svg.encode()).decode()}"
    
    html_content = template.render(**batch_dict)
    
    # Generate PDF
    html_doc = HTML(string=html_content, base_url=str(Path(__file__).parent))
    pdf_bytes = html_doc.write_pdf()
    
    return pdf_bytes

def generate_pdf_reportlab(batch: PredictionBatch) -> bytes:
    """Fallback PDF generation using ReportLab"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph(f"<b>Veridica AI - Target Activity Report</b><br/>{batch.compound_name}", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Metadata
    meta_text = f"Generated: {batch.timestamp} | Model: {batch.model_name} (R² = {batch.r2:.3f})"
    meta = Paragraph(meta_text, styles['Normal'])
    story.append(meta)
    story.append(Spacer(1, 12))
    
    # Warning banner
    warning = Paragraph("<i>🔍 In-silico estimates – wet-lab validation required before decision making.</i>", styles['Normal'])
    story.append(warning)
    story.append(Spacer(1, 12))
    
    # Table data
    data = [['Target', 'Potency', 'pIC50', 'Selectivity', 'Confidence', 'Assay']]
    for row in batch.predictions:
        data.append([
            row.protein,
            row.display_ic50,
            row.pic50,
            row.selectivity_label,
            f"{row.confidence_percent}%",
            row.assay
        ])
    
    # Create table
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 12))
    
    # Summary
    summary = Paragraph(f"<b>Summary:</b> Average Confidence: {batch.avg_confidence}% | Highest Potency: {batch.highest_potency_target} ({batch.highest_potency_value})", styles['Normal'])
    story.append(summary)
    
    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

def generate_prediction_pdf(predictions_data: Dict[str, Any]) -> bytes:
    """
    Main PDF generation function with automatic fallback
    """
    # Format data for PDF
    batch = format_predictions_for_pdf(predictions_data)
    
    # Try WeasyPrint first, fallback to ReportLab
    if WEASYPRINT_AVAILABLE:
        try:
            return generate_pdf_weasyprint(batch)
        except Exception as e:
            print(f"WeasyPrint failed: {e}, falling back to ReportLab")
    
    if REPORTLAB_AVAILABLE:
        return generate_pdf_reportlab(batch)
    else:
        raise RuntimeError("No PDF generation library available. Install weasyprint or reportlab.")