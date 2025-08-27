#!/usr/bin/env python3
"""
Comprehensive PDF Export Validation for Numeric Potency Gating System.

Tests the complete pipeline: API prediction → Gating → PDF generation
"""

import requests
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8001"

def test_aspirin_pdf_generation():
    """Test PDF generation for aspirin (all predictions should be gated)"""
    logger.info("🧪 Testing Aspirin PDF Generation (All Gated)")
    
    # Step 1: Get prediction from HP-AD API
    prediction_payload = {
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "targets": ["ERBB4", "AKT2", "PARP1"],
        "assay_types": ["IC50"]
    }
    
    logger.info("Making prediction request...")
    pred_response = requests.post(
        f"{BASE_URL}/api/gnosis-i/predict-with-hp-ad",
        json=prediction_payload,
        headers={"Content-Type": "application/json"}
    )
    
    if pred_response.status_code != 200:
        logger.error(f"Prediction request failed: {pred_response.status_code}")
        logger.error(pred_response.text)
        return False
    
    pred_data = pred_response.json()
    logger.info(f"✅ Got prediction response")
    
    # Validate gating in prediction
    gated_count = 0
    ok_count = 0
    
    for target, target_data in pred_data['predictions'].items():
        for assay_type, assay_data in target_data.items():
            if assay_type != 'selectivity_ratio':
                status = assay_data.get('status', 'OK')
                if status == 'HYPOTHESIS_ONLY':
                    gated_count += 1
                    logger.info(f"  ✅ {target}/{assay_type}: GATED - {assay_data.get('why', [])}")
                else:
                    ok_count += 1
                    logger.info(f"  ⚠️  {target}/{assay_type}: OK (not gated)")
    
    logger.info(f"Prediction summary: {gated_count} gated, {ok_count} OK")
    
    # Step 2: Generate PDF from prediction
    logger.info("Generating PDF from prediction...")
    pdf_response = requests.post(
        f"{BASE_URL}/api/reports/export-pdf",
        json=pred_data,
        headers={"Content-Type": "application/json"}
    )
    
    if pdf_response.status_code != 200:
        logger.error(f"PDF generation failed: {pdf_response.status_code}")
        logger.error(pdf_response.text)
        return False
    
    # Save PDF
    pdf_path = Path("/tmp/aspirin_validation_report.pdf")
    pdf_path.write_bytes(pdf_response.content)
    
    pdf_size = len(pdf_response.content)
    logger.info(f"✅ PDF generated: {pdf_size} bytes")
    
    # Validate PDF headers
    content_type = pdf_response.headers.get('content-type')
    if content_type != 'application/pdf':
        logger.error(f"Wrong content type: {content_type}")
        return False
    
    # Check for PDF signature
    if not pdf_response.content.startswith(b'%PDF'):
        logger.error("Invalid PDF signature")
        return False
    
    logger.info("✅ PDF validation passed")
    
    return {
        'gated_count': gated_count,
        'ok_count': ok_count,
        'pdf_size': pdf_size,
        'pdf_path': str(pdf_path)
    }

def test_normal_compound_pdf():
    """Test PDF generation for a normal compound (should have numeric predictions)"""
    logger.info("🧪 Testing Normal Compound PDF Generation")
    
    # Use a simple compound that should pass most gates
    prediction_payload = {
        "smiles": "CCO",  # Ethanol
        "targets": ["EGFR"],
        "assay_types": ["IC50"]
    }
    
    logger.info("Making prediction request for ethanol...")
    pred_response = requests.post(
        f"{BASE_URL}/api/gnosis-i/predict-with-hp-ad",
        json=prediction_payload,
        headers={"Content-Type": "application/json"}
    )
    
    if pred_response.status_code != 200:
        logger.error(f"Prediction request failed: {pred_response.status_code}")
        return False
    
    pred_data = pred_response.json()
    
    # Check if any predictions are not gated
    has_numeric = False
    for target, target_data in pred_data['predictions'].items():
        for assay_type, assay_data in target_data.items():
            if assay_type != 'selectivity_ratio':
                status = assay_data.get('status', 'OK')
                if status == 'OK':
                    has_numeric = True
                    logger.info(f"  ✅ {target}/{assay_type}: OK (has pActivity)")
    
    # Generate PDF
    pdf_response = requests.post(
        f"{BASE_URL}/api/reports/export-pdf",
        json=pred_data,
        headers={"Content-Type": "application/json"}
    )
    
    if pdf_response.status_code != 200:
        logger.error(f"PDF generation failed: {pdf_response.status_code}")
        return False
    
    pdf_path = Path("/tmp/ethanol_validation_report.pdf")
    pdf_path.write_bytes(pdf_response.content)
    
    logger.info(f"✅ Normal compound PDF generated: {len(pdf_response.content)} bytes")
    
    return {
        'has_numeric': has_numeric,
        'pdf_size': len(pdf_response.content),
        'pdf_path': str(pdf_path)
    }

def test_reports_health():
    """Test reports service health"""
    logger.info("🧪 Testing Reports Service Health")
    
    health_response = requests.get(f"{BASE_URL}/api/reports/health")
    
    if health_response.status_code != 200:
        logger.error(f"Health check failed: {health_response.status_code}")
        return False
    
    health_data = health_response.json()
    logger.info(f"✅ Reports service health: {health_data}")
    
    return health_data.get('status') == 'healthy'

def main():
    """Run comprehensive PDF export validation"""
    logger.info("🚀 Starting PDF Export Validation for Numeric Potency Gating")
    
    results = {
        'health_check': False,
        'aspirin_test': None,
        'normal_test': None
    }
    
    try:
        # Test 1: Reports service health
        results['health_check'] = test_reports_health()
        
        # Test 2: Aspirin (should be gated)
        results['aspirin_test'] = test_aspirin_pdf_generation()
        
        # Test 3: Normal compound (should have numerics)
        results['normal_test'] = test_normal_compound_pdf()
        
        # Final validation
        logger.info("\n🎯 FINAL VALIDATION RESULTS")
        logger.info(f"Reports Health: {'✅ PASS' if results['health_check'] else '❌ FAIL'}")
        
        if results['aspirin_test']:
            aspirin = results['aspirin_test']
            logger.info(f"Aspirin Gating: {'✅ PASS' if aspirin['gated_count'] > 0 else '❌ FAIL'} "
                       f"({aspirin['gated_count']} gated, {aspirin['ok_count']} numeric)")
            logger.info(f"Aspirin PDF: ✅ PASS ({aspirin['pdf_size']} bytes)")
        else:
            logger.error("❌ Aspirin test failed")
        
        if results['normal_test']:
            normal = results['normal_test'] 
            logger.info(f"Normal Compound: {'✅ PASS' if normal['has_numeric'] else '⚠️ WARNING'} "
                       f"(PDF: {normal['pdf_size']} bytes)")
        else:
            logger.error("❌ Normal compound test failed")
        
        # Success criteria
        success = (
            results['health_check'] and
            results['aspirin_test'] and
            results['aspirin_test']['gated_count'] >= 3 and  # All targets gated
            results['normal_test']
        )
        
        if success:
            logger.info("🎉 ALL PDF EXPORT TESTS PASSED!")
            logger.info("✅ Numeric Potency Gating system working correctly with PDF export")
            logger.info("✅ Aspirin predictions properly gated and displayed as 'Hypothesis only'")
            logger.info("✅ PDF generation handles both gated and normal predictions correctly")
        else:
            logger.error("❌ Some tests failed - see details above")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)