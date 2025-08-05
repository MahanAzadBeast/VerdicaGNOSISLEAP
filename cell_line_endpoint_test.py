#!/usr/bin/env python3
"""
Test Cell Line Response Model endpoints with correct format
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

def test_cell_line_predict():
    """Test Cell Line predict endpoint with correct format"""
    print("=== Testing Cell Line Predict Endpoint ===")
    
    # Use the correct format based on examples
    payload = {
        "drug_name": "Erlotinib",
        "smiles": "Cc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCOC",
        "cell_line": {
            "cell_line_name": "A549",
            "cancer_type": "LUNG",
            "genomic_features": {
                "mutations": {
                    "TP53": 1,
                    "KRAS": 1,
                    "EGFR": 0,
                    "BRAF": 0
                },
                "cnvs": {
                    "MYC": 1,
                    "CDKN2A": -1,
                    "PTEN": 0
                },
                "expression": {
                    "EGFR": -0.5,
                    "KRAS": 1.2,
                    "TP53": -1.8
                }
            }
        }
    }
    
    try:
        response = requests.post(f"{API_BASE}/cell-line/predict", 
                               json=payload,
                               headers={'Content-Type': 'application/json'},
                               timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS: Cell Line Predict endpoint working")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_cell_line_compare():
    """Test Cell Line compare endpoint with correct format"""
    print("\n=== Testing Cell Line Compare Endpoint ===")
    
    # Use the correct format based on examples
    payload = {
        "drug_name": "Trametinib",
        "smiles": "CC(C)[C@H](C(=O)N1CC2=C(C1=O)SC(=N2)C3=CC(=C(C=C3)F)F)N4C(=O)C5=C(C4=O)C=CC=C5I",
        "cell_lines": [
            {
                "cell_line_name": "A549",
                "cancer_type": "LUNG",
                "genomic_features": {
                    "mutations": {
                        "TP53": 1,
                        "KRAS": 1,
                        "EGFR": 0,
                        "BRAF": 0
                    },
                    "cnvs": {
                        "MYC": 1,
                        "CDKN2A": -1,
                        "PTEN": 0
                    },
                    "expression": {
                        "EGFR": -0.5,
                        "KRAS": 1.2,
                        "TP53": -1.8
                    }
                }
            },
            {
                "cell_line_name": "MCF7",
                "cancer_type": "BREAST",
                "genomic_features": {
                    "mutations": {
                        "TP53": 0,
                        "PIK3CA": 1,
                        "KRAS": 0,
                        "EGFR": 0
                    },
                    "cnvs": {
                        "MYC": 0,
                        "CDKN2A": 0,
                        "PTEN": 0
                    },
                    "expression": {
                        "EGFR": 0.3,
                        "KRAS": -0.2,
                        "TP53": 0.8
                    }
                }
            }
        ]
    }
    
    try:
        response = requests.post(f"{API_BASE}/cell-line/compare", 
                               json=payload,
                               headers={'Content-Type': 'application/json'},
                               timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS: Cell Line Compare endpoint working")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 CELL LINE RESPONSE MODEL ENDPOINT TESTING")
    print("=" * 60)
    print(f"🌐 Backend URL: {BACKEND_URL}")
    print(f"📡 API Base: {API_BASE}")
    
    predict_success = test_cell_line_predict()
    compare_success = test_cell_line_compare()
    
    print("\n" + "=" * 60)
    print("🎯 CELL LINE ENDPOINT TEST SUMMARY")
    print("=" * 60)
    
    if predict_success and compare_success:
        print("✅ ALL TESTS PASSED: Cell Line Response Model endpoints are working correctly")
    elif predict_success or compare_success:
        print("🟡 PARTIAL SUCCESS: Some Cell Line endpoints are working")
    else:
        print("❌ ALL TESTS FAILED: Cell Line endpoints need attention")
    
    print(f"   • Predict endpoint: {'✅ PASS' if predict_success else '❌ FAIL'}")
    print(f"   • Compare endpoint: {'✅ PASS' if compare_success else '❌ FAIL'}")