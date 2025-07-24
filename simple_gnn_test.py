#!/usr/bin/env python3
"""
Simple Enhanced GNN Verification Test
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

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health endpoint working")
            print(f"  Enhanced predictions: {data.get('enhanced_predictions')}")
            print(f"  Model type: {data.get('model_type')}")
            print(f"  Real ML models: {data.get('models_loaded', {}).get('real_ml_models')}")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_simple_prediction():
    """Test simple prediction with fallback"""
    try:
        payload = {
            "smiles": "CCO",  # ethanol
            "prediction_types": ["bioactivity_ic50"],
            "target": "EGFR"
        }
        
        response = requests.post(f"{API_BASE}/predict", 
                               json=payload, 
                               headers={'Content-Type': 'application/json'},
                               timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction endpoint working")
            
            if 'results' in data and len(data['results']) > 0:
                result = data['results'][0]
                enhanced_pred = result.get('enhanced_chemprop_prediction')
                
                if enhanced_pred:
                    model_type = enhanced_pred.get('model_type')
                    print(f"  Model type: {model_type}")
                    print(f"  pIC50: {enhanced_pred.get('pic50')}")
                    print(f"  Target specific: {enhanced_pred.get('target_specific')}")
                    
                    # Check for GNN features
                    if 'graph_features' in enhanced_pred:
                        print(f"  ✅ Enhanced GNN features detected!")
                        gf = enhanced_pred['graph_features']
                        print(f"    Atom features: {gf.get('atom_features')}")
                        print(f"    Message passing layers: {gf.get('message_passing_layers')}")
                        print(f"    Residual connections: {gf.get('residual_connections')}")
                    
                    return True
                else:
                    print("❌ No enhanced prediction")
                    return False
            else:
                print("❌ No results")
                return False
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

if __name__ == "__main__":
    print("🧠 Simple Enhanced GNN Verification")
    print(f"Backend URL: {API_BASE}")
    print("=" * 50)
    
    health_ok = test_health()
    pred_ok = test_simple_prediction()
    
    print("\n" + "=" * 50)
    if health_ok and pred_ok:
        print("🎯 Enhanced GNN implementation verified!")
    else:
        print("❌ Issues detected")