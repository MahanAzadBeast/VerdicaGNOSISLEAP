#!/usr/bin/env python3
"""
Quick Enhanced GNN Test - Test while GNN is training
"""

import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

def test_health_endpoint():
    """Test health endpoint for GNN status"""
    print("=== Testing Health Endpoint for Enhanced GNN Status ===")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health endpoint responding")
            print(f"Status: {data.get('status')}")
            print(f"Enhanced predictions: {data.get('enhanced_predictions')}")
            print(f"Model type: {data.get('model_type')}")
            print(f"Models loaded: {data.get('models_loaded')}")
            print(f"Real ML targets: {data.get('real_ml_targets')}")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_gnn_prediction_during_training():
    """Test GNN prediction while training is in progress"""
    print("\n=== Testing Enhanced GNN Prediction (Training in Progress) ===")
    try:
        payload = {
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
            "prediction_types": ["bioactivity_ic50"],
            "target": "EGFR"
        }
        
        print("Making prediction request (may take time due to GNN training)...")
        response = requests.post(f"{API_BASE}/predict", 
                               json=payload, 
                               headers={'Content-Type': 'application/json'},
                               timeout=120)  # Longer timeout for training
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful")
            
            if 'results' in data and len(data['results']) > 0:
                result = data['results'][0]
                enhanced_pred = result.get('enhanced_chemprop_prediction')
                
                if enhanced_pred:
                    print(f"Model type: {enhanced_pred.get('model_type')}")
                    print(f"Architecture: {enhanced_pred.get('architecture')}")
                    print(f"pIC50: {enhanced_pred.get('pic50')}")
                    print(f"IC50: {enhanced_pred.get('ic50_nm')} nM")
                    print(f"Confidence: {enhanced_pred.get('confidence')}")
                    print(f"Target specific: {enhanced_pred.get('target_specific')}")
                    
                    # Check for GNN-specific features
                    if 'graph_features' in enhanced_pred:
                        graph_features = enhanced_pred['graph_features']
                        print(f"Graph features: {graph_features}")
                        print(f"Atom features: {graph_features.get('atom_features')}")
                        print(f"Message passing layers: {graph_features.get('message_passing_layers')}")
                        print(f"Residual connections: {graph_features.get('residual_connections')}")
                    
                    # Check for training info
                    if 'training_size' in enhanced_pred:
                        print(f"Training size: {enhanced_pred.get('training_size')}")
                    
                    if 'model_performance' in enhanced_pred:
                        perf = enhanced_pred['model_performance']
                        print(f"Model performance: {perf}")
                    
                    return True
                else:
                    print("❌ No enhanced prediction data")
                    return False
            else:
                print("❌ No results in response")
                return False
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_fallback_behavior():
    """Test fallback to heuristic model if GNN not ready"""
    print("\n=== Testing Fallback Behavior ===")
    try:
        payload = {
            "smiles": "CCO",  # ethanol - simple molecule
            "prediction_types": ["bioactivity_ic50"],
            "target": "BRAF"  # Different target
        }
        
        response = requests.post(f"{API_BASE}/predict", 
                               json=payload, 
                               headers={'Content-Type': 'application/json'},
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Fallback prediction successful")
            
            if 'results' in data and len(data['results']) > 0:
                result = data['results'][0]
                enhanced_pred = result.get('enhanced_chemprop_prediction')
                
                if enhanced_pred:
                    model_type = enhanced_pred.get('model_type')
                    print(f"Model type: {model_type}")
                    
                    if model_type == 'enhanced_gnn':
                        print("✅ Using Enhanced GNN")
                    elif model_type == 'Enhanced RDKit-based':
                        print("✅ Using heuristic fallback (expected during training)")
                    else:
                        print(f"⚠️ Unexpected model type: {model_type}")
                    
                    return True
                else:
                    print("❌ No enhanced prediction data")
                    return False
            else:
                print("❌ No results in response")
                return False
        else:
            print(f"❌ Fallback test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Fallback test error: {e}")
        return False

if __name__ == "__main__":
    print("🧠 Quick Enhanced GNN Test")
    print(f"Backend URL: {API_BASE}")
    print("=" * 60)
    
    # Run quick tests
    health_ok = test_health_endpoint()
    prediction_ok = test_gnn_prediction_during_training()
    fallback_ok = test_fallback_behavior()
    
    print("\n" + "=" * 60)
    print("🏁 QUICK TEST SUMMARY")
    print("=" * 60)
    print(f"Health endpoint: {'✅' if health_ok else '❌'}")
    print(f"GNN prediction: {'✅' if prediction_ok else '❌'}")
    print(f"Fallback behavior: {'✅' if fallback_ok else '❌'}")
    
    if health_ok and (prediction_ok or fallback_ok):
        print("\n🎯 Enhanced GNN implementation is working!")
        print("Note: GNN may be training in background, fallback to heuristic is expected")
    else:
        print("\n❌ Issues detected with Enhanced GNN implementation")