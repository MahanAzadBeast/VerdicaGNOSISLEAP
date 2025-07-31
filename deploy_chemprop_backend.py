#!/usr/bin/env python3
"""
Ensure Chemprop Model is Deployed and Available
"""

import modal
import sys
import os
from pathlib import Path

# Import the production inference system
sys.path.append('/app/modal_training')

def deploy_chemprop_for_backend():
    """Deploy Chemprop model for backend integration"""
    
    print("🚀 DEPLOYING CHEMPROP FOR BACKEND INTEGRATION")
    print("=" * 60)
    
    try:
        from chemprop_production_inference import app, predict_oncoprotein_activity, get_model_info
        
        print("📦 Deploying Chemprop production inference...")
        
        with app.run():
            # Test prediction
            print("🧪 Testing prediction...")
            test_result = predict_oncoprotein_activity.remote("CC(=O)OC1=CC=CC=C1C(=O)O")
            
            if test_result.get('status') == 'success':
                print("✅ Prediction test successful!")
                predictions = test_result.get('predictions', {})
                print(f"   Targets predicted: {len(predictions)}")
                
                # Show sample prediction
                if 'EGFR' in predictions:
                    egfr_pred = predictions['EGFR']
                    ic50_nm = egfr_pred.get('IC50_nM', 0)
                    print(f"   Sample (EGFR): IC50 {ic50_nm:.1f} nM")
                
                print("✅ Chemprop model deployed and ready!")
                return True
            else:
                print(f"❌ Prediction test failed: {test_result}")
                return False
                
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return False

def test_backend_connection():
    """Test backend API connection to Chemprop"""
    
    print("\n🔗 TESTING BACKEND CONNECTION")
    print("=" * 40)
    
    import requests
    
    # Test status
    try:
        response = requests.get("http://localhost:8001/api/chemprop-real/status", timeout=10)
        print(f"Status endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Available: {data.get('available', False)}")
        else:
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   Status check failed: {e}")
    
    # Test prediction
    try:
        response = requests.post("http://localhost:8001/api/chemprop-real/predict", 
                               json={"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
                               timeout=60)
        print(f"Prediction endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Prediction status: {data.get('status', 'unknown')}")
            predictions = data.get('predictions', {})
            if predictions:
                print(f"   Targets predicted: {len(predictions)}")
        else:
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   Prediction test failed: {e}")

if __name__ == "__main__":
    print("🔧 Starting Chemprop deployment for backend...")
    
    # Deploy the model
    success = deploy_chemprop_for_backend()
    
    if success:
        print("\n✅ Deployment successful!")
        
        # Restart backend to pick up changes
        print("🔄 Restarting backend...")
        os.system("sudo supervisorctl restart backend")
        
        # Wait and test connection
        import time
        time.sleep(10)
        
        test_backend_connection()
        
        print("\n🎯 READY FOR MODEL ARCHITECTURE COMPARISON!")
        print("Both ChemBERTa and Chemprop should now be available in the UI")
        
    else:
        print("\n❌ Deployment failed - check logs for details")