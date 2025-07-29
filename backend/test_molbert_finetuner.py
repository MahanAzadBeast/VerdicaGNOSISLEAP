"""
Test script for MolBERT Fine-tuner
Tests the basic functionality of the fine-tuning approach
"""

import asyncio
import sys
import os

# Add paths
sys.path.append('/app/backend')
sys.path.append('/app/MolBERT')

from molbert_finetuner import molbert_finetuner

async def test_molbert_finetuner():
    """Test the MolBERT fine-tuner"""
    print("🧪 Testing MolBERT Fine-tuner...")
    
    # Test initialization
    print(f"✅ Fine-tuner initialized with pretrained model: {molbert_finetuner.pretrained_model_path}")
    print(f"✅ Device: {molbert_finetuner.device}")
    
    # Test model initialization (this will trigger fine-tuning)
    print("\n🔄 Testing model initialization...")
    success = await molbert_finetuner.initialize_models("EGFR")
    
    if success:
        print("✅ Model initialization successful!")
        
        # Test prediction
        print("\n🧬 Testing prediction...")
        test_smiles = "CCO"  # Ethanol
        result = await molbert_finetuner.predict_ic50_gnn(test_smiles, "EGFR")
        
        print(f"📊 Prediction result for {test_smiles}:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("❌ Model initialization failed!")

if __name__ == "__main__":
    asyncio.run(test_molbert_finetuner())