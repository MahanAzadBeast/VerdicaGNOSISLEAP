"""
Test script for SimpleMolBERT Fine-tuner
"""

import asyncio
import sys
import os

# Add paths
sys.path.append('/app/backend')

from simple_molbert_finetuner import simple_molbert_finetuner

async def test_simple_molbert():
    """Test the SimpleMolBERT fine-tuner"""
    print("🧪 Testing SimpleMolBERT Fine-tuner...")
    
    # Test initialization
    print(f"✅ Fine-tuner initialized")
    print(f"✅ Device: {simple_molbert_finetuner.device}")
    print(f"✅ Vocabulary size: {simple_molbert_finetuner.tokenizer.vocab_size}")
    
    # Test tokenization
    test_smiles = "CCO"
    tokens = simple_molbert_finetuner.tokenizer.tokenize(test_smiles)
    print(f"✅ Tokenization test for '{test_smiles}': {len(tokens)} tokens")
    
    # Test model initialization
    print("\n🔄 Testing model initialization...")
    success = await simple_molbert_finetuner.initialize_models("EGFR")
    
    if success:
        print("✅ Model initialization successful!")
        
        # Test prediction
        print(f"\n🧬 Testing prediction for {test_smiles}...")
        result = await simple_molbert_finetuner.predict_ic50_gnn(test_smiles, "EGFR")
        
        print(f"📊 Prediction result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("❌ Model initialization failed!")

if __name__ == "__main__":
    asyncio.run(test_simple_molbert())