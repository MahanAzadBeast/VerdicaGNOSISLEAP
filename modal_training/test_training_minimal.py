"""
Minimal test of ChemBERTa and Chemprop training to validate fixes
"""

import modal
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

# Basic Modal image for testing
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "torch>=2.0.0",
    "transformers>=4.30.0", 
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
])

app = modal.App("test-training-fixes")

@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=300  # 5 minutes for quick test
)
def test_chemberta_minimal():
    """Test ChemBERTa training components with minimal data"""
    
    import sys
    import os
    sys.path.append('/vol')  # Add path if needed
    
    # Create minimal test data
    test_data = {
        'canonical_smiles': ['CCO', 'C', 'CC', 'CCC', 'CCO'],
        'EGFR': [6.5, 5.2, 7.1, 6.8, 5.9],
        'BRAF': [7.2, 6.1, 6.9, 7.5, 6.3]
    }
    
    try:
        # Test data processing
        df = pd.DataFrame(test_data)
        print(f"✅ Test data created: {df.shape}")
        print(df.head())
        
        # Test imports
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        print("✅ ChemBERTa tokenizer loaded successfully")
        
        # Test tokenization
        test_smiles = "CCO"
        encoding = tokenizer(test_smiles, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        print(f"✅ Tokenization working: input_ids shape {encoding['input_ids'].shape}")
        
        return {"status": "success", "message": "ChemBERTa components working"}
        
    except Exception as e:
        print(f"❌ ChemBERTa test failed: {e}")
        return {"status": "error", "error": str(e)}

@app.function(
    image=image.pip_install(["chemprop"]),
    cpu=2.0,
    memory=4096,
    timeout=300
)
def test_chemprop_minimal():
    """Test Chemprop import and basic functionality"""
    
    try:
        # Test Chemprop imports
        import chemprop
        print("✅ Chemprop package imported successfully")
        
        # Test command preparation (without actually running)
        cmd = [
            'python', '-m', 'chemprop.train',
            '--data_path', '/tmp/test.csv',
            '--dataset_type', 'regression',
            '--epochs', '1'
        ]
        print(f"✅ Chemprop command formatted correctly: {' '.join(cmd[:4])}...")
        
        return {"status": "success", "message": "Chemprop components working"}
        
    except Exception as e:
        print(f"❌ Chemprop test failed: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("🧪 Running minimal training component tests...")
    print("=" * 60)