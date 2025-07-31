"""
Simple Chemprop test to diagnose issues
"""

import modal
import pandas as pd
import subprocess
import tempfile
from pathlib import Path

# Basic Modal image
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "chemprop>=1.7.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
])

app = modal.App("chemprop-test")

datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=2.0,
    memory=8192,
    timeout=600
)
def test_chemprop_basic():
    """Test basic Chemprop functionality"""
    
    print("🧪 Testing basic Chemprop functionality...")
    
    try:
        # Test Chemprop import
        import chemprop
        print("✅ Chemprop imported successfully")
        
        # Create simple test data
        test_data = {
            'smiles': ['CCO', 'C', 'CC', 'CCC'],
            'target1': [6.5, 5.2, 7.1, 6.8],
            'target2': [7.2, 6.1, 6.9, 7.5]
        }
        
        df = pd.DataFrame(test_data)
        print(f"✅ Test data created: {df.shape}")
        
        # Save to temp file
        temp_dir = Path(tempfile.mkdtemp())
        train_path = temp_dir / "train.csv"
        df.to_csv(train_path, index=False)
        print(f"✅ Data saved to: {train_path}")
        
        # Test Chemprop command
        cmd = [
            'python', '-m', 'chemprop.train',
            '--data_path', str(train_path),
            '--dataset_type', 'regression',
            '--epochs', '1',
            '--batch_size', '2',
            '--hidden_size', '32',
            '--depth', '2',
            '--num_workers', '0',
            '--save_dir', str(temp_dir / "model"),
            '--quiet'
        ]
        
        print("🔧 Testing Chemprop command...")
        print(f"Command: {' '.join(cmd[:5])}...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✅ Chemprop command successful!")
            print(f"STDOUT: {result.stdout[:500]}...")
        else:
            print(f"❌ Chemprop command failed with return code: {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
        return {
            "status": "success" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting Chemprop basic test...")