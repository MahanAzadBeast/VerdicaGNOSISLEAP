"""
Simple Chemprop test to diagnose issues
"""

import modal
import pandas as pd
import subprocess
import tempfile
from pathlib import Path

# Basic Modal image - Updated for v2.2.0
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "chemprop>=2.2.0",  # Updated to latest version
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
        
        # Create simple test data - Need more samples for splits to work
        test_data = {
            'smiles': [
                'CCO', 'C', 'CC', 'CCC', 'CCCC', 'CCCCC',  # Simple alkanes and ethanol
                'C1=CC=CC=C1', 'CC(=O)O', 'CCN', 'CCC(=O)O'  # Benzene, acetic acid, ethylamine, etc
            ],
            'target1': [6.5, 5.2, 7.1, 6.8, 5.9, 6.3, 7.8, 5.5, 6.7, 7.2],
            'target2': [7.2, 6.1, 6.9, 7.5, 6.8, 7.0, 8.1, 6.2, 7.1, 7.8]
        }
        
        df = pd.DataFrame(test_data)
        print(f"✅ Test data created: {df.shape}")
        
        # Save to temp file
        temp_dir = Path(tempfile.mkdtemp())
        train_path = temp_dir / "train.csv"
        df.to_csv(train_path, index=False)
        print(f"✅ Data saved to: {train_path}")
        
        # Test Chemprop command - Updated for v2.2.0 CLI with correct arguments
        cmd = [
            'chemprop', 'train',
            '--data-path', str(train_path),
            '--task-type', 'regression',
            '--epochs', '3',  # Increased to be higher than warmup epochs (default 2)
            '--batch-size', '2',
            '--message-hidden-dim', '32',  # Changed from --hidden-size
            '--depth', '2',
            '--num-workers', '0',
            '--tracking-metric', 'mse',  # Use mse instead of val_loss
            '--save-dir', str(temp_dir / "model")
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