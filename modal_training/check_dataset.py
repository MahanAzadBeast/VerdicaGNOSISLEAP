"""
Check Model 2 dataset structure
"""

import modal
import pandas as pd
from pathlib import Path

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install(["pandas"])
app = modal.App("check-dataset")
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume}
)
def check_model2_dataset():
    """Check Model 2 dataset structure"""
    
    datasets_dir = Path("/vol/datasets")
    training_file = datasets_dir / "gnosis_model2_cytotox_training.csv"
    
    if training_file.exists():
        df = pd.read_csv(training_file)
        print(f"Dataset shape: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
        print(f"Normal data columns: {[col for col in df.columns if 'normal' in col.lower()]}")
        print(f"AC50 columns: {[col for col in df.columns if 'ac50' in col.lower()]}")
        print(f"Selectivity columns: {[col for col in df.columns if 'select' in col.lower()]}")
        print(f"Sample first row:")
        print(df.iloc[0].to_dict())
        
        return {
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'normal_columns': [col for col in df.columns if 'normal' in col.lower()]
        }
    else:
        print(f"Dataset not found at: {training_file}")
        return {'error': 'Dataset not found'}

if __name__ == "__main__":
    print("🔍 Checking Model 2 dataset...")