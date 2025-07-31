"""
Test Chemprop CLI arguments to understand the new format
"""

import modal

# Basic Modal image
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "chemprop>=2.2.0",
])

app = modal.App("chemprop-help-test")

@app.function(
    image=image,
    cpu=1.0,
    memory=4096,
    timeout=300
)
def get_chemprop_help():
    """Get Chemprop CLI help to understand new arguments"""
    
    import subprocess
    
    try:
        # Get general help
        print("=== CHEMPROP GENERAL HELP ===")
        result = subprocess.run(['chemprop', '--help'], capture_output=True, text=True)
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        print("\n=== CHEMPROP TRAIN HELP ===")
        # Get train-specific help
        result2 = subprocess.run(['chemprop', 'train', '--help'], capture_output=True, text=True)
        print(f"Return code: {result2.returncode}")
        print(f"STDOUT:\n{result2.stdout}")
        if result2.stderr:
            print(f"STDERR:\n{result2.stderr}")
            
        return {"general": result.stdout, "train": result2.stdout}
        
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("Getting Chemprop help...")