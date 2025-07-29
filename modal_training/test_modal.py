"""
Quick Modal Deployment Test
Run this to verify everything is working before full training
"""

import modal
import sys

app = modal.App("molbert-training-test")

# Same image as main training
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "torch>=2.1.0",
    "transformers>=4.21.0", 
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "joblib>=1.3.0",
    "requests>=2.31.0",
    "rdkit-pypi>=2022.9.5"
])

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1),
    timeout=600  # 10 minutes max for test
)
def test_gpu_setup():
    """Test GPU setup and dependencies"""
    import torch
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ GPU Available: {gpu_name}")
        logger.info(f"✅ GPU Memory: {gpu_memory:.1f} GB")
        
        # Test tensor operations
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x.T)
        logger.info(f"✅ GPU Tensor Operations: Working")
        
        return {
            "gpu_available": True,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory,
            "status": "ready_for_training"
        }
    else:
        logger.error("❌ No GPU detected")
        return {
            "gpu_available": False,
            "status": "error_no_gpu"
        }

@app.local_entrypoint()
def main():
    """Test Modal setup"""
    print("🧪 Testing Modal GPU Setup...")
    result = test_gpu_setup.remote()
    print(f"📊 Test Result: {result}")
    
    if result["gpu_available"]:
        print("✅ Modal GPU setup successful!")
        print("🚀 Ready to deploy full MolBERT training!")
    else:
        print("❌ GPU setup failed")
    
    return result