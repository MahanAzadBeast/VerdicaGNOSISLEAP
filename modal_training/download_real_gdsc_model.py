"""
Download the real GDSC trained model from Modal storage to local models directory
"""

import modal
import shutil
import os

app = modal.App("download-real-gdsc-model")

model_volume = modal.Volume.from_name("trained-models")

@app.function(
    volumes={"/models": model_volume},
    timeout=300
)
def download_model():
    """Download the real GDSC model from Modal storage"""
    
    source_path = "/models/real_gdsc_chemberta_cytotox_v1.pth"
    
    if os.path.exists(source_path):
        with open(source_path, 'rb') as f:
            model_data = f.read()
        
        print(f"✅ Model found: {source_path}")
        print(f"📊 Size: {len(model_data):,} bytes")
        
        return model_data
    else:
        print(f"❌ Model not found: {source_path}")
        # List available models
        try:
            files = os.listdir("/models")
            print("Available models:")
            for file in files:
                if file.endswith('.pth'):
                    file_path = os.path.join("/models", file)
                    size = os.path.getsize(file_path)
                    print(f"  📄 {file} ({size:,} bytes)")
        except:
            print("Cannot list models directory")
        
        return None

if __name__ == "__main__":
    with app.run():
        model_data = download_model.remote()
        
        if model_data:
            # Save to local models directory
            local_path = "/app/models/real_gdsc_chemberta_cytotox_v1.pth"
            
            with open(local_path, 'wb') as f:
                f.write(model_data)
            
            print(f"✅ Model downloaded successfully!")
            print(f"💾 Saved to: {local_path}")
            print(f"📊 Size: {len(model_data):,} bytes")
            
            # Verify the file
            if os.path.exists(local_path):
                local_size = os.path.getsize(local_path)
                print(f"✅ Verification: {local_size:,} bytes")
            else:
                print("❌ File not found after download")
        else:
            print("❌ Download failed - model not found in Modal storage")