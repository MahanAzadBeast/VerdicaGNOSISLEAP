"""
Simple check: Can we access expanded-datasets > gdsc_dataset?
Just list files, no heavy processing
"""

import modal
import os

app = modal.App("simple-gdsc-dataset-check")
image = modal.Image.debian_slim()
data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    volumes={"/vol": data_volume},
    timeout=60
)
def check_gdsc_dataset_path():
    """Simple check of expanded-datasets > gdsc_dataset path"""
    
    print("🔍 CHECKING: expanded-datasets > gdsc_dataset")
    print("Path: /vol/gdsc_dataset/")
    print()
    
    # Check if gdsc_dataset directory exists
    gdsc_path = "/vol/gdsc_dataset"
    
    if os.path.exists(gdsc_path):
        print("✅ FOUND: gdsc_dataset directory exists!")
        
        try:
            files = os.listdir(gdsc_path)
            print(f"📁 Files in gdsc_dataset ({len(files)}):")
            
            for file in sorted(files):
                file_path = os.path.join(gdsc_path, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"   📄 {file} ({size:,} bytes)")
                elif os.path.isdir(file_path):
                    print(f"   📁 {file}/ (directory)")
            
            return {"found": True, "files": files}
            
        except Exception as e:
            print(f"❌ Error reading directory: {e}")
            return {"found": True, "error": str(e)}
    
    else:
        print("❌ NOT FOUND: gdsc_dataset directory does not exist")
        
        # Check what's in root
        print("\n📁 Root volume contents:")
        try:
            root_files = os.listdir("/vol")
            for file in sorted(root_files):
                print(f"   {file}")
        except Exception as e:
            print(f"❌ Can't read root: {e}")
        
        return {"found": False}

if __name__ == "__main__":
    with app.run():
        result = check_gdsc_dataset_path.remote()
        
        if result.get("found"):
            print("\n✅ YES - Can access gdsc_dataset directory")
        else:
            print("\n❌ NO - Cannot find gdsc_dataset directory")