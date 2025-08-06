"""
Read the README.txt file from Modal expanded-datasets
This should guide us to the GDSC dataset with >600 compounds
"""

import modal
import os

app = modal.App("read-gdsc-readme")

image = modal.Image.debian_slim()
data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    volumes={"/vol": data_volume},
    timeout=60
)
def read_gdsc_readme():
    """Read the README.txt file to understand the data structure"""
    
    print("📄 READING GDSC README.TXT")
    print("=" * 50)
    
    readme_path = "/vol/README.txt"
    
    if os.path.exists(readme_path):
        print("✅ Found README.txt")
        print()
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print("📄 README CONTENT:")
                print("-" * 30)
                print(content)
                print("-" * 30)
                
            return {"readme_content": content, "status": "success"}
            
        except Exception as e:
            print(f"❌ Error reading README.txt: {e}")
            return {"error": str(e)}
    else:
        print("❌ README.txt not found!")
        
        # List what files are available
        vol_path = "/vol"
        if os.path.exists(vol_path):
            files = os.listdir(vol_path)
            print(f"📁 Available files: {files}")
        
        return {"error": "README.txt not found"}

if __name__ == "__main__":
    with app.run():
        result = read_gdsc_readme.remote()
        if result.get("status") == "success":
            print("✅ README read successfully!")
        else:
            print(f"❌ Failed to read README: {result.get('error')}")