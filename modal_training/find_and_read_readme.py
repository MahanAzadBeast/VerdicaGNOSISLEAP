"""
Find and read any README or documentation files in Modal
"""

import modal
import os

app = modal.App("find-and-read-readme")

image = modal.Image.debian_slim()
data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    volumes={"/vol": data_volume},
    timeout=60
)
def find_and_read_documentation():
    """Find and read any documentation files"""
    
    print("🔍 SEARCHING FOR DOCUMENTATION FILES")
    print("=" * 50)
    
    vol_path = "/vol"
    
    if not os.path.exists(vol_path):
        print("❌ Volume path not accessible")
        return {"error": "Volume not accessible"}
    
    # List all files
    files = os.listdir(vol_path)
    print(f"📁 All files found: {files}")
    print()
    
    # Look for documentation files
    doc_files = []
    for file in files:
        file_lower = file.lower()
        if any(ext in file_lower for ext in ['.txt', '.md', '.readme', 'readme', 'info', 'manifest']):
            doc_files.append(file)
    
    print(f"📄 Documentation files found: {doc_files}")
    print()
    
    # Read each documentation file
    for doc_file in doc_files:
        file_path = os.path.join(vol_path, doc_file)
        print(f"📄 READING: {doc_file}")
        print("-" * 30)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content)
                print()
        except Exception as e:
            print(f"❌ Error reading {doc_file}: {e}")
        
        print("-" * 50)
        print()
    
    return {"documentation_files": doc_files, "all_files": files}

if __name__ == "__main__":
    with app.run():
        result = find_and_read_documentation.remote()
        doc_files = result.get("documentation_files", [])
        print(f"✅ Found {len(doc_files)} documentation files")