"""
List ALL files in Modal expanded-datasets volume
See exactly what's available
"""

import modal
import os

app = modal.App("list-modal-files")

image = modal.Image.debian_slim()
data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    volumes={"/vol": data_volume},
    timeout=60
)
def list_all_modal_files():
    """List all files in Modal volume"""
    
    print("📁 LISTING ALL FILES IN MODAL expanded-datasets VOLUME")
    print("=" * 60)
    
    vol_path = "/vol"
    if os.path.exists(vol_path):
        all_files = os.listdir(vol_path)
        print(f"Total files: {len(all_files)}")
        print()
        
        for i, filename in enumerate(sorted(all_files), 1):
            file_path = os.path.join(vol_path, filename)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"{i:2d}. {filename}")
                print(f"    Size: {size:,} bytes ({size/1024:.1f} KB)")
                
                # Show file extension info
                if filename.endswith('.csv'):
                    print(f"    Type: CSV Data File")
                elif filename.endswith('.txt'):
                    print(f"    Type: Text File")
                elif filename.endswith('.md'):
                    print(f"    Type: Markdown File")
                print()
    else:
        print("❌ Volume path /vol does not exist!")
    
    return {"status": "listed"}

if __name__ == "__main__":
    with app.run():
        result = list_all_modal_files.remote()
        print("✅ File listing complete!")