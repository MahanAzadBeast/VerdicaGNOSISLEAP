import modal
from pathlib import Path

app = modal.App("volume-check")
volume = modal.Volume.from_name("chembl-database", create_if_missing=False)

@app.function(
    volumes={"/vol/chembl": volume},
    timeout=300
)
def check_volume_contents():
    chembl_dir = Path("/vol/chembl")
    if chembl_dir.exists():
        files = list(chembl_dir.glob("*"))
        print(f"Files in chembl volume: {[f.name for f in files]}")
        for f in files:
            if f.is_file():
                size_gb = f.stat().st_size / 1e9
                print(f"  {f.name}: {size_gb:.2f} GB")
        return {"files": [f.name for f in files]}
    else:
        print("ChEMBL volume directory not found")
        return {"files": []}

if __name__ == "__main__":
    with app.run():
        result = check_volume_contents.remote()
        print(f"Result: {result}")