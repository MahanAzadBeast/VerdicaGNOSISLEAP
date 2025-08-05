"""
Investigate Model 2 Training Issues and Available Data
"""

import modal
import pandas as pd
import numpy as np
from pathlib import Path

app = modal.App("investigate-model2-issues")

# Setup volumes
expanded_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=False)
cell_line_volume = modal.Volume.from_name("cell-line-models", create_if_missing=False)

@app.function(
    image=modal.Image.debian_slim().pip_install(["pandas", "numpy"]),
    volumes={
        "/vol/expanded": expanded_volume,
        "/vol/cell_line": cell_line_volume
    }
)
def investigate_model2_data():
    """Investigate what cytotoxicity/GDSC data is available"""
    
    print("🔍 INVESTIGATING MODEL 2 (CYTOTOXICITY) DATA")
    print("=" * 60)
    
    # Check expanded datasets volume
    expanded_dir = Path("/vol/expanded")
    cell_line_dir = Path("/vol/cell_line")
    
    print("\n📁 EXPANDED DATASETS VOLUME:")
    print("-" * 40)
    
    if expanded_dir.exists():
        all_files = list(expanded_dir.glob("*"))
        csv_files = [f for f in all_files if f.suffix.lower() == '.csv']
        
        print(f"Total files: {len(all_files)}")
        print(f"CSV files: {len(csv_files)}")
        
        # Look for cytotoxicity/GDSC related files
        cytotox_files = [f for f in csv_files if any(keyword in f.name.lower() 
                        for keyword in ['cytotox', 'gdsc', 'cancer', 'cell', 'ic50', 'viability'])]
        
        print(f"\n🎯 CYTOTOXICITY-RELATED FILES:")
        for file in cytotox_files:
            try:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  📄 {file.name} ({size_mb:.1f}MB)")
                
                # Quick analysis of each file
                if size_mb < 100:  # Only analyze smaller files to avoid memory issues
                    df = pd.read_csv(file)
                    print(f"    📊 {len(df):,} rows × {len(df.columns)} columns")
                    print(f"    🔍 Columns: {list(df.columns)[:5]}...")
                    
                    # Look for cancer-specific indicators
                    cancer_indicators = []
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(term in col_lower for term in ['cancer', 'tumor', 'gdsc', 'cell_line', 'tissue']):
                            cancer_indicators.append(col)
                    
                    if cancer_indicators:
                        print(f"    🧬 Cancer indicators: {cancer_indicators}")
                    
                    # Check for IC50/cytotoxicity data
                    ic50_cols = [col for col in df.columns if 'ic50' in col.lower()]
                    cytotox_cols = [col for col in df.columns if 'cytotox' in col.lower() or 'viability' in col.lower()]
                    
                    if ic50_cols:
                        print(f"    💊 IC50 columns: {ic50_cols}")
                    if cytotox_cols:
                        print(f"    🔬 Cytotoxicity columns: {cytotox_cols}")
                        
                print()
                
            except Exception as e:
                print(f"    ❌ Error reading {file.name}: {e}")
        
        if not cytotox_files:
            print("  ❌ No cytotoxicity-related files found")
    
    else:
        print("❌ Expanded datasets volume not accessible")
    
    print("\n📁 CELL LINE MODELS VOLUME:")
    print("-" * 40)
    
    if cell_line_dir.exists():
        cell_files = list(cell_line_dir.glob("*"))
        print(f"Cell line files: {len(cell_files)}")
        
        for file in cell_files[:10]:  # Show first 10
            size = file.stat().st_size / (1024 * 1024)
            print(f"  📄 {file.name} ({size:.1f}MB)")
    else:
        print("❌ Cell line models volume not accessible")
    
    # Analysis of Model 2 requirements
    print("\n📋 MODEL 2 REQUIREMENTS ANALYSIS:")
    print("-" * 40)
    print("✅ REQUIRED: Cancer cell line IC50 data")
    print("✅ REQUIRED: SMILES structures") 
    print("✅ REQUIRED: Cell line genomic features (optional but preferred)")
    print("✅ TARGET: Cancer IC50 R² > 0.6")
    print("❌ AVOID: Normal cell cytotoxicity data")
    print()
    
    print("🎯 RECOMMENDATION:")
    if cytotox_files:
        print("- Found cytotoxicity files - need to verify if cancer-specific")
        print("- Check for GDSC data (gold standard for cancer cell lines)")
        print("- Ensure data has cancer cell line identifiers")
    else:
        print("- No cytotoxicity files found in Modal volumes")
        print("- Need to extract/upload GDSC cancer cell line data")
        print("- Consider using GDSC1 or GDSC2 public datasets")
    
    return {
        "cytotox_files_found": len(cytotox_files) if 'cytotox_files' in locals() else 0,
        "data_available": expanded_dir.exists(),
        "investigation_complete": True
    }

@app.local_entrypoint()
def main():
    result = investigate_model2_data.remote()
    print("\n🎉 INVESTIGATION COMPLETE")
    return result