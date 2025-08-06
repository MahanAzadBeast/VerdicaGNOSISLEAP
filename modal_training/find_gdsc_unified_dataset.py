"""
Find the gdsc_unified dataset that likely contains >600 compounds
Based on the Modal path showing "gdsc_unified 6"
"""

import modal
import pandas as pd
import os

app = modal.App("find-gdsc-unified")

image = modal.Image.debian_slim().pip_install(["pandas==2.1.0"])
data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    volumes={"/vol": data_volume},
    timeout=120
)
def find_gdsc_unified():
    """Find gdsc_unified or similar files with >600 compounds"""
    
    print("🔍 SEARCHING FOR gdsc_unified DATASET")
    print("=" * 50)
    
    vol_path = "/vol"
    
    if not os.path.exists(vol_path):
        return {"error": "Volume not accessible"}
    
    files = os.listdir(vol_path)
    print(f"All files: {files}")
    print()
    
    # Look specifically for gdsc_unified or similar patterns
    target_patterns = [
        'gdsc_unified',
        'gdsc_unique',
        'gdsc_drugs',
        'gdsc_compounds',
        'gdsc_sample',
        'unified',
        'drugs_with_smiles'
    ]
    
    candidate_files = []
    for file in files:
        file_lower = file.lower()
        if file.endswith('.csv'):
            for pattern in target_patterns:
                if pattern.lower() in file_lower:
                    candidate_files.append(file)
                    break
    
    print(f"📄 Candidate files: {candidate_files}")
    print()
    
    best_file = None
    max_compounds = 0
    
    # Check each candidate file for compound count
    for file in candidate_files:
        file_path = os.path.join(vol_path, file)
        print(f"🔍 Checking: {file}")
        
        try:
            # Load the file
            df = pd.read_csv(file_path)
            print(f"  Shape: {len(df):,} rows × {len(df.columns)} columns")
            
            # Look for SMILES or compound columns
            smiles_cols = []
            for col in df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['smiles', 'canonical', 'structure']):
                    smiles_cols.append(col)
            
            print(f"  SMILES columns: {smiles_cols}")
            
            for smiles_col in smiles_cols:
                unique_count = df[smiles_col].nunique()
                non_null = df[smiles_col].count()
                print(f"    {smiles_col}: {unique_count:,} unique, {non_null:,} non-null")
                
                if unique_count > max_compounds:
                    max_compounds = unique_count
                    best_file = {
                        'filename': file,
                        'smiles_column': smiles_col,
                        'unique_compounds': unique_count,
                        'total_rows': len(df),
                        'all_columns': list(df.columns)
                    }
                    print(f"    🏆 NEW BEST: {unique_count:,} compounds!")
                
                # Show sample SMILES to verify
                if unique_count > 0:
                    sample_smiles = df[smiles_col].dropna().head(2).tolist()
                    print(f"    Sample SMILES: {sample_smiles}")
            
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()
    
    # Also check the comprehensive file we know exists
    comprehensive_file = "gdsc_comprehensive_training_data.csv"
    if comprehensive_file in files and comprehensive_file not in candidate_files:
        print(f"🔍 Also checking: {comprehensive_file}")
        try:
            file_path = os.path.join(vol_path, comprehensive_file)
            df = pd.read_csv(file_path)
            
            # Look more thoroughly at all columns
            print(f"  All columns: {list(df.columns)}")
            
            # Check every column that might contain compounds
            for col in df.columns:
                unique_count = df[col].nunique()
                if unique_count > 50:  # Potentially interesting
                    print(f"    {col}: {unique_count:,} unique values")
                    
                    # Check if it might be SMILES
                    sample_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else ""
                    if isinstance(sample_val, str) and len(sample_val) > 10:
                        might_be_smiles = any(c in sample_val for c in ['C', 'N', 'O', '(', ')', '='])
                        if might_be_smiles and unique_count > max_compounds:
                            max_compounds = unique_count
                            best_file = {
                                'filename': comprehensive_file,
                                'smiles_column': col,
                                'unique_compounds': unique_count,
                                'total_rows': len(df),
                                'all_columns': list(df.columns)
                            }
                            print(f"      🏆 FOUND BETTER: {unique_count:,} compounds in {col}!")
        except Exception as e:
            print(f"  ❌ Error checking comprehensive file: {e}")
    
    print("🏆 FINAL RESULTS")
    print("-" * 30)
    print(f"Maximum compounds found: {max_compounds:,}")
    
    if best_file:
        print(f"Best file: {best_file['filename']}")
        print(f"SMILES column: {best_file['smiles_column']}")
        print(f"Unique compounds: {best_file['unique_compounds']:,}")
        print(f"Total rows: {best_file['total_rows']:,}")
        
        if best_file['unique_compounds'] >= 600:
            print("✅ MEETS >600 REQUIREMENT!")
        else:
            print("❌ Still below 600 compounds")
    
    return {
        'max_compounds': max_compounds,
        'best_dataset': best_file,
        'all_files': files
    }

if __name__ == "__main__":
    with app.run():
        result = find_gdsc_unified.remote()
        max_found = result.get('max_compounds', 0)
        print(f"✅ Search complete! Max compounds: {max_found:,}")
        
        if max_found >= 600:
            best = result.get('best_dataset')
            print(f"🎉 FOUND >600 COMPOUNDS!")
            print(f"File: {best['filename']}")
            print(f"Column: {best['smiles_column']}")
        else:
            print("⚠️ Still need to find dataset with >600 compounds")