"""
Examine the gdsc_comprehensive_training_data.csv file specifically
Find why it only shows 15 unique SMILES when it should have >600
"""

import modal
import pandas as pd

app = modal.App("examine-gdsc-comprehensive")

image = modal.Image.debian_slim().pip_install(["pandas==2.1.0"])
data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    volumes={"/vol": data_volume},
    timeout=180
)
def examine_gdsc_comprehensive_specifically():
    """Examine the specific GDSC comprehensive file"""
    
    print("🔍 EXAMINING gdsc_comprehensive_training_data.csv SPECIFICALLY")
    print("=" * 60)
    
    file_path = "/vol/gdsc_comprehensive_training_data.csv"
    
    try:
        # Load the file
        df = pd.read_csv(file_path)
        print(f"✅ Loaded successfully: {len(df):,} rows × {len(df.columns)} columns")
        print()
        
        # Show ALL columns
        print(f"📋 ALL {len(df.columns)} COLUMNS:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        print()
        
        # Find ALL potential SMILES columns
        smiles_candidates = []
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['smiles', 'canonical', 'structure', 'mol', 'compound']):
                smiles_candidates.append(col)
        
        print(f"🧬 POTENTIAL SMILES COLUMNS ({len(smiles_candidates)}):")
        for i, col in enumerate(smiles_candidates, 1):
            print(f"{i}. {col}")
            
            # Detailed analysis of each SMILES column
            unique_count = df[col].nunique()
            non_null_count = df[col].count()
            null_count = df[col].isnull().sum()
            
            print(f"   Unique values: {unique_count:,}")
            print(f"   Non-null values: {non_null_count:,}")
            print(f"   Null values: {null_count:,}")
            print(f"   Data type: {df[col].dtype}")
            
            if unique_count > 0:
                # Show sample values
                sample_values = df[col].dropna().head(5).tolist()
                print(f"   Sample values:")
                for j, val in enumerate(sample_values, 1):
                    val_str = str(val)[:50]  # Truncate long SMILES
                    print(f"     {j}. {val_str}")
                
                # Check if values look like SMILES
                if unique_count > 0:
                    first_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    if first_val:
                        val_str = str(first_val)
                        looks_like_smiles = any(char in val_str for char in ['C', 'N', 'O', '(', ')', '=', '[', ']'])
                        print(f"   Looks like SMILES: {'✅ YES' if looks_like_smiles else '❌ NO'}")
            print()
        
        # If no obvious SMILES columns, show sample data
        if not smiles_candidates:
            print("❌ No obvious SMILES columns found!")
            print("📋 Showing first 3 rows for inspection:")
            print(df.head(3))
        
        # Look for the column with most unique values that could be compounds
        print("📊 COLUMNS WITH MOST UNIQUE VALUES (potential compound identifiers):")
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count > 100:  # Potentially interesting columns
                print(f"   {col}: {unique_count:,} unique values")
                sample_vals = df[col].unique()[:3]
                print(f"     Sample: {sample_vals}")
        print()
        
        # Look for drug/compound name columns
        drug_candidates = []
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['drug', 'compound', 'name', 'id']):
                drug_candidates.append(col)
        
        if drug_candidates:
            print(f"💊 POTENTIAL DRUG/COMPOUND COLUMNS ({len(drug_candidates)}):")
            for col in drug_candidates:
                unique_count = df[col].nunique()
                print(f"   {col}: {unique_count:,} unique values")
                if unique_count <= 20:
                    sample_vals = df[col].unique()
                    print(f"     All values: {sample_vals}")
                else:
                    sample_vals = df[col].unique()[:5]
                    print(f"     Sample: {sample_vals}")
        
        print("=" * 60)
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'smiles_candidates': smiles_candidates,
            'drug_candidates': drug_candidates
        }
        
    except Exception as e:
        print(f"❌ Error examining file: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    with app.run():
        result = examine_gdsc_comprehensive_specifically.remote()
        print("✅ Examination complete!")