"""
Download the real GDSC dataset and train Model 2 with ChemBERTa transfer learning
Using ONLY real experimental IC50 data from GDSC
"""

import modal
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("download-real-gdsc-for-training")

image = modal.Image.debian_slim().pip_install([
    "pandas==2.1.0", "numpy==1.24.3"
])

data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    timeout=600,
    volumes={"/vol": data_volume}
)
def download_real_gdsc_data():
    """Download the real GDSC dataset to local for training"""
    
    logger.info("📁 Downloading REAL GDSC dataset...")
    
    gdsc_path = "/vol/gdsc_dataset/gdsc_sample_10k.csv"
    
    try:
        # Load the real GDSC dataset
        df = pd.read_csv(gdsc_path)
        logger.info(f"✅ Real GDSC data loaded: {df.shape}")
        
        # Display columns for verification
        logger.info(f"Columns: {list(df.columns)}")
        
        # Find key columns
        smiles_col = None
        ic50_col = None
        cell_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if not smiles_col and any(term in col_lower for term in ['smiles', 'canonical']):
                smiles_col = col
                logger.info(f"✅ Found SMILES column: {col}")
            if not ic50_col and any(term in col_lower for term in ['ic50', 'ic_50']):
                ic50_col = col
                logger.info(f"✅ Found IC50 column: {col}")
            if not cell_col and any(term in col_lower for term in ['cell', 'line', 'cosmic', 'sample']):
                cell_col = col
                logger.info(f"✅ Found cell line column: {col}")
        
        if not all([smiles_col, ic50_col, cell_col]):
            logger.error("❌ Missing required columns")
            return {"success": False, "error": "Missing columns"}
        
        # Filter to complete records only
        complete_mask = (
            df[smiles_col].notna() & 
            df[ic50_col].notna() & 
            df[cell_col].notna()
        )
        
        clean_df = df[complete_mask].copy()
        logger.info(f"Complete records: {len(clean_df):,}")
        logger.info(f"Unique compounds: {clean_df[smiles_col].nunique()}")
        logger.info(f"Unique cell lines: {clean_df[cell_col].nunique()}")
        
        # Standardize column names
        clean_df = clean_df.rename(columns={
            smiles_col: 'SMILES',
            ic50_col: 'IC50',
            cell_col: 'CELL_LINE'
        })
        
        # Show sample data
        logger.info("Sample data:")
        logger.info(clean_df[['SMILES', 'IC50', 'CELL_LINE']].head().to_string())
        
        # Convert to JSON for transfer
        data_dict = {
            'data': clean_df.to_dict('records'),
            'metadata': {
                'total_records': len(clean_df),
                'unique_compounds': clean_df['SMILES'].nunique(),
                'unique_cell_lines': clean_df['CELL_LINE'].nunique(),
                'ic50_range': [float(clean_df['IC50'].min()), float(clean_df['IC50'].max())],
                'source': 'Real GDSC experimental data'
            }
        }
        
        return {"success": True, "data": data_dict}
        
    except Exception as e:
        logger.error(f"❌ Error loading GDSC data: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    with app.run():
        result = download_real_gdsc_data.remote()
        
        if result.get('success'):
            data_dict = result['data']
            metadata = data_dict['metadata']
            
            print("✅ REAL GDSC DATA DOWNLOADED")
            print(f"Total records: {metadata['total_records']:,}")
            print(f"Unique compounds: {metadata['unique_compounds']}")
            print(f"Unique cell lines: {metadata['unique_cell_lines']}")
            print(f"IC50 range: {metadata['ic50_range']}")
            
            # Save to local file for training
            import json
            with open('/app/modal_training/real_gdsc_data.json', 'w') as f:
                json.dump(data_dict, f, indent=2)
            
            print("💾 Saved to: /app/modal_training/real_gdsc_data.json")
            print("🎯 Ready for ChemBERTa transfer learning!")
            
        else:
            print("❌ Download failed:")
            print(f"Error: {result.get('error')}")