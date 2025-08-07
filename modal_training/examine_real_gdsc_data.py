"""
Examine the specific GDSC dataset with real IC50 values
https://modal.com/api/volumes/mahanazad19/main/expanded-datasets/files/content?path=gdsc_dataset%2Fgdsc_sample_10k.csv
"""

import modal
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("examine-real-gdsc-dataset")

image = modal.Image.debian_slim().pip_install([
    "pandas==2.1.0", "numpy==1.24.3"
])

data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    timeout=900,
    volumes={"/vol": data_volume}
)
def examine_real_gdsc_data():
    """Examine the gdsc_sample_10k.csv file with real IC50 data"""
    
    logger.info("🎯 EXAMINING REAL GDSC DATASET")
    logger.info("File: gdsc_dataset/gdsc_sample_10k.csv")
    logger.info("Expected: Real compounds + SMILES + actual IC50 values")
    
    # Target file
    file_path = "/vol/gdsc_dataset/gdsc_sample_10k.csv"
    
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        logger.info(f"✅ Dataset loaded successfully")
        logger.info(f"Shape: {df.shape} (rows × columns)")
        
        # Display column information
        logger.info(f"Columns ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            logger.info(f"  {i+1:2d}. {col}")
        
        # Show first few rows
        logger.info(f"\n📊 FIRST 5 ROWS:")
        logger.info(df.head().to_string())
        
        # Look for key columns
        smiles_cols = []
        ic50_cols = []
        cell_line_cols = []
        compound_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            if any(term in col_lower for term in ['smiles', 'canonical']):
                smiles_cols.append(col)
            
            if any(term in col_lower for term in ['ic50', 'ic_50', 'pic50', 'pic_50']):
                ic50_cols.append(col)
                
            if any(term in col_lower for term in ['cell', 'line', 'cosmic', 'sample']):
                cell_line_cols.append(col)
                
            if any(term in col_lower for term in ['drug', 'compound', 'name']):
                compound_cols.append(col)
        
        logger.info(f"\n🎯 KEY COLUMNS IDENTIFIED:")
        logger.info(f"SMILES columns: {smiles_cols}")
        logger.info(f"IC50 columns: {ic50_cols}")
        logger.info(f"Cell line columns: {cell_line_cols}")
        logger.info(f"Compound columns: {compound_cols}")
        
        # Analyze SMILES data
        if smiles_cols:
            for smiles_col in smiles_cols:
                unique_smiles = df[smiles_col].dropna().nunique()
                total_smiles = df[smiles_col].notna().sum()
                logger.info(f"\n🧬 SMILES ANALYSIS - {smiles_col}:")
                logger.info(f"  Total records with SMILES: {total_smiles:,}")
                logger.info(f"  Unique SMILES: {unique_smiles:,}")
                logger.info(f"  Coverage: {total_smiles/len(df)*100:.1f}%")
                
                # Show some SMILES examples
                valid_smiles = df[smiles_col].dropna()
                if len(valid_smiles) > 0:
                    examples = valid_smiles.head(3).tolist()
                    logger.info(f"  Examples: {examples}")
        
        # Analyze IC50 data  
        if ic50_cols:
            for ic50_col in ic50_cols:
                valid_ic50 = df[ic50_col].dropna()
                logger.info(f"\n📈 IC50 ANALYSIS - {ic50_col}:")
                logger.info(f"  Total records with IC50: {len(valid_ic50):,}")
                logger.info(f"  IC50 range: {valid_ic50.min():.4f} - {valid_ic50.max():.4f}")
                logger.info(f"  IC50 mean: {valid_ic50.mean():.4f}")
                logger.info(f"  IC50 std: {valid_ic50.std():.4f}")
                
                # Check if values look like IC50 (should be positive, often in µM range)
                if valid_ic50.min() >= 0:
                    logger.info(f"  ✅ Values are positive (good for IC50)")
                else:
                    logger.info(f"  ⚠️  Contains negative values")
                
                # Check range - typical IC50 can be 0.001 to 100+ µM
                if valid_ic50.max() > 10:
                    logger.info(f"  ✅ Contains high values (typical IC50 range)")
                
        # Analyze cell line data
        if cell_line_cols:
            for cell_col in cell_line_cols:
                unique_cells = df[cell_col].nunique()
                total_cells = df[cell_col].notna().sum()
                logger.info(f"\n🧪 CELL LINE ANALYSIS - {cell_col}:")
                logger.info(f"  Total records with cell info: {total_cells:,}")
                logger.info(f"  Unique cell lines: {unique_cells:,}")
                
                # Show some cell line examples
                if total_cells > 0:
                    examples = df[cell_col].value_counts().head(5)
                    logger.info(f"  Top 5 cell lines:")
                    for cell, count in examples.items():
                        logger.info(f"    {cell}: {count} records")
        
        # Check data completeness
        logger.info(f"\n📊 DATA COMPLETENESS:")
        
        # Find rows with all key data
        has_smiles = df[smiles_cols[0]].notna() if smiles_cols else pd.Series([False]*len(df))
        has_ic50 = df[ic50_cols[0]].notna() if ic50_cols else pd.Series([False]*len(df))
        has_cell = df[cell_line_cols[0]].notna() if cell_line_cols else pd.Series([False]*len(df))
        
        complete_data = has_smiles & has_ic50 & has_cell
        complete_count = complete_data.sum()
        
        logger.info(f"Records with SMILES: {has_smiles.sum():,}")
        logger.info(f"Records with IC50: {has_ic50.sum():,}")
        logger.info(f"Records with cell line: {has_cell.sum():,}")
        logger.info(f"Records with ALL three: {complete_count:,}")
        logger.info(f"Completeness: {complete_count/len(df)*100:.1f}%")
        
        # Create training-ready dataset
        if complete_count > 0:
            logger.info(f"\n🎯 CREATING TRAINING DATASET:")
            
            # Extract complete records
            training_df = df[complete_data].copy()
            
            # Standardize column names
            if smiles_cols:
                training_df = training_df.rename(columns={smiles_cols[0]: 'SMILES'})
            if ic50_cols:
                training_df = training_df.rename(columns={ic50_cols[0]: 'IC50'})
            if cell_line_cols:
                training_df = training_df.rename(columns={cell_line_cols[0]: 'CELL_LINE'})
            
            # Check IC50 values and convert to pIC50 if needed
            ic50_values = training_df['IC50']
            
            # If IC50 values are large (>10), likely in nM, convert to µM then pIC50
            if ic50_values.mean() > 10:
                logger.info("Converting IC50 (nM) to pIC50...")
                # Convert nM to µM, then to pIC50: pIC50 = -log10(IC50_µM)
                training_df['pIC50'] = -np.log10(ic50_values / 1000)  # nM to µM conversion
            else:
                logger.info("Converting IC50 (µM) to pIC50...")
                # IC50 already in µM
                training_df['pIC50'] = -np.log10(ic50_values)
            
            # Remove invalid pIC50 values
            training_df = training_df[training_df['pIC50'].notna()]
            training_df = training_df[np.isfinite(training_df['pIC50'])]
            
            logger.info(f"Training dataset shape: {training_df.shape}")
            logger.info(f"pIC50 range: {training_df['pIC50'].min():.2f} - {training_df['pIC50'].max():.2f}")
            logger.info(f"Unique compounds: {training_df['SMILES'].nunique():,}")
            logger.info(f"Unique cell lines: {training_df['CELL_LINE'].nunique():,}")
            
            # Show sample of final data
            logger.info(f"\nFINAL TRAINING DATA SAMPLE:")
            sample_cols = ['SMILES', 'IC50', 'pIC50', 'CELL_LINE']
            available_cols = [col for col in sample_cols if col in training_df.columns]
            logger.info(training_df[available_cols].head(3).to_string())
            
            return {
                'success': True,
                'total_records': len(df),
                'training_records': len(training_df),
                'unique_compounds': training_df['SMILES'].nunique(),
                'unique_cell_lines': training_df['CELL_LINE'].nunique(),
                'pic50_range': (training_df['pIC50'].min(), training_df['pIC50'].max()),
                'columns': list(df.columns),
                'smiles_cols': smiles_cols,
                'ic50_cols': ic50_cols,
                'cell_line_cols': cell_line_cols,
                'ready_for_training': True
            }
        
        else:
            logger.warning("❌ No complete records found for training")
            return {
                'success': False,
                'error': 'No records with all required data (SMILES + IC50 + cell line)'
            }
            
    except Exception as e:
        logger.error(f"❌ Error examining dataset: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    with app.run():
        result = examine_real_gdsc_data.remote()
        
        print("\n" + "="*80)
        print("🔍 REAL GDSC DATASET ANALYSIS")
        print("="*80)
        
        if result.get('success'):
            print("✅ DATASET ANALYSIS SUCCESSFUL")
            print(f"📊 Total records: {result['total_records']:,}")
            print(f"🎯 Training-ready records: {result['training_records']:,}")
            print(f"🧬 Unique compounds: {result['unique_compounds']:,}")
            print(f"🧪 Unique cell lines: {result['unique_cell_lines']:,}")
            print(f"📈 pIC50 range: {result['pic50_range'][0]:.2f} - {result['pic50_range'][1]:.2f}")
            print(f"✅ Ready for ChemBERTa training: {result['ready_for_training']}")
            
            if result['unique_compounds'] >= 600:
                print("🎉 MEETS TARGET: >600 unique compounds!")
            else:
                print(f"⚠️  Below target: {result['unique_compounds']} < 600 compounds")
                print("But we have REAL IC50 data - much better than synthetic!")
                
        else:
            print("❌ ANALYSIS FAILED")
            print(f"Error: {result.get('error')}")
            
        print("\nNext step: Train ChemBERTa with this REAL data!")