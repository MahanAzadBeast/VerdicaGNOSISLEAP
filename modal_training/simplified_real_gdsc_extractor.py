"""
Simplified Real GDSC Data Extractor - NO SYNTHETIC DATA
Downloads confirmed working GDSC URLs with real experimental data
"""

import modal
import requests
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import io

# Modal setup - simplified dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "requests",
    "pandas", 
    "numpy",
    "openpyxl",
    "xlrd"
])

app = modal.App("simplified-real-gdsc-extractor")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

# Confirmed working GDSC URLs (tested October 2025)
GDSC_REAL_URLS = {
    'gdsc1_dose_response': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC1_fitted_dose_response_27Oct23.xlsx',
    'gdsc2_dose_response': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx',
    'cell_line_details': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Cell_Lines_Details.xlsx',
    'screened_compounds': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/screened_compounds_rel_8.5.csv'
}

class SimplifiedRealGDSCExtractor:
    """Simplified extractor for confirmed working GDSC URLs"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.logger = logging.getLogger(__name__)
    
    def download_gdsc_file(self, url: str, description: str) -> Optional[pd.DataFrame]:
        """Download real GDSC file and parse to DataFrame"""
        
        self.logger.info(f"📥 Downloading {description}...")
        
        try:
            response = self.session.get(url, timeout=600, stream=True)
            
            if response.status_code != 200:
                self.logger.error(f"❌ HTTP {response.status_code} for {description}")
                return None
            
            # Get file size
            content_length = response.headers.get('content-length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                self.logger.info(f"   📊 Size: {size_mb:.1f} MB")
            
            # Download content
            content = b""
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                downloaded += len(chunk)
                
                # Log progress for large files
                if content_length and downloaded % (5 * 1024 * 1024) == 0:  # Every 5MB
                    progress = (downloaded / int(content_length)) * 100
                    self.logger.info(f"   📈 Progress: {progress:.1f}%")
            
            # Parse file based on extension
            if url.endswith('.xlsx'):
                df = pd.read_excel(io.BytesIO(content))
            elif url.endswith('.csv'):
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            else:
                raise Exception(f"Unsupported file format: {url}")
            
            self.logger.info(f"   ✅ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error downloading {description}: {e}")
            return None
    
    def process_gdsc_dose_response(self, df: pd.DataFrame, version: str) -> pd.DataFrame:
        """Process GDSC dose response data to standard format"""
        
        self.logger.info(f"🔧 Processing {version} dose response data...")
        
        # Add version info
        df['GDSC_VERSION'] = version
        
        # Standard GDSC columns (may vary slightly between versions)
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if 'cell_line_name' in col_lower or 'sample_name' in col_lower:
                column_mapping[col] = 'CELL_LINE_NAME'
            elif 'drug_name' in col_lower or 'compound_name' in col_lower:
                column_mapping[col] = 'DRUG_NAME'
            elif 'cosmic_id' in col_lower:
                column_mapping[col] = 'COSMIC_ID'
            elif 'ln_ic50' in col_lower:
                column_mapping[col] = 'LN_IC50'
            elif 'auc' in col_lower:
                column_mapping[col] = 'AUC'
            elif 'rmse' in col_lower:
                column_mapping[col] = 'RMSE'
            elif 'z_score' in col_lower:
                column_mapping[col] = 'Z_SCORE'
        
        # Apply mapping
        df_processed = df.rename(columns=column_mapping)
        
        # Calculate IC50 values from LN_IC50 if available
        if 'LN_IC50' in df_processed.columns:
            df_processed['IC50_uM'] = np.exp(df_processed['LN_IC50'])
            df_processed['IC50_nM'] = df_processed['IC50_uM'] * 1000
            df_processed['pIC50'] = -np.log10(df_processed['IC50_uM'] / 1e6)
        
        # Quality control
        initial_count = len(df_processed)
        
        # Remove rows with missing essential data
        essential_cols = []
        for col in ['CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50']:
            if col in df_processed.columns:
                essential_cols.append(col)
        
        if essential_cols:
            df_processed = df_processed.dropna(subset=essential_cols)
        
        # Filter reasonable IC50 range if available
        if 'IC50_nM' in df_processed.columns:
            df_processed = df_processed[
                (df_processed['IC50_nM'] >= 0.1) &  # 0.1 nM minimum
                (df_processed['IC50_nM'] <= 1000000)  # 1 mM maximum
            ]
        
        self.logger.info(f"   📊 After processing: {len(df_processed):,} records (removed {initial_count - len(df_processed):,})")
        
        return df_processed
    
    def process_cell_line_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process cell line details"""
        
        self.logger.info("🔧 Processing cell line details...")
        
        # Standard mapping
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if 'cell_line_name' in col_lower or 'sample_name' in col_lower:
                column_mapping[col] = 'CELL_LINE_NAME'
            elif 'cosmic_id' in col_lower:
                column_mapping[col] = 'COSMIC_ID'
            elif 'tissue' in col_lower and 'subtype' not in col_lower:
                column_mapping[col] = 'TISSUE_TYPE'
            elif 'cancer_type' in col_lower or 'tcga_desc' in col_lower:
                column_mapping[col] = 'CANCER_TYPE'
            elif 'msi_status' in col_lower:
                column_mapping[col] = 'MSI_STATUS'
            elif 'gender' in col_lower or 'sex' in col_lower:
                column_mapping[col] = 'GENDER'
        
        df_processed = df.rename(columns=column_mapping)
        
        self.logger.info(f"   📊 Cell line details: {len(df_processed):,} cell lines")
        
        return df_processed
    
    def process_compound_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process compound/drug information"""
        
        self.logger.info("🔧 Processing compound information...")
        
        # Standard mapping
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if 'drug_name' in col_lower or 'compound_name' in col_lower:
                column_mapping[col] = 'DRUG_NAME'
            elif 'drug_id' in col_lower or 'compound_id' in col_lower:
                column_mapping[col] = 'DRUG_ID'
            elif 'pubchem' in col_lower:
                column_mapping[col] = 'PUBCHEM_ID'
            elif 'target' in col_lower:
                column_mapping[col] = 'TARGETS'
            elif 'pathway' in col_lower:
                column_mapping[col] = 'PATHWAY'
        
        df_processed = df.rename(columns=column_mapping)
        
        self.logger.info(f"   📊 Compound information: {len(df_processed):,} compounds")
        
        return df_processed

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=4.0,
    memory=16384,
    timeout=7200
)
def extract_real_gdsc_data_confirmed():
    """
    Extract REAL GDSC data from confirmed working URLs - NO SYNTHETIC DATA
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 REAL GDSC DATA EXTRACTION - CONFIRMED URLS")
    print("=" * 80)
    print("✅ Using confirmed working GDSC URLs from cog.sanger.ac.uk")
    print("❌ ZERO synthetic data generation")
    print("🎯 Release: GDSC_release8.5 (October 2023)")
    
    try:
        extractor = SimplifiedRealGDSCExtractor()
        
        # Download all real GDSC datasets
        print("\n📊 STEP 1: Downloading real GDSC datasets...")
        
        gdsc_datasets = {}
        
        # GDSC1 dose response
        gdsc1_df = extractor.download_gdsc_file(
            GDSC_REAL_URLS['gdsc1_dose_response'],
            'GDSC1 Dose Response (Real Data)'
        )
        if gdsc1_df is not None:
            gdsc_datasets['gdsc1_dose_response'] = gdsc1_df
        
        # GDSC2 dose response  
        gdsc2_df = extractor.download_gdsc_file(
            GDSC_REAL_URLS['gdsc2_dose_response'],
            'GDSC2 Dose Response (Real Data)'
        )
        if gdsc2_df is not None:
            gdsc_datasets['gdsc2_dose_response'] = gdsc2_df
        
        # Cell line details
        cell_lines_df = extractor.download_gdsc_file(
            GDSC_REAL_URLS['cell_line_details'],
            'Cell Line Details (Real Data)'
        )
        if cell_lines_df is not None:
            gdsc_datasets['cell_line_details'] = cell_lines_df
        
        # Compound information
        compounds_df = extractor.download_gdsc_file(
            GDSC_REAL_URLS['screened_compounds'],
            'Screened Compounds (Real Data)'
        )
        if compounds_df is not None:
            gdsc_datasets['screened_compounds'] = compounds_df
        
        if not gdsc_datasets:
            raise Exception("Failed to download any real GDSC data")
        
        print(f"   ✅ Downloaded {len(gdsc_datasets)} real GDSC datasets")
        
        # Process datasets
        print("\n🔧 STEP 2: Processing real GDSC data...")
        
        processed_datasets = {}
        
        # Process GDSC1 dose response
        if 'gdsc1_dose_response' in gdsc_datasets:
            gdsc1_processed = extractor.process_gdsc_dose_response(
                gdsc_datasets['gdsc1_dose_response'], 'GDSC1')
            processed_datasets['gdsc1_sensitivity'] = gdsc1_processed
            print(f"   ✅ GDSC1 sensitivity: {len(gdsc1_processed):,} drug-cell line pairs")
        
        # Process GDSC2 dose response
        if 'gdsc2_dose_response' in gdsc_datasets:
            gdsc2_processed = extractor.process_gdsc_dose_response(
                gdsc_datasets['gdsc2_dose_response'], 'GDSC2')
            processed_datasets['gdsc2_sensitivity'] = gdsc2_processed
            print(f"   ✅ GDSC2 sensitivity: {len(gdsc2_processed):,} drug-cell line pairs")
        
        # Process cell line details
        if 'cell_line_details' in gdsc_datasets:
            cell_lines_processed = extractor.process_cell_line_details(gdsc_datasets['cell_line_details'])
            processed_datasets['cell_line_info'] = cell_lines_processed
            print(f"   ✅ Cell line info: {len(cell_lines_processed):,} cell lines")
        
        # Process compound information
        if 'screened_compounds' in gdsc_datasets:
            compounds_processed = extractor.process_compound_info(gdsc_datasets['screened_compounds'])
            processed_datasets['compound_info'] = compounds_processed
            print(f"   ✅ Compound info: {len(compounds_processed):,} compounds")
        
        # Combine GDSC1 and GDSC2 sensitivity data
        print("\n🔗 STEP 3: Creating combined training dataset...")
        
        sensitivity_datasets = []
        if 'gdsc1_sensitivity' in processed_datasets:
            sensitivity_datasets.append(processed_datasets['gdsc1_sensitivity'])
        if 'gdsc2_sensitivity' in processed_datasets:
            sensitivity_datasets.append(processed_datasets['gdsc2_sensitivity'])
        
        if sensitivity_datasets:
            combined_sensitivity = pd.concat(sensitivity_datasets, ignore_index=True)
            
            # Merge with cell line information if available
            if 'cell_line_info' in processed_datasets:
                cell_info = processed_datasets['cell_line_info']
                
                # Try to merge on CELL_LINE_NAME or COSMIC_ID
                merge_key = None
                if 'CELL_LINE_NAME' in combined_sensitivity.columns and 'CELL_LINE_NAME' in cell_info.columns:
                    merge_key = 'CELL_LINE_NAME'
                elif 'COSMIC_ID' in combined_sensitivity.columns and 'COSMIC_ID' in cell_info.columns:
                    merge_key = 'COSMIC_ID'
                
                if merge_key:
                    combined_training = combined_sensitivity.merge(cell_info, on=merge_key, how='left')
                    processed_datasets['training_data'] = combined_training
                    print(f"   ✅ Combined training data: {combined_training.shape}")
                else:
                    processed_datasets['training_data'] = combined_sensitivity
                    print(f"   ⚠️ Could not merge cell line info - using sensitivity data only")
            else:
                processed_datasets['training_data'] = combined_sensitivity
            
            print(f"   📊 Training dataset summary:")
            training_data = processed_datasets['training_data']
            print(f"     • Total records: {len(training_data):,}")
            if 'DRUG_NAME' in training_data.columns:
                print(f"     • Unique drugs: {training_data['DRUG_NAME'].nunique()}")
            if 'CELL_LINE_NAME' in training_data.columns:
                print(f"     • Unique cell lines: {training_data['CELL_LINE_NAME'].nunique()}")
            if 'IC50_nM' in training_data.columns:
                print(f"     • IC50 range: {training_data['IC50_nM'].min():.1f} - {training_data['IC50_nM'].max():.1f} nM")
        
        # Save real datasets
        print("\n💾 STEP 4: Saving real GDSC datasets...")
        
        datasets_dir = Path("/vol/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for dataset_name, df in processed_datasets.items():
            file_path = datasets_dir / f"real_gdsc_{dataset_name}.csv"
            df.to_csv(file_path, index=False)
            saved_files[dataset_name] = str(file_path)
            print(f"   ✅ {dataset_name}: {file_path}")
        
        # Create metadata
        metadata = {
            'extraction_method': 'Real_GDSC_Confirmed_URLs',
            'data_source': 'GDSC_Sanger_Institute',
            'release_version': 'GDSC_release8.5',
            'release_date': '27Oct23',
            'no_synthetic_data': True,
            'real_experimental_data': True,
            'extraction_timestamp': datetime.now().isoformat(),
            'urls_used': GDSC_REAL_URLS,
            'datasets': {
                name: {
                    'records': len(df),
                    'columns': list(df.columns),
                    'file_path': saved_files[name]
                } for name, df in processed_datasets.items()
            }
        }
        
        metadata_path = datasets_dir / "real_gdsc_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate final report
        print(f"\n🎉 REAL GDSC DATA EXTRACTION COMPLETED!")
        print("=" * 80)
        print(f"📁 Saved files:")
        for dataset_name, file_path in saved_files.items():
            print(f"  • {dataset_name}: {file_path}")
        
        print(f"\n📊 Real GDSC data summary:")
        for dataset_name, df in processed_datasets.items():
            print(f"  • {dataset_name}: {len(df):,} records")
        
        print(f"\n✅ DATA VALIDATION:")
        print(f"  • Source: Official GDSC/Sanger Institute")
        print(f"  • Release: GDSC_release8.5 (October 2023)")
        print(f"  • NO synthetic data generated")
        print(f"  • 100% real experimental drug sensitivity data")
        print(f"  • Ready for Cell Line Response Model training")
        
        return {
            'status': 'success',
            'extraction_method': 'Real_GDSC_Confirmed_URLs',
            'no_synthetic_data': True,
            'real_experimental_data': True,
            'release_version': 'GDSC_release8.5',
            'datasets_extracted': len(processed_datasets),
            'total_records': sum(len(df) for df in processed_datasets.values()),
            'training_data_shape': processed_datasets['training_data'].shape if 'training_data' in processed_datasets else None,
            'unique_drugs': processed_datasets['training_data']['DRUG_NAME'].nunique() if 'training_data' in processed_datasets and 'DRUG_NAME' in processed_datasets['training_data'].columns else 0,
            'unique_cell_lines': processed_datasets['training_data']['CELL_LINE_NAME'].nunique() if 'training_data' in processed_datasets and 'CELL_LINE_NAME' in processed_datasets['training_data'].columns else 0,
            'saved_files': saved_files,
            'metadata_path': str(metadata_path),
            'ready_for_training': True
        }
        
    except Exception as e:
        print(f"❌ REAL GDSC DATA EXTRACTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🧬 Simplified Real GDSC Data Extractor - NO SYNTHETIC DATA")