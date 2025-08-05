"""
Real DepMap and GDSC Data Extractor - NO SYNTHETIC DATA
Implements user's specific instructions for real data download
"""

import modal
import requests
import pandas as pd
import numpy as np
import logging
import json
import time
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import io
from bs4 import BeautifulSoup

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "requests",
    "pandas", 
    "numpy",
    "beautifulsoup4",
    "synapseclient",
    "rdkit-pypi",
    "lxml"
])

app = modal.App("real-data-extractor-no-synthetic")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

# URLs and configuration
DEPMAP_DOWNLOAD_PAGE = "https://depmap.org/portal/download/"
GDSC_SYNAPSE_ID = "syn23466441"

class RealDataExtractor:
    """Real data extractor - NO SYNTHETIC DATA GENERATION"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.logger = logging.getLogger(__name__)
    
    def scrape_depmap_download_page(self) -> Optional[Dict[str, str]]:
        """Scrape DepMap download page for latest DepMap_Public_*.zip"""
        
        self.logger.info("🔍 Scraping DepMap download page...")
        
        try:
            response = self.session.get(DEPMAP_DOWNLOAD_PAGE, timeout=30)
            
            if response.status_code != 200:
                self.logger.error(f"❌ Failed to access DepMap download page: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for DepMap_Public download links
            depmap_links = []
            
            # Search for links containing "DepMap_Public" and ending with ".zip"
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'DepMap_Public' in href and href.endswith('.zip'):
                    depmap_links.append({
                        'url': href if href.startswith('http') else f"https://depmap.org{href}",
                        'text': link.get_text(strip=True)
                    })
            
            # Also check for download buttons or specific patterns
            for element in soup.find_all(['button', 'div', 'span'], class_=re.compile(r'download|button', re.I)):
                text = element.get_text(strip=True)
                if 'DepMap_Public' in text and '.zip' in text:
                    # Try to find associated link
                    parent = element.find_parent('a')
                    if parent and parent.get('href'):
                        depmap_links.append({
                            'url': parent['href'] if parent['href'].startswith('http') else f"https://depmap.org{parent['href']}",
                            'text': text
                        })
            
            if not depmap_links:
                # Try alternative approach - look for any ZIP files
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    text = link.get_text(strip=True)
                    if href.endswith('.zip') and any(keyword in text.lower() for keyword in ['depmap', 'public', '25q2', '24q4']):
                        depmap_links.append({
                            'url': href if href.startswith('http') else f"https://depmap.org{href}",
                            'text': text
                        })
            
            if depmap_links:
                # Find the latest release (look for version patterns like 25Q2, 24Q4, etc.)
                latest_link = None
                latest_version = ""
                
                for link in depmap_links:
                    # Extract version from text or URL
                    version_match = re.search(r'(\d{2}Q\d)', link['text'] + link['url'])
                    if version_match:
                        version = version_match.group(1)
                        if version > latest_version:
                            latest_version = version
                            latest_link = link
                
                if not latest_link:
                    latest_link = depmap_links[0]  # Take first one if no version found
                
                self.logger.info(f"   ✅ Found DepMap download: {latest_link['text']}")
                self.logger.info(f"   🔗 URL: {latest_link['url']}")
                
                return {
                    'url': latest_link['url'],
                    'filename': latest_link['text'],
                    'version': latest_version or 'unknown'
                }
            else:
                self.logger.error("❌ No DepMap_Public download links found")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error scraping DepMap download page: {e}")
            return None
    
    def download_depmap_data(self, download_info: Dict[str, str]) -> Optional[Dict[str, pd.DataFrame]]:
        """Download and extract DepMap data"""
        
        self.logger.info(f"📥 Downloading DepMap data: {download_info['filename']}")
        
        try:
            response = self.session.get(download_info['url'], timeout=1800, stream=True)  # 30 min timeout
            
            if response.status_code != 200:
                self.logger.error(f"❌ Download failed: HTTP {response.status_code}")
                return None
            
            # Get file size if available
            content_length = response.headers.get('content-length')
            if content_length:
                file_size_mb = int(content_length) / (1024 * 1024)
                self.logger.info(f"   📊 File size: {file_size_mb:.1f} MB")
            
            # Download in chunks
            content = b""
            chunk_size = 8192
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                content += chunk
                downloaded += len(chunk)
                if content_length and downloaded % (1024 * 1024) == 0:  # Log every MB
                    progress = (downloaded / int(content_length)) * 100
                    self.logger.info(f"   📈 Downloaded: {progress:.1f}%")
            
            self.logger.info(f"   ✅ Download completed: {len(content)} bytes")
            
            # Extract ZIP file
            datasets = {}
            
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                file_list = z.namelist()
                self.logger.info(f"   📋 Files in ZIP: {len(file_list)}")
                
                # Target files as specified by user
                target_files = [
                    'Drug_sensitivity.csv',
                    'Cell_line_annotations.csv', 
                    'sample_info.csv'
                ]
                
                for target_file in target_files:
                    found_file = None
                    
                    # Look for exact match first
                    if target_file in file_list:
                        found_file = target_file
                    else:
                        # Look for similar files (case insensitive, different naming)
                        for file_name in file_list:
                            if target_file.lower() in file_name.lower():
                                found_file = file_name
                                break
                    
                    if found_file:
                        self.logger.info(f"   📄 Extracting: {found_file}")
                        
                        with z.open(found_file) as f:
                            df = pd.read_csv(f)
                            datasets[target_file.replace('.csv', '')] = df
                            self.logger.info(f"     ✅ {target_file}: {df.shape[0]:,} rows, {df.shape[1]} columns")
                    else:
                        self.logger.warning(f"   ⚠️ Target file not found: {target_file}")
                
                # Log all available files for debugging
                self.logger.info("   📋 All files in ZIP:")
                for file_name in file_list[:20]:  # Show first 20 files
                    self.logger.info(f"     • {file_name}")
                if len(file_list) > 20:
                    self.logger.info(f"     ... and {len(file_list) - 20} more files")
            
            return datasets
            
        except Exception as e:
            self.logger.error(f"❌ Error downloading DepMap data: {e}")
            return None
    
    def download_gdsc_via_synapse(self) -> Optional[pd.DataFrame]:
        """Download GDSC data via Synapse client"""
        
        self.logger.info(f"🔬 Downloading GDSC data via Synapse: {GDSC_SYNAPSE_ID}")
        
        try:
            import synapseclient
            
            # Initialize Synapse client (anonymous access)
            syn = synapseclient.Synapse()
            
            # Try anonymous login first
            try:
                syn.login(silent=True)
                self.logger.info("   ✅ Synapse anonymous login successful")
            except Exception as e:
                self.logger.warning(f"   ⚠️ Synapse login failed: {e}")
                # Continue anyway, some datasets might be publicly accessible
            
            # Download the dataset
            entity = syn.get(GDSC_SYNAPSE_ID, downloadLocation='/tmp')
            
            self.logger.info(f"   ✅ Downloaded: {entity.path}")
            
            # Read the data
            if entity.path.endswith('.csv'):
                df = pd.read_csv(entity.path)
            elif entity.path.endswith('.xlsx'):
                df = pd.read_excel(entity.path)
            elif entity.path.endswith('.zip'):
                # Extract and read
                with zipfile.ZipFile(entity.path) as z:
                    # Look for CSV files
                    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                    if csv_files:
                        with z.open(csv_files[0]) as f:
                            df = pd.read_csv(f)
                    else:
                        raise Exception("No CSV files found in ZIP")
            else:
                raise Exception(f"Unsupported file format: {entity.path}")
            
            self.logger.info(f"   ✅ GDSC data: {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error downloading GDSC via Synapse: {e}")
            return None
    
    def process_depmap_drug_sensitivity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DepMap drug sensitivity data to standard format"""
        
        self.logger.info("🔧 Processing DepMap drug sensitivity data...")
        
        # Standard column mapping for DepMap drug sensitivity
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower()
            if 'depmap_id' in col_lower:
                column_mapping[col] = 'CELL_LINE_ID'
            elif 'cell_line' in col_lower:
                column_mapping[col] = 'CELL_LINE_NAME'
            elif 'compound' in col_lower or 'drug' in col_lower:
                column_mapping[col] = 'DRUG_NAME'
            elif 'ic50' in col_lower:
                column_mapping[col] = 'IC50'
            elif 'auc' in col_lower:
                column_mapping[col] = 'AUC'
            elif 'log_fold_change' in col_lower or 'viability' in col_lower:
                column_mapping[col] = 'LOG_FOLD_CHANGE'
        
        df_processed = df.rename(columns=column_mapping)
        
        # Quality control
        initial_count = len(df_processed)
        
        # Remove rows with missing essential data
        essential_cols = [col for col in ['CELL_LINE_ID', 'DRUG_NAME', 'IC50', 'LOG_FOLD_CHANGE'] if col in df_processed.columns]
        if essential_cols:
            df_processed = df_processed.dropna(subset=essential_cols[:2])  # At least cell line and drug
        
        # Calculate additional metrics if needed
        if 'LOG_FOLD_CHANGE' in df_processed.columns and 'IC50' not in df_processed.columns:
            # Convert log fold change to approximate IC50 values
            # This is a rough conversion for standardization
            df_processed['IC50_APPROX'] = 10 ** (-df_processed['LOG_FOLD_CHANGE'])
        
        self.logger.info(f"   📊 Processed: {len(df_processed):,} records (removed {initial_count - len(df_processed):,})")
        
        return df_processed
    
    def process_cell_line_annotations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process cell line annotations"""
        
        self.logger.info("🔧 Processing cell line annotations...")
        
        # Standard column mapping
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower()
            if 'depmap_id' in col_lower:
                column_mapping[col] = 'CELL_LINE_ID'
            elif 'cell_line' in col_lower or 'stripped_cell_line_name' in col_lower:
                column_mapping[col] = 'CELL_LINE_NAME'
            elif 'tissue' in col_lower or 'lineage' in col_lower:
                column_mapping[col] = 'TISSUE_TYPE'
            elif 'disease' in col_lower or 'cancer_type' in col_lower:
                column_mapping[col] = 'CANCER_TYPE'
        
        df_processed = df.rename(columns=column_mapping)
        
        self.logger.info(f"   📊 Cell line annotations: {len(df_processed):,} cell lines")
        
        return df_processed

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=4.0,
    memory=16384,
    timeout=7200
)
def extract_real_data_no_synthetic():
    """
    Extract REAL DepMap and GDSC data - NO SYNTHETIC DATA
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 REAL DATA EXTRACTION - NO SYNTHETIC DATA")
    print("=" * 80)
    print("🎯 DepMap: Scrape download page for DepMap_Public_*.zip")  
    print("🎯 GDSC: Download via Synapse syn23466441")
    print("❌ ZERO synthetic data generation")
    
    try:
        extractor = RealDataExtractor()
        
        # STEP 1: DepMap data extraction
        print("\n📊 STEP 1: DepMap data extraction...")
        
        # Scrape download page
        download_info = extractor.scrape_depmap_download_page()
        
        if not download_info:
            raise Exception("Failed to find DepMap download links")
        
        print(f"   ✅ Found DepMap release: {download_info['version']}")
        print(f"   📁 File: {download_info['filename']}")
        
        # Download and extract
        depmap_datasets = extractor.download_depmap_data(download_info)
        
        if not depmap_datasets:
            raise Exception("Failed to download DepMap data")
        
        print(f"   ✅ Extracted {len(depmap_datasets)} DepMap datasets")
        
        # STEP 2: GDSC data extraction via Synapse
        print("\n🔬 STEP 2: GDSC data extraction via Synapse...")
        
        gdsc_df = extractor.download_gdsc_via_synapse()
        
        if gdsc_df is not None:
            print(f"   ✅ GDSC data: {len(gdsc_df):,} records")
        else:
            print("   ⚠️ GDSC data download failed - continuing with DepMap only")
        
        # STEP 3: Process and standardize data
        print("\n🔧 STEP 3: Processing real data...")
        
        processed_datasets = {}
        
        # Process DepMap drug sensitivity
        if 'Drug_sensitivity' in depmap_datasets:
            drug_sensitivity_df = extractor.process_depmap_drug_sensitivity(depmap_datasets['Drug_sensitivity'])
            processed_datasets['drug_sensitivity'] = drug_sensitivity_df
            print(f"   ✅ Drug sensitivity: {len(drug_sensitivity_df):,} records")
        
        # Process cell line annotations
        if 'Cell_line_annotations' in depmap_datasets:
            cell_annotations_df = extractor.process_cell_line_annotations(depmap_datasets['Cell_line_annotations'])
            processed_datasets['cell_annotations'] = cell_annotations_df
            print(f"   ✅ Cell annotations: {len(cell_annotations_df):,} cell lines")
        
        # Process sample info
        if 'sample_info' in depmap_datasets:
            processed_datasets['sample_info'] = depmap_datasets['sample_info']
            print(f"   ✅ Sample info: {len(depmap_datasets['sample_info']):,} samples")
        
        # Add GDSC data if available
        if gdsc_df is not None:
            processed_datasets['gdsc_data'] = gdsc_df
            print(f"   ✅ GDSC data: {len(gdsc_df):,} records")
        
        # STEP 4: Create integrated training dataset
        print("\n🔗 STEP 4: Creating integrated training dataset...")
        
        if 'drug_sensitivity' in processed_datasets and 'cell_annotations' in processed_datasets:
            # Merge drug sensitivity with cell line annotations
            drug_df = processed_datasets['drug_sensitivity']
            cell_df = processed_datasets['cell_annotations']
            
            # Merge on cell line ID
            merge_key = None
            if 'CELL_LINE_ID' in drug_df.columns and 'CELL_LINE_ID' in cell_df.columns:
                merge_key = 'CELL_LINE_ID'
            elif 'CELL_LINE_NAME' in drug_df.columns and 'CELL_LINE_NAME' in cell_df.columns:
                merge_key = 'CELL_LINE_NAME'
            
            if merge_key:
                integrated_df = drug_df.merge(cell_df, on=merge_key, how='left')
                processed_datasets['integrated_training'] = integrated_df
                print(f"   ✅ Integrated dataset: {integrated_df.shape}")
            else:
                print("   ⚠️ Could not merge datasets - different key columns")
                processed_datasets['integrated_training'] = drug_df
        
        # STEP 5: Save real datasets
        print("\n💾 STEP 5: Saving real datasets...")
        
        datasets_dir = Path("/vol/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for dataset_name, df in processed_datasets.items():
            file_path = datasets_dir / f"real_{dataset_name}.csv"
            df.to_csv(file_path, index=False)
            saved_files[dataset_name] = str(file_path)
            print(f"   ✅ {dataset_name}: {file_path}")
        
        # Create metadata
        metadata = {
            'extraction_method': 'Real_Data_Only_No_Synthetic',
            'data_sources': {
                'depmap': {
                    'version': download_info['version'],
                    'filename': download_info['filename'],
                    'url': download_info['url'],
                    'datasets_extracted': list(depmap_datasets.keys()) if depmap_datasets else []
                },
                'gdsc': {
                    'synapse_id': GDSC_SYNAPSE_ID,
                    'available': gdsc_df is not None,
                    'records': len(gdsc_df) if gdsc_df is not None else 0
                }
            },
            'no_synthetic_data': True,
            'real_experimental_data': True,
            'extraction_timestamp': datetime.now().isoformat(),
            'datasets': {
                name: {
                    'records': len(df),
                    'columns': list(df.columns),
                    'file_path': saved_files[name]
                } for name, df in processed_datasets.items()
            }
        }
        
        metadata_path = datasets_dir / "real_data_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate final report
        print(f"\n🎉 REAL DATA EXTRACTION COMPLETED!")
        print("=" * 80)
        print(f"📁 Saved files:")
        for dataset_name, file_path in saved_files.items():
            print(f"  • {dataset_name}: {file_path}")
        
        print(f"\n📊 Real data summary:")
        for dataset_name, df in processed_datasets.items():
            print(f"  • {dataset_name}: {len(df):,} records, {len(df.columns)} columns")
        
        print(f"\n✅ DATA SOURCES:")
        print(f"  • DepMap version: {download_info['version']}")
        print(f"  • DepMap datasets: {len(depmap_datasets)}")
        if gdsc_df is not None:
            print(f"  • GDSC records: {len(gdsc_df):,}")
        print(f"  • NO synthetic data generated")
        print(f"  • 100% real experimental data")
        
        return {
            'status': 'success',
            'extraction_method': 'Real_Data_Only_No_Synthetic',
            'no_synthetic_data': True,
            'real_experimental_data': True,
            'depmap_version': download_info['version'],
            'depmap_datasets': len(depmap_datasets),
            'gdsc_available': gdsc_df is not None,
            'gdsc_records': len(gdsc_df) if gdsc_df is not None else 0,
            'total_datasets': len(processed_datasets),
            'saved_files': saved_files,
            'metadata_path': str(metadata_path),
            'ready_for_training': True
        }
        
    except Exception as e:
        print(f"❌ REAL DATA EXTRACTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🧬 Real Data Extractor - NO SYNTHETIC DATA")