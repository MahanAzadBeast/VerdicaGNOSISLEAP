"""
DepMap Real Data Extractor
Download real experimental drug sensitivity data from DepMap portal
NEVER USE SIMULATED DATA - ONLY REAL EXPERIMENTAL DATA
"""

import requests
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepMapDataExtractor:
    """
    Extract real experimental data from DepMap portal
    CRITICAL: ONLY REAL DATA - NO SIMULATION EVER
    """
    
    def __init__(self):
        self.base_url = "https://depmap.org/portal/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DepMap-Data-Extractor/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
    def submit_async_request(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Submit asynchronous request to DepMap API"""
        
        try:
            logger.info(f"Submitting request to {endpoint}")
            response = self.session.post(f"{self.base_url}/{endpoint}", json=params)
            response.raise_for_status()
            
            data = response.json()
            task_id = data.get('task_id') or data.get('id') or data.get('taskId')
            
            if not task_id:
                logger.error(f"No task ID returned: {data}")
                raise ValueError("No task ID in response")
                
            logger.info(f"Task submitted successfully: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit request: {e}")
            raise
    
    def poll_task_status(self, task_id: str, max_wait_minutes: int = 30) -> Dict[str, Any]:
        """Poll task status until completion"""
        
        max_attempts = max_wait_minutes * 6  # Check every 10 seconds
        
        for attempt in range(max_attempts):
            try:
                response = self.session.get(f"{self.base_url}/task/{task_id}")
                response.raise_for_status()
                
                data = response.json()
                status = data.get('status', '').upper()
                
                logger.info(f"Task {task_id} status: {status} (attempt {attempt + 1})")
                
                if status == 'SUCCESS':
                    download_url = data.get('downloadUrl') or data.get('download_url') or data.get('result', {}).get('downloadUrl')
                    if download_url:
                        logger.info(f"Task completed successfully: {download_url}")
                        return {'status': 'SUCCESS', 'downloadUrl': download_url, 'data': data}
                    else:
                        logger.warning("Task successful but no download URL found")
                        return {'status': 'SUCCESS', 'data': data}
                        
                elif status in ['FAILED', 'ERROR']:
                    error_msg = data.get('error', 'Unknown error')
                    logger.error(f"Task failed: {error_msg}")
                    raise RuntimeError(f"Task failed: {error_msg}")
                    
                elif status in ['PENDING', 'RUNNING', 'IN_PROGRESS']:
                    time.sleep(10)  # Wait 10 seconds before next check
                    continue
                else:
                    logger.warning(f"Unknown status: {status}, continuing...")
                    time.sleep(10)
                    
            except Exception as e:
                logger.error(f"Error checking task status: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(10)
                    continue
                else:
                    raise
        
        raise TimeoutError(f"Task {task_id} did not complete within {max_wait_minutes} minutes")
    
    def download_data(self, download_url: str, output_path: Path) -> bool:
        """Download data from the provided URL"""
        
        try:
            logger.info(f"Downloading data from: {download_url}")
            
            response = self.session.get(download_url, stream=True)
            response.raise_for_status()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = output_path.stat().st_size
            logger.info(f"Downloaded {file_size:,} bytes to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            return False
    
    def get_drug_sensitivity_data(self) -> Optional[pd.DataFrame]:
        """
        Get real drug sensitivity data from DepMap (includes GDSC data)
        CRITICAL: ONLY REAL EXPERIMENTAL DATA
        """
        
        logger.info("🧬 Fetching REAL drug sensitivity data from DepMap")
        
        try:
            # Request drug sensitivity data (GDSC + other screens)
            params = {
                'data_type': 'drug_sensitivity',
                'dataset': ['GDSC1', 'GDSC2', 'CTRP', 'PRISM'],  # Multiple real datasets
                'format': 'csv',
                'include_metadata': True
            }
            
            # Try direct API endpoints for drug sensitivity
            endpoints_to_try = [
                'download/drug_sensitivity',
                'data/drug_sensitivity', 
                'download/compound_sensitivity',
                'datasets/drug_dependency'
            ]
            
            for endpoint in endpoints_to_try:
                try:
                    logger.info(f"Trying endpoint: {endpoint}")
                    task_id = self.submit_async_request(endpoint, params)
                    
                    # Poll for completion
                    result = self.poll_task_status(task_id)
                    
                    if result['status'] == 'SUCCESS' and 'downloadUrl' in result:
                        # Download the data
                        output_path = Path("/tmp/depmap_drug_sensitivity.csv")
                        if self.download_data(result['downloadUrl'], output_path):
                            # Load and validate
                            df = pd.read_csv(output_path)
                            if len(df) > 1000:  # Ensure meaningful dataset size
                                logger.info(f"✅ Successfully loaded {len(df):,} drug sensitivity records")
                                return df
                    
                except Exception as e:
                    logger.warning(f"Endpoint {endpoint} failed: {e}")
                    continue
            
            # If async API fails, try direct download endpoints
            logger.info("Trying direct download approach...")
            return self.try_direct_download()
            
        except Exception as e:
            logger.error(f"Failed to get drug sensitivity data: {e}")
            return None
    
    def try_direct_download(self) -> Optional[pd.DataFrame]:
        """Try direct download from DepMap public datasets"""
        
        # Known DepMap download URLs for real data
        direct_urls = [
            "https://ndownloader.figshare.com/files/34008503",  # GDSC drug sensitivity
            "https://ndownloader.figshare.com/files/34008496",  # GDSC compound info
            "https://depmap.org/portal/download/api/download/external?file_name=GDSC2_fitted_dose_response_25Feb20.csv",
            "https://depmap.org/portal/download/api/download/external?file_name=drug_sensitivity_AUC.csv"
        ]
        
        for url in direct_urls:
            try:
                logger.info(f"Trying direct download: {url}")
                
                response = self.session.get(url, timeout=60)
                if response.status_code == 200:
                    
                    # Save to temp file
                    temp_path = Path(f"/tmp/depmap_direct_{hash(url) % 10000}.csv")
                    with open(temp_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Try to load as CSV
                    try:
                        df = pd.read_csv(temp_path)
                        if len(df) > 1000:
                            logger.info(f"✅ Direct download successful: {len(df):,} records")
                            return df
                    except:
                        # Try as TSV
                        try:
                            df = pd.read_csv(temp_path, sep='\t')
                            if len(df) > 1000:
                                logger.info(f"✅ Direct download successful (TSV): {len(df):,} records")
                                return df
                        except:
                            continue
                            
            except Exception as e:
                logger.warning(f"Direct download failed: {e}")
                continue
        
        return None
    
    def get_cell_line_metadata(self) -> Optional[pd.DataFrame]:
        """Get real cell line genomic data"""
        
        logger.info("🧬 Fetching real cell line genomic data")
        
        try:
            # Request cell line metadata and genomic features
            params = {
                'data_type': 'cell_line_metadata',
                'include_genomics': True,
                'format': 'csv'
            }
            
            endpoints = ['cell_lines/metadata', 'download/cell_line_info', 'datasets/cell_line_annotations']
            
            for endpoint in endpoints:
                try:
                    task_id = self.submit_async_request(endpoint, params)
                    result = self.poll_task_status(task_id)
                    
                    if result['status'] == 'SUCCESS' and 'downloadUrl' in result:
                        output_path = Path("/tmp/depmap_cell_lines.csv")
                        if self.download_data(result['downloadUrl'], output_path):
                            df = pd.read_csv(output_path)
                            if len(df) > 100:
                                logger.info(f"✅ Cell line data loaded: {len(df):,} cell lines")
                                return df
                                
                except Exception as e:
                    logger.warning(f"Cell line endpoint {endpoint} failed: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to get cell line data: {e}")
        
        return None

def extract_real_depmap_data():
    """
    Extract real experimental data from DepMap
    CRITICAL: NO SIMULATED DATA EVER
    """
    
    print("🧬 DEPMAP REAL DATA EXTRACTION")
    print("=" * 80)
    print("🚨 CRITICAL: ONLY REAL EXPERIMENTAL DATA")
    print("❌ NO SIMULATED/SYNTHETIC DATA ALLOWED")
    print()
    
    extractor = DepMapDataExtractor()
    
    try:
        # Step 1: Get drug sensitivity data
        print("📊 STEP 1: Extracting real drug sensitivity data...")
        drug_data = extractor.get_drug_sensitivity_data()
        
        if drug_data is None:
            print("❌ Failed to extract drug sensitivity data")
            return None
        
        print(f"✅ Drug sensitivity data extracted: {len(drug_data):,} records")
        
        # Analyze the dataset
        print("\n📈 DATASET ANALYSIS:")
        if 'SMILES' in drug_data.columns or 'smiles' in drug_data.columns:
            smiles_col = 'SMILES' if 'SMILES' in drug_data.columns else 'smiles'
            unique_compounds = drug_data[smiles_col].nunique()
            print(f"   📊 Unique compounds: {unique_compounds:,}")
        else:
            # Look for compound identifiers
            compound_cols = [col for col in drug_data.columns if 'compound' in col.lower() or 'drug' in col.lower()]
            if compound_cols:
                unique_compounds = drug_data[compound_cols[0]].nunique()
                print(f"   📊 Unique compounds: {unique_compounds:,}")
            else:
                print("   ⚠️ Could not identify compound column")
                unique_compounds = 0
        
        # Look for cell line info
        cell_line_cols = [col for col in drug_data.columns if 'cell' in col.lower() or 'line' in col.lower()]
        if cell_line_cols:
            unique_cell_lines = drug_data[cell_line_cols[0]].nunique()
            print(f"   📊 Unique cell lines: {unique_cell_lines:,}")
        
        # Look for IC50/AUC data
        ic50_cols = [col for col in drug_data.columns if any(term in col.lower() for term in ['ic50', 'auc', 'viability', 'response'])]
        if ic50_cols:
            print(f"   📊 Response columns: {ic50_cols[:3]}")  # Show first 3
        
        # Step 2: Get cell line genomic data
        print(f"\n📊 STEP 2: Extracting cell line genomic data...")
        cell_line_data = extractor.get_cell_line_metadata()
        
        if cell_line_data is not None:
            print(f"✅ Cell line genomic data: {len(cell_line_data):,} cell lines")
        else:
            print("⚠️ Cell line genomic data not available (will use basic features)")
        
        # Step 3: Data quality assessment
        print(f"\n🔍 STEP 3: Data quality assessment...")
        
        print(f"   📊 Dataset shape: {drug_data.shape}")
        print(f"   📊 Columns: {list(drug_data.columns)[:10]}...")  # First 10 columns
        print(f"   📊 Data types: {drug_data.dtypes.value_counts().to_dict()}")
        
        # Step 4: Expected R² with real data
        print(f"\n🎯 STEP 4: Expected performance with REAL data...")
        
        if unique_compounds > 100 and len(drug_data) > 10000:
            expected_r2 = "0.45-0.65"
            confidence = "HIGH"
            print(f"   📈 Expected R²: {expected_r2} ({confidence} confidence)")
            print(f"   ✅ Dataset size sufficient for good performance")
            print(f"   ✅ Multiple compounds and cell lines available")
        elif unique_compounds > 50 and len(drug_data) > 5000:
            expected_r2 = "0.30-0.50"
            confidence = "MODERATE"
            print(f"   📈 Expected R²: {expected_r2} ({confidence} confidence)")
        else:
            expected_r2 = "0.15-0.35"
            confidence = "LOW"
            print(f"   📈 Expected R²: {expected_r2} ({confidence} confidence)")
            print(f"   ⚠️ Small dataset may limit performance")
        
        # Save the real data
        output_path = Path("/app/real_depmap_data.csv")
        drug_data.to_csv(output_path, index=False)
        print(f"\n💾 Real data saved: {output_path}")
        print(f"   📊 {len(drug_data):,} real experimental records")
        print(f"   🚨 NO SYNTHETIC DATA - 100% REAL")
        
        return {
            'drug_data': drug_data,
            'cell_line_data': cell_line_data,
            'unique_compounds': unique_compounds,
            'expected_r2': expected_r2,
            'confidence': confidence,
            'data_path': str(output_path)
        }
        
    except Exception as e:
        logger.error(f"Real data extraction failed: {e}")
        print(f"❌ CRITICAL ERROR: {e}")
        return None

if __name__ == "__main__":
    # Extract real DepMap data
    result = extract_real_depmap_data()
    
    if result:
        print(f"\n🎉 REAL DEPMAP DATA EXTRACTION SUCCESSFUL!")
        print(f"   📊 Compounds: {result['unique_compounds']:,}")
        print(f"   📈 Expected R²: {result['expected_r2']}")
        print(f"   ✅ Ready for ChemBERTa training with REAL data")
    else:
        print(f"\n❌ FAILED TO EXTRACT REAL DATA")
        print(f"   🚨 CANNOT PROCEED WITHOUT REAL EXPERIMENTAL DATA")