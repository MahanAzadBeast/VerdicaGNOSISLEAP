"""
Real BindingDB API Extractor
Uses BindingDB RESTful API for IC50/Ki/Kd binding affinity data
"""

import modal
import requests
import pandas as pd
import numpy as np
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "requests",
    "pandas", 
    "numpy",
    "rdkit-pypi"
])

app = modal.App("real-bindingdb-api-extractor")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

# BindingDB API endpoints
BINDINGDB_API_BASE = "https://bindingdb.org/rest"
BINDINGDB_ENDPOINTS = {
    'by_uniprot': f"{BINDINGDB_API_BASE}/getLigandsByUniprot",
    'by_uniprots': f"{BINDINGDB_API_BASE}/getLigandsByUniprots", 
    'by_pdbs': f"{BINDINGDB_API_BASE}/getLigandsByPDBs",
    'by_compound': "https://bindingdb.org/axis2/services/BDBService/getTargetByCompound"
}

# Oncology protein targets (UniProt IDs)
ONCOLOGY_TARGETS = {
    'EGFR': 'P00533',      # Epidermal Growth Factor Receptor
    'ERBB2': 'P04626',     # HER2/neu
    'ERBB3': 'P21860',     # HER3
    'ERBB4': 'Q15303',     # HER4
    'VEGFR1': 'P17948',    # VEGF Receptor 1
    'VEGFR2': 'P35968',    # VEGF Receptor 2
    'VEGFR3': 'P35916',    # VEGF Receptor 3
    'PDGFRA': 'P16234',    # PDGF Receptor Alpha
    'PDGFRB': 'P09619',    # PDGF Receptor Beta
    'KIT': 'P10721',       # c-Kit
    'FLT3': 'P36888',      # FLT3
    'RET': 'P07949',       # RET proto-oncogene
    'ALK': 'Q9UM73',       # Anaplastic Lymphoma Kinase
    'ROS1': 'P08922',      # ROS1 proto-oncogene
    'MET': 'P08581',       # Hepatocyte Growth Factor Receptor
    'AXL': 'P30530',       # AXL receptor tyrosine kinase
    'BRAF': 'P15056',      # B-Raf proto-oncogene
    'RAF1': 'P04049',      # Raf-1 proto-oncogene
    'PIK3CA': 'P42336',    # PI3K catalytic subunit alpha
    'AKT1': 'P31749',      # AKT1 kinase
    'MTOR': 'P42345',      # mTOR kinase
    'TP53': 'P04637',      # p53 tumor suppressor
    'MDM2': 'Q00987'       # MDM2 proto-oncogene
}

class RealBindingDBExtractor:
    """Real BindingDB API extractor with retry logic"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GNOSIS-Therapeutic-Index/1.0',
            'Accept': 'application/json'
        })
        self.logger = logging.getLogger(__name__)
    
    def test_api_connectivity(self) -> bool:
        """Test BindingDB API connectivity"""
        
        self.logger.info("🔍 Testing BindingDB API connectivity...")
        
        # Test with a simple, small query
        test_url = f"{BINDINGDB_ENDPOINTS['by_uniprot']}?uniprot=P00533;10000&response=application/json"
        
        try:
            response = self.session.get(test_url, timeout=45)
            
            if response.status_code == 200:
                # Try to parse as JSON
                try:
                    data = response.json()
                    self.logger.info("   ✅ BindingDB API accessible and returns JSON")
                    return True
                except:
                    # Might be valid data but not JSON
                    if len(response.text) > 10:
                        self.logger.info("   ✅ BindingDB API accessible (non-JSON response)")
                        return True
            
            self.logger.warning(f"   ⚠️ BindingDB API returned HTTP {response.status_code}")
            return False
            
        except requests.Timeout:
            self.logger.warning("   ⚠️ BindingDB API timeout - may be slow but accessible")
            return True  # Assume accessible, just slow
        except Exception as e:
            self.logger.error(f"   ❌ BindingDB API test failed: {e}")
            return False
    
    def extract_ligands_by_uniprot(self, uniprot_id: str, target_name: str, 
                                 affinity_cutoff: int = 10000, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Extract ligands for a specific UniProt target with retry logic"""
        
        self.logger.info(f"📥 Extracting {target_name} ligands (UniProt: {uniprot_id})...")
        
        # API URL with JSON response
        api_url = f"{BINDINGDB_ENDPOINTS['by_uniprot']}?uniprot={uniprot_id};{affinity_cutoff}&response=application/json"
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"   🔄 Attempt {attempt + 1}/{max_retries}")
                
                # Longer timeout for BindingDB (can be slow)
                response = self.session.get(api_url, timeout=120)
                
                if response.status_code == 200:
                    # Try to parse JSON
                    try:
                        data = response.json()
                        
                        if isinstance(data, list) and len(data) > 0:
                            # Convert to DataFrame
                            df = pd.DataFrame(data)
                            
                            # Add metadata
                            df['target_name'] = target_name
                            df['uniprot_id'] = uniprot_id
                            df['affinity_cutoff'] = affinity_cutoff
                            df['extraction_date'] = datetime.now().isoformat()
                            
                            self.logger.info(f"   ✅ {target_name}: {len(df):,} ligand records")
                            return df
                        
                        elif isinstance(data, dict):
                            # Single record response
                            df = pd.DataFrame([data])
                            df['target_name'] = target_name
                            df['uniprot_id'] = uniprot_id
                            self.logger.info(f"   ✅ {target_name}: 1 ligand record")
                            return df
                        
                        else:
                            self.logger.warning(f"   ⚠️ {target_name}: Empty or invalid response")
                            return pd.DataFrame()
                    
                    except json.JSONDecodeError:
                        # Try to parse as XML or other format
                        response_text = response.text.strip()
                        
                        if len(response_text) > 10 and not response_text.startswith('<'):
                            self.logger.warning(f"   ⚠️ {target_name}: Non-JSON response received")
                            # Could implement XML parsing here if needed
                        
                        return pd.DataFrame()
                
                else:
                    self.logger.warning(f"   ⚠️ {target_name}: HTTP {response.status_code}")
                    
                    if response.status_code == 404:
                        # No data for this target
                        return pd.DataFrame()
                
            except requests.Timeout:
                self.logger.warning(f"   ⚠️ {target_name}: Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 10)  # 10, 20, 30 second delays
            
            except Exception as e:
                self.logger.warning(f"   ⚠️ {target_name}: Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 5)
        
        self.logger.error(f"   ❌ {target_name}: All retry attempts failed")
        return None
    
    def extract_ligands_batch_uniprots(self, uniprot_dict: Dict[str, str], 
                                     affinity_cutoff: int = 10000) -> Optional[pd.DataFrame]:
        """Extract ligands for multiple UniProt targets in batch"""
        
        self.logger.info(f"📥 Batch extracting ligands for {len(uniprot_dict)} targets...")
        
        # Create comma-separated UniProt list
        uniprot_list = ",".join(uniprot_dict.values())
        target_names = ",".join(uniprot_dict.keys())
        
        api_url = f"{BINDINGDB_ENDPOINTS['by_uniprots']}?uniprot={uniprot_list}&cutoff={affinity_cutoff}&response=application/json"
        
        try:
            self.logger.info("   🔄 Batch API call...")
            response = self.session.get(api_url, timeout=180)  # 3 minute timeout for batch
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    if isinstance(data, list) and len(data) > 0:
                        df = pd.DataFrame(data)
                        df['affinity_cutoff'] = affinity_cutoff
                        df['extraction_date'] = datetime.now().isoformat()
                        df['batch_extraction'] = True
                        
                        self.logger.info(f"   ✅ Batch extraction: {len(df):,} total ligand records")
                        return df
                    else:
                        self.logger.warning("   ⚠️ Batch extraction: Empty response")
                        return pd.DataFrame()
                
                except json.JSONDecodeError:
                    self.logger.warning("   ⚠️ Batch extraction: Invalid JSON response")
                    return pd.DataFrame()
            
            else:
                self.logger.warning(f"   ⚠️ Batch extraction: HTTP {response.status_code}")
                return None
        
        except Exception as e:
            self.logger.error(f"   ❌ Batch extraction failed: {e}")
            return None
    
    def process_bindingdb_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and standardize BindingDB data"""
        
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        self.logger.info(f"🔧 Processing BindingDB data: {len(df):,} records")
        
        # Standardize column names (BindingDB may vary)
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'smiles' in col_lower:
                column_mapping[col] = 'SMILES'
            elif 'monomerid' in col_lower:
                column_mapping[col] = 'monomer_id'
            elif 'ic50' in col_lower:
                column_mapping[col] = 'IC50'
            elif col_lower in ['ki', 'ki_nm']:
                column_mapping[col] = 'Ki'
            elif col_lower in ['kd', 'kd_nm']:
                column_mapping[col] = 'Kd'
            elif 'affinitytype' in col_lower:
                column_mapping[col] = 'affinity_type'
            elif 'affinityvalue' in col_lower:
                column_mapping[col] = 'affinity_value'
        
        # Apply mapping
        df = df.rename(columns=column_mapping)
        
        # Process affinity values
        affinity_records = []
        
        for idx, row in df.iterrows():
            base_record = {
                'monomer_id': row.get('monomer_id'),
                'SMILES': row.get('SMILES'),
                'target_name': row.get('target_name'),
                'uniprot_id': row.get('uniprot_id'),
                'extraction_date': row.get('extraction_date'),
                'data_source': 'BindingDB_API'
            }
            
            # Extract affinity values
            affinity_types = ['IC50', 'Ki', 'Kd']
            
            for aff_type in affinity_types:
                if aff_type in row and pd.notna(row[aff_type]):
                    try:
                        aff_value = float(str(row[aff_type]).replace('>', '').replace('<', '').replace('~', ''))
                        
                        # Convert to standard units (nM)
                        if aff_value > 1000000:  # Likely in pM, convert to nM
                            aff_value = aff_value / 1000
                        elif aff_value < 0.001:  # Likely in M, convert to nM
                            aff_value = aff_value * 1e9
                        elif aff_value < 1:  # Likely in μM, convert to nM
                            aff_value = aff_value * 1000
                        
                        record = base_record.copy()
                        record.update({
                            'affinity_type': aff_type,
                            'affinity_value_nm': aff_value,
                            'log_affinity': np.log10(aff_value),
                            'pic50': -np.log10(aff_value / 1e9) if aff_type == 'IC50' else None
                        })
                        
                        affinity_records.append(record)
                        
                    except (ValueError, TypeError):
                        continue
            
            # If no specific affinity values, use generic affinity_value
            if not any(aff_type in row for aff_type in affinity_types):
                if 'affinity_value' in row and pd.notna(row['affinity_value']):
                    try:
                        aff_value = float(row['affinity_value'])
                        aff_type = row.get('affinity_type', 'Unknown')
                        
                        record = base_record.copy()
                        record.update({
                            'affinity_type': aff_type,
                            'affinity_value_nm': aff_value,
                            'log_affinity': np.log10(aff_value) if aff_value > 0 else None
                        })
                        
                        affinity_records.append(record)
                        
                    except (ValueError, TypeError):
                        continue
        
        if affinity_records:
            processed_df = pd.DataFrame(affinity_records)
            
            # Quality control
            initial_count = len(processed_df)
            
            # Remove invalid SMILES
            processed_df = processed_df.dropna(subset=['SMILES'])
            processed_df = processed_df[processed_df['SMILES'].str.len() > 5]
            
            # Remove unreasonable affinity values
            processed_df = processed_df[
                (processed_df['affinity_value_nm'] >= 0.01) &  # 10 pM minimum
                (processed_df['affinity_value_nm'] <= 1000000)  # 1 mM maximum
            ]
            
            self.logger.info(f"   📊 Processed: {len(processed_df):,} records (removed {initial_count - len(processed_df):,})")
            
            return processed_df
        
        else:
            self.logger.warning("   ⚠️ No valid affinity records found")
            return pd.DataFrame()

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=4.0,
    memory=16384,
    timeout=7200
)
def extract_real_bindingdb_data():
    """
    Extract real BindingDB data via RESTful API
    Target: IC50/Ki/Kd values for oncology proteins
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 REAL BINDINGDB API EXTRACTION")
    print("=" * 80)
    print("✅ Using BindingDB RESTful API")
    print("🎯 Target: IC50/Ki/Kd values for oncology proteins")
    print(f"📊 Targets: {len(ONCOLOGY_TARGETS)} oncology proteins")
    
    try:
        extractor = RealBindingDBExtractor()
        
        # Test API connectivity
        print("\n🔍 STEP 1: Testing BindingDB API connectivity...")
        if not extractor.test_api_connectivity():
            print("   ⚠️ API connectivity issues detected - proceeding with retry logic")
        
        # Extract data for oncology targets
        print(f"\n📥 STEP 2: Extracting data for {len(ONCOLOGY_TARGETS)} oncology targets...")
        
        all_datasets = []
        successful_targets = []
        failed_targets = []
        
        # Try batch extraction first
        print("\n   🔄 Attempting batch extraction...")
        batch_df = extractor.extract_ligands_batch_uniprots(ONCOLOGY_TARGETS, affinity_cutoff=50000)
        
        if batch_df is not None and len(batch_df) > 0:
            print(f"   ✅ Batch extraction successful: {len(batch_df):,} records")
            all_datasets.append(batch_df)
        else:
            print("   ⚠️ Batch extraction failed - falling back to individual extraction")
            
            # Individual target extraction
            for target_name, uniprot_id in ONCOLOGY_TARGETS.items():
                print(f"\n   📊 {target_name} ({uniprot_id})")
                
                target_df = extractor.extract_ligands_by_uniprot(
                    uniprot_id, target_name, affinity_cutoff=50000
                )
                
                if target_df is not None and len(target_df) > 0:
                    all_datasets.append(target_df)
                    successful_targets.append(target_name)
                else:
                    failed_targets.append(target_name)
                
                # Rate limiting between targets
                time.sleep(2)
        
        # Combine all datasets
        if all_datasets:
            combined_df = pd.concat(all_datasets, ignore_index=True)
            print(f"\n   ✅ Combined raw data: {len(combined_df):,} records")
        else:
            raise Exception("No BindingDB data successfully extracted")
        
        # Process data
        print("\n🔧 STEP 3: Processing BindingDB data...")
        processed_df = extractor.process_bindingdb_data(combined_df)
        
        if len(processed_df) == 0:
            raise Exception("No valid binding affinity data after processing")
        
        print(f"   ✅ Processed data: {len(processed_df):,} binding affinity records")
        
        # Data quality summary
        print(f"\n📊 Data quality summary:")
        print(f"   • Unique compounds: {processed_df['SMILES'].nunique()}")
        print(f"   • Unique targets: {processed_df['target_name'].nunique()}")
        print(f"   • Affinity types: {processed_df['affinity_type'].value_counts().to_dict()}")
        print(f"   • Median affinity: {processed_df['affinity_value_nm'].median():.1f} nM")
        
        # Save data
        print("\n💾 STEP 4: Saving BindingDB data...")
        
        datasets_dir = Path("/vol/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw data
        raw_path = datasets_dir / "bindingdb_raw_data.csv"
        combined_df.to_csv(raw_path, index=False)
        
        # Save processed data
        processed_path = datasets_dir / "bindingdb_processed_data.csv"
        processed_df.to_csv(processed_path, index=False)
        
        # Save for training pipeline
        training_path = datasets_dir / "bindingdb_training_data.csv"
        processed_df.to_csv(training_path, index=False)
        
        # Create metadata
        metadata = {
            'extraction_method': 'BindingDB_RESTful_API',
            'api_base': BINDINGDB_API_BASE,
            'extraction_date': datetime.now().isoformat(),
            'targets_attempted': len(ONCOLOGY_TARGETS),
            'targets_successful': len(successful_targets) if successful_targets else 'batch_mode',
            'targets_failed': len(failed_targets) if failed_targets else 0,
            'raw_records': len(combined_df),
            'processed_records': len(processed_df),
            'unique_compounds': int(processed_df['SMILES'].nunique()),
            'unique_targets': int(processed_df['target_name'].nunique()),
            'affinity_types': processed_df['affinity_type'].value_counts().to_dict(),
            'median_affinity_nm': float(processed_df['affinity_value_nm'].median()),
            'oncology_targets': ONCOLOGY_TARGETS,
            'files_created': {
                'raw_data': str(raw_path),
                'processed_data': str(processed_path),
                'training_data': str(training_path)
            }
        }
        
        metadata_path = datasets_dir / "bindingdb_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate report
        print(f"\n🎉 REAL BINDINGDB EXTRACTION COMPLETED!")
        print("=" * 80)
        print(f"📁 Files saved:")
        print(f"  • Raw data: {raw_path}")
        print(f"  • Processed data: {processed_path}")
        print(f"  • Training ready: {training_path}")
        print(f"  • Metadata: {metadata_path}")
        
        print(f"\n📊 BindingDB summary:")
        print(f"  • Raw records: {len(combined_df):,}")
        print(f"  • Processed records: {len(processed_df):,}")
        print(f"  • Unique compounds: {processed_df['SMILES'].nunique()}")
        print(f"  • Unique targets: {processed_df['target_name'].nunique()}")
        print(f"  • Median binding affinity: {processed_df['affinity_value_nm'].median():.1f} nM")
        
        print(f"\n✅ REAL IC50/KI/KD DATA:")
        print(f"  • Source: BindingDB RESTful API")
        print(f"  • Oncology-focused: 23 target proteins")
        print(f"  • Binding affinity types: IC50, Ki, Kd")
        print(f"  • Ready for ligand-protein training")
        
        return {
            'status': 'success',
            'extraction_method': 'BindingDB_RESTful_API',
            'raw_records': len(combined_df),
            'processed_records': len(processed_df),
            'unique_compounds': int(processed_df['SMILES'].nunique()),
            'unique_targets': int(processed_df['target_name'].nunique()),
            'median_affinity_nm': float(processed_df['affinity_value_nm'].median()),
            'files_created': metadata['files_created'],
            'metadata_path': str(metadata_path),
            'ready_for_training': True
        }
        
    except Exception as e:
        print(f"❌ BINDINGDB EXTRACTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🧬 Real BindingDB API Extractor")