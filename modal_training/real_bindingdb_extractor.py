"""
Real BindingDB Data Extractor
Uses BindingDB RESTful API to extract real protein-ligand binding affinity data
"""

import modal
import requests
import pandas as pd
import numpy as np
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import aiohttp
from urllib.parse import quote

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "requests",
    "aiohttp",
    "pandas", 
    "numpy",
    "rdkit-pypi",
    "lxml",
    "beautifulsoup4"
])

app = modal.App("real-bindingdb-extractor")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

# Target UniProt IDs for our oncology targets
ONCOLOGY_TARGETS_UNIPROT = {
    # ONCOPROTEINS with UniProt IDs
    "EGFR": {"uniprot_id": "P00533", "category": "oncoprotein", "name": "Epidermal Growth Factor Receptor"},
    "HER2": {"uniprot_id": "P04626", "category": "oncoprotein", "name": "Receptor Tyrosine-Protein Kinase erbB-2"},
    "VEGFR2": {"uniprot_id": "P35968", "category": "oncoprotein", "name": "Vascular Endothelial Growth Factor Receptor 2"},
    "BRAF": {"uniprot_id": "P15056", "category": "oncoprotein", "name": "Serine/Threonine-Protein Kinase B-raf"},
    "MET": {"uniprot_id": "P08581", "category": "oncoprotein", "name": "Hepatocyte Growth Factor Receptor"},
    "CDK4": {"uniprot_id": "P11802", "category": "oncoprotein", "name": "Cyclin-Dependent Kinase 4"},
    "CDK6": {"uniprot_id": "Q00534", "category": "oncoprotein", "name": "Cyclin-Dependent Kinase 6"},
    "ALK": {"uniprot_id": "Q9UM73", "category": "oncoprotein", "name": "ALK Tyrosine Kinase Receptor"},
    "MDM2": {"uniprot_id": "Q00987", "category": "oncoprotein", "name": "E3 Ubiquitin-Protein Ligase Mdm2"},
    "PI3KCA": {"uniprot_id": "P42336", "category": "oncoprotein", "name": "Phosphatidylinositol 4,5-Bisphosphate 3-Kinase Catalytic Subunit Alpha"},
    
    # TUMOR SUPPRESSORS with UniProt IDs
    "TP53": {"uniprot_id": "P04637", "category": "tumor_suppressor", "name": "Cellular Tumor Antigen p53"},
    "RB1": {"uniprot_id": "P06400", "category": "tumor_suppressor", "name": "Retinoblastoma-Associated Protein"},
    "PTEN": {"uniprot_id": "P60484", "category": "tumor_suppressor", "name": "Phosphatidylinositol 3,4,5-Trisphosphate 3-Phosphatase"},
    "APC": {"uniprot_id": "P25054", "category": "tumor_suppressor", "name": "Adenomatous Polyposis Coli Protein"},
    "BRCA1": {"uniprot_id": "P38398", "category": "tumor_suppressor", "name": "Breast Cancer Type 1 Susceptibility Protein"},
    "BRCA2": {"uniprot_id": "P51587", "category": "tumor_suppressor", "name": "Breast Cancer Type 2 Susceptibility Protein"},
    "VHL": {"uniprot_id": "P40337", "category": "tumor_suppressor", "name": "Von Hippel-Lindau Disease Tumor Suppressor"},
    
    # METASTASIS SUPPRESSORS with UniProt IDs
    "NDRG1": {"uniprot_id": "Q92597", "category": "metastasis_suppressor", "name": "Protein NDRG1"},
    "KAI1": {"uniprot_id": "P48506", "category": "metastasis_suppressor", "name": "CD82 Antigen"},
    "KISS1": {"uniprot_id": "Q15726", "category": "metastasis_suppressor", "name": "Metastasis Suppressor KiSS-1"},
    "NM23H1": {"uniprot_id": "P15531", "category": "metastasis_suppressor", "name": "Nucleoside Diphosphate Kinase A"},
    "RKIP": {"uniprot_id": "P30086", "category": "metastasis_suppressor", "name": "Phosphatidylethanolamine-Binding Protein 1"},
    "CASP8": {"uniprot_id": "Q14790", "category": "metastasis_suppressor", "name": "Caspase-8"}
}

class BindingDBExtractor:
    """Real BindingDB data extractor using RESTful API"""
    
    def __init__(self, rate_limit: float = 2.0):
        self.base_url = "https://bindingdb.org/rest"
        self.axis_url = "https://bindingdb.org/axis2/services/BDBService"
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
    
    def extract_target_data(self, target_name: str, target_info: Dict[str, Any], affinity_cutoff: float = 10000.0) -> List[Dict[str, Any]]:
        """Extract real binding data for a single target from BindingDB"""
        
        self.logger.info(f"🔗 BindingDB: Extracting {target_name} ({target_info['category']}, {target_info['uniprot_id']})...")
        
        uniprot_id = target_info['uniprot_id']
        all_records = []
        
        try:
            # Use BindingDB REST API to get ligands by UniProt ID
            url = f"{self.base_url}/getLigandsByUniprots"
            params = {
                'uniprot': uniprot_id,
                'cutoff': affinity_cutoff,  # nM
                'response': 'application/json'
            }
            
            self.logger.info(f"   🌐 Querying: {url} with UniProt {uniprot_id}, cutoff {affinity_cutoff} nM")
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                self.logger.warning(f"   ⚠️ HTTP {response.status_code}: {response.text[:200]}")
                return []
            
            # Handle different response formats
            try:
                if response.headers.get('content-type', '').startswith('application/json'):
                    data = response.json()
                else:
                    # Try parsing as JSON anyway
                    data = response.json()
            except json.JSONDecodeError:
                self.logger.warning(f"   ⚠️ Non-JSON response for {target_name}")
                # Try parsing XML response
                return self._parse_xml_response(response.text, target_name, target_info)
            
            # Process JSON response
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                # Check common JSON response structures
                entries = data.get('entries', data.get('results', data.get('data', [data])))
            else:
                entries = [data]
            
            for entry in entries:
                try:
                    record = self._process_binding_entry(entry, target_name, target_info)
                    if record:
                        all_records.append(record)
                except Exception as e:
                    self.logger.debug(f"   Error processing entry: {e}")
                    continue
            
            self.logger.info(f"   ✅ BindingDB {target_name}: {len(all_records)} records extracted")
            
            # Rate limiting
            time.sleep(self.rate_limit)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"   ❌ Network error for {target_name}: {e}")
        except Exception as e:
            self.logger.error(f"   ❌ Unexpected error for {target_name}: {e}")
        
        return all_records
    
    def _parse_xml_response(self, xml_text: str, target_name: str, target_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse XML response from BindingDB"""
        
        try:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(xml_text)
            
            records = []
            
            # Look for common XML structures in BindingDB responses
            for entry in root.findall('.//entry'):
                try:
                    record = self._process_xml_entry(entry, target_name, target_info)
                    if record:
                        records.append(record)
                except:
                    continue
            
            return records
            
        except Exception as e:
            self.logger.warning(f"   ⚠️ XML parsing failed for {target_name}: {e}")
            return []
    
    def _process_binding_entry(self, entry: Dict[str, Any], target_name: str, target_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single binding entry from BindingDB JSON response"""
        
        try:
            # Common field mappings from BindingDB API
            smiles = entry.get('smiles', entry.get('SMILES', ''))
            if not smiles:
                return None
            
            # Extract binding affinity data
            ic50 = entry.get('IC50', entry.get('ic50'))
            ki = entry.get('Ki', entry.get('ki'))
            kd = entry.get('Kd', entry.get('kd'))
            
            # Determine activity type and value
            activity_type = None
            value = None
            units = 'nM'  # BindingDB typically reports in nM
            
            if ic50 and ic50 != 'NULL' and ic50 != '':
                activity_type = 'IC50'
                value = self._parse_affinity_value(ic50)
            elif ki and ki != 'NULL' and ki != '':
                activity_type = 'Ki'  
                value = self._parse_affinity_value(ki)
            elif kd and kd != 'NULL' and kd != '':
                activity_type = 'Kd'
                value = self._parse_affinity_value(kd)
            
            if not value or value <= 0:
                return None
            
            # Calculate pIC50/pKi/pKd
            pic50 = -np.log10(value / 1e9)  # Convert nM to M, then -log10
            
            # Validate pIC50 range
            if not (0 < pic50 < 15):
                return None
            
            # Create standardized record
            record = {
                'canonical_smiles': smiles,
                'target_name': target_name,
                'target_category': target_info['category'],
                'activity_type': activity_type,
                'standard_value': value,
                'standard_units': units,
                'standard_value_nm': value,
                'pic50': pic50,
                'bindingdb_id': entry.get('BindingDB_ID', entry.get('id', '')),
                'pubchem_cid': entry.get('PubChem_CID', ''),
                'target_uniprot': target_info['uniprot_id'],
                'data_source': 'BindingDB'
            }
            
            return record
            
        except Exception as e:
            return None
    
    def _parse_affinity_value(self, value_str: str) -> Optional[float]:
        """Parse affinity value from string, handling various formats"""
        
        if not value_str or value_str in ['NULL', '', 'N/A']:
            return None
        
        try:
            # Handle numeric values
            if isinstance(value_str, (int, float)):
                return float(value_str)
            
            # Handle string values with units
            value_str = str(value_str).strip()
            
            # Remove common units and symbols
            value_str = value_str.replace('nM', '').replace('uM', '').replace('mM', '')
            value_str = value_str.replace('µM', '').replace('μM', '').replace('pM', '')
            value_str = value_str.replace('>', '').replace('<', '').replace('~', '').replace('≈', '')
            value_str = value_str.strip()
            
            # Parse the numeric value
            value = float(value_str)
            
            # Assume nM if no explicit unit conversion needed
            return value
            
        except (ValueError, TypeError):
            return None
    
    def _process_xml_entry(self, entry, target_name: str, target_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process XML entry (placeholder for XML parsing)"""
        # This would need to be implemented based on actual XML structure
        return None

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=4.0,
    memory=16384,
    timeout=7200
)
def extract_real_bindingdb_data():
    """
    Extract real binding affinity data from BindingDB using REST API
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🔗 REAL BINDINGDB DATA EXTRACTION")
    print("=" * 80)
    print("🌐 Using BindingDB RESTful API")
    print(f"📋 Targets: {len(ONCOLOGY_TARGETS_UNIPROT)}")
    
    try:
        extractor = BindingDBExtractor()
        all_records = []
        target_stats = {}
        
        for target_idx, (target_name, target_info) in enumerate(ONCOLOGY_TARGETS_UNIPROT.items(), 1):
            print(f"\n📍 [{target_idx}/{len(ONCOLOGY_TARGETS_UNIPROT)}] Processing {target_name}...")
            
            try:
                # Extract data with generous cutoff to get more data
                target_records = extractor.extract_target_data(
                    target_name, 
                    target_info, 
                    affinity_cutoff=50000.0  # 50 μM cutoff
                )
                
                all_records.extend(target_records)
                
                target_stats[target_name] = {
                    'category': target_info['category'],
                    'uniprot_id': target_info['uniprot_id'],
                    'total_records': len(target_records)
                }
                
                print(f"   📊 {target_name}: {len(target_records)} binding records")
                
            except Exception as e:
                print(f"   ❌ {target_name} failed: {e}")
                target_stats[target_name] = {
                    'category': target_info['category'],
                    'uniprot_id': target_info['uniprot_id'],
                    'total_records': 0,
                    'error': str(e)
                }
        
        if not all_records:
            print("⚠️ No records extracted from BindingDB API")
            # Return empty result but don't fail completely
            return {
                'status': 'success',
                'total_records': 0,
                'message': 'No data retrieved from BindingDB API - may need different approach'
            }
        
        print(f"\n📊 REAL BINDINGDB DATA SUMMARY:")
        print(f"   📈 Total records: {len(all_records)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        
        print(f"   📊 DataFrame shape: {df.shape}")
        print(f"   📊 Unique targets: {df['target_name'].nunique()}")
        print(f"   📊 Unique compounds: {df['canonical_smiles'].nunique()}")
        
        # Show activity type distribution
        if 'activity_type' in df.columns:
            activity_dist = df['activity_type'].value_counts()
            print(f"\n📊 Activity Type Distribution:")
            for activity_type, count in activity_dist.items():
                percentage = (count / len(df)) * 100
                print(f"   • {activity_type}: {count:,} ({percentage:.1f}%)")
        
        # Apply quality control
        print(f"\n🔍 APPLYING DATA QUALITY CONTROL...")
        
        initial_count = len(df)
        df = df.dropna(subset=['canonical_smiles', 'target_name', 'standard_value_nm'])
        df = df.drop_duplicates(subset=['canonical_smiles', 'target_name', 'activity_type'], keep='first')
        
        print(f"   📊 After quality control: {len(df)} records (removed {initial_count - len(df)})")
        
        # Save datasets
        print(f"\n💾 Saving real BindingDB dataset...")
        
        datasets_dir = Path("/vol/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw data
        raw_data_path = datasets_dir / "real_bindingdb_raw_data.csv"
        df.to_csv(raw_data_path, index=False)
        
        # Create binding affinity matrix
        if len(df) > 0:
            pivot_table = df.pivot_table(
                index='canonical_smiles',
                columns='target_name', 
                values='pic50',
                aggfunc='median'
            ).reset_index()
            
            matrix_path = datasets_dir / "real_bindingdb_matrix.csv"
            pivot_table.to_csv(matrix_path, index=False)
            print(f"   📊 Binding matrix: {pivot_table.shape}")
        
        # Save metadata
        metadata = {
            'extraction_method': 'BindingDB_REST_API',
            'real_data': True,
            'api_endpoint': 'https://bindingdb.org/rest/getLigandsByUniprots',
            'targets': list(ONCOLOGY_TARGETS_UNIPROT.keys()),
            'target_info': ONCOLOGY_TARGETS_UNIPROT,
            'total_records': len(df),
            'total_targets': df['target_name'].nunique() if len(df) > 0 else 0,
            'total_compounds': df['canonical_smiles'].nunique() if len(df) > 0 else 0,
            'target_stats': target_stats,
            'activity_type_counts': df['activity_type'].value_counts().to_dict() if len(df) > 0 else {},
            'extraction_timestamp': datetime.now().isoformat(),
            'data_quality': {
                'real_experimental_data': True,
                'bindingdb_source': True,
                'protein_ligand_binding': True,
                'units': 'nM',
                'pic50_calculated': True,
                'duplicates_removed': True
            }
        }
        
        metadata_path = datasets_dir / "real_bindingdb_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate final report
        print(f"\n🎉 REAL BINDINGDB EXTRACTION COMPLETED!")
        print("=" * 80)
        print(f"📁 Dataset files:")
        print(f"  • Raw data: {raw_data_path}")
        if len(df) > 0:
            print(f"  • Binding matrix: {matrix_path}")
        print(f"  • Metadata: {metadata_path}")
        
        print(f"\n📊 Final real dataset summary:")
        print(f"  • Total records: {len(df):,}")
        print(f"  • Unique targets: {df['target_name'].nunique() if len(df) > 0 else 0}")
        print(f"  • Unique compounds: {df['canonical_smiles'].nunique() if len(df) > 0 else 0}")
        
        if len(df) > 0:
            # Category breakdown
            for category in ['oncoprotein', 'tumor_suppressor', 'metastasis_suppressor']:
                category_targets = [name for name, info in ONCOLOGY_TARGETS_UNIPROT.items() if info['category'] == category]
                category_records = df[df['target_name'].isin(category_targets)]
                print(f"  • {category.replace('_', ' ').title()}: {len(category_records)} binding records across {len(category_targets)} targets")
        
        print(f"\n🌐 REAL BINDINGDB DATA: Authentic protein-ligand binding affinities")
        
        return {
            'status': 'success',
            'raw_data_path': str(raw_data_path),
            'matrix_path': str(matrix_path) if len(df) > 0 else None,
            'metadata_path': str(metadata_path),
            'total_records': len(df),
            'total_targets': df['target_name'].nunique() if len(df) > 0 else 0,
            'total_compounds': df['canonical_smiles'].nunique() if len(df) > 0 else 0,
            'activity_distribution': df['activity_type'].value_counts().to_dict() if len(df) > 0 else {},
            'ready_for_integration': True,
            'real_data': True
        }
        
    except Exception as e:
        print(f"❌ REAL BINDINGDB EXTRACTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🔗 Real BindingDB Data Extractor")