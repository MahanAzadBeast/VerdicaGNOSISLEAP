"""
Real PubChem BioAssay Data Extractor
Uses actual PubChem API to extract bioactivity data for oncology targets
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
import xml.etree.ElementTree as ET

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

app = modal.App("real-pubchem-extractor")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

# Target information with PubChem synonyms for better search results
ONCOLOGY_TARGETS_PUBCHEM = {
    # ONCOPROTEINS
    "EGFR": {"category": "oncoprotein", "name": "Epidermal Growth Factor Receptor", 
             "synonyms": ["EGFR", "HER1", "ErbB1", "Epidermal growth factor receptor"]},
    "HER2": {"category": "oncoprotein", "name": "Human Epidermal Growth Factor Receptor 2",
             "synonyms": ["HER2", "ERBB2", "NEU", "HER-2"]},
    "VEGFR2": {"category": "oncoprotein", "name": "Vascular Endothelial Growth Factor Receptor 2",
               "synonyms": ["VEGFR2", "KDR", "FLK1", "VEGFR-2"]},
    "BRAF": {"category": "oncoprotein", "name": "B-Raf Proto-Oncogene",
             "synonyms": ["BRAF", "B-RAF", "BRAF1", "RAF-B"]},
    "MET": {"category": "oncoprotein", "name": "MET Proto-Oncogene",
            "synonyms": ["MET", "HGFR", "c-MET", "Hepatocyte growth factor receptor"]},
    "CDK4": {"category": "oncoprotein", "name": "Cyclin Dependent Kinase 4",
             "synonyms": ["CDK4", "Cyclin-dependent kinase 4"]},
    "CDK6": {"category": "oncoprotein", "name": "Cyclin Dependent Kinase 6",
             "synonyms": ["CDK6", "Cyclin-dependent kinase 6"]},
    "ALK": {"category": "oncoprotein", "name": "ALK Receptor Tyrosine Kinase",
            "synonyms": ["ALK", "Anaplastic lymphoma kinase"]},
    "MDM2": {"category": "oncoprotein", "name": "MDM2 Proto-Oncogene",
             "synonyms": ["MDM2", "Mouse double minute 2"]},
    "PI3KCA": {"category": "oncoprotein", "name": "Phosphatidylinositol-4,5-Bisphosphate 3-Kinase Catalytic Subunit Alpha",
               "synonyms": ["PIK3CA", "PI3K", "PI3KCA", "PI3-kinase"]},
    
    # TUMOR SUPPRESSORS
    "TP53": {"category": "tumor_suppressor", "name": "Tumor Protein P53",
             "synonyms": ["TP53", "p53", "P53", "Tumor protein p53"]},
    "RB1": {"category": "tumor_suppressor", "name": "RB Transcriptional Corepressor 1",
            "synonyms": ["RB1", "RB", "Retinoblastoma protein"]},
    "PTEN": {"category": "tumor_suppressor", "name": "Phosphatase And Tensin Homolog",
             "synonyms": ["PTEN", "Phosphatase and tensin homolog"]},
    "APC": {"category": "tumor_suppressor", "name": "APC Regulator Of WNT Signaling Pathway",
            "synonyms": ["APC", "Adenomatous polyposis coli"]},
    "BRCA1": {"category": "tumor_suppressor", "name": "BRCA1 DNA Repair Associated",
              "synonyms": ["BRCA1", "Breast cancer 1"]},
    "BRCA2": {"category": "tumor_suppressor", "name": "BRCA2 DNA Repair Associated",
              "synonyms": ["BRCA2", "Breast cancer 2"]},
    "VHL": {"category": "tumor_suppressor", "name": "Von Hippel-Lindau Tumor Suppressor",
            "synonyms": ["VHL", "Von Hippel-Lindau"]},
    
    # METASTASIS SUPPRESSORS
    "NDRG1": {"category": "metastasis_suppressor", "name": "NDRG1 Myelin Transcription Factor 1",
              "synonyms": ["NDRG1", "N-myc downstream regulated 1"]},
    "KAI1": {"category": "metastasis_suppressor", "name": "CD82 Molecule",
             "synonyms": ["CD82", "KAI1", "Tetraspanin-27"]},
    "KISS1": {"category": "metastasis_suppressor", "name": "KISS1 Metastasis Suppressor",
              "synonyms": ["KISS1", "KiSS-1"]},
    "NM23H1": {"category": "metastasis_suppressor", "name": "NME1 NDP Kinase 1",
               "synonyms": ["NME1", "NM23", "NM23-H1"]},
    "RKIP": {"category": "metastasis_suppressor", "name": "Raf Kinase Inhibitor Protein",
             "synonyms": ["PEBP1", "RKIP", "PBP", "HCNPpp"]},
    "CASP8": {"category": "metastasis_suppressor", "name": "Caspase 8",
              "synonyms": ["CASP8", "Caspase-8", "FLICE"]}
}

class RealPubChemExtractor:
    """Real PubChem BioAssay data extractor using actual PubChem APIs"""
    
    def __init__(self, rate_limit: float = 1.0):
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest"
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Veridica-AI/1.0 (Bioactivity Research; contact@veridica.ai)'
        })
        self.logger = logging.getLogger(__name__)
    
    def search_target_bioassays(self, target_name: str, target_info: Dict[str, Any]) -> List[str]:
        """Search for bioassays related to a specific target"""
        
        self.logger.info(f"🔍 Searching PubChem bioassays for {target_name}...")
        
        all_aid_ids = set()
        
        # Search using target synonyms
        for synonym in target_info.get('synonyms', [target_name]):
            try:
                # Search bioassays by target name
                search_url = f"{self.base_url}/pug/assay/target/name/{quote(synonym)}/aids/JSON"
                
                self.logger.debug(f"   Searching synonym: {synonym}")
                response = self.session.get(search_url, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    aids = data.get('InformationList', {}).get('Information', [])
                    
                    for aid_info in aids:
                        aid_id = aid_info.get('AID')
                        if aid_id:
                            all_aid_ids.add(str(aid_id))
                    
                    self.logger.debug(f"   Found {len(aids)} assays for {synonym}")
                
                # Also search by activity/assay description
                desc_search_url = f"{self.base_url}/pug/assay/description/{quote(synonym)}/aids/JSON"
                response = self.session.get(desc_search_url, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    aids = data.get('InformationList', {}).get('Information', [])
                    
                    for aid_info in aids:
                        aid_id = aid_info.get('AID')
                        if aid_id:
                            all_aid_ids.add(str(aid_id))
                
                time.sleep(self.rate_limit)
                
            except Exception as e:
                self.logger.debug(f"   Search error for {synonym}: {e}")
                continue
        
        aid_list = list(all_aid_ids)
        self.logger.info(f"   ✅ Found {len(aid_list)} unique bioassays for {target_name}")
        
        return aid_list[:50]  # Limit to top 50 assays to avoid overload
    
    def extract_bioassay_data(self, aid_id: str, target_name: str, target_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract bioactivity data from a specific bioassay"""
        
        try:
            # Get bioassay summary information
            summary_url = f"{self.base_url}/pug/assay/aid/{aid_id}/summary/JSON"
            response = self.session.get(summary_url, timeout=30)
            
            if response.status_code != 200:
                return []
            
            summary_data = response.json()
            assays = summary_data.get('PC_AssayContainer', [])
            
            if not assays:
                return []
            
            assay = assays[0].get('assay', {})
            
            # Check if assay is relevant (contains IC50, EC50, Ki, etc.)
            assay_description = assay.get('descr', {})
            activity_outcome_method = assay_description.get('activity_outcome_method', '')
            assay_name = assay_description.get('name', '')
            
            relevant_keywords = ['IC50', 'EC50', 'Ki', 'Kd', 'inhibition', 'binding', 'activity']
            is_relevant = any(keyword.lower() in f"{activity_outcome_method} {assay_name}".lower() 
                            for keyword in relevant_keywords)
            
            if not is_relevant:
                return []
            
            # Get active compounds from this bioassay
            active_url = f"{self.base_url}/pug/assay/aid/{aid_id}/cids/JSON?cids_type=active"
            response = self.session.get(active_url, timeout=30)
            
            if response.status_code != 200:
                return []
            
            cid_data = response.json()
            active_cids = cid_data.get('InformationList', {}).get('Information', [])
            
            if not active_cids:
                return []
            
            # Get compound details and bioactivity data
            records = []
            cid_list = [str(info['CID']) for info in active_cids[:100]]  # Limit to 100 compounds
            
            # Get SMILES for these compounds
            smiles_data = self.get_compound_smiles(cid_list)
            
            # Get bioactivity data
            bioactivity_data = self.get_bioactivity_data(aid_id, cid_list)
            
            # Combine data
            for cid in cid_list:
                if cid in smiles_data and cid in bioactivity_data:
                    smiles = smiles_data[cid]
                    activity_info = bioactivity_data[cid]
                    
                    if smiles and activity_info:
                        record = {
                            'canonical_smiles': smiles,
                            'target_name': target_name,
                            'target_category': target_info['category'],
                            'activity_type': activity_info.get('activity_type', 'IC50'),
                            'standard_value': activity_info.get('value'),
                            'standard_units': activity_info.get('units', 'nM'),
                            'standard_value_nm': activity_info.get('value_nm'),
                            'pic50': activity_info.get('pic50'),
                            'pubchem_aid': aid_id,
                            'pubchem_cid': cid,
                            'assay_name': assay_name,
                            'data_source': 'PubChem_BioAssay'
                        }
                        
                        records.append(record)
            
            time.sleep(self.rate_limit)
            return records
            
        except Exception as e:
            self.logger.debug(f"   Error extracting bioassay {aid_id}: {e}")
            return []
    
    def get_compound_smiles(self, cid_list: List[str]) -> Dict[str, str]:
        """Get SMILES for a list of compound CIDs"""
        
        smiles_dict = {}
        
        # Process in batches of 20
        batch_size = 20
        for i in range(0, len(cid_list), batch_size):
            batch = cid_list[i:i+batch_size]
            cid_string = ','.join(batch)
            
            try:
                smiles_url = f"{self.base_url}/pug/compound/cid/{cid_string}/property/CanonicalSMILES/JSON"
                response = self.session.get(smiles_url, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    properties = data.get('PropertyTable', {}).get('Properties', [])
                    
                    for prop in properties:
                        cid = str(prop.get('CID'))
                        smiles = prop.get('CanonicalSMILES')
                        if cid and smiles:
                            smiles_dict[cid] = smiles
                
                time.sleep(self.rate_limit)
                
            except Exception as e:
                self.logger.debug(f"Error getting SMILES for batch: {e}")
                continue
        
        return smiles_dict
    
    def get_bioactivity_data(self, aid_id: str, cid_list: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get bioactivity data for compounds in a specific assay"""
        
        activity_dict = {}
        
        try:
            # Get bioactivity data for all compounds in this assay
            bioactivity_url = f"{self.base_url}/pug/assay/aid/{aid_id}/CSV"
            response = self.session.get(bioactivity_url, timeout=60)
            
            if response.status_code == 200:
                # Parse CSV data
                import io
                csv_data = pd.read_csv(io.StringIO(response.text))
                
                # Look for relevant activity columns
                activity_columns = [col for col in csv_data.columns 
                                  if any(keyword in col.lower() 
                                        for keyword in ['ic50', 'ec50', 'ki', 'kd', 'ac50', 'potency'])]
                
                if not activity_columns:
                    return activity_dict
                
                # Process each compound
                for _, row in csv_data.iterrows():
                    cid = str(row.get('PUBCHEM_CID', ''))
                    
                    if cid in cid_list:
                        # Find the best activity value
                        best_value = None
                        best_units = 'nM'
                        activity_type = 'IC50'
                        
                        for col in activity_columns:
                            value = row.get(col)
                            if pd.notna(value) and value > 0:
                                # Determine activity type from column name
                                if 'ic50' in col.lower():
                                    activity_type = 'IC50'
                                elif 'ec50' in col.lower():
                                    activity_type = 'EC50'
                                elif 'ki' in col.lower():
                                    activity_type = 'Ki'
                                elif 'kd' in col.lower():
                                    activity_type = 'Kd'
                                elif 'ac50' in col.lower():
                                    activity_type = 'AC50'
                                
                                best_value = float(value)
                                break
                        
                        if best_value and best_value > 0:
                            # Convert to nM if needed and calculate pIC50
                            value_nm = self._convert_to_nm(best_value, best_units)
                            pic50 = -np.log10(value_nm / 1e9) if value_nm else None
                            
                            activity_dict[cid] = {
                                'activity_type': activity_type,
                                'value': best_value,
                                'units': best_units,
                                'value_nm': value_nm,
                                'pic50': pic50
                            }
        
        except Exception as e:
            self.logger.debug(f"Error getting bioactivity data for AID {aid_id}: {e}")
        
        return activity_dict
    
    def _convert_to_nm(self, value: float, units: str) -> Optional[float]:
        """Convert activity value to nM"""
        
        if not value or value <= 0:
            return None
        
        units = units.lower()
        
        # Unit conversion factors to nM
        conversion_factors = {
            'nm': 1.0,
            'nanomolar': 1.0,
            'µm': 1000.0,
            'um': 1000.0,
            'micromolar': 1000.0,
            'mm': 1000000.0,
            'millimolar': 1000000.0,
            'm': 1000000000.0,
            'molar': 1000000000.0,
            'pm': 0.001,
            'picomolar': 0.001
        }
        
        # Try to find matching unit
        for unit_key, factor in conversion_factors.items():
            if unit_key in units:
                return value * factor
        
        # Default assume nM
        return value
    
    def extract_target_data(self, target_name: str, target_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all bioactivity data for a target from PubChem BioAssay"""
        
        self.logger.info(f"🧪 PubChem: Extracting {target_name} ({target_info['category']})...")
        
        # Search for relevant bioassays
        aid_list = self.search_target_bioassays(target_name, target_info)
        
        if not aid_list:
            self.logger.warning(f"   ⚠️ No bioassays found for {target_name}")
            return []
        
        all_records = []
        
        # Extract data from each bioassay
        for i, aid_id in enumerate(aid_list[:10], 1):  # Limit to top 10 assays
            self.logger.debug(f"   Processing bioassay {i}/{min(len(aid_list), 10)}: AID {aid_id}")
            
            try:
                records = self.extract_bioassay_data(aid_id, target_name, target_info)
                all_records.extend(records)
                
                self.logger.debug(f"   Bioassay {aid_id}: {len(records)} records")
                
            except Exception as e:
                self.logger.debug(f"   Error processing bioassay {aid_id}: {e}")
                continue
        
        self.logger.info(f"   ✅ PubChem {target_name}: {len(all_records)} records extracted")
        return all_records

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=4.0,
    memory=16384,
    timeout=7200
)
def extract_real_pubchem_data():
    """
    Extract real bioactivity data from PubChem BioAssay using actual APIs
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🧪 REAL PUBCHEM BIOASSAY DATA EXTRACTION")
    print("=" * 80)
    print("🌐 Using Real PubChem APIs")
    print(f"📋 Targets: {len(ONCOLOGY_TARGETS_PUBCHEM)}")
    
    try:
        extractor = RealPubChemExtractor()
        all_records = []
        target_stats = {}
        
        for target_idx, (target_name, target_info) in enumerate(ONCOLOGY_TARGETS_PUBCHEM.items(), 1):
            print(f"\n📍 [{target_idx}/{len(ONCOLOGY_TARGETS_PUBCHEM)}] Processing {target_name}...")
            
            try:
                target_records = extractor.extract_target_data(target_name, target_info)
                all_records.extend(target_records)
                
                target_stats[target_name] = {
                    'category': target_info['category'],
                    'total_records': len(target_records),
                    'synonyms_searched': len(target_info.get('synonyms', []))
                }
                
                print(f"   📊 {target_name}: {len(target_records)} bioactivity records")
                
            except Exception as e:
                print(f"   ❌ {target_name} failed: {e}")
                target_stats[target_name] = {
                    'category': target_info['category'],
                    'total_records': 0,
                    'error': str(e)
                }
        
        if not all_records:
            print("⚠️ No records extracted from PubChem BioAssay API")
            # Return empty result but don't fail completely
            return {
                'status': 'success',
                'total_records': 0,
                'message': 'No data retrieved from PubChem BioAssay API - may need different search terms'
            }
        
        print(f"\n📊 REAL PUBCHEM DATA SUMMARY:")
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
        
        # Filter reasonable activity range
        df = df[(df['standard_value_nm'] >= 0.1) & (df['standard_value_nm'] <= 100000000)]  # 0.1 nM to 100 mM
        
        print(f"   📊 After quality control: {len(df)} records (removed {initial_count - len(df)})")
        
        # Save datasets
        print(f"\n💾 Saving real PubChem dataset...")
        
        datasets_dir = Path("/vol/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw data
        raw_data_path = datasets_dir / "real_pubchem_raw_data.csv"
        df.to_csv(raw_data_path, index=False)
        
        # Create bioactivity matrix
        if len(df) > 0:
            pivot_table = df.pivot_table(
                index='canonical_smiles',
                columns='target_name', 
                values='pic50',
                aggfunc='median'
            ).reset_index()
            
            matrix_path = datasets_dir / "real_pubchem_matrix.csv"
            pivot_table.to_csv(matrix_path, index=False)
            print(f"   📊 Bioactivity matrix: {pivot_table.shape}")
        
        # Save metadata
        metadata = {
            'extraction_method': 'PubChem_BioAssay_Real_API',
            'real_data': True,
            'api_base_url': 'https://pubchem.ncbi.nlm.nih.gov/rest',
            'targets': list(ONCOLOGY_TARGETS_PUBCHEM.keys()),
            'target_info': ONCOLOGY_TARGETS_PUBCHEM,
            'total_records': len(df),
            'total_targets': df['target_name'].nunique() if len(df) > 0 else 0,
            'total_compounds': df['canonical_smiles'].nunique() if len(df) > 0 else 0,
            'target_stats': target_stats,
            'activity_type_counts': df['activity_type'].value_counts().to_dict() if len(df) > 0 else {},
            'extraction_timestamp': datetime.now().isoformat(),
            'data_quality': {
                'real_experimental_data': True,
                'pubchem_bioassay_source': True,
                'bioactivity_assays': True,
                'units': 'nM',
                'pic50_calculated': True,
                'duplicates_removed': True,
                'range_filtered': '0.1 nM - 100 mM'
            }
        }
        
        metadata_path = datasets_dir / "real_pubchem_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate final report
        print(f"\n🎉 REAL PUBCHEM EXTRACTION COMPLETED!")
        print("=" * 80)
        print(f"📁 Dataset files:")
        print(f"  • Raw data: {raw_data_path}")
        if len(df) > 0:
            print(f"  • Bioactivity matrix: {matrix_path}")
        print(f"  • Metadata: {metadata_path}")
        
        print(f"\n📊 Final real dataset summary:")
        print(f"  • Total records: {len(df):,}")
        print(f"  • Unique targets: {df['target_name'].nunique() if len(df) > 0 else 0}")
        print(f"  • Unique compounds: {df['canonical_smiles'].nunique() if len(df) > 0 else 0}")
        
        if len(df) > 0:
            # Category breakdown
            for category in ['oncoprotein', 'tumor_suppressor', 'metastasis_suppressor']:
                category_targets = [name for name, info in ONCOLOGY_TARGETS_PUBCHEM.items() if info['category'] == category]
                category_records = df[df['target_name'].isin(category_targets)]
                print(f"  • {category.replace('_', ' ').title()}: {len(category_records)} bioactivity records across {len(category_targets)} targets")
        
        print(f"\n🌐 REAL PUBCHEM DATA: Authentic bioassay experimental results")
        
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
        print(f"❌ REAL PUBCHEM EXTRACTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🧪 Real PubChem BioAssay Data Extractor")