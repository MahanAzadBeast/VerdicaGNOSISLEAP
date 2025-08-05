"""
BindingDB SDF Extractor - Extract from SDF format
Handle the actual BindingDB_All_2D.sdf file format
"""

import modal
import requests
import pandas as pd
import numpy as np
import logging
import zipfile
import io
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
from rdkit import Chem
from rdkit.Chem import Descriptors

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "requests",
    "pandas", 
    "numpy",
    "rdkit-pypi"
])

app = modal.App("bindingdb-sdf-extractor")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

# Oncology-relevant proteins
ONCOLOGY_PROTEINS = {
    'EGFR', 'ERBB2', 'ERBB3', 'ERBB4',  # EGFR family
    'ALK', 'ROS1', 'MET', 'RET',        # Receptor tyrosine kinases
    'BRAF', 'RAF1', 'ARAF',             # RAF kinases  
    'PI3KCA', 'PIK3CB', 'PIK3CG',       # PI3K
    'AKT1', 'AKT2', 'AKT3',             # AKT
    'MTOR',                              # mTOR
    'STAT3', 'STAT5A', 'STAT5B',        # STAT
    'MYC', 'MYCN', 'MYCL',              # MYC family
    'CTNNB1',                            # Beta-catenin
    'MDM2', 'MDM4',                     # MDM
    'BCL2', 'BCLXL', 'MCL1',           # Anti-apoptotic
    'VEGFR1', 'VEGFR2', 'VEGFR3',      # VEGF receptors
    'PDGFRA', 'PDGFRB',                 # PDGF receptors
    'KIT', 'FLT3',                      # Other RTKs
    'JAK1', 'JAK2', 'JAK3',            # JAK kinases
    'SRC', 'YES1', 'FYN',              # SRC family
    'ABL1', 'ABL2',                     # ABL kinases
    'CDK2', 'CDK4', 'CDK6', 'CDK9',    # Cell cycle kinases
    'AURKA', 'AURKB',                   # Aurora kinases
    'PARP1', 'PARP2',                   # DNA repair
    'TP53', 'RB1', 'PTEN', 'BRCA1', 'BRCA2'  # Tumor suppressors
}

class BindingDBSDFExtractor:
    """BindingDB SDF extractor for oncology proteins"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GNOSIS-BindingDB-SDF-Extractor/1.0'
        })
        self.logger = logging.getLogger(__name__)
    
    def download_and_extract_sdf(self) -> Optional[List[Dict[str, Any]]]:
        """Download and extract BindingDB SDF file"""
        
        self.logger.info("📥 Downloading BindingDB_All_2D.sdf.zip...")
        
        url = "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_2D_202508_sdf.zip"
        
        try:
            response = self.session.get(url, stream=True, timeout=1800)
            
            if response.status_code != 200:
                self.logger.error(f"   ❌ Download failed: HTTP {response.status_code}")
                return None
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            self.logger.info(f"   📦 File size: {total_size / (1024*1024*1024):.1f} GB")
            
            # Download with progress
            zip_content = io.BytesIO()
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192*16):
                if chunk:
                    zip_content.write(chunk)
                    downloaded += len(chunk)
                    
                    if downloaded % (1024*1024*100) == 0:  # Progress every 100MB
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            self.logger.info(f"     📊 Progress: {progress:.1f}%")
            
            self.logger.info(f"   ✅ Download completed: {downloaded/(1024*1024*1024):.1f} GB")
            
            # Extract SDF
            zip_content.seek(0)
            
            with zipfile.ZipFile(zip_content) as zip_file:
                file_list = zip_file.namelist()
                self.logger.info(f"   📁 ZIP contains: {file_list}")
                
                # Find SDF file
                sdf_file = None
                for filename in file_list:
                    if filename.endswith('.sdf'):
                        sdf_file = filename
                        break
                
                if not sdf_file:
                    self.logger.error("   ❌ SDF file not found in ZIP")
                    return None
                
                self.logger.info(f"   📄 Extracting: {sdf_file}")
                
                # Process SDF file
                with zip_file.open(sdf_file) as f:
                    return self.process_sdf_content(f)
                        
        except Exception as e:
            self.logger.error(f"   ❌ BindingDB download failed: {e}")
            return None
    
    def process_sdf_content(self, sdf_file) -> List[Dict[str, Any]]:
        """Process SDF file content to extract oncology binding data"""
        
        self.logger.info("🔧 Processing SDF content for oncology targets...")
        
        records = []
        mol_count = 0
        oncology_count = 0
        
        # Read SDF in chunks using RDKit
        suppl = Chem.ForwardSDMolSupplier(sdf_file)
        
        for mol in suppl:
            if mol is None:
                continue
                
            mol_count += 1
            
            try:
                # Get molecular properties
                smiles = Chem.MolToSmiles(mol)
                if not smiles or len(smiles) < 5:
                    continue
                
                # Extract BindingDB properties from SDF
                props = mol.GetPropsAsDict()
                
                # Look for target information in various property fields
                target_info = ""
                uniprot_id = ""
                
                # Common BindingDB property names for targets
                target_fields = ['Target Name', 'UniProt ID', 'Target', 'Protein Name']
                
                for field in target_fields:
                    if field in props:
                        if 'uniprot' in field.lower():
                            uniprot_id = str(props[field])
                        else:
                            target_info = str(props[field])
                        break
                
                # Check if this is an oncology target
                is_oncology = False
                for protein in ONCOLOGY_PROTEINS:
                    if (protein.upper() in target_info.upper() or 
                        protein.upper() in uniprot_id.upper()):
                        is_oncology = True
                        break
                
                if not is_oncology:
                    continue
                
                oncology_count += 1
                
                # Extract binding affinity data
                affinity_data = []
                
                # Common BindingDB affinity fields
                affinity_fields = {
                    'Ki (nM)': 'Ki',
                    'IC50 (nM)': 'IC50', 
                    'EC50 (nM)': 'EC50',
                    'Kd (nM)': 'Kd',
                    'Ki': 'Ki',
                    'IC50': 'IC50',
                    'EC50': 'EC50',
                    'Kd': 'Kd'
                }
                
                for field_name, assay_type in affinity_fields.items():
                    if field_name in props:
                        try:
                            value = float(props[field_name])
                            if 0.01 <= value <= 1e8:  # Reasonable range for nM
                                affinity_data.append({
                                    'assay_type': assay_type,
                                    'affinity_nm': value
                                })
                        except:
                            continue
                
                # Create records for each affinity measurement
                for affinity in affinity_data:
                    record = {
                        'SMILES': smiles,
                        'UniProt_ID': uniprot_id or target_info,
                        'target_name': target_info,
                        'Assay_Type': affinity['assay_type'],
                        'Original_Value_nM': affinity['affinity_nm'],
                        'pAffinity': -np.log10(affinity['affinity_nm'] / 1e9),
                        'SourceDB': 'BindingDB_SDF'
                    }
                    records.append(record)
                
            except Exception as e:
                continue
            
            # Progress tracking
            if mol_count % 50000 == 0:
                self.logger.info(f"   📊 Processed {mol_count:,} molecules, found {oncology_count:,} oncology targets, extracted {len(records)} binding records")
            
            # Limit for testing - remove in production
            if mol_count >= 200000:  # First 200K molecules
                break
        
        self.logger.info(f"   ✅ SDF processing complete:")
        self.logger.info(f"     • Total molecules: {mol_count:,}")
        self.logger.info(f"     • Oncology targets: {oncology_count:,}")
        self.logger.info(f"     • Binding records: {len(records)}")
        
        return records

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=8.0,     # More CPU for processing
    memory=32768, # 32GB RAM for large dataset
    timeout=14400 # 4 hours for full download and processing
)
def extract_bindingdb_sdf():
    """
    Extract BindingDB oncology data from SDF format
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 BINDINGDB SDF EXTRACTION")
    print("=" * 80)
    print("📂 Source: BindingDB_All_2D.sdf (SDF FORMAT)")
    print("🎯 Target: Oncology proteins (oncoproteins + tumor suppressors)")
    print("📊 Expected: Thousands of binding records")
    
    try:
        extractor = BindingDBSDFExtractor()
        
        # Download and extract SDF
        print("\n📥 STEP 1: Downloading and processing SDF...")
        
        records = extractor.download_and_extract_sdf()
        
        if not records or len(records) == 0:
            raise Exception("No binding records extracted from SDF")
        
        # Convert to DataFrame
        print(f"\n📊 STEP 2: Creating dataset...")
        
        df = pd.DataFrame(records)
        
        # Deduplicate by (SMILES, UniProt_ID, Assay_Type)
        print(f"\n🔧 STEP 3: Deduplicating records...")
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=['SMILES', 'UniProt_ID', 'Assay_Type'], keep='first')
        
        print(f"   Removed {initial_count - len(df)} duplicates")
        
        # Save dataset
        print(f"\n💾 STEP 4: Saving BindingDB SDF dataset...")
        
        datasets_dir = Path("/vol/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        bindingdb_sdf_path = datasets_dir / "bindingdb_sdf_oncology.csv"
        df.to_csv(bindingdb_sdf_path, index=False)
        
        # Generate report
        print(f"\n🎉 BINDINGDB SDF EXTRACTION COMPLETED!")
        print("=" * 80)
        print(f"📊 BindingDB SDF Results:")
        print(f"  • Total Records: {len(df):,}")
        print(f"  • Unique Compounds: {df['SMILES'].nunique():,}")
        print(f"  • Unique Targets: {df['UniProt_ID'].nunique()}")
        
        print(f"\n📊 Assay Distribution:")
        for assay, count in df['Assay_Type'].value_counts().items():
            print(f"    • {assay}: {count:,} records")
        
        print(f"\n📊 Top Targets:")
        for target, count in df['UniProt_ID'].value_counts().head(10).items():
            print(f"    • {target}: {count} records")
        
        print(f"\n✅ BINDINGDB SDF READY:")
        print(f"  • File: bindingdb_sdf_oncology.csv")
        print(f"  • Ready for ChEMBL merger")
        
        return {
            'status': 'success',
            'source': 'BindingDB_SDF',
            'total_records': len(df),
            'unique_compounds': int(df['SMILES'].nunique()),
            'unique_targets': int(df['UniProt_ID'].nunique()),
            'assay_distribution': df['Assay_Type'].value_counts().to_dict(),
            'ready_for_merger': True
        }
        
    except Exception as e:
        print(f"❌ BINDINGDB SDF EXTRACTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🧬 BindingDB SDF Extractor")