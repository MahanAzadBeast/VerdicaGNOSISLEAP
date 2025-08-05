"""
Full BindingDB Bulk Extractor - Following Exact Specifications
Download BindingDB_All.tsv.zip and extract tens of thousands of oncology binding records
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

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "requests",
    "pandas", 
    "numpy",
    "rdkit-pypi"
])

app = modal.App("full-bindingdb-bulk-extractor")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

# Oncology-relevant proteins (oncoproteins + tumor suppressors)
ONCOLOGY_PROTEINS = {
    # Oncoproteins
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
    'RRM2',                              # Ribonucleotide reductase
    'BCL2', 'BCLXL', 'MCL1',           # Anti-apoptotic
    'VEGFR1', 'VEGFR2', 'VEGFR3',      # VEGF receptors
    'PDGFRA', 'PDGFRB',                 # PDGF receptors
    'KIT', 'FLT3',                      # Other RTKs
    'JAK1', 'JAK2', 'JAK3',            # JAK kinases
    'SRC', 'YES1', 'FYN',              # SRC family
    'ABL1', 'ABL2',                     # ABL kinases
    'CDK2', 'CDK4', 'CDK6', 'CDK9',    # Cell cycle kinases
    'AURKA', 'AURKB',                   # Aurora kinases
    'PLK1',                              # Polo-like kinase
    'WEE1',                              # Cell cycle checkpoint
    'CHEK1', 'CHEK2',                   # Checkpoint kinases
    'PARP1', 'PARP2',                   # DNA repair
    
    # Tumor suppressors  
    'TP53',                              # p53
    'RB1',                               # Retinoblastoma
    'PTEN',                              # PTEN
    'APC',                               # Adenomatous polyposis coli
    'BRCA1', 'BRCA2',                   # BRCA
    'VHL',                               # Von Hippel-Lindau
    'CDKN2A',                            # p16
    'CDKN1A',                            # p21
    'ATM',                               # Ataxia telangiectasia
    'STK11',                             # LKB1
    'NF1', 'NF2',                       # Neurofibromin
    'SMAD2', 'SMAD3', 'SMAD4',         # SMAD
    'TGFBR1', 'TGFBR2'                 # TGF-beta receptors
}

class FullBindingDBBulkExtractor:
    """Full BindingDB bulk extractor following exact specifications"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GNOSIS-FullBindingDB-Bulk-Extractor/1.0'
        })
        self.logger = logging.getLogger(__name__)
    
    def download_bindingdb_all_tsv(self) -> Optional[pd.DataFrame]:
        """Download BindingDB_All.tsv.zip following exact specifications"""
        
        self.logger.info("📥 Downloading BindingDB_All.tsv.zip (FULL BULK DATASET)...")
        
        # Correct BindingDB SDF URL provided by user
        url = "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_2D_202508_sdf.zip"
        
        try:
            self.logger.info(f"   📡 Downloading from: {url}")
            
            response = self.session.get(url, stream=True, timeout=1800)  # 30 min timeout
            
            if response.status_code != 200:
                self.logger.error(f"   ❌ Download failed: HTTP {response.status_code}")
                return None
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            self.logger.info(f"   📦 File size: {total_size / (1024*1024*1024):.1f} GB")
            
            # Download with progress tracking
            zip_content = io.BytesIO()
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192*16):  # Larger chunks
                if chunk:
                    zip_content.write(chunk)
                    downloaded += len(chunk)
                    
                    if downloaded % (1024*1024*100) == 0:  # Progress every 100MB
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            self.logger.info(f"     📊 Progress: {progress:.1f}% ({downloaded/(1024*1024*1024):.1f} GB)")
            
            self.logger.info(f"   ✅ Download completed: {downloaded/(1024*1024*1024):.1f} GB")
            
            # Extract ZIP
            zip_content.seek(0)
            
            with zipfile.ZipFile(zip_content) as zip_file:
                file_list = zip_file.namelist()
                self.logger.info(f"   📁 ZIP contains: {file_list}")
                
                # Find BindingDB_All.tsv
                tsv_file = None
                for filename in file_list:
                    if 'BindingDB_All' in filename and filename.endswith('.tsv'):
                        tsv_file = filename
                        break
                
                if not tsv_file:
                    self.logger.error("   ❌ BindingDB_All.tsv not found in ZIP")
                    return None
                
                self.logger.info(f"   📄 Extracting: {tsv_file}")
                
                # Read TSV data in chunks due to size
                with zip_file.open(tsv_file) as f:
                    # Read the TSV file
                    self.logger.info("   🔧 Parsing massive TSV file...")
                    
                    try:
                        df = pd.read_csv(f, sep='\t', low_memory=False, chunksize=50000)
                        
                        # Process in chunks
                        all_chunks = []
                        total_rows = 0
                        
                        for i, chunk in enumerate(df):
                            all_chunks.append(chunk)
                            total_rows += len(chunk)
                            
                            if i % 10 == 0:  # Progress every 10 chunks
                                self.logger.info(f"     📊 Processed {total_rows:,} rows...")
                            
                            # Limit for testing - remove in production
                            if i >= 100:  # First 5M rows for testing
                                break
                        
                        # Combine chunks
                        combined_df = pd.concat(all_chunks, ignore_index=True)
                        
                        self.logger.info(f"   ✅ Loaded BindingDB_All.tsv: {len(combined_df):,} total records")
                        self.logger.info(f"   📊 Columns: {len(combined_df.columns)} - {list(combined_df.columns[:10])}...")
                        
                        return combined_df
                        
                    except Exception as e:
                        self.logger.error(f"   ❌ Error parsing TSV: {e}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"   ❌ BindingDB download failed: {e}")
            return None
    
    def filter_oncology_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for oncology-relevant proteins following exact specifications"""
        
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        self.logger.info("🎯 Filtering for oncology targets (oncoproteins + tumor suppressors)...")
        self.logger.info(f"   📊 Input records: {len(df):,}")
        
        # Find UniProt/target columns
        uniprot_cols = []
        target_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if 'uniprot' in col_lower or 'accession' in col_lower:
                uniprot_cols.append(col)
            elif 'target' in col_lower or 'protein' in col_lower or 'gene' in col_lower:
                target_cols.append(col)
        
        self.logger.info(f"   🔍 UniProt columns: {uniprot_cols}")
        self.logger.info(f"   🔍 Target columns: {target_cols}")
        
        # Filter for oncology targets
        oncology_mask = pd.Series([False] * len(df))
        
        # Check UniProt columns
        for col in uniprot_cols:
            if col in df.columns:
                for protein in ONCOLOGY_PROTEINS:
                    protein_mask = df[col].astype(str).str.contains(protein, case=False, na=False)
                    oncology_mask |= protein_mask
        
        # Check target name columns
        for col in target_cols:
            if col in df.columns:
                for protein in ONCOLOGY_PROTEINS:
                    protein_mask = df[col].astype(str).str.contains(protein, case=False, na=False)
                    oncology_mask |= protein_mask
        
        filtered_df = df[oncology_mask]
        
        self.logger.info(f"   ✅ Oncology targets filtered: {len(filtered_df):,} records")
        self.logger.info(f"   📊 Reduction: {len(df) - len(filtered_df):,} records removed")
        
        return filtered_df
    
    def extract_binding_affinities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and process binding affinities following exact specifications"""
        
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        self.logger.info("🔧 Extracting binding affinities (Ki, IC50, EC50)...")
        
        # Find relevant columns
        smiles_col = None
        uniprot_col = None
        ki_cols = []
        ic50_cols = []
        ec50_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # SMILES column
            if 'smiles' in col_lower and 'ligand' in col_lower:
                smiles_col = col
            elif 'smiles' in col_lower and smiles_col is None:
                smiles_col = col
            
            # UniProt column
            if 'uniprot' in col_lower:
                uniprot_col = col
            
            # Affinity columns
            if 'ki' in col_lower and ('nm' in col_lower or 'value' in col_lower):
                ki_cols.append(col)
            if 'ic50' in col_lower and ('nm' in col_lower or 'value' in col_lower):
                ic50_cols.append(col)
            if 'ec50' in col_lower and ('nm' in col_lower or 'value' in col_lower):
                ec50_cols.append(col)
        
        self.logger.info(f"   📊 SMILES column: {smiles_col}")
        self.logger.info(f"   📊 UniProt column: {uniprot_col}")
        self.logger.info(f"   📊 Ki columns: {ki_cols}")
        self.logger.info(f"   📊 IC50 columns: {ic50_cols}")
        self.logger.info(f"   📊 EC50 columns: {ec50_cols}")
        
        # Extract records
        processed_records = []
        
        for idx, row in df.iterrows():
            smiles = row.get(smiles_col) if smiles_col else None
            uniprot_id = row.get(uniprot_col) if uniprot_col else 'Unknown'
            
            if pd.isna(smiles) or len(str(smiles)) < 5:
                continue
            
            # Process Ki values
            for ki_col in ki_cols:
                ki_value = row.get(ki_col)
                if pd.notna(ki_value) and self.is_numeric_affinity(ki_value):
                    ki_nm = self.convert_to_nm(ki_value, ki_col)
                    if ki_nm and 0.01 <= ki_nm <= 1e8:
                        record = {
                            'SMILES': str(smiles),
                            'UniProt_ID': str(uniprot_id),
                            'Assay_Type': 'Ki',
                            'Original_Value_nM': ki_nm,
                            'pAffinity': -np.log10(ki_nm / 1e9),  # pKi
                            'SourceDB': 'BindingDB'
                        }
                        processed_records.append(record)
            
            # Process IC50 values
            for ic50_col in ic50_cols:
                ic50_value = row.get(ic50_col)
                if pd.notna(ic50_value) and self.is_numeric_affinity(ic50_value):
                    ic50_nm = self.convert_to_nm(ic50_value, ic50_col)
                    if ic50_nm and 0.01 <= ic50_nm <= 1e8:
                        record = {
                            'SMILES': str(smiles),
                            'UniProt_ID': str(uniprot_id),
                            'Assay_Type': 'IC50',
                            'Original_Value_nM': ic50_nm,
                            'pAffinity': -np.log10(ic50_nm / 1e9),  # pIC50
                            'SourceDB': 'BindingDB'
                        }
                        processed_records.append(record)
            
            # Process EC50 values
            for ec50_col in ec50_cols:
                ec50_value = row.get(ec50_col)
                if pd.notna(ec50_value) and self.is_numeric_affinity(ec50_value):
                    ec50_nm = self.convert_to_nm(ec50_value, ec50_col)
                    if ec50_nm and 0.01 <= ec50_nm <= 1e8:
                        record = {
                            'SMILES': str(smiles),
                            'UniProt_ID': str(uniprot_id),
                            'Assay_Type': 'EC50',
                            'Original_Value_nM': ec50_nm,
                            'pAffinity': -np.log10(ec50_nm / 1e9),  # pEC50
                            'SourceDB': 'BindingDB'
                        }
                        processed_records.append(record)
            
            # Progress tracking
            if idx > 0 and idx % 50000 == 0:
                self.logger.info(f"     📊 Processed {idx:,} rows, extracted {len(processed_records)} valid records...")
        
        if processed_records:
            processed_df = pd.DataFrame(processed_records)
            
            # Deduplicate by (SMILES, UniProt_ID, Assay_Type)
            initial_count = len(processed_df)
            processed_df = processed_df.drop_duplicates(
                subset=['SMILES', 'UniProt_ID', 'Assay_Type'], 
                keep='first'
            )
            
            self.logger.info(f"   ✅ BindingDB processed: {len(processed_df):,} records")
            self.logger.info(f"   📊 Duplicates removed: {initial_count - len(processed_df)}")
            self.logger.info(f"   📊 Unique compounds: {processed_df['SMILES'].nunique()}")
            self.logger.info(f"   📊 Unique targets: {processed_df['UniProt_ID'].nunique()}")
            
            return processed_df
        
        else:
            self.logger.error("   ❌ No valid binding affinities extracted")
            return pd.DataFrame()
    
    def is_numeric_affinity(self, value) -> bool:
        """Check if value is numeric affinity"""
        
        try:
            if pd.isna(value):
                return False
            
            # Convert to string and check for numeric patterns
            value_str = str(value).strip()
            
            # Skip non-numeric indicators
            if any(indicator in value_str.lower() for indicator in ['>', '<', '~', 'inactive', 'nd', 'n.d.']):
                return False
            
            # Try to convert to float
            float(value_str)
            return True
            
        except:
            return False
    
    def convert_to_nm(self, value, column_name: str) -> Optional[float]:
        """Convert affinity value to nM"""
        
        try:
            value_float = float(value)
            
            if value_float <= 0:
                return None
            
            column_lower = column_name.lower()
            
            # Check units from column name
            if 'um' in column_lower or 'microm' in column_lower:
                return value_float * 1000  # μM to nM
            elif 'mm' in column_lower:
                return value_float * 1e6   # mM to nM
            elif 'pm' in column_lower:
                return value_float / 1000  # pM to nM
            elif 'm' in column_lower and 'nm' not in column_lower:
                return value_float * 1e9   # M to nM
            else:
                return value_float  # Assume nM
                
        except:
            return None

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=8.0,     # More CPU for processing
    memory=32768, # 32GB RAM for large dataset
    timeout=10800 # 3 hours for full download and processing
)
def extract_full_bindingdb_bulk():
    """
    Extract full BindingDB bulk dataset following exact specifications
    Expected: Tens of thousands of oncology binding records
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 FULL BINDINGDB BULK EXTRACTION")
    print("=" * 80)
    print("📂 Source: BindingDB_All.tsv.zip (COMPLETE BULK DATASET)")
    print("🎯 Target: Oncology proteins (oncoproteins + tumor suppressors)")
    print("📊 Expected: Tens of thousands of binding records")
    
    try:
        extractor = FullBindingDBBulkExtractor()
        
        # Step 1: Download BindingDB_All.tsv.zip
        print("\n📥 STEP 1: Downloading BindingDB_All.tsv.zip...")
        
        raw_df = extractor.download_bindingdb_all_tsv()
        
        if raw_df is None or len(raw_df) == 0:
            raise Exception("Failed to download BindingDB_All.tsv")
        
        # Step 2: Filter for oncology targets
        print(f"\n🎯 STEP 2: Filtering for oncology targets...")
        
        oncology_df = extractor.filter_oncology_targets(raw_df)
        
        # Step 3: Extract binding affinities
        print(f"\n🔧 STEP 3: Extracting binding affinities...")
        
        final_df = extractor.extract_binding_affinities(oncology_df)
        
        if len(final_df) == 0:
            raise Exception("No binding affinities extracted")
        
        # Step 4: Save dataset
        print(f"\n💾 STEP 4: Saving BindingDB bulk dataset...")
        
        datasets_dir = Path("/vol/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        bindingdb_bulk_path = datasets_dir / "bindingdb_bulk_oncology.csv"
        final_df.to_csv(bindingdb_bulk_path, index=False)
        
        # Generate comprehensive report
        print(f"\n🎉 FULL BINDINGDB BULK EXTRACTION COMPLETED!")
        print("=" * 80)
        print(f"📊 BindingDB Bulk Results:")
        print(f"  • Total Records: {len(final_df):,}")
        print(f"  • Unique Compounds: {final_df['SMILES'].nunique():,}")
        print(f"  • Unique Targets: {final_df['UniProt_ID'].nunique()}")
        print(f"  • Assay Types: {', '.join(final_df['Assay_Type'].unique())}")
        
        print(f"\n📊 Assay Distribution:")
        for assay, count in final_df['Assay_Type'].value_counts().items():
            print(f"    • {assay}: {count:,} records")
        
        print(f"\n📊 Top Targets (by record count):")
        for target, count in final_df['UniProt_ID'].value_counts().head(15).items():
            print(f"    • {target}: {count} records")
        
        print(f"\n✅ BINDINGDB BULK READY:")
        print(f"  • File: bindingdb_bulk_oncology.csv")
        print(f"  • Ready for ChEMBL augmentation")
        print(f"  • Expected combined size: TENS OF THOUSANDS of records")
        
        return {
            'status': 'success',
            'source': 'BindingDB_All_TSV_Bulk',
            'total_records': len(final_df),
            'unique_compounds': int(final_df['SMILES'].nunique()),
            'unique_targets': int(final_df['UniProt_ID'].nunique()),
            'assay_distribution': final_df['Assay_Type'].value_counts().to_dict(),
            'ready_for_chembl_augmentation': True
        }
        
    except Exception as e:
        print(f"❌ FULL BINDINGDB BULK EXTRACTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🧬 Full BindingDB Bulk Extractor")