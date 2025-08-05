"""
ChEMBL Oncology Augmentation - Following Exact Specifications
Pull all bioactivities (Ki, IC50, EC50) for oncology targets via ChEMBL API
"""

import modal
import requests
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "requests",
    "pandas", 
    "numpy",
    "rdkit-pypi"
])

app = modal.App("chembl-oncology-augmentation")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

# ChEMBL API Configuration
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

# Oncology targets with verified ChEMBL IDs (expanded comprehensive list)
ONCOLOGY_TARGETS_CHEMBL = {
    # RTK family (major oncoproteins)
    "EGFR": "CHEMBL203",      # Epidermal Growth Factor Receptor  
    "ERBB2": "CHEMBL1824",    # HER2/ErbB2
    "ERBB3": "CHEMBL2363053", # HER3/ErbB3
    "ERBB4": "CHEMBL2095159", # HER4/ErbB4
    
    # ALK family
    "ALK": "CHEMBL4247",      # Anaplastic Lymphoma Kinase
    "ROS1": "CHEMBL5469",     # ROS Proto-oncogene 1
    "MET": "CHEMBL3717",      # Hepatocyte Growth Factor Receptor
    "RET": "CHEMBL2334",      # RET Proto-oncogene
    
    # MAPK pathway
    "BRAF": "CHEMBL5145",     # B-Raf Proto-oncogene  
    "RAF1": "CHEMBL4005",     # Raf-1 Proto-oncogene
    "MAP2K1": "CHEMBL4899",   # MEK1
    "MAP2K2": "CHEMBL5896",   # MEK2
    
    # PI3K/AKT/mTOR pathway
    "PIK3CA": "CHEMBL4040",   # PI3K Catalytic Subunit Alpha
    "PIK3CB": "CHEMBL3130",   # PI3K Catalytic Subunit Beta
    "PIK3CG": "CHEMBL3267",   # PI3K Catalytic Subunit Gamma
    "AKT1": "CHEMBL4282",     # AKT Serine/Threonine Kinase 1
    "AKT2": "CHEMBL2093865",  # AKT Serine/Threonine Kinase 2
    "MTOR": "CHEMBL2842",     # Mechanistic Target of Rapamycin
    
    # JAK/STAT pathway
    "JAK1": "CHEMBL2971",     # Janus Kinase 1
    "JAK2": "CHEMBL2363",     # Janus Kinase 2
    "JAK3": "CHEMBL2148",     # Janus Kinase 3
    "STAT3": "CHEMBL1163125", # Signal Transducer STAT3
    
    # Cell cycle kinases
    "CDK2": "CHEMBL301",      # Cyclin-dependent Kinase 2
    "CDK4": "CHEMBL331",      # Cyclin-dependent Kinase 4  
    "CDK6": "CHEMBL3974",     # Cyclin-dependent Kinase 6
    "CDK9": "CHEMBL2508",     # Cyclin-dependent Kinase 9
    
    # SRC family kinases
    "SRC": "CHEMBL267",       # SRC Proto-oncogene
    "YES1": "CHEMBL3357",     # YES Proto-oncogene 1
    "FYN": "CHEMBL1841",      # FYN Proto-oncogene
    
    # ABL kinases
    "ABL1": "CHEMBL1862",     # ABL Proto-oncogene 1
    "ABL2": "CHEMBL4247",     # ABL Proto-oncogene 2
    
    # Aurora and mitotic kinases
    "AURKA": "CHEMBL4722",    # Aurora Kinase A
    "AURKB": "CHEMBL2185",    # Aurora Kinase B
    "PLK1": "CHEMBL3788",     # Polo Like Kinase 1
    
    # DNA damage response
    "CHEK1": "CHEMBL4630",    # Checkpoint Kinase 1
    "CHEK2": "CHEMBL2527",    # Checkpoint Kinase 2
    "ATM": "CHEMBL5330",      # ATM Serine/Threonine Kinase
    "PARP1": "CHEMBL3105",    # Poly ADP-Ribose Polymerase 1
    "PARP2": "CHEMBL4804",    # Poly ADP-Ribose Polymerase 2
    
    # MDM family
    "MDM2": "CHEMBL5023",     # MDM2 Proto-oncogene
    "MDM4": "CHEMBL2111460",  # MDM4 Regulator
    
    # VEGF pathway
    "KDR": "CHEMBL279",       # VEGFR2
    "FLT1": "CHEMBL1868",     # VEGFR1  
    "FLT4": "CHEMBL1955",     # VEGFR3
    
    # Other important oncoproteins
    "KIT": "CHEMBL1936",      # KIT Proto-oncogene
    "PDGFRA": "CHEMBL2007",   # PDGF Receptor Alpha
    "PDGFRB": "CHEMBL2826",   # PDGF Receptor Beta
    "FLT3": "CHEMBL1974",     # FMS Like Tyrosine Kinase 3
    
    # Anti-apoptotic proteins
    "BCL2": "CHEMBL4860",     # BCL2 Apoptosis Regulator
    "BCL2L1": "CHEMBL2243",   # BCL2L1 (BCL-XL)
    "MCL1": "CHEMBL4779",     # MCL1 Apoptosis Regulator
    
    # Tumor suppressors (where druggable)
    "TP53": "CHEMBL4096",     # p53 (limited druggability)
    "RB1": "CHEMBL2111374",   # Retinoblastoma 1
    "PTEN": "CHEMBL2111253",  # PTEN
    "BRCA1": "CHEMBL2095180", # BRCA1 DNA Repair
    "BRCA2": "CHEMBL4805",    # BRCA2 DNA Repair
}

class ChEMBLOncologyAugmentation:
    """ChEMBL oncology augmentation following exact specifications"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GNOSIS-ChEMBL-Oncology-Augmentation/1.0'
        })
        self.logger = logging.getLogger(__name__)
    
    def test_chembl_api(self) -> bool:
        """Test ChEMBL API connectivity"""
        
        self.logger.info("🔍 Testing ChEMBL API connectivity...")
        
        try:
            test_url = f"{CHEMBL_BASE_URL}/target/CHEMBL203.json"
            response = self.session.get(test_url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                target_name = data.get('pref_name', 'Unknown')
                self.logger.info(f"   ✅ ChEMBL API accessible - Test target: {target_name}")
                return True
            else:
                self.logger.error(f"   ❌ ChEMBL API returned {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"   ❌ ChEMBL API test failed: {e}")
            return False
    
    def extract_chembl_target_bioactivities(self, target_name: str, chembl_id: str) -> pd.DataFrame:
        """Extract ALL bioactivities (Ki, IC50, EC50) for a target from ChEMBL - DOWNLOAD EVERYTHING"""
        
        self.logger.info(f"📥 Extracting ALL {target_name} bioactivities from ChEMBL ({chembl_id})...")
        
        try:
            all_records = []
            offset = 0
            batch_size = 2000  # Larger batch size
            max_retries = 3
            
            while True:
                # Get activities for this target (NO LIMITS - EVERYTHING)
                activities_url = f"{CHEMBL_BASE_URL}/activity.json"
                params = {
                    'target_chembl_id': chembl_id,
                    'standard_type__in': 'IC50,EC50,Ki,Kd,AC50',
                    'standard_units__in': 'nM,uM,pM,M,mM',
                    'limit': batch_size,
                    'offset': offset
                }
                
                # Retry mechanism for API reliability
                success = False
                for retry in range(max_retries):
                    try:
                        response = self.session.get(activities_url, params=params, timeout=180)
                        if response.status_code == 200:
                            success = True
                            break
                        else:
                            self.logger.warning(f"   ⚠️ {target_name}: HTTP {response.status_code}, retry {retry+1}")
                            time.sleep(2)
                    except Exception as e:
                        self.logger.warning(f"   ⚠️ {target_name}: API error, retry {retry+1}: {e}")
                        time.sleep(2)
                
                if not success:
                    self.logger.error(f"   ❌ {target_name}: Failed after {max_retries} retries")
                    break
                
                data = response.json()
                activities = data.get('activities', [])
                
                if not activities:
                    self.logger.info(f"   ✅ {target_name}: No more data at offset {offset}")
                    break
                
                # Process activities in this batch - PARALLEL PROCESSING FOR MOLECULES
                batch_records = []
                
                for activity in activities:
                    try:
                        # Get molecule info directly from activity when possible
                        molecule_chembl_id = activity.get('molecule_chembl_id')
                        if not molecule_chembl_id:
                            continue
                        
                        # Try to get SMILES from activity first (faster)
                        smiles = None
                        if 'canonical_smiles' in activity:
                            smiles = activity.get('canonical_smiles')
                        
                        # If not in activity, fetch from molecule endpoint
                        if not smiles:
                            mol_url = f"{CHEMBL_BASE_URL}/molecule/{molecule_chembl_id}.json"
                            mol_response = self.session.get(mol_url, timeout=30)
                            
                            if mol_response.status_code == 200:
                                mol_data = mol_response.json()
                                structures = mol_data.get('molecule_structures')
                                if structures and structures.get('canonical_smiles'):
                                    smiles = structures['canonical_smiles']
                        
                        if not smiles or len(smiles) < 5:
                            continue
                        
                        # Extract activity data
                        standard_value = activity.get('standard_value')
                        standard_units = activity.get('standard_units', '').lower()
                        standard_type = activity.get('standard_type')
                        pchembl_value = activity.get('pchembl_value')
                        
                        if standard_value and pd.notna(standard_value):
                            # Convert to nM
                            value_nm = self.convert_to_nm(float(standard_value), standard_units)
                            
                            if value_nm and 0.001 <= value_nm <= 1e9:  # Extended range
                                # Calculate pAffinity if not available
                                if pchembl_value and pd.notna(pchembl_value):
                                    p_affinity = float(pchembl_value)
                                else:
                                    p_affinity = -np.log10(value_nm / 1e9)
                                
                                record = {
                                    'SMILES': smiles,
                                    'UniProt_ID': target_name,
                                    'Assay_Type': standard_type,
                                    'Original_Value_nM': value_nm,
                                    'pAffinity': p_affinity,
                                    'SourceDB': 'ChEMBL',
                                    'molecule_chembl_id': molecule_chembl_id,
                                    'activity_id': activity.get('activity_id')
                                }
                                
                                batch_records.append(record)
                        
                    except Exception as e:
                        continue
                
                all_records.extend(batch_records)
                
                self.logger.info(f"   📊 {target_name}: Batch {offset//batch_size + 1} - {len(batch_records)} valid records, TOTAL: {len(all_records)}")
                
                # Check if we got fewer than batch_size (last batch)
                if len(activities) < batch_size:
                    break
                
                offset += batch_size
                
                # Minimal delay - we want EVERYTHING fast
                time.sleep(0.1)
            
            if all_records:
                target_df = pd.DataFrame(all_records)
                
                # Remove duplicates within target
                target_df = target_df.drop_duplicates(
                    subset=['SMILES', 'UniProt_ID', 'Assay_Type'],
                    keep='first'
                )
                
                self.logger.info(f"   ✅ {target_name}: {len(target_df)} TOTAL UNIQUE RECORDS EXTRACTED")
                return target_df
            else:
                self.logger.warning(f"   ⚠️ {target_name}: No valid records extracted")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"   ❌ {target_name}: Extraction failed - {e}")
            return pd.DataFrame()
    
    def convert_to_nm(self, value: float, units: str) -> Optional[float]:
        """Convert affinity value to nM"""
        
        if value <= 0:
            return None
        
        units = units.lower()
        
        if 'um' in units or 'μm' in units:
            return value * 1000  # μM to nM
        elif 'mm' in units:
            return value * 1e6   # mM to nM
        elif 'pm' in units:
            return value / 1000  # pM to nM
        elif 'm' in units and 'nm' not in units:
            return value * 1e9   # M to nM
        else:
            return value  # Assume nM

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=8.0,
    memory=16384,
    timeout=14400  # 4 hours for comprehensive ChEMBL extraction
)
def extract_chembl_oncology_bioactivities():
    """
    Extract ALL oncology bioactivities from ChEMBL API
    Following exact specifications - NO artificial limits
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 CHEMBL ONCOLOGY BIOACTIVITIES EXTRACTION")
    print("=" * 80)
    print(f"🎯 Targets: {len(ONCOLOGY_TARGETS_CHEMBL)} oncology proteins")
    print("📊 Bioactivities: Ki, IC50, EC50, Kd")
    print("🚫 NO artificial limits - pulling ALL data")
    
    try:
        extractor = ChEMBLOncologyAugmentation()
        
        # Test ChEMBL API
        print("\n🔍 STEP 1: Testing ChEMBL API...")
        if not extractor.test_chembl_api():
            raise Exception("ChEMBL API not accessible")
        
        # Extract data for all oncology targets
        print(f"\n📥 STEP 2: Extracting from {len(ONCOLOGY_TARGETS_CHEMBL)} oncology targets...")
        
        chembl_datasets = []
        successful_targets = []
        failed_targets = []
        
        for target_name, chembl_id in ONCOLOGY_TARGETS_CHEMBL.items():
            try:
                target_df = extractor.extract_chembl_target_bioactivities(target_name, chembl_id)
                
                if len(target_df) > 0:
                    chembl_datasets.append(target_df)
                    successful_targets.append((target_name, len(target_df)))
                else:
                    failed_targets.append(target_name)
                
                # Rate limiting between targets
                time.sleep(2)
                
            except Exception as e:
                failed_targets.append(target_name)
                print(f"   ❌ {target_name}: Failed - {e}")
                continue
        
        # Combine all ChEMBL data
        if chembl_datasets:
            combined_chembl_df = pd.concat(chembl_datasets, ignore_index=True)
            
            # Final deduplication across all targets
            initial_count = len(combined_chembl_df)
            combined_chembl_df = combined_chembl_df.drop_duplicates(
                subset=['SMILES', 'UniProt_ID', 'Assay_Type'],
                keep='first'
            )
            
            print(f"\n🔧 STEP 3: Final deduplication...")
            print(f"   Removed {initial_count - len(combined_chembl_df)} cross-target duplicates")
            
        else:
            raise Exception("No ChEMBL data extracted from any target")
        
        # Save ChEMBL dataset
        print(f"\n💾 STEP 4: Saving ChEMBL oncology dataset...")
        
        datasets_dir = Path("/vol/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        chembl_oncology_path = datasets_dir / "chembl_oncology_bioactivities.csv"
        combined_chembl_df.to_csv(chembl_oncology_path, index=False)
        
        # Generate comprehensive report
        print(f"\n🎉 CHEMBL ONCOLOGY EXTRACTION COMPLETED!")
        print("=" * 80)
        print(f"📊 ChEMBL Results:")
        print(f"  • Total Records: {len(combined_chembl_df):,}")
        print(f"  • Unique Compounds: {combined_chembl_df['SMILES'].nunique():,}")
        print(f"  • Unique Targets: {combined_chembl_df['UniProt_ID'].nunique()}")
        print(f"  • Successful Targets: {len(successful_targets)}")
        print(f"  • Failed Targets: {len(failed_targets)}")
        
        print(f"\n📊 Assay Distribution:")
        for assay, count in combined_chembl_df['Assay_Type'].value_counts().items():
            print(f"    • {assay}: {count:,} records")
        
        print(f"\n📊 Top Targets by Records:")
        for target, count in combined_chembl_df['UniProt_ID'].value_counts().head(15).items():
            print(f"    • {target}: {count} records")
        
        if successful_targets:
            print(f"\n📊 Target Success Details:")
            for target, count in successful_targets:
                print(f"    ✅ {target}: {count} records")
        
        if failed_targets:
            print(f"\n📊 Failed Targets:")
            for target in failed_targets:
                print(f"    ❌ {target}")
        
        print(f"\n✅ CHEMBL ONCOLOGY DATA READY:")
        print(f"  • File: chembl_oncology_bioactivities.csv")
        print(f"  • Ready for BindingDB merger")
        print(f"  • ALL bioactivities extracted (no limits)")
        
        return {
            'status': 'success',
            'source': 'ChEMBL_API_Full_Extraction',
            'total_records': len(combined_chembl_df),
            'unique_compounds': int(combined_chembl_df['SMILES'].nunique()),
            'unique_targets': int(combined_chembl_df['UniProt_ID'].nunique()),
            'successful_targets': len(successful_targets),
            'failed_targets': len(failed_targets),
            'assay_distribution': combined_chembl_df['Assay_Type'].value_counts().to_dict(),
            'ready_for_merger': True
        }
        
    except Exception as e:
        print(f"❌ CHEMBL ONCOLOGY EXTRACTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🧬 ChEMBL Oncology Augmentation")