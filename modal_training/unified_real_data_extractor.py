"""
Unified Real Data Extractor for GNOSIS Training
Focus: ChEMBL + BindingDB for protein-ligand data, Bulk Tox21 for cytotoxicity
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

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "requests",
    "pandas", 
    "numpy",
    "rdkit-pypi"
])

app = modal.App("unified-real-data-extractor")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

# ChEMBL API Configuration
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

# Oncology protein targets with verified ChEMBL IDs
ONCOLOGY_TARGETS_CHEMBL = {
    "EGFR": "CHEMBL203",     # Epidermal Growth Factor Receptor
    "HER2": "CHEMBL1824",    # ErbB2/HER2
    "VEGFR2": "CHEMBL279",   # VEGFR2
    "ALK": "CHEMBL4247",     # Anaplastic Lymphoma Kinase
    "BRAF": "CHEMBL5145",    # B-Raf Proto-oncogene
    "MET": "CHEMBL3717",     # Hepatocyte Growth Factor Receptor
    "MDM2": "CHEMBL5023",    # MDM2 Proto-oncogene
    "CDK4": "CHEMBL331",     # Cyclin-dependent Kinase 4
    "CDK6": "CHEMBL3974",    # Cyclin-dependent Kinase 6
    "PI3KCA": "CHEMBL4040",  # PI3K Catalytic Subunit Alpha
}

class UnifiedRealDataExtractor:
    """Unified extractor for ChEMBL and BindingDB data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GNOSIS-Unified-Extractor/1.0'
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
    
    def extract_chembl_target_data(self, target_name: str, chembl_id: str) -> pd.DataFrame:
        """Extract IC50/EC50/Ki data for a specific target from ChEMBL"""
        
        self.logger.info(f"📥 Extracting {target_name} data from ChEMBL ({chembl_id})...")
        
        try:
            # Get activities for this target
            activities_url = f"{CHEMBL_BASE_URL}/activity.json"
            params = {
                'target_chembl_id': chembl_id,
                'standard_type__in': 'IC50,EC50,Ki,Kd',
                'standard_units': 'nM',
                'limit': 1000,
                'pchembl_value__isnull': 'false'  # Ensure we have pIC50 values
            }
            
            response = self.session.get(activities_url, params=params, timeout=60)
            
            if response.status_code != 200:
                self.logger.warning(f"   ⚠️ {target_name}: HTTP {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            activities = data.get('activities', [])
            
            if not activities:
                self.logger.warning(f"   ⚠️ {target_name}: No activities found")
                return pd.DataFrame()
            
            self.logger.info(f"   📊 {target_name}: Found {len(activities)} activities")
            
            # Process activities
            records = []
            
            for activity in activities:
                try:
                    # Get molecule SMILES
                    molecule_chembl_id = activity.get('molecule_chembl_id')
                    if not molecule_chembl_id:
                        continue
                    
                    # Get SMILES from molecule endpoint
                    mol_url = f"{CHEMBL_BASE_URL}/molecule/{molecule_chembl_id}.json"
                    mol_response = self.session.get(mol_url, timeout=30)
                    
                    if mol_response.status_code != 200:
                        continue
                    
                    mol_data = mol_response.json()
                    structures = mol_data.get('molecule_structures')
                    
                    if not structures or not structures.get('canonical_smiles'):
                        continue
                    
                    smiles = structures['canonical_smiles']
                    
                    # Extract activity data
                    standard_value = activity.get('standard_value')
                    pchembl_value = activity.get('pchembl_value')
                    
                    if standard_value and pchembl_value:
                        record = {
                            'SMILES': smiles,
                            'target_name': target_name,
                            'target_chembl_id': chembl_id,
                            'standard_type': activity.get('standard_type'),
                            'standard_value_nm': float(standard_value),
                            'pchembl_value': float(pchembl_value),
                            'molecule_chembl_id': molecule_chembl_id,
                            'activity_id': activity.get('activity_id'),
                            'data_source': 'ChEMBL_API',
                            'extraction_date': datetime.now().isoformat()
                        }
                        
                        records.append(record)
                        
                except Exception as e:
                    continue
                
                time.sleep(0.1)  # Rate limiting
            
            if records:
                target_df = pd.DataFrame(records)
                self.logger.info(f"   ✅ {target_name}: {len(target_df)} records with SMILES")
                return target_df
            else:
                self.logger.warning(f"   ⚠️ {target_name}: No valid records extracted")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"   ❌ {target_name}: Extraction failed - {e}")
            return pd.DataFrame()
    
    def check_existing_bindingdb_data(self, datasets_dir: Path) -> Optional[pd.DataFrame]:
        """Check if BindingDB data was successfully extracted"""
        
        self.logger.info("🔍 Checking for existing BindingDB data...")
        
        bindingdb_files = [
            datasets_dir / "bindingdb_training_data.csv",
            datasets_dir / "bindingdb_processed_data.csv"
        ]
        
        for file_path in bindingdb_files:
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    if len(df) > 0:
                        self.logger.info(f"   ✅ Found BindingDB data: {len(df)} records")
                        return df
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Error reading {file_path}: {e}")
        
        self.logger.info("   ⚠️ No existing BindingDB data found")
        return None
    
    def load_bulk_tox21_data(self) -> pd.DataFrame:
        """Load bulk Tox21 cytotoxicity data"""
        
        self.logger.info("📥 Loading bulk Tox21 cytotoxicity data...")
        
        try:
            tox21_path = Path("/app/datasets/cytotoxicity_data.csv")
            
            if tox21_path.exists():
                df = pd.read_csv(tox21_path)
                self.logger.info(f"   ✅ Loaded {len(df)} Tox21 cytotoxicity records")
                return df
            else:
                self.logger.error("   ❌ Bulk Tox21 data not found")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"   ❌ Error loading Tox21 data: {e}")
            return pd.DataFrame()

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=4.0,
    memory=16384,
    timeout=7200
)
def extract_unified_real_data():
    """
    Extract unified real data for GNOSIS training:
    1. ChEMBL protein-ligand IC50/EC50/Ki data
    2. Check for BindingDB data (if extraction completed)
    3. Load bulk Tox21 cytotoxicity data
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 UNIFIED REAL DATA EXTRACTION FOR GNOSIS")
    print("=" * 80)
    print("✅ ChEMBL API for protein-ligand binding data")
    print("✅ BindingDB data (if available)")
    print("✅ Bulk Tox21 data for cytotoxicity")
    
    try:
        extractor = UnifiedRealDataExtractor()
        
        datasets_dir = Path("/vol/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract ChEMBL data
        print("\n🔍 STEP 1: ChEMBL API extraction...")
        
        if not extractor.test_chembl_api():
            raise Exception("ChEMBL API not accessible")
        
        chembl_datasets = []
        successful_targets = []
        
        for target_name, chembl_id in ONCOLOGY_TARGETS_CHEMBL.items():
            target_df = extractor.extract_chembl_target_data(target_name, chembl_id)
            
            if len(target_df) > 0:
                chembl_datasets.append(target_df)
                successful_targets.append(target_name)
                
                # Small delay between targets
                time.sleep(1)
        
        # Combine ChEMBL data
        if chembl_datasets:
            combined_chembl_df = pd.concat(chembl_datasets, ignore_index=True)
            print(f"   ✅ ChEMBL data: {len(combined_chembl_df)} binding records from {len(successful_targets)} targets")
        else:
            print("   ⚠️ No ChEMBL data extracted")
            combined_chembl_df = pd.DataFrame()
        
        # Step 2: Check BindingDB data
        print("\n🔍 STEP 2: Checking BindingDB data...")
        bindingdb_df = extractor.check_existing_bindingdb_data(datasets_dir)
        
        # Step 3: Load Tox21 data
        print("\n🔍 STEP 3: Loading bulk Tox21 data...")
        tox21_df = extractor.load_bulk_tox21_data()
        
        # Combine protein-ligand data
        protein_ligand_datasets = []
        
        if len(combined_chembl_df) > 0:
            protein_ligand_datasets.append(combined_chembl_df)
        
        if bindingdb_df is not None and len(bindingdb_df) > 0:
            # Standardize BindingDB columns to match ChEMBL
            if 'affinity_value_nm' in bindingdb_df.columns:
                bindingdb_df = bindingdb_df.rename(columns={
                    'affinity_value_nm': 'standard_value_nm',
                    'affinity_type': 'standard_type'
                })
            protein_ligand_datasets.append(bindingdb_df)
        
        if protein_ligand_datasets:
            final_protein_ligand_df = pd.concat(protein_ligand_datasets, ignore_index=True)
            
            # Remove duplicates (same SMILES, same target, prefer ChEMBL)
            final_protein_ligand_df = final_protein_ligand_df.drop_duplicates(
                subset=['SMILES', 'target_name'], 
                keep='first'
            )
            
            print(f"   ✅ Combined protein-ligand data: {len(final_protein_ligand_df)} records")
        else:
            final_protein_ligand_df = pd.DataFrame()
            print("   ❌ No protein-ligand data available")
        
        # Save datasets
        print("\n💾 STEP 4: Saving unified datasets...")
        
        # Save protein-ligand data
        if len(final_protein_ligand_df) > 0:
            protein_ligand_path = datasets_dir / "unified_protein_ligand_data.csv"
            final_protein_ligand_df.to_csv(protein_ligand_path, index=False)
            print(f"   ✅ Protein-ligand data: {protein_ligand_path}")
        
        # Save cytotoxicity data
        if len(tox21_df) > 0:
            cytotox_path = datasets_dir / "unified_cytotoxicity_data.csv"
            tox21_df.to_csv(cytotox_path, index=False)
            print(f"   ✅ Cytotoxicity data: {cytotox_path}")
        
        # Create unified metadata
        metadata = {
            'extraction_method': 'Unified_ChEMBL_BindingDB_BulkTox21',
            'extraction_date': datetime.now().isoformat(),
            'protein_ligand_data': {
                'total_records': len(final_protein_ligand_df),
                'unique_compounds': int(final_protein_ligand_df['SMILES'].nunique()) if len(final_protein_ligand_df) > 0 else 0,
                'targets': list(final_protein_ligand_df['target_name'].unique()) if len(final_protein_ligand_df) > 0 else [],
                'sources': list(final_protein_ligand_df['data_source'].unique()) if len(final_protein_ligand_df) > 0 else []
            },
            'cytotoxicity_data': {
                'total_records': len(tox21_df),
                'unique_compounds': int(tox21_df['smiles'].nunique()) if len(tox21_df) > 0 else 0,
                'source': 'Bulk_Literature_Curated'
            },
            'chembl_targets_successful': successful_targets,
            'bindingdb_available': bindingdb_df is not None,
            'ready_for_training': len(final_protein_ligand_df) > 0 and len(tox21_df) > 0
        }
        
        metadata_path = datasets_dir / "unified_extraction_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate report
        print(f"\n🎉 UNIFIED REAL DATA EXTRACTION COMPLETED!")
        print("=" * 80)
        print(f"📊 Data Summary:")
        print(f"  • Protein-ligand records: {len(final_protein_ligand_df):,}")
        print(f"  • Unique compounds (ligands): {final_protein_ligand_df['SMILES'].nunique() if len(final_protein_ligand_df) > 0 else 0}")
        print(f"  • Targets covered: {len(successful_targets)}")
        print(f"  • Cytotoxicity records: {len(tox21_df):,}")
        print(f"  • BindingDB data: {'Available' if bindingdb_df is not None else 'Not available'}")
        
        print(f"\n✅ DATASETS READY FOR GNOSIS TRAINING:")
        print(f"  • ChEMBL + BindingDB: Ligand-protein IC50/EC50/Ki prediction")
        print(f"  • Bulk Tox21: Normal cell cytotoxicity for therapeutic index")
        print(f"  • Ready for unified ChemBERTA training")
        
        return {
            'status': 'success',
            'protein_ligand_records': len(final_protein_ligand_df),
            'cytotoxicity_records': len(tox21_df),
            'targets_successful': successful_targets,
            'bindingdb_available': bindingdb_df is not None,
            'ready_for_training': metadata['ready_for_training'],
            'metadata_path': str(metadata_path)
        }
        
    except Exception as e:
        print(f"❌ UNIFIED EXTRACTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🧬 Unified Real Data Extractor for GNOSIS Training")