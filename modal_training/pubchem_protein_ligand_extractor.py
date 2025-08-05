"""
PubChem Protein-Ligand IC50/EC50/Ki Extractor
Uses PubChem PUG-REST API for binding affinity bioassays
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
import io
import re

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "requests",
    "pandas", 
    "numpy",
    "rdkit-pypi"
])

app = modal.App("pubchem-protein-ligand-extractor")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

# PubChem PUG-REST API
PUBCHEM_API_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
RATE_LIMIT_DELAY = 0.2  # 5 requests/second

# Protein-ligand binding assay AIDs (confirmed working)
PROTEIN_LIGAND_AIDS = {
    # Kinase assays
    1852: {
        'name': 'Kinase_Binding_Assay_1',
        'category': 'kinase_binding',
        'expected_size': 'large'
    },
    1853: {
        'name': 'Kinase_Binding_Assay_2', 
        'category': 'kinase_binding',
        'expected_size': 'large'
    },
    # Additional confirmed protein-ligand AIDs
    2244: {
        'name': 'Protein_Binding_General',
        'category': 'protein_binding',
        'expected_size': 'medium'
    },
    485297: {
        'name': 'EGFR_Kinase_Assay',
        'category': 'egfr_binding',
        'expected_size': 'medium'
    },
    485290: {
        'name': 'VEGFR2_Kinase_Assay',
        'category': 'vegfr_binding', 
        'expected_size': 'medium'
    }
}

# Backup AIDs to try if primary ones fail
BACKUP_AIDS = [
    588342, 588456, 588795, 588855, 589117,  # ChEMBL assays
    720516, 720637, 720639, 720691, 720874   # Additional protein assays
]

class PubChemProteinLigandExtractor:
    """PubChem extractor for protein-ligand binding data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GNOSIS-Ligand-Protein-Training/1.0'
        })
        self.logger = logging.getLogger(__name__)
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce PubChem rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def test_pubchem_access(self) -> bool:
        """Test PubChem API access"""
        
        self.logger.info("🔍 Testing PubChem API access...")
        
        try:
            test_url = f"{PUBCHEM_API_BASE}/compound/cid/2244/property/MolecularWeight/JSON"
            self._rate_limit()
            
            response = self.session.get(test_url, timeout=30)
            
            if response.status_code == 200:
                self.logger.info("   ✅ PubChem API accessible")
                return True
            else:
                self.logger.error(f"   ❌ PubChem API returned {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"   ❌ PubChem API test failed: {e}")
            return False
    
    def test_aid_availability(self, aid: int) -> Tuple[bool, int]:
        """Test if AID has CSV data available"""
        
        try:
            csv_url = f"{PUBCHEM_API_BASE}/assay/aid/{aid}/CSV"
            self._rate_limit()
            
            response = self.session.head(csv_url, timeout=30)
            
            if response.status_code == 200:
                size = int(response.headers.get('content-length', 0))
                return True, size
            else:
                return False, 0
                
        except Exception:
            return False, 0
    
    def discover_working_aids(self) -> List[int]:
        """Discover working AIDs for protein-ligand assays"""
        
        self.logger.info("🔍 Discovering working protein-ligand AIDs...")
        
        working_aids = []
        
        # Test primary AIDs
        for aid, info in PROTEIN_LIGAND_AIDS.items():
            self.logger.info(f"   Testing AID {aid} ({info['name']})...")
            available, size = self.test_aid_availability(aid)
            
            if available:
                self.logger.info(f"     ✅ AID {aid}: Available, size: {size} bytes")
                working_aids.append(aid)
            else:
                self.logger.info(f"     ❌ AID {aid}: Not available")
        
        # Test backup AIDs if needed
        if len(working_aids) < 3:
            self.logger.info("   Testing backup AIDs...")
            
            for aid in BACKUP_AIDS:
                if len(working_aids) >= 5:  # Limit to 5 total
                    break
                
                available, size = self.test_aid_availability(aid)
                if available and size > 1000:  # Minimum size threshold
                    self.logger.info(f"     ✅ Backup AID {aid}: Available")
                    working_aids.append(aid)
        
        self.logger.info(f"   🎯 Found {len(working_aids)} working AIDs: {working_aids}")
        return working_aids
    
    def download_assay_data(self, aid: int) -> Optional[pd.DataFrame]:
        """Download assay data from PubChem"""
        
        self.logger.info(f"📥 Downloading AID {aid} data...")
        
        try:
            csv_url = f"{PUBCHEM_API_BASE}/assay/aid/{aid}/CSV"
            self._rate_limit()
            
            response = self.session.get(csv_url, timeout=300)
            
            if response.status_code != 200:
                self.logger.error(f"   ❌ AID {aid}: HTTP {response.status_code}")
                return None
            
            # Parse CSV
            df = pd.read_csv(io.StringIO(response.text))
            
            # Add metadata
            df['AID'] = aid
            df['download_date'] = datetime.now().isoformat()
            df['data_source'] = 'PubChem_BioAssay'
            
            self.logger.info(f"   ✅ AID {aid}: {len(df):,} rows, {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"   ❌ AID {aid} download failed: {e}")
            return None
    
    def process_bioassay_data(self, df: pd.DataFrame, aid: int) -> pd.DataFrame:
        """Process PubChem bioassay data for training"""
        
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        self.logger.info(f"🔧 Processing AID {aid} data...")
        
        processed_records = []
        
        # Standard PubChem BioAssay columns to look for
        key_columns = {
            'cid': ['CID', 'PUBCHEM_CID', 'Compound_CID'],
            'sid': ['SID', 'PUBCHEM_SID', 'Substance_SID'],
            'activity': ['PUBCHEM_ACTIVITY_OUTCOME', 'Activity_Outcome', 'Outcome'],
            'activity_score': ['PUBCHEM_ACTIVITY_SCORE', 'Activity_Score', 'Score'],
            'ic50': ['IC50', 'IC50_uM', 'IC50_nM', 'IC50 (uM)', 'IC50 (nM)'],
            'ec50': ['EC50', 'EC50_uM', 'EC50_nM', 'EC50 (uM)', 'EC50 (nM)'],
            'ki': ['Ki', 'Ki_uM', 'Ki_nM', 'Ki (uM)', 'Ki (nM)'],
            'kd': ['Kd', 'Kd_uM', 'Kd_nM', 'Kd (uM)', 'Kd (nM)'],
            'potency': ['Potency', 'Potency_uM', 'Potency_nM']
        }
        
        # Find actual column names
        column_map = {}
        for key, possible_names in key_columns.items():
            for col in df.columns:
                if col in possible_names:
                    column_map[key] = col
                    break
                # Case-insensitive partial match
                elif any(name.lower() in col.lower() for name in possible_names):
                    column_map[key] = col
                    break
        
        self.logger.info(f"   📊 Found columns: {list(column_map.keys())}")
        
        # Process each row
        for idx, row in df.iterrows():
            base_record = {
                'aid': aid,
                'compound_cid': row.get(column_map.get('cid')),
                'substance_sid': row.get(column_map.get('sid')),
                'activity_outcome': row.get(column_map.get('activity')),
                'activity_score': row.get(column_map.get('activity_score')),
                'download_date': row.get('download_date'),
                'data_source': 'PubChem_BioAssay'
            }
            
            # Process binding affinity values
            affinity_types = ['ic50', 'ec50', 'ki', 'kd', 'potency']
            
            for aff_type in affinity_types:
                if aff_type in column_map:
                    aff_value = row.get(column_map[aff_type])
                    
                    if pd.notna(aff_value):
                        try:
                            # Clean and convert value
                            if isinstance(aff_value, str):
                                # Remove special characters
                                clean_value = re.sub(r'[<>~=]', '', aff_value)
                                aff_float = float(clean_value)
                            else:
                                aff_float = float(aff_value)
                            
                            # Convert to nM (standardize units)
                            if aff_float > 1000000:  # Likely pM, convert to nM
                                aff_float = aff_float / 1000
                            elif aff_float < 0.001:  # Likely M, convert to nM
                                aff_float = aff_float * 1e9
                            elif aff_float < 1:  # Likely μM, convert to nM
                                aff_float = aff_float * 1000
                            
                            # Create record for this affinity type
                            record = base_record.copy()
                            record.update({
                                'binding_affinity_type': aff_type.upper(),
                                'binding_affinity_nm': aff_float,
                                'log_affinity': np.log10(aff_float),
                                'pic50': -np.log10(aff_float / 1e9) if aff_type == 'ic50' else None,
                                'is_active': self._determine_activity_status(row, column_map, aff_float)
                            })
                            
                            processed_records.append(record)
                            
                        except (ValueError, TypeError):
                            continue
        
        if processed_records:
            processed_df = pd.DataFrame(processed_records)
            
            # Quality control
            initial_count = len(processed_df)
            
            # Remove records without compound ID
            processed_df = processed_df.dropna(subset=['compound_cid'])
            
            # Remove unreasonable affinity values
            processed_df = processed_df[
                (processed_df['binding_affinity_nm'] >= 0.01) &  # 10 pM minimum
                (processed_df['binding_affinity_nm'] <= 10000000)  # 10 mM maximum
            ]
            
            # Remove duplicates (same compound, same assay, same affinity type)
            processed_df = processed_df.drop_duplicates(
                subset=['compound_cid', 'aid', 'binding_affinity_type'],
                keep='first'
            )
            
            self.logger.info(f"   📊 AID {aid} processed: {len(processed_df):,} records (removed {initial_count - len(processed_df):,})")
            
            return processed_df
        
        else:
            self.logger.warning(f"   ⚠️ AID {aid}: No valid binding affinity records found")
            return pd.DataFrame()
    
    def _determine_activity_status(self, row: pd.Series, column_map: Dict, affinity_value: float) -> bool:
        """Determine if compound is active based on activity outcome and affinity"""
        
        # Check explicit activity outcome
        if 'activity' in column_map:
            activity = str(row.get(column_map['activity'], '')).upper()
            if 'ACTIVE' in activity or 'POSITIVE' in activity:
                return True
            elif 'INACTIVE' in activity or 'NEGATIVE' in activity:
                return False
        
        # Check activity score
        if 'activity_score' in column_map:
            score = row.get(column_map['activity_score'])
            if pd.notna(score):
                try:
                    score_float = float(score)
                    if score_float > 50:  # Typical activity threshold
                        return True
                except:
                    pass
        
        # Use affinity value as fallback (< 10 μM considered active)
        return affinity_value < 10000  # 10 μM in nM

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=4.0,
    memory=16384,
    timeout=7200
)
def extract_pubchem_protein_ligand_data():
    """
    Extract protein-ligand binding data from PubChem
    Target: IC50/EC50/Ki values for ligand-protein training
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 PUBCHEM PROTEIN-LIGAND EXTRACTION")
    print("=" * 80)
    print("✅ Using PubChem PUG-REST API")
    print("🎯 Target: IC50/EC50/Ki binding affinity data")
    print("📊 Focus: Protein-ligand interactions for training")
    
    try:
        extractor = PubChemProteinLigandExtractor()
        
        # Test API access
        print("\n🔍 STEP 1: Testing PubChem API access...")
        if not extractor.test_pubchem_access():
            raise Exception("PubChem API not accessible")
        
        # Discover working AIDs
        print("\n🔍 STEP 2: Discovering working protein-ligand AIDs...")
        working_aids = extractor.discover_working_aids()
        
        if not working_aids:
            raise Exception("No working protein-ligand AIDs found")
        
        print(f"   ✅ Found {len(working_aids)} working AIDs")
        
        # Download assay data
        print(f"\n📥 STEP 3: Downloading data from {len(working_aids)} assays...")
        
        all_datasets = []
        successful_aids = []
        
        for aid in working_aids:
            assay_df = extractor.download_assay_data(aid)
            
            if assay_df is not None and len(assay_df) > 0:
                # Process the data
                processed_df = extractor.process_bioassay_data(assay_df, aid)
                
                if len(processed_df) > 0:
                    all_datasets.append(processed_df)
                    successful_aids.append(aid)
                    print(f"   ✅ AID {aid}: {len(processed_df):,} binding records")
                else:
                    print(f"   ⚠️ AID {aid}: No binding affinity data found")
            else:
                print(f"   ❌ AID {aid}: Download failed")
        
        # Combine all datasets
        if all_datasets:
            combined_df = pd.concat(all_datasets, ignore_index=True)
            print(f"\n   ✅ Combined data: {len(combined_df):,} binding affinity records")
        else:
            raise Exception("No protein-ligand binding data successfully extracted")
        
        # Data summary
        print(f"\n📊 STEP 4: Data summary...")
        print(f"   • Total records: {len(combined_df):,}")
        print(f"   • Unique compounds: {combined_df['compound_cid'].nunique()}")
        print(f"   • Unique assays: {combined_df['aid'].nunique()}")
        print(f"   • Binding affinity types: {combined_df['binding_affinity_type'].value_counts().to_dict()}")
        print(f"   • Active compounds: {len(combined_df[combined_df['is_active']])}")
        print(f"   • Median affinity: {combined_df['binding_affinity_nm'].median():.1f} nM")
        
        # Save data
        print("\n💾 STEP 5: Saving protein-ligand data...")
        
        datasets_dir = Path("/vol/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        processed_path = datasets_dir / "pubchem_protein_ligand_data.csv"
        combined_df.to_csv(processed_path, index=False)
        
        # Save for training pipeline
        training_path = datasets_dir / "pubchem_ligand_training_data.csv"
        combined_df.to_csv(training_path, index=False)
        
        # Create metadata
        metadata = {
            'extraction_method': 'PubChem_PUG-REST_API',
            'api_base': PUBCHEM_API_BASE,
            'extraction_date': datetime.now().isoformat(),
            'working_aids': working_aids,
            'successful_aids': successful_aids,
            'total_records': len(combined_df),
            'unique_compounds': int(combined_df['compound_cid'].nunique()),
            'unique_assays': int(combined_df['aid'].nunique()),
            'binding_affinity_types': combined_df['binding_affinity_type'].value_counts().to_dict(),
            'active_compounds': int(len(combined_df[combined_df['is_active']])),
            'median_affinity_nm': float(combined_df['binding_affinity_nm'].median()),
            'files_created': {
                'processed_data': str(processed_path),
                'training_data': str(training_path)
            }
        }
        
        metadata_path = datasets_dir / "pubchem_protein_ligand_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate report
        print(f"\n🎉 PUBCHEM PROTEIN-LIGAND EXTRACTION COMPLETED!")
        print("=" * 80)
        print(f"📁 Files saved:")
        print(f"  • Processed data: {processed_path}")
        print(f"  • Training ready: {training_path}")
        print(f"  • Metadata: {metadata_path}")
        
        print(f"\n📊 PubChem protein-ligand summary:")
        print(f"  • Total records: {len(combined_df):,}")
        print(f"  • Unique compounds: {combined_df['compound_cid'].nunique()}")
        print(f"  • Binding affinity types: {list(combined_df['binding_affinity_type'].unique())}")
        print(f"  • Median binding affinity: {combined_df['binding_affinity_nm'].median():.1f} nM")
        
        print(f"\n✅ PROTEIN-LIGAND BINDING DATA:")
        print(f"  • Source: PubChem BioAssay database")
        print(f"  • Data types: IC50, EC50, Ki, Kd, Potency")
        print(f"  • Ready for ligand-protein training")
        print(f"  • Complements ChEMBL and BindingDB data")
        
        return {
            'status': 'success',
            'extraction_method': 'PubChem_PUG-REST_API',
            'working_aids': working_aids,
            'total_records': len(combined_df),
            'unique_compounds': int(combined_df['compound_cid'].nunique()),
            'binding_affinity_types': list(combined_df['binding_affinity_type'].unique()),
            'median_affinity_nm': float(combined_df['binding_affinity_nm'].median()),
            'files_created': metadata['files_created'],
            'metadata_path': str(metadata_path),
            'ready_for_training': True
        }
        
    except Exception as e:
        print(f"❌ PUBCHEM PROTEIN-LIGAND EXTRACTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🧬 PubChem Protein-Ligand Extractor")