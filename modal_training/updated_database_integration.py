"""
Updated Database Integration Pipeline
Integrates ChEMBL + Real PubChem + Real BindingDB + GDSC (NO DTC)
Focus on real API data sources only
"""

import modal
import pandas as pd
import numpy as np
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Modal setup  
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy",
    "rdkit-pypi"
])

app = modal.App("updated-database-integration")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=8.0,
    memory=32768,  # 32GB for large dataset processing
    timeout=7200   # 2 hours
)
def launch_real_data_extraction():
    """
    Launch extraction of all real data sources: PubChem, BindingDB, GDSC
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🚀 LAUNCHING REAL DATA EXTRACTION PIPELINE")
    print("=" * 80)
    print("🎯 Sources: Real PubChem + Real BindingDB + GDSC")
    print("❌ REMOVED: DTC (as requested)")
    
    try:
        extraction_results = {}
        
        # Step 1: Extract Real PubChem Data
        print("\n📊 STEP 1: Extracting Real PubChem BioAssay Data...")
        print("-" * 60)
        
        try:
            from real_pubchem_extractor import app as pubchem_app, extract_real_pubchem_data
            
            with pubchem_app.run() as app_run:
                pubchem_result = extract_real_pubchem_data.remote()
            
            extraction_results['pubchem'] = pubchem_result
            
            if pubchem_result['status'] == 'success':
                print(f"   ✅ PubChem: {pubchem_result['total_records']} real bioassay records")
            else:
                print(f"   ⚠️ PubChem: {pubchem_result.get('message', 'No data extracted')}")
        
        except Exception as e:
            print(f"   ❌ PubChem extraction failed: {e}")
            extraction_results['pubchem'] = {'status': 'failed', 'error': str(e)}
        
        # Step 2: Extract Real BindingDB Data
        print("\n📊 STEP 2: Extracting Real BindingDB Data...")
        print("-" * 60)
        
        try:
            from real_bindingdb_extractor import app as bindingdb_app, extract_real_bindingdb_data
            
            with bindingdb_app.run() as app_run:
                bindingdb_result = extract_real_bindingdb_data.remote()
            
            extraction_results['bindingdb'] = bindingdb_result
            
            if bindingdb_result['status'] == 'success':
                print(f"   ✅ BindingDB: {bindingdb_result['total_records']} real binding records")
            else:
                print(f"   ⚠️ BindingDB: {bindingdb_result.get('message', 'No data extracted')}")
        
        except Exception as e:
            print(f"   ❌ BindingDB extraction failed: {e}")
            extraction_results['bindingdb'] = {'status': 'failed', 'error': str(e)}
        
        # Step 3: Extract GDSC Data
        print("\n📊 STEP 3: Extracting GDSC Cancer Data...")
        print("-" * 60)
        
        try:
            from gdsc_cancer_extractor import app as gdsc_app, extract_gdsc_cancer_data
            
            with gdsc_app.run() as app_run:
                gdsc_result = extract_gdsc_cancer_data.remote()
            
            extraction_results['gdsc'] = gdsc_result
            
            if gdsc_result['status'] == 'success':
                print(f"   ✅ GDSC: {gdsc_result['sensitivity_data']['total_records']} cell line sensitivity records")
                if gdsc_result['genomics_data']['available']:
                    print(f"   ✅ GDSC Genomics: {gdsc_result['genomics_data']['cell_lines']} cell lines with genomic features")
            else:
                print(f"   ⚠️ GDSC: {gdsc_result.get('message', 'No data extracted')}")
        
        except Exception as e:
            print(f"   ❌ GDSC extraction failed: {e}")
            extraction_results['gdsc'] = {'status': 'failed', 'error': str(e)}
        
        # Generate extraction summary
        print(f"\n📊 REAL DATA EXTRACTION SUMMARY:")
        print("=" * 80)
        
        successful_extractions = []
        failed_extractions = []
        
        for source, result in extraction_results.items():
            if result['status'] == 'success':
                successful_extractions.append(source.upper())
                if source == 'gdsc':
                    records = result['sensitivity_data']['total_records']
                else:
                    records = result['total_records']
                print(f"   ✅ {source.upper()}: {records:,} records extracted")
            else:
                failed_extractions.append(source.upper())
                print(f"   ❌ {source.upper()}: Failed - {result.get('error', 'Unknown error')}")
        
        print(f"\n📈 Success Rate: {len(successful_extractions)}/{len(extraction_results)} sources")
        print(f"   • Successful: {', '.join(successful_extractions) if successful_extractions else 'None'}")
        print(f"   • Failed: {', '.join(failed_extractions) if failed_extractions else 'None'}")
        
        return {
            'status': 'success',
            'extractions_completed': len(successful_extractions),
            'total_sources': len(extraction_results),
            'successful_sources': successful_extractions,
            'failed_sources': failed_extractions,
            'extraction_results': extraction_results,
            'ready_for_integration': len(successful_extractions) >= 2,  # Need at least 2 sources
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ REAL DATA EXTRACTION PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=8.0,
    memory=32768,
    timeout=7200
)
def integrate_real_databases():
    """
    Integrate all real databases: ChEMBL + PubChem + BindingDB + GDSC (NO DTC)
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🔗 REAL DATABASE INTEGRATION PIPELINE")
    print("=" * 80)
    print("🎯 Integrating: ChEMBL + Real PubChem + Real BindingDB + GDSC")
    print("❌ EXCLUDED: DTC (removed as requested)")
    
    try:
        datasets_dir = Path("/vol/datasets")
        
        # Step 1: Load all available real datasets
        print("📊 STEP 1: Loading real datasets...")
        print("-" * 60)
        
        available_databases = {}
        
        # Load ChEMBL (existing)
        chembl_path = datasets_dir / "expanded_fixed_raw_data.csv"
        if chembl_path.exists():
            print("📊 Loading ChEMBL dataset...")
            chembl_df = pd.read_csv(chembl_path)
            chembl_df['data_source'] = 'ChEMBL'
            available_databases['ChEMBL'] = {
                'dataframe': chembl_df,
                'records': len(chembl_df),
                'targets': chembl_df['target_name'].nunique() if 'target_name' in chembl_df.columns else 0,
                'file': 'expanded_fixed_raw_data.csv'
            }
            print(f"   ✅ ChEMBL: {len(chembl_df):,} records, {available_databases['ChEMBL']['targets']} targets")
        else:
            print("   ❌ ChEMBL dataset not found")
        
        # Load Real PubChem
        pubchem_path = datasets_dir / "real_pubchem_raw_data.csv"
        if pubchem_path.exists():
            print("📊 Loading Real PubChem dataset...")
            pubchem_df = pd.read_csv(pubchem_path)
            pubchem_df['data_source'] = 'PubChem_BioAssay'
            available_databases['PubChem'] = {
                'dataframe': pubchem_df,
                'records': len(pubchem_df),
                'targets': pubchem_df['target_name'].nunique() if 'target_name' in pubchem_df.columns else 0,
                'file': 'real_pubchem_raw_data.csv'
            }
            print(f"   ✅ Real PubChem: {len(pubchem_df):,} records, {available_databases['PubChem']['targets']} targets")
        else:
            print("   ❌ Real PubChem dataset not found")
        
        # Load Real BindingDB
        bindingdb_path = datasets_dir / "real_bindingdb_raw_data.csv"
        if bindingdb_path.exists():
            print("📊 Loading Real BindingDB dataset...")
            bindingdb_df = pd.read_csv(bindingdb_path)
            bindingdb_df['data_source'] = 'BindingDB'
            available_databases['BindingDB'] = {
                'dataframe': bindingdb_df,
                'records': len(bindingdb_df),
                'targets': bindingdb_df['target_name'].nunique() if 'target_name' in bindingdb_df.columns else 0,
                'file': 'real_bindingdb_raw_data.csv'
            }
            print(f"   ✅ Real BindingDB: {len(bindingdb_df):,} records, {available_databases['BindingDB']['targets']} targets")
        else:
            print("   ❌ Real BindingDB dataset not found")
        
        # Load GDSC (drug sensitivity - different format)
        gdsc_path = datasets_dir / "gdsc_drug_sensitivity.csv"
        if gdsc_path.exists():
            print("📊 Loading GDSC dataset...")
            gdsc_df = pd.read_csv(gdsc_path)
            gdsc_df['data_source'] = 'GDSC'
            available_databases['GDSC'] = {
                'dataframe': gdsc_df,
                'records': len(gdsc_df),
                'targets': gdsc_df['DRUG_NAME'].nunique() if 'DRUG_NAME' in gdsc_df.columns else 0,
                'file': 'gdsc_drug_sensitivity.csv',
                'type': 'cell_line_sensitivity'
            }
            print(f"   ✅ GDSC: {len(gdsc_df):,} cell line sensitivity records, {available_databases['GDSC']['targets']} drugs")
        else:
            print("   ❌ GDSC dataset not found")
        
        if not available_databases:
            raise Exception("No real databases found for integration")
        
        print(f"\n📊 Found {len(available_databases)} real databases for integration")
        
        # Step 2: Create Two Integration Tracks
        print(f"\n🔧 STEP 2: Creating dual integration tracks...")
        print("-" * 60)
        
        # Track 1: Protein-Ligand Activity (ChEMBL + PubChem + BindingDB)
        protein_ligand_databases = {}
        for db_name in ['ChEMBL', 'PubChem', 'BindingDB']:
            if db_name in available_databases:
                protein_ligand_databases[db_name] = available_databases[db_name]
        
        print(f"🎯 Track 1: Protein-Ligand Activity ({len(protein_ligand_databases)} databases)")
        for db_name in protein_ligand_databases:
            print(f"   • {db_name}: {protein_ligand_databases[db_name]['records']:,} records")
        
        # Track 2: Cell Line Drug Sensitivity (GDSC)
        cell_line_databases = {}
        if 'GDSC' in available_databases:
            cell_line_databases['GDSC'] = available_databases['GDSC']
        
        print(f"🎯 Track 2: Cell Line Drug Sensitivity ({len(cell_line_databases)} databases)")
        for db_name in cell_line_databases:
            print(f"   • {db_name}: {cell_line_databases[db_name]['records']:,} records")
        
        # Step 3: Integrate Protein-Ligand Activity Data
        print(f"\n🔗 STEP 3: Integrating protein-ligand activity data...")
        print("-" * 60)
        
        if protein_ligand_databases:
            protein_ligand_integrated = integrate_protein_ligand_data(protein_ligand_databases)
            
            # Save protein-ligand integrated data
            protein_ligand_path = datasets_dir / "real_protein_ligand_integrated.csv"
            protein_ligand_integrated.to_csv(protein_ligand_path, index=False)
            
            print(f"   ✅ Protein-ligand integration: {len(protein_ligand_integrated):,} records")
            print(f"   📊 Unique targets: {protein_ligand_integrated['target_name'].nunique()}")
            print(f"   📊 Unique compounds: {protein_ligand_integrated['canonical_smiles'].nunique()}")
        else:
            protein_ligand_integrated = pd.DataFrame()
            print("   ⚠️ No protein-ligand databases available for integration")
        
        # Step 4: Process Cell Line Data (separate track)
        print(f"\n🧬 STEP 4: Processing cell line drug sensitivity data...")
        print("-" * 60)
        
        if cell_line_databases:
            cell_line_processed = process_cell_line_data(cell_line_databases)
            
            # Save cell line data
            cell_line_path = datasets_dir / "real_cell_line_drug_sensitivity.csv"
            cell_line_processed.to_csv(cell_line_path, index=False)
            
            print(f"   ✅ Cell line processing: {len(cell_line_processed):,} records")
            print(f"   📊 Unique cell lines: {cell_line_processed['CELL_LINE_NAME'].nunique()}")
            print(f"   📊 Unique drugs: {cell_line_processed['DRUG_NAME'].nunique()}")
        else:
            cell_line_processed = pd.DataFrame()
            print("   ⚠️ No cell line databases available for processing")
        
        # Step 5: Create comprehensive metadata
        print(f"\n💾 STEP 5: Saving comprehensive integration metadata...")
        print("-" * 60)
        
        # Create comprehensive metadata
        metadata = {
            'integration_method': 'Real_Database_Dual_Track_Integration',
            'integration_timestamp': datetime.now().isoformat(),
            'dtc_removed': True,
            'real_data_only': True,
            'integration_tracks': {
                'protein_ligand_activity': {
                    'databases': list(protein_ligand_databases.keys()),
                    'total_records': len(protein_ligand_integrated) if not protein_ligand_integrated.empty else 0,
                    'unique_targets': protein_ligand_integrated['target_name'].nunique() if not protein_ligand_integrated.empty else 0,
                    'unique_compounds': protein_ligand_integrated['canonical_smiles'].nunique() if not protein_ligand_integrated.empty else 0,
                    'file': 'real_protein_ligand_integrated.csv'
                },
                'cell_line_drug_sensitivity': {
                    'databases': list(cell_line_databases.keys()),
                    'total_records': len(cell_line_processed) if not cell_line_processed.empty else 0,
                    'unique_cell_lines': cell_line_processed['CELL_LINE_NAME'].nunique() if not cell_line_processed.empty else 0,
                    'unique_drugs': cell_line_processed['DRUG_NAME'].nunique() if not cell_line_processed.empty else 0,
                    'file': 'real_cell_line_drug_sensitivity.csv'
                }
            },
            'available_databases': {
                db_name: {
                    'records': info['records'],
                    'targets': info['targets'],
                    'file': info['file'],
                    'type': info.get('type', 'protein_ligand')
                } for db_name, info in available_databases.items()
            },
            'data_quality': {
                'real_experimental_data': True,
                'api_sources': True,
                'synthetic_data_removed': True,
                'dtc_excluded': True,
                'dual_track_architecture': True,
                'cell_line_genomics_ready': 'GDSC' in available_databases
            }
        }
        
        metadata_path = datasets_dir / "real_database_integration_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate final report
        print(f"\n🎉 REAL DATABASE INTEGRATION COMPLETED!")
        print("=" * 80)
        print(f"📁 Integration files:")
        if not protein_ligand_integrated.empty:
            print(f"  • Protein-ligand data: {protein_ligand_path}")
        if not cell_line_processed.empty:
            print(f"  • Cell line data: {cell_line_path}")
        print(f"  • Metadata: {metadata_path}")
        
        print(f"\n📊 Integration summary:")
        print(f"  • Available databases: {len(available_databases)}")
        print(f"  • Protein-ligand track: {len(protein_ligand_databases)} databases, {len(protein_ligand_integrated):,} records")
        print(f"  • Cell line track: {len(cell_line_databases)} databases, {len(cell_line_processed):,} records")
        
        print(f"\n🚀 DUAL-TRACK ARCHITECTURE ESTABLISHED:")
        print(f"  • ✅ Track 1: Protein-ligand activity prediction (ChEMBL + PubChem + BindingDB)")
        print(f"  • ✅ Track 2: Cell line drug sensitivity with genomics (GDSC)")
        print(f"  • ❌ DTC completely removed from pipeline")
        print(f"  • 🌐 All data from real API sources")
        
        return {
            'status': 'success',
            'integration_type': 'dual_track',
            'protein_ligand_records': len(protein_ligand_integrated),
            'cell_line_records': len(cell_line_processed),
            'databases_integrated': list(available_databases.keys()),
            'dtc_removed': True,
            'real_data_only': True,
            'metadata_path': str(metadata_path),
            'ready_for_training': True
        }
        
    except Exception as e:
        print(f"❌ REAL DATABASE INTEGRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def integrate_protein_ligand_data(databases: Dict[str, Dict]) -> pd.DataFrame:
    """Integrate protein-ligand activity data from multiple sources"""
    
    print("🔗 Integrating protein-ligand activity data...")
    
    # Common columns for protein-ligand data
    standard_columns = [
        'canonical_smiles', 'target_name', 'target_category', 'activity_type',
        'standard_value', 'standard_units', 'standard_value_nm', 'pic50', 'data_source'
    ]
    
    standardized_datasets = []
    
    for db_name, db_info in databases.items():
        df = db_info['dataframe'].copy()
        print(f"   Processing {db_name}...")
        
        # Add missing columns with defaults
        for col in standard_columns:
            if col not in df.columns:
                if col == 'data_source':
                    df[col] = db_name
                elif col == 'target_category':
                    # Assign based on target name patterns
                    df[col] = df['target_name'].apply(lambda x: assign_target_category(x) if pd.notna(x) else 'oncoprotein')
                else:
                    df[col] = None
        
        # Select and clean standard columns
        standardized_df = df[standard_columns].copy()
        standardized_df = standardized_df.dropna(subset=['canonical_smiles', 'target_name'])
        
        standardized_datasets.append(standardized_df)
        print(f"     ✅ {db_name}: {len(standardized_df):,} records standardized")
    
    # Combine all datasets
    combined_df = pd.concat(standardized_datasets, ignore_index=True)
    print(f"   📊 Combined raw records: {len(combined_df):,}")
    
    # Apply deduplication with source priority: ChEMBL > PubChem > BindingDB
    print("   🔄 Applying cross-source deduplication...")
    deduplicated_df = apply_protein_ligand_deduplication(combined_df)
    
    print(f"   ✅ Integration complete: {len(deduplicated_df):,} records")
    
    return deduplicated_df

def process_cell_line_data(databases: Dict[str, Dict]) -> pd.DataFrame:
    """Process cell line drug sensitivity data"""
    
    print("🧬 Processing cell line drug sensitivity data...")
    
    # Currently only GDSC in this track
    gdsc_df = databases['GDSC']['dataframe'].copy()
    
    # Basic quality control
    initial_count = len(gdsc_df)
    gdsc_df = gdsc_df.dropna(subset=['IC50_nM', 'CELL_LINE_NAME', 'DRUG_NAME'])
    gdsc_df = gdsc_df[(gdsc_df['IC50_nM'] >= 1) & (gdsc_df['IC50_nM'] <= 100000000)]  # 1 nM to 100 mM
    
    print(f"   📊 After quality control: {len(gdsc_df):,} records (removed {initial_count - len(gdsc_df):,})")
    
    return gdsc_df

def apply_protein_ligand_deduplication(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deduplication for protein-ligand activity data"""
    
    source_priority = {'ChEMBL': 1, 'PubChem_BioAssay': 2, 'BindingDB': 3}
    
    # Group by compound-target-activity combinations
    grouped = df.groupby(['canonical_smiles', 'target_name', 'activity_type'])
    
    deduplicated_records = []
    discarded_count = 0
    
    for (smiles, target, activity_type), group in grouped:
        if len(group) == 1:
            deduplicated_records.append(group.iloc[0].to_dict())
            continue
        
        # Multiple measurements - prioritize by source
        group = group.copy()
        group['source_priority'] = group['data_source'].map(source_priority).fillna(999)
        group_sorted = group.sort_values('source_priority')
        
        # Check for reasonable agreement if multiple sources
        values = group['standard_value_nm'].dropna()
        if len(values) > 1:
            max_val = np.max(values)
            min_val = np.min(values)
            
            if max_val / min_val > 100:  # >100-fold difference
                discarded_count += len(group)
                continue
        
        # Use highest priority source
        best_record = group_sorted.iloc[0].to_dict()
        best_record['cross_source_data'] = len(group) > 1
        best_record['source_count'] = len(group)
        
        deduplicated_records.append(best_record)
    
    result_df = pd.DataFrame(deduplicated_records)
    
    print(f"     📊 Deduplication: {len(df)} → {len(result_df)} records")
    print(f"     🗑️ Discarded (high variance): {discarded_count}")
    
    return result_df

def assign_target_category(target_name):
    """Assign target category based on target name"""
    
    target_name = str(target_name).upper()
    
    # Oncoproteins
    oncoproteins = ['EGFR', 'HER2', 'VEGFR2', 'BRAF', 'MET', 'CDK4', 'CDK6', 'ALK', 'MDM2', 'PI3KCA', 'PI3K']
    if any(onco in target_name for onco in oncoproteins):
        return 'oncoprotein'
    
    # Tumor suppressors
    tumor_suppressors = ['TP53', 'RB1', 'PTEN', 'APC', 'BRCA1', 'BRCA2', 'VHL']
    if any(ts in target_name for ts in tumor_suppressors):
        return 'tumor_suppressor'
    
    # Metastasis suppressors
    metastasis_suppressors = ['NDRG1', 'KAI1', 'KISS1', 'NM23H1', 'RKIP', 'CASP8']
    if any(ms in target_name for ms in metastasis_suppressors):
        return 'metastasis_suppressor'
    
    return 'oncoprotein'  # Default

if __name__ == "__main__":
    print("🔗 Updated Database Integration Pipeline - Real Data Only")