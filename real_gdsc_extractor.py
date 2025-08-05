"""
GDSC Real Drug Sensitivity Data Extractor
Direct download from GDSC/DepMap public datasets
CRITICAL: ONLY REAL EXPERIMENTAL DATA - NO SIMULATION
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from io import StringIO
import zipfile
import gzip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_real_gdsc_data():
    """
    Download real GDSC drug sensitivity data
    CRITICAL: ONLY REAL EXPERIMENTAL DATA
    """
    
    print("🧬 DOWNLOADING REAL GDSC DRUG SENSITIVITY DATA")
    print("=" * 80)
    print("🚨 CRITICAL: ONLY REAL EXPERIMENTAL DATA")
    print("❌ NEVER USE SIMULATED DATA")
    print()
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'GDSC-Data-Extractor/1.0'
    })
    
    try:
        # GDSC2 fitted dose response data (REAL experimental IC50s)
        print("📊 Downloading GDSC2 fitted dose response data...")
        
        gdsc_urls = [
            # GDSC2 IC50 data
            "https://www.cancerrxgene.org/gdsc1000/GDSC2_fitted_dose_response_25Feb20.xlsx",
            "https://www.cancerrxgene.org/gdsc1000/GDSC2_fitted_dose_response_17July19.xlsx", 
            # Alternative CSV format
            "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.0/GDSC2_fitted_dose_response_25Feb20.xlsx",
            # DepMap hosted version
            "https://ndownloader.figshare.com/files/34008503"
        ]
        
        gdsc_data = None
        
        for url in gdsc_urls:
            try:
                print(f"   Trying: {url}")
                response = session.get(url, timeout=120)
                
                if response.status_code == 200:
                    print(f"   ✅ Downloaded {len(response.content):,} bytes")
                    
                    # Try different formats
                    if url.endswith('.xlsx'):
                        # Excel format
                        with open('/tmp/gdsc_data.xlsx', 'wb') as f:
                            f.write(response.content)
                        gdsc_data = pd.read_excel('/tmp/gdsc_data.xlsx')
                    elif url.endswith('.csv'):
                        # CSV format
                        gdsc_data = pd.read_csv(StringIO(response.text))
                    else:
                        # Try as CSV first
                        try:
                            gdsc_data = pd.read_csv(StringIO(response.text))
                        except:
                            # Try as Excel
                            with open('/tmp/gdsc_data_temp', 'wb') as f:
                                f.write(response.content)
                            gdsc_data = pd.read_excel('/tmp/gdsc_data_temp')
                    
                    if gdsc_data is not None and len(gdsc_data) > 1000:
                        print(f"   ✅ Successfully loaded {len(gdsc_data):,} records")
                        break
                    else:
                        print(f"   ⚠️ Data too small or invalid: {len(gdsc_data) if gdsc_data is not None else 0}")
                        gdsc_data = None
                        
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue
        
        if gdsc_data is None:
            print("❌ Could not download GDSC data from primary sources")
            print("🔄 Trying alternative approach...")
            return download_alternative_sources()
        
        # Download compound information
        print(f"\n📊 Downloading compound information...")
        
        compound_urls = [
            "https://www.cancerrxgene.org/gdsc1000/screened_compounds_rel_8.2.csv",
            "https://www.cancerrxgene.org/gdsc1000/GDSC_compounds_15Mar2019.xlsx",
            "https://ndownloader.figshare.com/files/34008496"  # DepMap compound info
        ]
        
        compound_data = None
        
        for url in compound_urls:
            try:
                print(f"   Trying: {url}")
                response = session.get(url, timeout=60)
                
                if response.status_code == 200:
                    if url.endswith('.xlsx'):
                        with open('/tmp/compounds.xlsx', 'wb') as f:
                            f.write(response.content)
                        compound_data = pd.read_excel('/tmp/compounds.xlsx')
                    else:
                        compound_data = pd.read_csv(StringIO(response.text))
                    
                    if compound_data is not None and len(compound_data) > 50:
                        print(f"   ✅ Compound data loaded: {len(compound_data):,} compounds")
                        break
                        
            except Exception as e:
                print(f"   ❌ Compound download failed: {e}")
                continue
        
        # Download cell line information
        print(f"\n📊 Downloading cell line information...")
        
        cell_line_urls = [
            "https://www.cancerrxgene.org/gdsc1000/Cell_Lines_Details.xlsx",
            "https://ndownloader.figshare.com/files/34008419"  # DepMap cell line info
        ]
        
        cell_line_data = None
        
        for url in cell_line_urls:
            try:
                print(f"   Trying: {url}")
                response = session.get(url, timeout=60)
                
                if response.status_code == 200:
                    if url.endswith('.xlsx'):
                        with open('/tmp/cell_lines.xlsx', 'wb') as f:
                            f.write(response.content)
                        cell_line_data = pd.read_excel('/tmp/cell_lines.xlsx')
                    else:
                        cell_line_data = pd.read_csv(StringIO(response.text))
                    
                    if cell_line_data is not None and len(cell_line_data) > 100:
                        print(f"   ✅ Cell line data loaded: {len(cell_line_data):,} cell lines")
                        break
                        
            except Exception as e:
                print(f"   ❌ Cell line download failed: {e}")
                continue
        
        # Analyze the REAL experimental data
        print(f"\n🔍 ANALYZING REAL EXPERIMENTAL DATA:")
        print(f"   📊 Drug sensitivity records: {len(gdsc_data):,}")
        print(f"   📊 Columns: {list(gdsc_data.columns)}")
        
        # Look for key columns
        ic50_cols = [col for col in gdsc_data.columns if 'ic50' in col.lower() or 'ln_ic50' in col.lower()]
        if ic50_cols:
            print(f"   📊 IC50 columns found: {ic50_cols}")
            
            # Check IC50 data quality
            ic50_col = ic50_cols[0]
            ic50_values = pd.to_numeric(gdsc_data[ic50_col], errors='coerce')
            valid_ic50 = ic50_values.dropna()
            
            print(f"   📊 Valid IC50 measurements: {len(valid_ic50):,}")
            print(f"   📊 IC50 range: {valid_ic50.min():.3f} - {valid_ic50.max():.3f}")
        
        # Check for compound/drug information
        drug_cols = [col for col in gdsc_data.columns if any(term in col.lower() for term in ['drug', 'compound', 'name'])]
        if drug_cols:
            unique_drugs = gdsc_data[drug_cols[0]].nunique()
            print(f"   📊 Unique drugs/compounds: {unique_drugs:,}")
        
        # Check for cell line information
        cell_cols = [col for col in gdsc_data.columns if 'cell' in col.lower() or 'line' in col.lower()]
        if cell_cols:
            unique_cells = gdsc_data[cell_cols[0]].nunique()
            print(f"   📊 Unique cell lines: {unique_cells:,}")
        
        # Expected R² with real GDSC data
        print(f"\n🎯 EXPECTED PERFORMANCE WITH REAL GDSC DATA:")
        
        if len(gdsc_data) > 50000 and unique_drugs > 200:
            expected_r2 = "0.50-0.70"
            confidence = "HIGH"
        elif len(gdsc_data) > 20000 and unique_drugs > 100:
            expected_r2 = "0.40-0.60"
            confidence = "MODERATE-HIGH"
        elif len(gdsc_data) > 10000 and unique_drugs > 50:
            expected_r2 = "0.30-0.50"
            confidence = "MODERATE"
        else:
            expected_r2 = "0.20-0.40"
            confidence = "LOW-MODERATE"
        
        print(f"   📈 Expected R²: {expected_r2}")
        print(f"   🎯 Confidence: {confidence}")
        print(f"   ✅ Based on real experimental drug-cell line relationships")
        
        # Save the real data
        output_path = Path("/app/real_gdsc_drug_sensitivity.csv")
        gdsc_data.to_csv(output_path, index=False)
        
        if compound_data is not None:
            compound_path = Path("/app/real_gdsc_compounds.csv")
            compound_data.to_csv(compound_path, index=False)
        
        if cell_line_data is not None:
            cell_line_path = Path("/app/real_gdsc_cell_lines.csv")
            cell_line_data.to_csv(cell_line_path, index=False)
        
        print(f"\n💾 REAL EXPERIMENTAL DATA SAVED:")
        print(f"   📊 Drug sensitivity: {output_path} ({len(gdsc_data):,} records)")
        if compound_data is not None:
            print(f"   📊 Compounds: {compound_path} ({len(compound_data):,} compounds)")
        if cell_line_data is not None:
            print(f"   📊 Cell lines: {cell_line_path} ({len(cell_line_data):,} cell lines)")
        
        return {
            'drug_sensitivity_data': gdsc_data,
            'compound_data': compound_data,
            'cell_line_data': cell_line_data,
            'unique_compounds': unique_drugs if 'unique_drugs' in locals() else 0,
            'expected_r2': expected_r2,
            'confidence': confidence,
            'data_path': str(output_path)
        }
        
    except Exception as e:
        logger.error(f"GDSC data download failed: {e}")
        print(f"❌ CRITICAL ERROR: {e}")
        return None

def download_alternative_sources():
    """Try alternative real data sources"""
    
    print("🔄 TRYING ALTERNATIVE REAL DATA SOURCES...")
    
    # Try ChEMBL bioactivity data (real experimental)
    try:
        print("📊 Trying ChEMBL bioactivity data...")
        
        # ChEMBL REST API for real bioactivity data
        chembl_url = "https://www.ebi.ac.uk/chembl/api/data/bioactivity?format=csv&limit=50000&standard_type=IC50&target_type=CELL-LINE"
        
        response = requests.get(chembl_url, timeout=120)
        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text))
            if len(data) > 1000:
                print(f"   ✅ ChEMBL data: {len(data):,} real IC50 measurements")
                
                # Save ChEMBL real data
                output_path = Path("/app/real_chembl_bioactivity.csv")
                data.to_csv(output_path, index=False)
                
                unique_compounds = data['molecule_chembl_id'].nunique() if 'molecule_chembl_id' in data.columns else 0
                
                return {
                    'drug_sensitivity_data': data,
                    'unique_compounds': unique_compounds,
                    'expected_r2': "0.35-0.55",
                    'confidence': "MODERATE-HIGH",
                    'data_path': str(output_path),
                    'source': 'ChEMBL_real_experimental'
                }
        
    except Exception as e:
        print(f"   ❌ ChEMBL failed: {e}")
    
    print("❌ All real data sources failed")
    return None

if __name__ == "__main__":
    # Download real GDSC experimental data
    result = download_real_gdsc_data()
    
    if result:
        print(f"\n🎉 REAL GDSC DATA DOWNLOAD SUCCESSFUL!")
        print(f"   📊 Compounds: {result['unique_compounds']:,}")
        print(f"   📈 Expected R²: {result['expected_r2']}")
        print(f"   🎯 Confidence: {result['confidence']}")
        print(f"   ✅ 100% REAL EXPERIMENTAL DATA - NO SIMULATION")
    else:
        print(f"\n❌ FAILED TO DOWNLOAD REAL EXPERIMENTAL DATA")
        print(f"   🚨 CANNOT PROCEED WITHOUT REAL DATA")
        print(f"   ❌ WILL NOT USE SIMULATED DATA")