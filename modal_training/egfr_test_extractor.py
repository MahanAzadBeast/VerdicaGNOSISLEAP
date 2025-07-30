"""
EGFR Test Data Extractor
Small-scale test to validate the complete pipeline before full 14-target download
"""

import modal
import requests
import pandas as pd
import numpy as np
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Modal setup - lightweight image for testing
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "requests",
    "pandas", 
    "numpy",
    "rdkit",
    "matplotlib",
    "seaborn"
])

app = modal.App("egfr-test-extractor")

# Persistent volume for test datasets
datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)

# ChEMBL API Configuration
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
REQUEST_DELAY = 0.2  # Faster for testing

# Test with just EGFR
TEST_TARGET = {
    "EGFR": "CHEMBL203"  # Verified working
}

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=2.0,
    memory=4096,
    timeout=1800  # 30 minutes
)
def extract_egfr_test_data():
    """
    Extract EGFR bioactivity data for pipeline testing
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Starting EGFR Test Data Extraction")
    logger.info("=" * 50)
    
    def make_api_request(url, params=None):
        """Make API request with error handling"""
        time.sleep(REQUEST_DELAY)
        
        if params is None:
            params = {}
        params['format'] = 'json'
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None
    
    # Step 1: Get EGFR activities
    logger.info("🎯 Extracting EGFR bioactivity data...")
    
    activities_url = f"{CHEMBL_BASE_URL}/activity"
    params = {
        'target_chembl_id': 'CHEMBL203',
        'limit': 1000,  # Get more data for better testing
        'format': 'json'
    }
    
    logger.info("🔍 Querying ChEMBL API for EGFR activities...")
    data = make_api_request(activities_url, params)
    
    if not data or 'activities' not in data:
        raise ValueError("No EGFR data found via API")
    
    activities = data['activities']
    logger.info(f"📊 Found {len(activities)} total EGFR activities")
    
    # Step 2: Process activities and get SMILES
    logger.info("🔬 Processing activities and extracting SMILES...")
    
    processed_data = []
    smiles_cache = {}  # Cache SMILES to avoid repeated API calls
    
    for i, activity in enumerate(activities):
        if i % 100 == 0:
            logger.info(f"  Processed {i}/{len(activities)} activities...")
        
        # Filter for relevant data
        if not activity.get('standard_value'):
            continue
        
        standard_type = activity.get('standard_type', '')
        standard_units = activity.get('standard_units', '')
        
        # Focus on IC50, Ki, EC50 in nM
        if standard_type not in ['IC50', 'Ki', 'EC50']:
            continue
        if standard_units != 'nM':
            continue
        
        molecule_chembl_id = activity.get('molecule_chembl_id')
        if not molecule_chembl_id:
            continue
        
        # Get SMILES (use cache to avoid repeated calls)
        if molecule_chembl_id not in smiles_cache:
            molecule_url = f"{CHEMBL_BASE_URL}/molecule/{molecule_chembl_id}"
            molecule_data = make_api_request(molecule_url)
            
            if not molecule_data or 'molecule_structures' not in molecule_data:
                smiles_cache[molecule_chembl_id] = None
                continue
            
            structures = molecule_data['molecule_structures']
            if not structures or not structures.get('canonical_smiles'):
                smiles_cache[molecule_chembl_id] = None
                continue
            
            smiles_cache[molecule_chembl_id] = structures['canonical_smiles']
        
        smiles = smiles_cache[molecule_chembl_id]
        if not smiles:
            continue
        
        # Validate SMILES with RDKit
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
        except:
            continue
        
        # Create record
        record = {
            'canonical_smiles': smiles,
            'target_name': 'EGFR',
            'target_chembl_id': 'CHEMBL203',
            'standard_type': standard_type,
            'standard_value': float(activity['standard_value']),
            'standard_units': standard_units,
            'pchembl_value': activity.get('pchembl_value'),
            'activity_id': activity['activity_id'],
            'molecule_chembl_id': molecule_chembl_id
        }
        
        processed_data.append(record)
    
    logger.info(f"✅ Processed {len(processed_data)} valid EGFR bioactivity records")
    
    if not processed_data:
        raise ValueError("No valid EGFR bioactivity data found")
    
    # Step 3: Create DataFrame and process
    logger.info("📊 Creating and processing dataset...")
    df = pd.DataFrame(processed_data)
    
    # Filter valid values
    df = df[df['standard_value'] > 0]
    df = df[df['standard_value'] <= 1000000]  # Remove extreme outliers
    
    # Convert to pIC50
    df['pIC50'] = -np.log10(df['standard_value'] / 1e9)  # Convert nM to M, then -log10
    
    # Remove pIC50 outliers
    df = df[df['pIC50'] >= 0]
    df = df[df['pIC50'] <= 12]
    
    logger.info(f"📈 After filtering: {len(df)} valid records")
    
    # Aggregate multiple measurements for same compound
    logger.info("🔄 Aggregating multiple measurements...")
    aggregated = df.groupby('canonical_smiles').agg({
        'pIC50': 'median',
        'standard_value': 'median',
        'activity_id': 'count',
        'standard_type': lambda x: ','.join(set(x))
    }).rename(columns={'activity_id': 'measurement_count'}).reset_index()
    
    logger.info(f"📈 After aggregation: {len(aggregated)} unique compounds")
    
    # Step 4: Save test dataset
    logger.info("💾 Saving EGFR test dataset...")
    
    datasets_dir = Path("/vol/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = datasets_dir / "egfr_test_dataset.csv"
    aggregated.to_csv(csv_path, index=False)
    
    # Save Parquet
    parquet_path = datasets_dir / "egfr_test_dataset.parquet"
    aggregated.to_parquet(parquet_path, index=False)
    
    # Save metadata
    metadata = {
        'extraction_method': 'ChEMBL_API_Test',
        'target': 'EGFR',
        'target_chembl_id': 'CHEMBL203',
        'total_compounds': len(aggregated),
        'total_activities_before_aggregation': len(df),
        'pIC50_range': [float(aggregated['pIC50'].min()), float(aggregated['pIC50'].max())],
        'measurement_stats': {
            'mean_measurements_per_compound': float(aggregated['measurement_count'].mean()),
            'max_measurements_per_compound': int(aggregated['measurement_count'].max())
        }
    }
    
    metadata_path = datasets_dir / "egfr_test_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("🎉 EGFR Test Data Extraction Completed!")
    logger.info(f"📁 Saved files:")
    logger.info(f"  • {csv_path}")
    logger.info(f"  • {parquet_path}")
    logger.info(f"  • {metadata_path}")
    
    return {
        'status': 'success',
        'target': 'EGFR',
        'total_compounds': len(aggregated),
        'dataset_path': str(csv_path),
        'parquet_path': str(parquet_path),
        'metadata_path': str(metadata_path),
        'pIC50_range': [float(aggregated['pIC50'].min()), float(aggregated['pIC50'].max())],
        'sample_compounds': aggregated.head(3)[['canonical_smiles', 'pIC50']].to_dict('records')
    }

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=1.0,
    memory=2048,
    timeout=300
)
def test_egfr_dataset():
    """
    Test the extracted EGFR dataset
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing EGFR dataset...")
    
    dataset_path = Path("/vol/datasets/egfr_test_dataset.csv")
    if not dataset_path.exists():
        return {'status': 'error', 'message': 'EGFR test dataset not found'}
    
    try:
        df = pd.read_csv(dataset_path)
        
        stats = {
            'total_compounds': len(df),
            'pIC50_stats': {
                'mean': float(df['pIC50'].mean()),
                'std': float(df['pIC50'].std()),
                'min': float(df['pIC50'].min()),
                'max': float(df['pIC50'].max())
            },
            'sample_smiles': df['canonical_smiles'].head(3).tolist(),
            'file_size_mb': dataset_path.stat().st_size / 1e6
        }
        
        logger.info(f"✅ Dataset loaded: {stats['total_compounds']} compounds")
        logger.info(f"✅ pIC50 range: {stats['pIC50_stats']['min']:.2f} - {stats['pIC50_stats']['max']:.2f}")
        
        return {'status': 'success', 'stats': stats}
        
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    # Local test
    pass