"""
ChEMBL API-based Data Extractor for Oncoproteins
Fast alternative to downloading the full ChEMBL database
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

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "requests",
    "pandas", 
    "numpy",
    "rdkit",
    "matplotlib",
    "seaborn"
])

app = modal.App("chembl-api-extractor")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)

# ChEMBL API Configuration
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
REQUEST_DELAY = 0.5  # Seconds between API requests to be respectful

# 14 Key Oncoprotein targets with ChEMBL IDs
ONCOPROTEIN_TARGETS = {
    "EGFR": "CHEMBL203",
    "HER2": "CHEMBL1824", 
    "VEGFR2": "CHEMBL279",
    "ALK": "CHEMBL3717",  # Updated ID
    "BRAF": "CHEMBL1823",
    "MET": "CHEMBL3717",
    "MDM2": "CHEMBL5023",
    "STAT3": "CHEMBL5407",
    "RRM2": "CHEMBL3352",
    "CTNNB1": "CHEMBL6132",  # β-catenin
    "MYC": "CHEMBL6130",
    "PI3KCA": "CHEMBL4040",  # PIK3CA
    "CDK4": "CHEMBL331",
    "CDK6": "CHEMBL3974"
}

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=4.0,
    memory=8192,
    timeout=3600  # 1 hour
)
def extract_oncoprotein_data_via_api():
    """
    Extract oncoprotein bioactivity data using ChEMBL API
    Much faster than downloading the full 5GB database
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting ChEMBL API-based oncoprotein data extraction")
    logger.info(f"📋 Extracting data for {len(ONCOPROTEIN_TARGETS)} targets")
    
    all_data = []
    target_stats = {}
    
    def make_api_request(url, params=None):
        """Make API request with error handling and rate limiting"""
        time.sleep(REQUEST_DELAY)
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None
    
    for target_name, chembl_id in ONCOPROTEIN_TARGETS.items():
        logger.info(f"🎯 Processing {target_name} ({chembl_id})...")
        
        # Get activities for this target
        activities_url = f"{CHEMBL_BASE_URL}/activity"
        params = {
            'target_chembl_id': chembl_id,
            'standard_type': 'IC50,Ki,EC50',
            'standard_units': 'nM',
            'limit': 10000,  # Max per request
            'format': 'json'
        }
        
        logger.info(f"🔍 Querying activities for {target_name}...")
        data = make_api_request(activities_url, params)
        
        if not data or 'activities' not in data:
            logger.warning(f"⚠️ No data found for {target_name}")
            target_stats[target_name] = {'total_activities': 0, 'with_smiles': 0}
            continue
        
        activities = data['activities']
        logger.info(f"📊 Found {len(activities)} activities for {target_name}")
        
        # Process activities and get SMILES
        target_data = []
        smiles_requests = 0
        
        for activity in activities:
            # Skip if no standard value
            if not activity.get('standard_value'):
                continue
            
            # Get compound SMILES
            molecule_chembl_id = activity.get('molecule_chembl_id')
            if not molecule_chembl_id:
                continue
            
            # Rate limit SMILES requests
            if smiles_requests % 50 == 0 and smiles_requests > 0:
                logger.info(f"  Processed {smiles_requests} compounds for {target_name}...")
                time.sleep(2)  # Longer break every 50 requests
            
            # Get molecule structure
            molecule_url = f"{CHEMBL_BASE_URL}/molecule/{molecule_chembl_id}"
            molecule_data = make_api_request(molecule_url)
            smiles_requests += 1
            
            if not molecule_data or 'molecule_structures' not in molecule_data:
                continue
            
            structures = molecule_data['molecule_structures']
            if not structures or not structures.get('canonical_smiles'):
                continue
            
            smiles = structures['canonical_smiles']
            
            # Create record
            record = {
                'canonical_smiles': smiles,
                'target_chembl_id': chembl_id,
                'target_name': target_name,
                'standard_type': activity['standard_type'],
                'standard_value': float(activity['standard_value']),
                'standard_units': activity['standard_units'],
                'pchembl_value': activity.get('pchembl_value'),
                'activity_id': activity['activity_id'],
                'molecule_chembl_id': molecule_chembl_id
            }
            
            target_data.append(record)
            all_data.append(record)
        
        target_stats[target_name] = {
            'total_activities': len(activities),
            'with_smiles': len(target_data)
        }
        
        logger.info(f"✅ {target_name}: {len(target_data)} records with SMILES")
        
        # Small break between targets
        time.sleep(1)
    
    # Create DataFrame
    logger.info("📊 Creating consolidated dataset...")
    df = pd.DataFrame(all_data)
    
    if df.empty:
        raise ValueError("No bioactivity data retrieved via API")
    
    logger.info(f"📈 Retrieved {len(df)} total bioactivity records via API")
    
    # Process the data (same logic as database version)
    logger.info("🔧 Processing and cleaning data...")
    
    # Filter valid values
    df = df[df['standard_value'].notna()]
    df = df[df['standard_value'] > 0]
    df = df[df['standard_value'] <= 1000000]  # Remove extreme outliers
    
    # Convert to pIC50
    df['pIC50'] = -np.log10(df['standard_value'] / 1e9)  # Convert nM to M, then -log10
    
    # Remove pIC50 outliers
    df = df[df['pIC50'] >= 0]  # Remove negative pIC50
    df = df[df['pIC50'] <= 12]  # Remove extremely high pIC50
    
    logger.info(f"📊 After filtering: {len(df)} valid records")
    
    # Aggregate multiple measurements for same compound-target pairs
    logger.info("🔄 Aggregating multiple measurements...")
    
    aggregated = df.groupby(['canonical_smiles', 'target_name']).agg({
        'pIC50': 'median',  # Use median for robustness
        'standard_value': 'median',
        'activity_id': 'count'  # Count of measurements
    }).rename(columns={'activity_id': 'measurement_count'}).reset_index()
    
    logger.info(f"📈 After aggregation: {len(aggregated)} unique compound-target pairs")
    
    # Create multi-target matrix
    logger.info("🔄 Creating multi-target dataset matrix...")
    
    pivot_table = aggregated.pivot(index='canonical_smiles', 
                                 columns='target_name', 
                                 values='pIC50')
    
    # Reset index to make SMILES a column
    pivot_table = pivot_table.reset_index()
    
    # Fill NaN values for compounds not tested against certain targets
    logger.info("📊 Dataset shape: {pivot_table.shape}")
    logger.info(f"📊 Targets covered: {', '.join(pivot_table.columns[1:])}")
    
    # Calculate coverage statistics
    coverage_stats = {}
    for target in ONCOPROTEIN_TARGETS.keys():
        if target in pivot_table.columns:
            coverage = pivot_table[target].notna().sum()
            total = len(pivot_table)
            coverage_stats[target] = {
                'compounds_tested': coverage,
                'coverage_percent': (coverage / total) * 100,
                'total_compounds': total
            }
    
    # Save datasets
    logger.info("💾 Saving datasets...")
    
    # Ensure directory exists
    datasets_dir = Path("/vol/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = datasets_dir / "oncoprotein_multitask_dataset_api.csv"
    pivot_table.to_csv(csv_path, index=False)
    logger.info(f"✅ Saved CSV: {csv_path}")
    
    # Save Parquet for efficiency
    parquet_path = datasets_dir / "oncoprotein_multitask_dataset_api.parquet"
    pivot_table.to_parquet(parquet_path, index=False)
    logger.info(f"✅ Saved Parquet: {parquet_path}")
    
    # Save metadata
    metadata = {
        'extraction_method': 'ChEMBL_API',
        'targets': list(ONCOPROTEIN_TARGETS.keys()),
        'total_compounds': len(pivot_table),
        'total_records_before_aggregation': len(df),
        'coverage_stats': coverage_stats,
        'target_stats': target_stats
    }
    
    metadata_path = datasets_dir / "extraction_metadata_api.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("🎉 ChEMBL API extraction completed!")
    
    return {
        'status': 'success',
        'dataset_path': str(csv_path),
        'parquet_path': str(parquet_path),
        'total_compounds': len(pivot_table),
        'targets_covered': len([t for t in ONCOPROTEIN_TARGETS.keys() if t in pivot_table.columns]),
        'coverage_stats': coverage_stats
    }

# Test function to verify API connectivity
@app.function(
    image=image,
    timeout=300
)
def test_chembl_api():
    """Test ChEMBL API connectivity"""
    
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing ChEMBL API connectivity...")
    
    try:
        # Test basic API endpoint
        test_url = f"{CHEMBL_BASE_URL}/target/CHEMBL203"
        response = requests.get(test_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        target_name = data.get('pref_name', 'Unknown')
        
        # Test activities endpoint
        activities_url = f"{CHEMBL_BASE_URL}/activity"
        params = {
            'target_chembl_id': 'CHEMBL203',
            'limit': 5,
            'format': 'json'
        }
        
        response = requests.get(activities_url, params=params, timeout=10)
        response.raise_for_status()
        activities_data = response.json()
        
        activity_count = len(activities_data.get('activities', []))
        
        return {
            'status': 'success',
            'api_accessible': True,
            'test_target': 'CHEMBL203',
            'target_name': target_name,
            'sample_activities_found': activity_count
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'api_accessible': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Local test
    pass