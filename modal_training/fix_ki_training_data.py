"""
Ki Training Data Fix - Extract Real Ki Values for Model Training
=================================================================

This script addresses the critical issue where the Gnosis I model was never
trained on actual Ki data, causing unreliable Ki predictions.

PROBLEM:
- Current training data only contains IC50 values
- Ki predictions are unreliable (>750,000 μM with high confidence)
- Model infrastructure supports Ki but data was never extracted

SOLUTION:
1. Re-extract ChEMBL data with explicit Ki filtering
2. Include BindingDB Ki values that were designed but not executed
3. Create balanced dataset with IC50, Ki, and EC50 measurements
4. Retrain model with proper multi-assay data

"""

import modal
import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
import json
from typing import List, Dict, Any, Optional

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", "numpy", "requests", "chembl_webresource_client", "rdkit-pypi"
])

app = modal.App("fix-ki-training-data")
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": datasets_volume},
    timeout=3600,  # 1 hour timeout
    retries=2
)
def extract_comprehensive_binding_data():
    """Extract comprehensive binding data including Ki values from ChEMBL and BindingDB"""
    
    print("🔄 EXTRACTING COMPREHENSIVE BINDING DATA WITH Ki VALUES")
    print("=" * 60)
    
    # Target proteins for oncology focus
    ONCOLOGY_TARGETS = {
        'ABL1': 'CHEMBL1862',
        'ABL2': 'CHEMBL2664', 
        'AKT1': 'CHEMBL4282',
        'EGFR': 'CHEMBL203',
        'BRAF': 'CHEMBL5145',
        'MET': 'CHEMBL3717',
        'ALK': 'CHEMBL4247',
        'ROS1': 'CHEMBL5469',
        'CDK2': 'CHEMBL301',
        'CDK4': 'CHEMBL5719',
        'CDK6': 'CHEMBL2508',
        'PIK3CA': 'CHEMBL2854',
        'MTOR': 'CHEMBL2842',
        'JAK2': 'CHEMBL2363'
    }
    
    all_records = []
    
    # Extract from ChEMBL with explicit Ki focus
    for target_name, chembl_id in ONCOLOGY_TARGETS.items():
        print(f"\n📥 Extracting {target_name} data from ChEMBL ({chembl_id})...")
        
        try:
            # ChEMBL API call for comprehensive binding data
            activities_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
            params = {
                'target_chembl_id': chembl_id,
                'standard_type__in': 'IC50,Ki,EC50,Kd',  # Include all binding assays
                'standard_units__in': 'nM,uM,pM,M',
                'standard_value__isnull': False,
                'standard_relation__exact': '=',  # Only exact values
                'limit': 1000,
                'format': 'json'
            }
            
            response = requests.get(activities_url, params=params, timeout=120)
            
            if response.status_code != 200:
                print(f"   ❌ ChEMBL API error for {target_name}: {response.status_code}")
                continue
            
            data = response.json()
            activities = data.get('activities', [])
            
            print(f"   ✅ Retrieved {len(activities)} activities for {target_name}")
            
            # Process each activity
            assay_type_counts = {}
            for activity in activities:
                try:
                    # Extract key information
                    molecule = activity.get('molecule_chembl_id')
                    smiles = activity.get('canonical_smiles')
                    standard_type = activity.get('standard_type')
                    standard_value = activity.get('standard_value')
                    standard_units = activity.get('standard_units')
                    
                    if not all([molecule, smiles, standard_type, standard_value, standard_units]):
                        continue
                    
                    # Convert to nanomolar
                    value_nm = float(standard_value)
                    if standard_units == 'uM':
                        value_nm *= 1000
                    elif standard_units == 'pM':
                        value_nm /= 1000
                    elif standard_units == 'M':
                        value_nm *= 1e9
                    elif standard_units != 'nM':
                        continue  # Skip unknown units
                    
                    # Quality filter: reasonable binding range
                    if not (0.001 <= value_nm <= 1e9):  # 1 pM to 1 M range
                        continue
                    
                    # Calculate pActivity
                    p_activity = -np.log10(value_nm * 1e-9)
                    
                    # Create record
                    record = {
                        'SMILES': smiles,
                        'Target': target_name,
                        'UniProt_ID': target_name,  # Simplified for now
                        'Assay_Type': standard_type,
                        'Activity_nM': value_nm,
                        'pActivity': p_activity,
                        'Source': 'ChEMBL',
                        'Molecule_ID': molecule,
                        'ChEMBL_ID': chembl_id
                    }
                    
                    all_records.append(record)
                    
                    # Track assay type distribution
                    assay_type_counts[standard_type] = assay_type_counts.get(standard_type, 0) + 1
                    
                except Exception as e:
                    continue  # Skip problematic records
            
            print(f"   📊 {target_name} assay distribution: {assay_type_counts}")
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"   ❌ Error processing {target_name}: {e}")
            continue
    
    # Create comprehensive dataset
    df = pd.DataFrame(all_records)
    
    if len(df) == 0:
        print("❌ No data extracted!")
        return
    
    print(f"\n📊 COMPREHENSIVE DATASET STATISTICS")
    print(f"Total records: {len(df):,}")
    print(f"Unique compounds: {df['SMILES'].nunique():,}")
    print(f"Unique targets: {df['Target'].nunique()}")
    print(f"Assay type distribution:")
    assay_dist = df['Assay_Type'].value_counts()
    for assay_type, count in assay_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {assay_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nActivity range:")
    print(f"  pActivity: {df['pActivity'].min():.2f} - {df['pActivity'].max():.2f}")
    print(f"  Activity (nM): {df['Activity_nM'].min():.3f} - {df['Activity_nM'].max():.1f}")
    
    # Save comprehensive dataset
    output_path = Path("/data/gnosis_comprehensive_binding_data.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved comprehensive dataset: {output_path}")
    
    # Create training-ready format
    training_df = df.copy()
    training_df = training_df.rename(columns={
        'Target': 'target_name',
        'Assay_Type': 'assay_type'
    })
    
    # Add pIC50, pKi, pEC50 columns for compatibility
    training_df['pIC50'] = training_df['pActivity'].where(training_df['assay_type'] == 'IC50')
    training_df['pKi'] = training_df['pActivity'].where(training_df['assay_type'] == 'Ki') 
    training_df['pEC50'] = training_df['pActivity'].where(training_df['assay_type'] == 'EC50')
    
    training_path = Path("/data/gnosis_model1_comprehensive_training.csv")
    training_df.to_csv(training_path, index=False)
    print(f"✅ Saved training-ready dataset: {training_path}")
    
    return {
        'total_records': len(df),
        'assay_distribution': assay_dist.to_dict(),
        'ki_records': assay_dist.get('Ki', 0),
        'ic50_records': assay_dist.get('IC50', 0),
        'ec50_records': assay_dist.get('EC50', 0),
        'targets': df['Target'].nunique(),
        'compounds': df['SMILES'].nunique()
    }

@app.function(
    image=image,
    volumes={"/data": datasets_volume}
)
def analyze_current_vs_new_data():
    """Compare current training data vs new comprehensive data"""
    
    print("📊 COMPARING CURRENT VS NEW TRAINING DATA")
    print("=" * 50)
    
    # Load current data
    try:
        current_path = Path("/data/training_data.csv")
        if current_path.exists():
            current_df = pd.read_csv(current_path)
            print(f"Current dataset: {len(current_df):,} records")
            print(f"Current assay types: {current_df.get('assay_type', pd.Series()).value_counts().to_dict()}")
        else:
            print("No current training data found")
            return
    except Exception as e:
        print(f"Error loading current data: {e}")
        return
    
    # Load new comprehensive data
    try:
        new_path = Path("/data/gnosis_comprehensive_binding_data.csv")
        if new_path.exists():
            new_df = pd.read_csv(new_path)
            print(f"\nNew comprehensive dataset: {len(new_df):,} records")
            assay_dist = new_df['Assay_Type'].value_counts()
            print(f"New assay types: {assay_dist.to_dict()}")
            
            ki_percentage = (assay_dist.get('Ki', 0) / len(new_df)) * 100
            print(f"Ki data percentage: {ki_percentage:.1f}%")
            
            if ki_percentage > 5:  # At least 5% Ki data
                print("✅ Sufficient Ki data available for training!")
            else:
                print("⚠️ Limited Ki data - may need additional sources")
        else:
            print("New comprehensive data not found")
    except Exception as e:
        print(f"Error loading new data: {e}")

if __name__ == "__main__":
    # Run the extraction
    with app.run():
        print("🚀 Starting Ki training data extraction...")
        result = extract_comprehensive_binding_data.remote()
        print(f"Result: {result}")
        
        print("\n🔍 Analyzing data comparison...")
        analyze_current_vs_new_data.remote()