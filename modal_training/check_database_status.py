"""
Database Status Checker
Verifies what datasets are actually downloaded, standardized and ready for training
"""

import modal
import json
from pathlib import Path
from datetime import datetime

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy"
])

app = modal.App("database-status-checker")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=2.0,
    memory=8192,
    timeout=300
)
def check_database_status():
    """
    Check what datasets are actually available and ready for training
    """
    
    print("🔍 CHECKING DATABASE STATUS")
    print("=" * 80)
    print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S AEST')}")
    print()
    
    try:
        datasets_dir = Path("/vol/datasets")
        
        if not datasets_dir.exists():
            print("❌ Datasets directory does not exist")
            return {'status': 'no_datasets_found'}
        
        # List all files in datasets directory
        all_files = list(datasets_dir.glob("*"))
        
        print(f"📁 Files in datasets directory ({len(all_files)} total):")
        for file in sorted(all_files):
            size_mb = file.stat().st_size / (1024 * 1024) if file.is_file() else 0
            file_type = "DIR" if file.is_dir() else f"{size_mb:.1f}MB"
            print(f"   • {file.name} ({file_type})")
        
        # Check for specific database files
        database_status = {
            'chembl': {
                'raw_data': None,
                'matrix': None,
                'metadata': None,
                'status': 'not_found'
            },
            'pubchem': {
                'raw_data': None,
                'matrix': None, 
                'metadata': None,
                'status': 'not_found'
            },
            'integrated': {
                'raw_data': None,
                'matrix': None,
                'metadata': None,
                'status': 'not_found'
            },
            'bindingdb': {
                'status': 'not_implemented'
            },
            'dtc': {
                'status': 'not_implemented'
            }
        }
        
        # Check ChEMBL files
        chembl_files = {
            'raw_data': datasets_dir / "expanded_fixed_raw_data.csv",
            'matrix': datasets_dir / "expanded_fixed_ic50_matrix.csv", 
            'metadata': datasets_dir / "expanded_fixed_metadata.json"
        }
        
        chembl_found = 0
        for file_type, file_path in chembl_files.items():
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                database_status['chembl'][file_type] = {
                    'path': str(file_path),
                    'size_mb': round(size_mb, 2),
                    'exists': True
                }
                chembl_found += 1
            else:
                database_status['chembl'][file_type] = {'exists': False}
        
        if chembl_found > 0:
            database_status['chembl']['status'] = 'available' if chembl_found == 3 else 'partial'
        
        # Check PubChem files
        pubchem_files = {
            'raw_data': datasets_dir / "pubchem_bioassay_raw_data.csv",
            'matrix': datasets_dir / "pubchem_bioassay_ic50_matrix.csv",
            'metadata': datasets_dir / "pubchem_bioassay_metadata.json"
        }
        
        pubchem_found = 0
        for file_type, file_path in pubchem_files.items():
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                database_status['pubchem'][file_type] = {
                    'path': str(file_path),
                    'size_mb': round(size_mb, 2),
                    'exists': True
                }
                pubchem_found += 1
            else:
                database_status['pubchem'][file_type] = {'exists': False}
        
        if pubchem_found > 0:
            database_status['pubchem']['status'] = 'available' if pubchem_found == 3 else 'partial'
        
        # Check integrated files
        integrated_files = {
            'raw_data': datasets_dir / "integrated_chembl_pubchem_raw_data.csv",
            'matrix': datasets_dir / "integrated_chembl_pubchem_ic50_matrix.csv",
            'metadata': datasets_dir / "integrated_chembl_pubchem_metadata.json"
        }
        
        integrated_found = 0
        for file_type, file_path in integrated_files.items():
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                database_status['integrated'][file_type] = {
                    'path': str(file_path),
                    'size_mb': round(size_mb, 2),
                    'exists': True
                }
                integrated_found += 1
            else:
                database_status['integrated'][file_type] = {'exists': False}
        
        if integrated_found > 0:
            database_status['integrated']['status'] = 'available' if integrated_found == 3 else 'partial'
        
        # Load metadata if available
        dataset_stats = {}
        
        if database_status['chembl']['status'] in ['available', 'partial']:
            metadata_path = datasets_dir / "expanded_fixed_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        chembl_metadata = json.load(f)
                    dataset_stats['chembl'] = {
                        'records': chembl_metadata.get('total_records', 0),
                        'targets': chembl_metadata.get('total_targets', 0),
                        'compounds': chembl_metadata.get('total_compounds', 0)
                    }
                except:
                    pass
        
        if database_status['integrated']['status'] in ['available', 'partial']:
            metadata_path = datasets_dir / "integrated_chembl_pubchem_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        integrated_metadata = json.load(f)
                    dataset_stats['integrated'] = {
                        'records': integrated_metadata.get('integrated_dataset', {}).get('total_records', 0),
                        'targets': integrated_metadata.get('integrated_dataset', {}).get('total_targets', 0),
                        'compounds': integrated_metadata.get('integrated_dataset', {}).get('total_compounds', 0)
                    }
                except:
                    pass
        
        # Generate status report
        print(f"\n📊 DATABASE STATUS SUMMARY:")
        print(f"   🧬 ChEMBL: {database_status['chembl']['status'].upper()}")
        if 'chembl' in dataset_stats:
            stats = dataset_stats['chembl']
            print(f"      • Records: {stats['records']:,}")
            print(f"      • Targets: {stats['targets']}")
            print(f"      • Compounds: {stats['compounds']:,}")
        
        print(f"   🧪 PubChem BioAssay: {database_status['pubchem']['status'].upper()}")
        if 'pubchem' in dataset_stats:
            stats = dataset_stats['pubchem']
            print(f"      • Records: {stats['records']:,}")
            print(f"      • Targets: {stats['targets']}")
            print(f"      • Compounds: {stats['compounds']:,}")
        
        print(f"   🔗 Integrated Dataset: {database_status['integrated']['status'].upper()}")
        if 'integrated' in dataset_stats:
            stats = dataset_stats['integrated']
            print(f"      • Records: {stats['records']:,}")
            print(f"      • Targets: {stats['targets']}")
            print(f"      • Compounds: {stats['compounds']:,}")
        
        print(f"   🔗 BindingDB: {database_status['bindingdb']['status'].upper()}")
        print(f"   🔗 DTC: {database_status['dtc']['status'].upper()}")
        
        # Training readiness assessment
        print(f"\n🚀 TRAINING READINESS:")
        
        training_ready = False
        best_dataset = None
        
        if database_status['integrated']['status'] == 'available':
            training_ready = True
            best_dataset = 'integrated'
            print(f"   ✅ READY: Integrated ChEMBL+PubChem dataset available")
        elif database_status['chembl']['status'] == 'available':
            training_ready = True
            best_dataset = 'chembl'
            print(f"   ✅ READY: ChEMBL dataset available (PubChem integration pending)")
        else:
            print(f"   ❌ NOT READY: No complete datasets found")
        
        result = {
            'status': 'success',
            'database_status': database_status,
            'dataset_stats': dataset_stats,
            'training_ready': training_ready,
            'best_dataset': best_dataset,
            'check_timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"❌ ERROR CHECKING DATABASE STATUS: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'error',
            'error': str(e),
            'check_timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🔍 Database Status Checker")