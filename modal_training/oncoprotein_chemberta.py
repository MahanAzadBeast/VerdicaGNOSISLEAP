"""
Multi-Task ChemBERTa for Oncoproteins - Complete Implementation
Downloads ChEMBL database, processes oncoprotein data, and trains multi-target model
"""

import modal
import os
from pathlib import Path

# Define Modal app
app = modal.App("oncoprotein-chemberta-multitask")

# Create comprehensive image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    # Core ML and chemistry
    "torch>=2.1.0",
    "transformers>=4.35.0",
    "pytorch-lightning>=2.0.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    
    # Chemistry and database
    "rdkit-pypi>=2022.9.5",
    "sqlalchemy>=2.0.0",
    # sqlite3 is included with Python
    
    # Data processing and visualization
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.15.0",
    "pyarrow>=13.0.0",  # For parquet files
    
    # Utilities
    "requests>=2.31.0",
    "tqdm>=4.65.0",
    "joblib>=1.3.0",
    "huggingface-hub>=0.19.0",
]).run_commands([
    # Install system dependencies
    "apt-get update && apt-get install -y wget curl unzip gzip tar",
    # Create directories
    "mkdir -p /vol /tmp/chembl"
])

# Create persistent volumes
chembl_volume = modal.Volume.from_name("chembl-database", create_if_missing=True)
datasets_volume = modal.Volume.from_name("oncoprotein-datasets", create_if_missing=True)
models_volume = modal.Volume.from_name("chemberta-models", create_if_missing=True)

# Target definitions
ONCOPROTEIN_TARGETS = {
    "EGFR": "CHEMBL203",
    "HER2": "CHEMBL1824",  # ERBB2
    "VEGFR2": "CHEMBL279",  # KDR
    "ALK": "CHEMBL3565",
    "BRAF": "CHEMBL1823",
    "MET": "CHEMBL3717",
    "MDM2": "CHEMBL5023",
    "STAT3": "CHEMBL5407",
    "RRM2": "CHEMBL3352",
    "CTNNB1": "CHEMBL6132",  # β-catenin
    "MYC": "CHEMBL6130",
    "PI3KCA": "CHEMBL4040",
    "CDK4": "CHEMBL331",
    "CDK6": "CHEMBL3974"
}

@app.function(
    image=image,
    volumes={"/vol/chembl": chembl_volume},
    cpu=4.0,
    memory=16384,
    timeout=600  # 10 minutes
)
def test_chembl_database():
    """
    Test function to verify ChEMBL database structure and accessibility
    """
    import sqlite3
    import logging
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing ChEMBL database accessibility...")
    
    # Check for database files
    chembl_dir = Path("/vol/chembl")
    if not chembl_dir.exists():
        return {"status": "error", "message": "ChEMBL volume directory not found"}
    
    db_files = list(chembl_dir.glob("*.db")) + list(chembl_dir.glob("*.sqlite"))
    if not db_files:
        available_files = list(chembl_dir.glob("*"))
        return {
            "status": "error", 
            "message": "No database files found",
            "available_files": [f.name for f in available_files]
        }
    
    db_file = db_files[0]
    logger.info(f"Testing database: {db_file}")
    
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        # Get basic info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Test a simple query
        if 'target_dictionary' in tables:
            cursor.execute("SELECT COUNT(*) FROM target_dictionary;")
            target_count = cursor.fetchone()[0]
        else:
            target_count = "table not found"
        
        if 'activities' in tables:
            cursor.execute("SELECT COUNT(*) FROM activities;")
            activity_count = cursor.fetchone()[0]
        else:
            activity_count = "table not found"
        
        conn.close()
        
        return {
            "status": "success",
            "database_file": str(db_file),
            "file_size_gb": db_file.stat().st_size / 1e9,
            "table_count": len(tables),
            "target_count": target_count,
            "activity_count": activity_count,
            "sample_tables": tables[:10]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "database_file": str(db_file),
            "error": str(e)
        }

@app.function(
    image=image,
    volumes={
        "/vol/chembl": chembl_volume,
        "/vol/datasets": datasets_volume,
        "/vol/models": models_volume
    },
    cpu=8.0,
    memory=32768,  # 32GB for ChEMBL processing
    timeout=7200   # 2 hours
)
def download_chembl_database():
    """
    Step 1: Download latest ChEMBL SQLite database
    """
    import requests
    import logging
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    chembl_path = Path("/vol/chembl")
    chembl_file = chembl_path / "chembl.sqlite"
    
    # Check if already downloaded
    if chembl_file.exists() and chembl_file.stat().st_size > 1e9:  # > 1GB
        logger.info(f"✅ ChEMBL database already exists: {chembl_file}")
        return {"status": "exists", "path": str(chembl_file), "size_gb": chembl_file.stat().st_size / 1e9}
    
    logger.info("📥 Downloading latest ChEMBL SQLite database...")
    
    # ChEMBL FTP URLs for latest version (ChEMBL v35 as of 2025)
    
    try:
        # Try different ChEMBL versions with correct FTP URLs
        chembl_urls = [
            "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_sqlite.tar.gz",
            "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_34/chembl_34_sqlite.tar.gz",
            "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_33/chembl_33_sqlite.tar.gz",
            # Alternative HTTPS URLs as fallback
            "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_sqlite.tar.gz",
            "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_34/chembl_34_sqlite.tar.gz"
        ]
        
        response = None
        chembl_url = None
        
        for url in chembl_urls:
            try:
                logger.info(f"🔍 Trying URL: {url}")
                
                # Handle FTP URLs differently from HTTP URLs
                if url.startswith('ftp://'):
                    # Convert FTP to HTTP for requests library
                    http_url = url.replace('ftp://', 'https://')
                    response = requests.get(http_url, stream=True, timeout=60)
                else:
                    response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                chembl_url = url
                logger.info(f"✅ Found ChEMBL at: {url}")
                break
            except requests.exceptions.RequestException as e:
                logger.info(f"❌ Failed: {e}")
                continue
        
        if response is None:
            raise ValueError("Could not find ChEMBL database at any known URL")
        
        # Download compressed file
        temp_file = Path("/tmp/chembl_sqlite.tar.gz")
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"📦 Downloading {total_size / 1e9:.1f} GB...")
        
        with open(temp_file, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (100 * 1024 * 1024) == 0:  # Log every 100MB
                        logger.info(f"   Downloaded {downloaded / 1e9:.1f} GB...")
        
        logger.info("📦 Extracting ChEMBL database...")
        
        # Extract tar.gz
        import tarfile
        with tarfile.open(temp_file, 'r:gz') as tar:
            # List contents before extraction for debugging
            members = tar.getnames()
            logger.info(f"Archive contents: {members}")
            tar.extractall("/tmp/chembl/")
        
        # Find and move the SQLite file (ChEMBL v35 uses .db extension)
        extracted_files = []
        
        # Look for common database file extensions
        for pattern in ["*.db", "*.sqlite", "*.sqlite3"]:
            extracted_files.extend(list(Path("/tmp/chembl/").glob(pattern)))
        
        # Also check subdirectories for database files
        if not extracted_files:
            logger.info("No database files in root, checking subdirectories...")
            for subdir in Path("/tmp/chembl/").iterdir():
                if subdir.is_dir():
                    logger.info(f"Checking subdirectory: {subdir}")
                    for pattern in ["*.db", "*.sqlite", "*.sqlite3"]:
                        extracted_files.extend(list(subdir.glob(pattern)))
        
        if not extracted_files:
            # List all files for debugging
            all_files = list(Path("/tmp/chembl/").rglob("*"))
            logger.error(f"Available files in archive: {[str(f) for f in all_files]}")
            raise FileNotFoundError("No SQLite (.db, .sqlite, .sqlite3) file found in extracted archive")
        
        # Use the first (and likely only) database file found
        sqlite_file = extracted_files[0]
        logger.info(f"Found database file: {sqlite_file} ({sqlite_file.stat().st_size / 1e9:.2f} GB)")
        
        # Move to expected location with standard name
        target_file = chembl_file  # This is /vol/chembl/chembl.sqlite
        sqlite_file.rename(target_file)
        
        # Cleanup
        temp_file.unlink()
        
        size_gb = chembl_file.stat().st_size / 1e9
        logger.info(f"✅ ChEMBL database downloaded: {size_gb:.1f} GB")
        
        return {
            "status": "downloaded",
            "path": str(chembl_file),
            "size_gb": size_gb
        }
        
    except Exception as e:
        logger.error(f"❌ ChEMBL download failed: {e}")
        raise

@app.function(
    image=image,
    volumes={
        "/vol/chembl": chembl_volume,
        "/vol/datasets": datasets_volume
    },
    cpu=8.0,
    memory=32768,
    timeout=3600  # 1 hour
)
def extract_oncoprotein_data():
    """
    Step 2: Extract IC50/Ki/EC50 data for oncoproteins from ChEMBL
    """
    import sqlite3
    import pandas as pd
    import numpy as np
    import logging
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    chembl_db = "/vol/chembl/chembl.sqlite"
    if not Path(chembl_db).exists():
        # Also check for alternative naming
        alt_paths = [
            "/vol/chembl/chembl_35.db",
            "/vol/chembl/chembl.db", 
            "/vol/chembl/chembl_35.sqlite"
        ]
        found_db = None
        for alt_path in alt_paths:
            if Path(alt_path).exists():
                found_db = alt_path
                chembl_db = alt_path
                logger.info(f"Found ChEMBL database at alternative path: {alt_path}")
                break
        
        if not found_db:
            available_files = list(Path("/vol/chembl/").glob("*"))
            logger.error(f"Available files in /vol/chembl/: {[f.name for f in available_files]}")
            raise FileNotFoundError("ChEMBL database not found. Run download_chembl_database() first.")
    
    logger.info(f"🔬 Connecting to ChEMBL database: {chembl_db}")
    
    # Test database connection and structure
    try:
        conn = sqlite3.connect(chembl_db)
        
        # Verify database has expected tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = ['activities', 'assays', 'target_dictionary', 'compound_structures']
        missing_tables = [table for table in required_tables if table not in tables]
        
        if missing_tables:
            logger.error(f"Missing required tables: {missing_tables}")
            logger.info(f"Available tables: {tables[:10]}...")  # Show first 10 tables
            raise ValueError(f"Database missing required tables: {missing_tables}")
        
        logger.info(f"✅ Database connection successful. Found {len(tables)} tables.")
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        conn.close() if 'conn' in locals() else None
        raise
    
    # Get target ChEMBL IDs
    target_ids = list(ONCOPROTEIN_TARGETS.values())
    target_names = list(ONCOPROTEIN_TARGETS.keys())
    
    logger.info(f"📊 Extracting bioactivity data for {len(target_ids)} oncoproteins...")
    logger.info(f"Targets: {', '.join([f'{name} ({tid})' for name, tid in zip(target_names, target_ids)])}")
    
    # First, verify that the target ChEMBL IDs exist in the database
    logger.info("🔍 Verifying target availability in database...")
    target_check_query = f"""
    SELECT td.chembl_id, td.pref_name, COUNT(*) as assay_count
    FROM target_dictionary td
    JOIN assays a ON td.tid = a.tid
    WHERE td.chembl_id IN ({','.join([f"'{tid}'" for tid in target_ids])})
    GROUP BY td.chembl_id, td.pref_name
    ORDER BY assay_count DESC
    """
    
    try:
        target_check_df = pd.read_sql_query(target_check_query, conn)
        logger.info(f"Found {len(target_check_df)} targets with assays:")
        for _, row in target_check_df.iterrows():
            logger.info(f"  {row['chembl_id']}: {row['pref_name']} ({row['assay_count']} assays)")
        
        if target_check_df.empty:
            logger.error("❌ None of the specified targets found in database")
            raise ValueError("No specified oncoprotein targets found in ChEMBL database")
            
    except Exception as e:
        logger.error(f"❌ Target verification failed: {e}")
        raise
    
    # SQL query to extract bioactivity data
    query = """
    SELECT DISTINCT
        cs.canonical_smiles,
        td.chembl_id as target_chembl_id,
        act.standard_type,
        act.standard_value,
        act.standard_units,
        act.pchembl_value,
        assays.confidence_score,
        act.activity_id
    FROM activities act
    JOIN assays ON act.assay_id = assays.assay_id
    JOIN target_dictionary td ON assays.tid = td.tid
    JOIN compound_structures cs ON act.molregno = cs.molregno
    WHERE td.chembl_id IN ({})
    AND act.standard_type IN ('IC50', 'Ki', 'EC50')
    AND act.standard_units = 'nM'
    AND act.standard_value IS NOT NULL
    AND cs.canonical_smiles IS NOT NULL
    AND assays.confidence_score >= 8
    AND act.standard_value > 0
    AND act.standard_value <= 1000000
    ORDER BY cs.canonical_smiles, td.chembl_id
    """.format(','.join([f"'{tid}'" for tid in target_ids]))
    
    logger.info("🗄️ Executing main bioactivity data query...")
    logger.info("This may take several minutes for large datasets...")
    
    try:
        df = pd.read_sql_query(query, conn)
        logger.info(f"📈 Retrieved {len(df)} bioactivity records")
        
    except Exception as e:
        logger.error(f"❌ Query execution failed: {e}")
        logger.info("💡 Trying simpler query without confidence score filter...")
        
        # Fallback query without confidence score filter
        fallback_query = query.replace("AND assays.confidence_score >= 8", "")
        try:
            df = pd.read_sql_query(fallback_query, conn)
            logger.info(f"📈 Retrieved {len(df)} bioactivity records (fallback query)")
        except Exception as e2:
            logger.error(f"❌ Fallback query also failed: {e2}")
            raise
    
    finally:
        conn.close()
    
    if df.empty:
        logger.error("❌ No bioactivity data retrieved from ChEMBL")
        logger.info("💡 This might be due to:")
        logger.info("  - Targets not having IC50/Ki/EC50 data in nM units")
        logger.info("  - High confidence score threshold (>= 8)")
        logger.info("  - Missing SMILES structures")
        raise ValueError("No bioactivity data retrieved from ChEMBL for specified targets")
    
    # Data preprocessing
    logger.info("🧹 Preprocessing bioactivity data...")
    
    # Convert to pIC50 values
    df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)  # Convert nM to M, then -log10
    
    # Remove outliers (pIC50 < 0 or > 12)
    initial_count = len(df)
    df = df[(df['pIC50'] >= 0) & (df['pIC50'] <= 12)]
    logger.info(f"🔍 Removed {initial_count - len(df)} outliers")
    
    # Map ChEMBL IDs to target names
    chembl_to_name = {v: k for k, v in ONCOPROTEIN_TARGETS.items()}
    df['target_name'] = df['target_chembl_id'].map(chembl_to_name)
    
    # For compounds with multiple measurements, take the median
    logger.info("📊 Aggregating multiple measurements per compound-target pair...")
    
    df_agg = df.groupby(['canonical_smiles', 'target_name']).agg({
        'pIC50': 'median',
        'standard_value': 'median',
        'activity_id': 'count'
    }).reset_index()
    
    df_agg.rename(columns={'activity_id': 'measurement_count'}, inplace=True)
    
    # Pivot to create multi-target dataset
    logger.info("🔄 Creating multi-target pivot table...")
    
    pivot_df = df_agg.pivot(index='canonical_smiles', columns='target_name', values='pIC50')
    pivot_df.reset_index(inplace=True)
    
    # Add SMILES column name
    pivot_df.rename(columns={'canonical_smiles': 'SMILES'}, inplace=True)
    
    # Fill NaN values (compounds not tested on certain targets)
    logger.info("📋 Dataset statistics:")
    for target in target_names:
        if target in pivot_df.columns:
            count = pivot_df[target].notna().sum()
            logger.info(f"   {target}: {count} compounds")
        else:
            logger.info(f"   {target}: 0 compounds (no data)")
            pivot_df[target] = np.nan
    
    total_compounds = len(pivot_df)
    logger.info(f"📊 Total unique compounds: {total_compounds}")
    
    # Save datasets
    logger.info("💾 Saving datasets...")
    
    csv_path = "/vol/datasets/oncoprotein_multitask_dataset.csv"
    parquet_path = "/vol/datasets/oncoprotein_multitask_dataset.parquet"
    
    pivot_df.to_csv(csv_path, index=False)
    pivot_df.to_parquet(parquet_path, index=False)
    
    # Save summary statistics
    summary_stats = {
        'total_compounds': total_compounds,
        'targets': {target: int(pivot_df[target].notna().sum()) if target in pivot_df.columns else 0 
                   for target in target_names},
        'raw_records': len(df),
        'aggregated_records': len(df_agg)
    }
    
    import json
    with open("/vol/datasets/dataset_summary.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logger.info("✅ Oncoprotein dataset extraction completed")
    
    return summary_stats

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=4.0,
    memory=16384,
    timeout=1800
)
def generate_dataset_report():
    """
    Step 3: Generate dataset statistics and visualizations
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import logging
    import json
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load dataset
    dataset_path = "/vol/datasets/oncoprotein_multitask_dataset.parquet"
    if not Path(dataset_path).exists():
        raise FileNotFoundError("Dataset not found. Run extract_oncoprotein_data() first.")
    
    logger.info("📊 Loading dataset for analysis...")
    df = pd.read_parquet(dataset_path)
    
    # Generate comprehensive report
    logger.info("📈 Generating dataset report...")
    
    # Basic statistics
    target_columns = [col for col in df.columns if col != 'SMILES']
    
    report = {
        'dataset_overview': {
            'total_compounds': len(df),
            'total_targets': len(target_columns),
            'total_measurements': df[target_columns].notna().sum().sum()
        },
        'target_statistics': {},
        'data_distribution': {}
    }
    
    # Per-target statistics
    for target in target_columns:
        target_data = df[target].dropna()
        if len(target_data) > 0:
            report['target_statistics'][target] = {
                'compound_count': len(target_data),
                'mean_pIC50': float(target_data.mean()),
                'std_pIC50': float(target_data.std()),
                'min_pIC50': float(target_data.min()),
                'max_pIC50': float(target_data.max()),
                'median_pIC50': float(target_data.median())
            }
        else:
            report['target_statistics'][target] = {
                'compound_count': 0,
                'mean_pIC50': None,
                'std_pIC50': None,
                'min_pIC50': None,
                'max_pIC50': None,
                'median_pIC50': None
            }
    
    # Create visualizations
    logger.info("📊 Creating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Compound count per target
    targets_with_data = [(target, report['target_statistics'][target]['compound_count']) 
                        for target in target_columns 
                        if report['target_statistics'][target]['compound_count'] > 0]
    
    if targets_with_data:
        targets, counts = zip(*sorted(targets_with_data, key=lambda x: x[1], reverse=True))
        
        axes[0, 0].bar(targets, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        axes[0, 0].set_title('Compound Count per Target', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Compounds')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, count in enumerate(counts):
            axes[0, 0].text(i, count + max(counts) * 0.01, str(count), 
                           ha='center', va='bottom', fontweight='bold')
    
    # 2. pIC50 distribution heatmap
    target_data_for_heatmap = []
    for target in target_columns:
        if report['target_statistics'][target]['compound_count'] > 0:
            target_data_for_heatmap.append([
                report['target_statistics'][target]['mean_pIC50'],
                report['target_statistics'][target]['std_pIC50'],
                report['target_statistics'][target]['compound_count']
            ])
    
    if target_data_for_heatmap:
        heatmap_data = np.array(target_data_for_heatmap).T
        valid_targets = [t for t in target_columns if report['target_statistics'][t]['compound_count'] > 0]
        
        sns.heatmap(heatmap_data, 
                   xticklabels=valid_targets,
                   yticklabels=['Mean pIC50', 'Std pIC50', 'Count (scaled)'],
                   annot=True, fmt='.2f', cmap='viridis', ax=axes[0, 1])
        axes[0, 1].set_title('Target Statistics Heatmap', fontsize=14, fontweight='bold')
    
    # 3. pIC50 distribution violin plot
    plot_data = []
    for target in target_columns:
        target_values = df[target].dropna()
        if len(target_values) > 5:  # Only plot if sufficient data
            for value in target_values:
                plot_data.append({'Target': target, 'pIC50': value})
    
    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        sns.violinplot(data=plot_df, x='Target', y='pIC50', ax=axes[1, 0])
        axes[1, 0].set_title('pIC50 Distribution per Target', fontsize=14, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Data coverage matrix
    coverage_matrix = df[target_columns].notna().astype(int)
    sns.heatmap(coverage_matrix.corr(), annot=True, cmap='RdYlBu', 
               xticklabels=target_columns, yticklabels=target_columns, ax=axes[1, 1])
    axes[1, 1].set_title('Target Data Coverage Correlation', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/vol/datasets/dataset_report.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed report
    with open('/vol/datasets/dataset_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("✅ Dataset report generated")
    
    return report

@app.function(
    image=image,
    gpu="A100-40GB",  # High-performance GPU for training
    volumes={
        "/vol/datasets": datasets_volume,
        "/vol/models": models_volume
    },
    cpu=8.0,
    memory=32768,
    timeout=14400  # 4 hours
)
def train_multitask_chemberta():
    """
    Step 4: Train multi-task ChemBERTa model
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
    from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
    import pandas as pd
    import numpy as np
    import logging
    from pathlib import Path
    from sklearn.metrics import r2_score, mean_squared_error
    import json
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Check for dataset
    dataset_path = "/vol/datasets/oncoprotein_multitask_dataset.parquet"
    if not Path(dataset_path).exists():
        raise FileNotFoundError("Dataset not found. Run extract_oncoprotein_data() first.")
    
    logger.info("🚀 Starting multi-task ChemBERTa training...")
    logger.info(f"🖥️ GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    # Load dataset
    logger.info("📊 Loading dataset...")
    df = pd.read_parquet(dataset_path)
    target_columns = [col for col in df.columns if col != 'SMILES']
    
    logger.info(f"📈 Dataset: {len(df)} compounds, {len(target_columns)} targets")
    
    # Load pre-trained ChemBERTa
    logger.info("📥 Loading pre-trained ChemBERTa...")
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    
    # Multi-task model definition
    class MultiTaskChemBERTa(nn.Module):
        def __init__(self, base_model, num_targets, hidden_dim=768, dropout=0.1):
            super().__init__()
            self.bert = base_model
            self.dropout = nn.Dropout(dropout)
            
            # Shared representation layer
            self.shared_layer = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # Individual task heads
            self.task_heads = nn.ModuleDict({
                target: nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 1)
                ) for target in target_columns
            })
            
        def forward(self, input_ids, attention_mask):
            # Get BERT embeddings
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            
            # Shared representation
            shared_repr = self.shared_layer(self.dropout(pooled_output))
            
            # Task-specific predictions
            predictions = {}
            for target, head in self.task_heads.items():
                predictions[target] = head(shared_repr).squeeze(-1)
            
            return predictions
    
    # Dataset class
    class OncoproteincDataset(Dataset):
        def __init__(self, dataframe, tokenizer, target_columns, max_length=128):
            self.data = dataframe.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.target_columns = target_columns
            self.max_length = max_length
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            smiles = row['SMILES']
            
            # Tokenize SMILES
            encoding = self.tokenizer(
                smiles,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Extract targets
            targets = {}
            for target in self.target_columns:
                value = row[target]
                if pd.isna(value):
                    targets[target] = float('nan')
                else:
                    targets[target] = float(value)
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': targets
            }
    
    # Prepare dataset
    logger.info("🔄 Preparing training dataset...")
    
    # Remove compounds with no target data
    df_clean = df.dropna(subset=target_columns, how='all')
    logger.info(f"📊 Cleaned dataset: {len(df_clean)} compounds with target data")
    
    # Create dataset
    full_dataset = OncoproteincDataset(df_clean, tokenizer, target_columns)
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"📊 Dataset split: {train_size} train, {val_size} val, {test_size} test")
    
    # Data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskChemBERTa(base_model, len(target_columns)).to(device)
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    
    total_steps = len(train_loader) * 10  # 10 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    # Masked MSE loss function
    def masked_mse_loss(predictions, targets, target_columns):
        total_loss = 0
        valid_losses = 0
        
        for target in target_columns:
            pred = predictions[target]
            true = targets[target]
            
            # Create mask for non-NaN values
            mask = ~torch.isnan(true)
            
            if mask.sum() > 0:  # If there are valid targets
                masked_pred = pred[mask]
                masked_true = true[mask]
                loss = nn.MSELoss()(masked_pred, masked_true)
                total_loss += loss
                valid_losses += 1
        
        return total_loss / max(valid_losses, 1)
    
    # Training loop
    logger.info("🎓 Starting training...")
    
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': [], 'target_r2': {}}
    
    for epoch in range(10):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = {target: batch['targets'][target].to(device) 
                      for target in target_columns}
            
            optimizer.zero_grad()
            
            predictions = model(input_ids, attention_mask)
            loss = masked_mse_loss(predictions, targets, target_columns)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                logger.info(f"   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_predictions = {target: [] for target in target_columns}
        val_targets = {target: [] for target in target_columns}
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = {target: batch['targets'][target].to(device) 
                          for target in target_columns}
                
                predictions = model(input_ids, attention_mask)
                loss = masked_mse_loss(predictions, targets, target_columns)
                total_val_loss += loss.item()
                
                # Collect predictions for R² calculation
                for target in target_columns:
                    pred = predictions[target].cpu().numpy()
                    true = targets[target].cpu().numpy()
                    
                    # Only collect valid (non-NaN) values
                    mask = ~np.isnan(true)
                    if mask.sum() > 0:
                        val_predictions[target].extend(pred[mask])
                        val_targets[target].extend(true[mask])
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Calculate R² scores
        target_r2 = {}
        for target in target_columns:
            if len(val_predictions[target]) > 0:
                r2 = r2_score(val_targets[target], val_predictions[target])
                target_r2[target] = r2
            else:
                target_r2[target] = 0.0
        
        avg_r2 = np.mean(list(target_r2.values()))
        
        logger.info(f"📈 Epoch {epoch + 1}/10:")
        logger.info(f"   Train Loss: {avg_train_loss:.4f}")
        logger.info(f"   Val Loss: {avg_val_loss:.4f}")
        logger.info(f"   Avg R²: {avg_r2:.4f}")
        
        # Log top R² scores
        sorted_r2 = sorted(target_r2.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"   Top R² scores: {sorted_r2}")
        
        # Save training history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['target_r2'][f'epoch_{epoch+1}'] = target_r2
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            model_save_dir = Path("/vol/models/chemberta_multitask")
            model_save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model state
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'target_r2': target_r2,
                'target_columns': target_columns
            }, model_save_dir / "best_model.pt")
            
            # Save tokenizer
            tokenizer.save_pretrained(model_save_dir)
            
            logger.info(f"💾 Best model saved (Val Loss: {avg_val_loss:.4f})")
    
    # Save training history
    with open("/vol/models/chemberta_multitask/training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("✅ Multi-task ChemBERTa training completed!")
    
    return {
        "status": "completed",
        "best_val_loss": best_val_loss,
        "final_target_r2": target_r2,
        "avg_r2": avg_r2,
        "model_path": "/vol/models/chemberta_multitask"
    }

@app.function(
    image=image,
    volumes={"/vol/models": models_volume},
    cpu=2.0,
    memory=8192,
    timeout=300
)
def create_inference_script():
    """
    Step 5: Create inference script for trained model
    """
    import logging
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("📝 Creating inference script...")
    
    inference_code = '''"""
Inference script for Multi-Task ChemBERTa Oncoprotein Model
Usage: python inference.py "CC(=O)Nc1ccc..."
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import argparse
import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class MultiTaskChemBERTa(nn.Module):
    def __init__(self, base_model, target_columns, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(dropout)
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Individual task heads
        self.task_heads = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1)
            ) for target in target_columns
        })
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        
        # Shared representation
        shared_repr = self.shared_layer(self.dropout(pooled_output))
        
        # Task-specific predictions
        predictions = {}
        for target, head in self.task_heads.items():
            predictions[target] = head(shared_repr).squeeze(-1)
        
        return predictions

def load_model(model_path="/vol/models/chemberta_multitask"):
    """Load trained multi-task ChemBERTa model"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model checkpoint
    checkpoint = torch.load(f"{model_path}/best_model.pt", map_location='cpu')
    target_columns = checkpoint['target_columns']
    
    # Load base model
    from transformers import AutoModel
    base_model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    
    # Initialize multi-task model
    model = MultiTaskChemBERTa(base_model, target_columns)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer, target_columns

def predict(smiles, model=None, tokenizer=None, target_columns=None, model_path="/vol/models/chemberta_multitask"):
    """
    Predict pIC50 values for all targets
    
    Args:
        smiles: SMILES string
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        target_columns: Target column names (optional)
        model_path: Path to model directory
    
    Returns:
        Dictionary of predicted pIC50 values
    """
    
    # Load model if not provided
    if model is None or tokenizer is None:
        model, tokenizer, target_columns = load_model(model_path)
    
    # Tokenize SMILES
    inputs = tokenizer(
        smiles,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # Make prediction
    with torch.no_grad():
        predictions = model(inputs['input_ids'], inputs['attention_mask'])
    
    # Convert to dictionary
    results = {}
    for target in target_columns:
        pred_value = predictions[target].item()
        # Convert back to IC50 in nM if needed
        ic50_nm = 10 ** (9 - pred_value)  # Convert pIC50 to nM
        results[target] = {
            'pIC50': round(pred_value, 3),
            'IC50_nM': round(ic50_nm, 1)
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Multi-Task ChemBERTa Inference')
    parser.add_argument('smiles', help='SMILES string to predict')
    parser.add_argument('--model-path', default='/vol/models/chemberta_multitask', 
                       help='Path to model directory')
    parser.add_argument('--format', choices=['json', 'table'], default='json',
                       help='Output format')
    
    args = parser.parse_args()
    
    try:
        # Load model
        print("Loading model...")
        model, tokenizer, target_columns = load_model(args.model_path)
        
        # Make prediction
        print(f"Predicting for SMILES: {args.smiles}")
        results = predict(args.smiles, model, tokenizer, target_columns)
        
        if args.format == 'json':
            print(json.dumps(results, indent=2))
        else:
            print(f"\\n{'Target':<10} {'pIC50':<8} {'IC50 (nM)':<12}")
            print("-" * 32)
            for target, values in results.items():
                print(f"{target:<10} {values['pIC50']:<8} {values['IC50_nM']:<12}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    # Save inference script
    script_path = "/vol/models/inference.py"
    with open(script_path, 'w') as f:
        f.write(inference_code)
    
    # Make executable
    import os
    os.chmod(script_path, 0o755)
    
    logger.info(f"✅ Inference script created: {script_path}")
    
    return {"status": "created", "path": script_path}

# Orchestration function
@app.local_entrypoint()
def run_complete_pipeline():
    """
    Run the complete multi-task ChemBERTa pipeline
    """
    print("🚀 Starting Multi-Task ChemBERTa Pipeline for Oncoproteins")
    print("=" * 60)
    
    # Step 1: Download ChEMBL
    print("📥 Step 1: Downloading ChEMBL database...")
    download_result = download_chembl_database.remote()
    print(f"✅ ChEMBL downloaded: {download_result}")
    
    # Step 2: Extract data
    print("🔬 Step 2: Extracting oncoprotein bioactivity data...")
    extract_result = extract_oncoprotein_data.remote()
    print(f"✅ Data extracted: {extract_result}")
    
    # Step 3: Generate report
    print("📊 Step 3: Generating dataset report...")
    report_result = generate_dataset_report.remote()
    print(f"✅ Report generated: {report_result}")
    
    # Step 4: Train model
    print("🎓 Step 4: Training multi-task ChemBERTa...")
    training_result = train_multitask_chemberta.remote()
    print(f"✅ Training completed: {training_result}")
    
    # Step 5: Create inference script
    print("📝 Step 5: Creating inference script...")
    inference_result = create_inference_script.remote()
    print(f"✅ Inference script created: {inference_result}")
    
    print("\n🎉 Multi-Task ChemBERTa Pipeline Completed!")
    print("📁 Deliverables:")
    print("  • /vol/datasets/oncoprotein_multitask_dataset.csv")
    print("  • /vol/datasets/oncoprotein_multitask_dataset.parquet")
    print("  • /vol/models/chemberta_multitask/best_model.pt")
    print("  • /vol/models/inference.py")
    print("  • /vol/datasets/dataset_report.png")
    
    return {
        "status": "completed",
        "deliverables": {
            "dataset": "/vol/datasets/oncoprotein_multitask_dataset.csv",
            "model": "/vol/models/chemberta_multitask/best_model.pt",
            "inference": "/vol/models/inference.py",
            "report": "/vol/datasets/dataset_report.png"
        },
        "results": {
            "download": download_result,
            "extraction": extract_result,
            "training": training_result
        }
    }

if __name__ == "__main__":
    print("🎯 Multi-Task ChemBERTa for Oncoproteins")
    print("Usage: modal run oncoprotein_chemberta.py")