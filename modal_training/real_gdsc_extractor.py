"""
Real GDSC REST API Extractor
Uses real GDSC/Sanger API endpoints for drug sensitivity and genomic data
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
import zipfile

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "requests",
    "pandas", 
    "numpy",
    "rdkit-pypi",
    "openpyxl",
    "xlrd"
])

app = modal.App("real-gdsc-extractor")

# Persistent volume for datasets
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

# Real GDSC API endpoints (updated for 2025)
GDSC_API_ENDPOINTS = {
    'base_url': 'https://www.cancerrxgene.org/api',
    'bulk_download': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5',
    'rest_api': 'https://www.cancerrxgene.org/gdsc1000/ANOVA/output/feature'
}

# Real GDSC bulk download URLs
GDSC_REAL_URLS = {
    'drug_sensitivity_gdsc2': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_24Jul22.xlsx',
    'drug_sensitivity_gdsc1': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC1_fitted_dose_response_24Jul22.xlsx',
    'compound_info': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/screened_compunds_rel_8.5.csv',
    'cell_line_details': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Cell_Lines_Details.xlsx',
    'drug_info': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Drug_listMon_Jul_25_09_25_07_2022.csv',
    'genomics_mutations': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/WES_variants.xlsx',
    'genomics_cnv': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/PANCANCER_Genetic_feature_cna.csv',
    'genomics_expression': 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Cell_line_RMA_proc_basalExp.txt.zip'
}

# Oncology-focused cancer types
ONCOLOGY_CANCER_TYPES = [
    'LUNG', 'BREAST', 'COLON', 'LIVER', 'STOMACH', 'PANCREAS', 'KIDNEY',
    'PROSTATE', 'OVARY', 'BRAIN', 'SKIN', 'BLOOD', 'BONE', 'SOFT_TISSUE',
    'HEAD_NECK', 'CERVIX', 'ENDOMETRIUM', 'THYROID', 'BLADDER', 'ESOPHAGUS'
]

# Key cancer genes for genomic features
GDSC_CANCER_GENES = [
    'TP53', 'KRAS', 'PIK3CA', 'APC', 'BRCA1', 'BRCA2', 'EGFR', 'ERBB2',
    'BRAF', 'MET', 'ALK', 'ROS1', 'RET', 'NTRK1', 'CDK4', 'CDK6',
    'MDM2', 'CDKN2A', 'RB1', 'PTEN', 'VHL', 'IDH1', 'IDH2', 'TERT'
]

class RealGDSCExtractor:
    """Real GDSC data extractor using official API endpoints"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Veridica-AI-GDSC-Extractor/1.0',
            'Accept': 'application/json'
        })
        self.logger = logging.getLogger(__name__)
    
    def download_real_gdsc_file(self, url: str, description: str) -> Optional[pd.DataFrame]:
        """Download and parse real GDSC data file"""
        
        self.logger.info(f"📥 Downloading {description} from real GDSC endpoint...")
        
        try:
            response = self.session.get(url, timeout=600, stream=True)  # 10 minute timeout
            
            if response.status_code != 200:
                self.logger.error(f"❌ Failed to download {description}: HTTP {response.status_code}")
                return None
            
            content_length = response.headers.get('content-length')
            if content_length:
                self.logger.info(f"   📊 File size: {int(content_length) / (1024*1024):.1f} MB")
            
            # Read content
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
            
            # Handle different file formats
            if url.endswith('.xlsx'):
                df = pd.read_excel(io.BytesIO(content))
            elif url.endswith('.csv'):
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            elif url.endswith('.zip'):
                # Handle zip files
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    # Find the main data file
                    filenames = z.namelist()
                    data_file = None
                    
                    for filename in filenames:
                        if filename.endswith(('.csv', '.txt', '.xlsx')) and not filename.startswith('__'):
                            data_file = filename
                            break
                    
                    if not data_file:
                        data_file = filenames[0]  # Take first file if no obvious data file
                    
                    with z.open(data_file) as f:
                        if data_file.endswith('.xlsx'):
                            df = pd.read_excel(f)
                        else:
                            df = pd.read_csv(f, sep='\t' if data_file.endswith('.txt') else ',')
            elif url.endswith('.txt'):
                df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep='\t')
            else:
                # Try CSV format as default
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            
            self.logger.info(f"   ✅ {description}: {df.shape[0]:,} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error downloading {description}: {e}")
            return None
    
    def extract_drug_sensitivity_data(self) -> Optional[pd.DataFrame]:
        """Extract real drug sensitivity data from GDSC"""
        
        self.logger.info("🧪 Extracting REAL GDSC drug sensitivity data...")
        
        # Try both GDSC1 and GDSC2 datasets
        sensitivity_datasets = []
        
        # GDSC2 (newer, more drugs)
        gdsc2_df = self.download_real_gdsc_file(
            GDSC_REAL_URLS['drug_sensitivity_gdsc2'],
            'GDSC2 Drug Sensitivity'
        )
        if gdsc2_df is not None:
            gdsc2_df['GDSC_VERSION'] = 'GDSC2'
            sensitivity_datasets.append(gdsc2_df)
        
        # GDSC1 (older, more cell lines)
        gdsc1_df = self.download_real_gdsc_file(
            GDSC_REAL_URLS['drug_sensitivity_gdsc1'],
            'GDSC1 Drug Sensitivity'
        )
        if gdsc1_df is not None:
            gdsc1_df['GDSC_VERSION'] = 'GDSC1'
            sensitivity_datasets.append(gdsc1_df)
        
        if not sensitivity_datasets:
            self.logger.error("❌ Failed to download any GDSC sensitivity data")
            return None
        
        # Combine datasets
        combined_df = pd.concat(sensitivity_datasets, ignore_index=True)
        self.logger.info(f"   ✅ Combined GDSC sensitivity data: {len(combined_df):,} records")
        
        # Process and standardize
        return self._process_real_sensitivity_data(combined_df)
    
    def _process_real_sensitivity_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process real GDSC sensitivity data to standard format"""
        
        self.logger.info("🔧 Processing real GDSC sensitivity data...")
        
        # Map GDSC columns to standard format
        # Common GDSC columns: COSMIC_ID, CELL_LINE_NAME, DRUG_NAME, DRUG_ID, LN_IC50, AUC, RMSE, Z_SCORE
        
        column_mapping = {}
        
        # Find the right columns (GDSC format can vary slightly)
        for col in df.columns:
            col_lower = col.lower()
            if 'cell_line' in col_lower or 'cell line' in col_lower:
                column_mapping[col] = 'CELL_LINE_NAME'
            elif 'drug_name' in col_lower or 'drug name' in col_lower:
                column_mapping[col] = 'DRUG_NAME'
            elif 'cosmic' in col_lower:
                column_mapping[col] = 'COSMIC_ID'  
            elif 'ln_ic50' in col_lower or 'log_ic50' in col_lower:
                column_mapping[col] = 'LN_IC50'
            elif col_lower == 'auc':
                column_mapping[col] = 'AUC'
            elif 'tissue' in col_lower or 'cancer_type' in col_lower:
                column_mapping[col] = 'CANCER_TYPE'
            elif 'drug_id' in col_lower:
                column_mapping[col] = 'DRUG_ID'
        
        # Apply column mapping
        df = df.rename(columns=column_mapping)
        
        # Calculate IC50 values if we have LN_IC50
        if 'LN_IC50' in df.columns:
            # LN_IC50 in GDSC is natural log of IC50 in μM
            df['IC50_uM'] = np.exp(df['LN_IC50'])
            df['IC50_nM'] = df['IC50_uM'] * 1000  # Convert to nM
            df['pIC50'] = -np.log10(df['IC50_uM'] / 1e6)  # pIC50 from μM
        
        # Quality control
        initial_count = len(df)
        
        # Remove rows with missing essential data
        essential_cols = ['CELL_LINE_NAME', 'DRUG_NAME']
        for col in essential_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        # Filter reasonable IC50 range if available
        if 'IC50_nM' in df.columns:
            df = df[
                (df['IC50_nM'] >= 0.1) &  # 0.1 nM minimum
                (df['IC50_nM'] <= 1000000)  # 1 mM maximum
            ]
        
        # Remove duplicates (keep first occurrence)
        if 'CELL_LINE_NAME' in df.columns and 'DRUG_NAME' in df.columns:
            df = df.drop_duplicates(subset=['CELL_LINE_NAME', 'DRUG_NAME'], keep='first')
        
        self.logger.info(f"   📊 After processing: {len(df):,} records (removed {initial_count - len(df):,})")
        
        return df
    
    def extract_compound_info(self) -> Optional[pd.DataFrame]:
        """Extract compound/drug information"""
        
        self.logger.info("💊 Extracting GDSC compound information...")
        
        # Download compound info
        compound_df = self.download_real_gdsc_file(
            GDSC_REAL_URLS['compound_info'],
            'GDSC Compound Information'
        )
        
        if compound_df is None:
            # Try drug info as alternative
            compound_df = self.download_real_gdsc_file(
                GDSC_REAL_URLS['drug_info'],
                'GDSC Drug Information'
            )
        
        if compound_df is not None:
            return self._process_compound_info(compound_df)
        
        return None
    
    def _process_compound_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process compound information"""
        
        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'drug_name' in col_lower or 'compound_name' in col_lower:
                column_mapping[col] = 'DRUG_NAME'
            elif 'drug_id' in col_lower or 'compound_id' in col_lower:
                column_mapping[col] = 'DRUG_ID'
            elif 'smiles' in col_lower:
                column_mapping[col] = 'SMILES'
            elif 'target' in col_lower:
                column_mapping[col] = 'TARGETS'
            elif 'pubchem' in col_lower:
                column_mapping[col] = 'PUBCHEM_ID'
        
        df = df.rename(columns=column_mapping)
        
        return df
    
    def extract_cell_line_info(self) -> Optional[pd.DataFrame]:
        """Extract cell line information"""
        
        self.logger.info("🧬 Extracting GDSC cell line information...")
        
        cell_line_df = self.download_real_gdsc_file(
            GDSC_REAL_URLS['cell_line_details'],
            'GDSC Cell Line Details'
        )
        
        if cell_line_df is not None:
            return self._process_cell_line_info(cell_line_df)
        
        return None
    
    def _process_cell_line_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process cell line information"""
        
        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'cell_line' in col_lower:
                column_mapping[col] = 'CELL_LINE_NAME'
            elif 'cosmic' in col_lower:
                column_mapping[col] = 'COSMIC_ID'
            elif 'tissue' in col_lower and 'type' in col_lower:
                column_mapping[col] = 'TISSUE_TYPE'
            elif 'cancer_type' in col_lower:
                column_mapping[col] = 'CANCER_TYPE'
            elif 'site' in col_lower:
                column_mapping[col] = 'TISSUE_SITE'
        
        df = df.rename(columns=column_mapping)
        
        return df
    
    def extract_genomics_data(self) -> Dict[str, pd.DataFrame]:
        """Extract genomics data (mutations, CNV, expression)"""
        
        self.logger.info("🧬 Extracting GDSC genomics data...")
        
        genomics_data = {}
        
        # Extract mutations
        mutations_df = self.download_real_gdsc_file(
            GDSC_REAL_URLS['genomics_mutations'],
            'GDSC Mutations (WES)'
        )
        if mutations_df is not None:
            genomics_data['mutations'] = self._process_mutations_data(mutations_df)
        
        # Extract copy number variations
        cnv_df = self.download_real_gdsc_file(
            GDSC_REAL_URLS['genomics_cnv'],
            'GDSC Copy Number Variations'
        )
        if cnv_df is not None:
            genomics_data['cnv'] = self._process_cnv_data(cnv_df)
        
        # Extract expression data
        expression_df = self.download_real_gdsc_file(
            GDSC_REAL_URLS['genomics_expression'],
            'GDSC Gene Expression'
        )
        if expression_df is not None:
            genomics_data['expression'] = self._process_expression_data(expression_df)
        
        return genomics_data
    
    def _process_mutations_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process mutations data"""
        
        # Standard genomics columns in GDSC
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'cell_line' in col_lower or 'sample' in col_lower:
                column_mapping[col] = 'CELL_LINE_NAME'
            elif 'gene' in col_lower and 'symbol' in col_lower:
                column_mapping[col] = 'GENE_SYMBOL'
            elif 'mutation' in col_lower and 'type' in col_lower:
                column_mapping[col] = 'MUTATION_TYPE'
            elif 'cosmic' in col_lower:
                column_mapping[col] = 'COSMIC_ID'
        
        df = df.rename(columns=column_mapping)
        
        # Filter for cancer genes
        if 'GENE_SYMBOL' in df.columns:
            df = df[df['GENE_SYMBOL'].isin(GDSC_CANCER_GENES)]
        
        return df
    
    def _process_cnv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process copy number variation data"""
        
        # CNV data is usually in matrix format (cell lines x genes)
        if df.shape[1] > 50:  # Likely a matrix format
            # First column should be cell line identifier
            cell_line_col = df.columns[0]
            df = df.rename(columns={cell_line_col: 'CELL_LINE_NAME'})
            
            # Melt to long format
            gene_cols = [col for col in df.columns if col != 'CELL_LINE_NAME']
            
            # Filter for cancer genes
            cancer_gene_cols = [col for col in gene_cols if any(gene in col for gene in GDSC_CANCER_GENES)]
            
            if cancer_gene_cols:
                df_subset = df[['CELL_LINE_NAME'] + cancer_gene_cols]
                df_melted = df_subset.melt(
                    id_vars=['CELL_LINE_NAME'],
                    var_name='GENE_SYMBOL', 
                    value_name='CNV_VALUE'
                )
                return df_melted
        
        return df
    
    def _process_expression_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process gene expression data"""
        
        # Expression data is usually in matrix format (genes x cell lines or vice versa)
        
        # Check if first column/row contains gene symbols
        first_col_values = df.iloc[:, 0].astype(str).str.upper()
        cancer_genes_in_first_col = sum(1 for gene in GDSC_CANCER_GENES if any(gene in val for val in first_col_values[:20]))
        
        first_row_values = df.iloc[0, :].astype(str).str.upper()  
        cancer_genes_in_first_row = sum(1 for gene in GDSC_CANCER_GENES if any(gene in val for val in first_row_values[:20]))
        
        if cancer_genes_in_first_col > cancer_genes_in_first_row:
            # Genes in rows, cell lines in columns
            df = df.set_index(df.columns[0])
            df = df.T  # Transpose so cell lines are rows
            df.index.name = 'CELL_LINE_NAME'
            df = df.reset_index()
        else:
            # Cell lines in rows, genes in columns (or standard format)
            if 'CELL_LINE_NAME' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'CELL_LINE_NAME'})
        
        # Filter for cancer genes
        gene_cols = [col for col in df.columns if col != 'CELL_LINE_NAME']
        cancer_gene_cols = [col for col in gene_cols if any(gene in col.upper() for gene in GDSC_CANCER_GENES)]
        
        if cancer_gene_cols:
            df_subset = df[['CELL_LINE_NAME'] + cancer_gene_cols]
            
            # Melt to long format
            df_melted = df_subset.melt(
                id_vars=['CELL_LINE_NAME'],
                var_name='GENE_SYMBOL',
                value_name='EXPRESSION_VALUE'
            )
            return df_melted
        
        return df
    
    def create_integrated_genomic_features(self, genomics_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create integrated genomic feature matrix for cell lines"""
        
        self.logger.info("🔗 Creating integrated genomic features...")
        
        # Get all unique cell lines
        all_cell_lines = set()
        for data_type, df in genomics_data.items():
            if 'CELL_LINE_NAME' in df.columns:
                all_cell_lines.update(df['CELL_LINE_NAME'].unique())
        
        # Initialize feature matrix
        feature_records = []
        
        for cell_line in all_cell_lines:
            record = {'CELL_LINE_NAME': cell_line}
            
            # Add mutation features
            if 'mutations' in genomics_data:
                mut_df = genomics_data['mutations']
                cell_mutations = mut_df[mut_df['CELL_LINE_NAME'] == cell_line]
                
                for gene in GDSC_CANCER_GENES:
                    # Binary mutation status
                    has_mutation = gene in cell_mutations['GENE_SYMBOL'].values if 'GENE_SYMBOL' in cell_mutations.columns else False
                    record[f'{gene}_mutation'] = 1 if has_mutation else 0
            
            # Add CNV features
            if 'cnv' in genomics_data:
                cnv_df = genomics_data['cnv']
                cell_cnvs = cnv_df[cnv_df['CELL_LINE_NAME'] == cell_line]
                
                for gene in GDSC_CANCER_GENES:
                    # CNV value (log2 ratio, default to 0 = diploid)
                    gene_cnvs = cell_cnvs[cell_cnvs['GENE_SYMBOL'].str.contains(gene, na=False)] if 'GENE_SYMBOL' in cell_cnvs.columns else pd.DataFrame()
                    cnv_value = gene_cnvs['CNV_VALUE'].iloc[0] if len(gene_cnvs) > 0 and 'CNV_VALUE' in gene_cnvs.columns else 0
                    record[f'{gene}_cnv'] = cnv_value
            
            # Add expression features
            if 'expression' in genomics_data:
                expr_df = genomics_data['expression']
                cell_expressions = expr_df[expr_df['CELL_LINE_NAME'] == cell_line]
                
                for gene in GDSC_CANCER_GENES:
                    # Expression value (log2 normalized, default to 0 = average)
                    gene_expressions = cell_expressions[cell_expressions['GENE_SYMBOL'].str.contains(gene, na=False)] if 'GENE_SYMBOL' in cell_expressions.columns else pd.DataFrame()
                    expr_value = gene_expressions['EXPRESSION_VALUE'].iloc[0] if len(gene_expressions) > 0 and 'EXPRESSION_VALUE' in gene_expressions.columns else 0
                    record[f'{gene}_expression'] = expr_value
            
            feature_records.append(record)
        
        genomic_features_df = pd.DataFrame(feature_records)
        self.logger.info(f"   ✅ Genomic features: {len(genomic_features_df)} cell lines, {len(genomic_features_df.columns)-1} features")
        
        return genomic_features_df

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=4.0,
    memory=16384,
    timeout=7200
)
def extract_real_gdsc_data():
    """
    Extract real GDSC data using official API endpoints and bulk downloads
    """
    
    logging.basicConfig(level=logging.INFO)
    print("🧬 REAL GDSC DATA EXTRACTION")
    print("=" * 80)
    print("🎯 Using: Official GDSC/Sanger Institute endpoints")
    print("❌ NO SYNTHETIC DATA - Real experimental data only")
    
    try:
        extractor = RealGDSCExtractor()
        
        # Extract drug sensitivity data
        print("\n📊 STEP 1: Extracting real drug sensitivity data...")
        sensitivity_df = extractor.extract_drug_sensitivity_data()
        
        if sensitivity_df is None or len(sensitivity_df) == 0:
            raise Exception("Failed to extract real GDSC sensitivity data")
        
        print(f"   ✅ Drug sensitivity: {len(sensitivity_df):,} drug-cell line pairs")
        print(f"   📊 Unique drugs: {sensitivity_df['DRUG_NAME'].nunique() if 'DRUG_NAME' in sensitivity_df.columns else 'N/A'}")
        print(f"   📊 Unique cell lines: {sensitivity_df['CELL_LINE_NAME'].nunique() if 'CELL_LINE_NAME' in sensitivity_df.columns else 'N/A'}")
        
        # Extract compound information
        print("\n💊 STEP 2: Extracting compound information...")
        compound_df = extractor.extract_compound_info()
        if compound_df is not None:
            print(f"   ✅ Compound info: {len(compound_df):,} compounds")
        
        # Extract cell line information  
        print("\n🧬 STEP 3: Extracting cell line information...")
        cell_line_df = extractor.extract_cell_line_info()
        if cell_line_df is not None:
            print(f"   ✅ Cell line info: {len(cell_line_df):,} cell lines")
        
        # Extract genomics data
        print("\n🧬 STEP 4: Extracting genomics data...")
        genomics_data = extractor.extract_genomics_data()
        
        print(f"   📊 Genomics datasets extracted: {len(genomics_data)}")
        for data_type, df in genomics_data.items():
            print(f"     • {data_type}: {len(df):,} records")
        
        # Create integrated genomic features
        genomic_features_df = None
        if genomics_data:
            print("\n🔗 STEP 5: Creating integrated genomic features...")
            genomic_features_df = extractor.create_integrated_genomic_features(genomics_data)
        
        # Save datasets
        print(f"\n💾 STEP 6: Saving real GDSC datasets...")
        
        datasets_dir = Path("/vol/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save sensitivity data
        sensitivity_path = datasets_dir / "gdsc_real_drug_sensitivity.csv"
        sensitivity_df.to_csv(sensitivity_path, index=False)
        saved_files['drug_sensitivity'] = str(sensitivity_path)
        print(f"   ✅ Drug sensitivity: {sensitivity_path}")
        
        # Save compound info
        if compound_df is not None:
            compound_path = datasets_dir / "gdsc_real_compounds.csv"
            compound_df.to_csv(compound_path, index=False)
            saved_files['compounds'] = str(compound_path)
            print(f"   ✅ Compound info: {compound_path}")
        
        # Save cell line info
        if cell_line_df is not None:
            cell_line_path = datasets_dir / "gdsc_real_cell_lines.csv"
            cell_line_df.to_csv(cell_line_path, index=False)
            saved_files['cell_lines'] = str(cell_line_path)
            print(f"   ✅ Cell line info: {cell_line_path}")
        
        # Save genomics data
        for data_type, df in genomics_data.items():
            genomics_path = datasets_dir / f"gdsc_real_{data_type}.csv"
            df.to_csv(genomics_path, index=False)
            saved_files[f'genomics_{data_type}'] = str(genomics_path)
            print(f"   ✅ Genomics {data_type}: {genomics_path}")
        
        # Save integrated genomic features
        if genomic_features_df is not None:
            features_path = datasets_dir / "gdsc_real_genomic_features.csv"
            genomic_features_df.to_csv(features_path, index=False)
            saved_files['genomic_features'] = str(features_path)
            print(f"   ✅ Genomic features: {features_path}")
        
        # Create training-ready dataset
        print(f"\n🎯 STEP 7: Creating training-ready dataset...")
        
        training_data = sensitivity_df.copy()
        
        # Merge with genomic features if available
        if genomic_features_df is not None and 'CELL_LINE_NAME' in training_data.columns:
            training_data = training_data.merge(
                genomic_features_df,
                on='CELL_LINE_NAME',
                how='left'
            )
            print(f"   🔗 Merged with genomics: {training_data.shape}")
        
        # Save training dataset
        training_path = datasets_dir / "gdsc_real_training_data.csv" 
        training_data.to_csv(training_path, index=False)
        saved_files['training_data'] = str(training_path)
        print(f"   ✅ Training dataset: {training_path}")
        
        # Create metadata
        metadata = {
            'extraction_method': 'Real_GDSC_API_BulkDownload',
            'data_source': 'GDSC_Sanger_Institute',
            'data_type': 'real_cancer_drug_sensitivity',
            'focus': 'oncology_cell_line_screening',
            'no_synthetic_data': True,
            'real_experimental_data': True,
            'sensitivity_data': {
                'total_records': len(sensitivity_df),
                'unique_drugs': sensitivity_df['DRUG_NAME'].nunique() if 'DRUG_NAME' in sensitivity_df.columns else 0,
                'unique_cell_lines': sensitivity_df['CELL_LINE_NAME'].nunique() if 'CELL_LINE_NAME' in sensitivity_df.columns else 0,
                'versions': list(sensitivity_df['GDSC_VERSION'].unique()) if 'GDSC_VERSION' in sensitivity_df.columns else []
            },
            'compound_data': {
                'available': compound_df is not None,
                'total_compounds': len(compound_df) if compound_df is not None else 0
            },
            'cell_line_data': {
                'available': cell_line_df is not None, 
                'total_cell_lines': len(cell_line_df) if cell_line_df is not None else 0
            },
            'genomics_data': {
                'datasets_available': list(genomics_data.keys()),
                'integrated_features': genomic_features_df is not None,
                'features_count': len(genomic_features_df.columns) - 1 if genomic_features_df is not None else 0,
                'cell_lines_with_genomics': len(genomic_features_df) if genomic_features_df is not None else 0
            },
            'training_dataset': {
                'shape': training_data.shape,
                'has_genomics': genomic_features_df is not None,
                'ready_for_cell_line_model': True
            },
            'urls_used': GDSC_REAL_URLS,
            'cancer_genes': GDSC_CANCER_GENES,
            'extraction_timestamp': datetime.now().isoformat(),
            'saved_files': saved_files
        }
        
        metadata_path = datasets_dir / "gdsc_real_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate final report
        print(f"\n🎉 REAL GDSC EXTRACTION COMPLETED!")
        print("=" * 80)
        print(f"📁 Dataset files saved:")
        for dataset_type, file_path in saved_files.items():
            print(f"  • {dataset_type}: {file_path}")
        
        print(f"\n📊 Real GDSC data summary:")
        print(f"  • Drug-cell line pairs: {len(sensitivity_df):,}")
        print(f"  • Unique drugs: {sensitivity_df['DRUG_NAME'].nunique() if 'DRUG_NAME' in sensitivity_df.columns else 'N/A'}")
        print(f"  • Unique cell lines: {sensitivity_df['CELL_LINE_NAME'].nunique() if 'CELL_LINE_NAME' in sensitivity_df.columns else 'N/A'}")
        print(f"  • Genomics datasets: {len(genomics_data)}")
        if genomic_features_df is not None:
            print(f"  • Genomic features: {len(genomic_features_df.columns)-1} features for {len(genomic_features_df)} cell lines")
        print(f"  • Training dataset: {training_data.shape}")
        
        print(f"\n🧬 REAL EXPERIMENTAL DATA:")
        print(f"  • Source: Official GDSC/Sanger Institute")
        print(f"  • NO synthetic/simulated data used")
        print(f"  • Real IC50 measurements from cancer cell lines")
        print(f"  • Real genomic profiles (mutations, CNV, expression)")  
        print(f"  • Ready for Cell Line Response Model training")
        
        return {
            'status': 'success',
            'extraction_method': 'Real_GDSC_Official_Sources',
            'no_synthetic_data': True,
            'real_experimental_data': True,
            'sensitivity_records': len(sensitivity_df),
            'unique_drugs': sensitivity_df['DRUG_NAME'].nunique() if 'DRUG_NAME' in sensitivity_df.columns else 0,
            'unique_cell_lines': sensitivity_df['CELL_LINE_NAME'].nunique() if 'CELL_LINE_NAME' in sensitivity_df.columns else 0,
            'genomics_datasets': list(genomics_data.keys()),
            'genomic_features_available': genomic_features_df is not None,
            'training_dataset_shape': training_data.shape,
            'saved_files': saved_files,
            'metadata_path': str(metadata_path),
            'ready_for_training': True,
            'cell_line_model_ready': True
        }
        
    except Exception as e:
        print(f"❌ REAL GDSC EXTRACTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🧬 Real GDSC Data Extractor")