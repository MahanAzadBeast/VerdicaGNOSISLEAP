"""
ChEMBL Data Connector
Fetches and enriches compound data from ChEMBL using webresource client
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_CLIENT_AVAILABLE = True
except ImportError:
    CHEMBL_CLIENT_AVAILABLE = False

from ..utils.rate_limit import qps_limiter
from ..utils.chem import canonicalize, validate_smiles

logger = logging.getLogger(__name__)


class ChEMBLFetcher:
    """
    ChEMBL data fetcher using webresource client
    """
    
    def __init__(self, max_qps: float = 10):
        self.max_qps = max_qps
        
        if CHEMBL_CLIENT_AVAILABLE:
            self.molecules = new_client.molecule
            self.activities = new_client.activity
            self.targets = new_client.target
            self.mechanisms = new_client.mechanism
        else:
            logger.error("ChEMBL webresource client not available")
            raise ImportError("chembl_webresource_client required")
    
    def load_existing_chembl_data(self, chembl_file: str) -> pd.DataFrame:
        """
        Load existing ChEMBL data from our collected dataset
        
        Args:
            chembl_file: Path to existing ChEMBL CSV file
            
        Returns:
            DataFrame with ChEMBL compound data
        """
        logger.info(f"Loading existing ChEMBL data from {chembl_file}")
        
        try:
            df = pd.read_csv(chembl_file)
            logger.info(f"✅ Loaded {len(df):,} ChEMBL compounds")
            
            # Validate required columns
            required_cols = ['chembl_id', 'primary_drug', 'smiles']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()
            
            # Validate SMILES
            valid_smiles_mask = df['smiles'].apply(validate_smiles)
            valid_count = valid_smiles_mask.sum()
            
            logger.info(f"✅ Valid SMILES: {valid_count:,}/{len(df):,} ({(valid_count/len(df)*100):.1f}%)")
            
            return df[valid_smiles_mask].copy()
            
        except Exception as e:
            logger.error(f"Error loading ChEMBL data: {e}")
            return pd.DataFrame()
    
    @qps_limiter(max_qps=10)
    def fetch_molecule_details(self, chembl_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed molecule information from ChEMBL
        
        Args:
            chembl_id: ChEMBL molecule identifier
            
        Returns:
            Dictionary with molecule details or None if not found
        """
        try:
            molecule = self.molecules.get(chembl_id)
            
            if not molecule:
                return None
            
            # Extract comprehensive molecule data
            molecule_data = {
                'chembl_id': molecule.get('molecule_chembl_id'),
                'pref_name': molecule.get('pref_name'),
                'molecule_type': molecule.get('molecule_type'),
                'max_phase': molecule.get('max_phase'),
                'first_approval': molecule.get('first_approval'),
                'oral': molecule.get('oral'),
                'parenteral': molecule.get('parenteral'),
                'topical': molecule.get('topical'),
                'black_box_warning': molecule.get('black_box_warning'),
                'availability_type': molecule.get('availability_type'),
                'chirality': molecule.get('chirality')
            }
            
            # Add structure information
            structures = molecule.get('molecule_structures')
            if structures:
                molecule_data.update({
                    'canonical_smiles': structures.get('canonical_smiles'),
                    'standard_inchi': structures.get('standard_inchi'),
                    'standard_inchi_key': structures.get('standard_inchi_key')
                })
            
            # Add molecular properties
            properties = molecule.get('molecule_properties')
            if properties:
                molecule_data.update({
                    'molecular_weight': properties.get('full_mwt'),
                    'alogp': properties.get('alogp'),
                    'hbd': properties.get('hbd'),
                    'hba': properties.get('hba'),
                    'psa': properties.get('psa'),
                    'rtb': properties.get('rtb'),
                    'aromatic_rings': properties.get('aromatic_rings'),
                    'heavy_atoms': properties.get('heavy_atoms'),
                    'num_rings': properties.get('num_rings')
                })
            
            return molecule_data
            
        except Exception as e:
            logger.error(f"Error fetching molecule {chembl_id}: {e}")
            return None
    
    @qps_limiter(max_qps=5)
    def fetch_bioactivity_data(self, chembl_id: str, target_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch bioactivity data for toxicity surrogates
        
        Args:
            chembl_id: ChEMBL molecule identifier
            target_types: List of target types to filter (e.g., ['KCNH2'] for hERG)
            
        Returns:
            List of bioactivity records
        """
        if target_types is None:
            target_types = ['KCNH2', 'CYP3A4', 'CYP2D6', 'CYP1A2']  # Common toxicity targets
        
        try:
            activities = self.activities.filter(molecule_chembl_id=chembl_id)
            
            bioactivity_records = []
            
            for activity in activities:
                target_chembl_id = activity.get('target_chembl_id')
                
                if target_chembl_id:
                    # Get target details
                    target = self.targets.get(target_chembl_id)
                    
                    if target and target.get('target_type') in target_types:
                        record = {
                            'chembl_id': chembl_id,
                            'target_chembl_id': target_chembl_id,
                            'target_type': target.get('target_type'),
                            'target_pref_name': target.get('pref_name'),
                            'standard_type': activity.get('standard_type'),
                            'standard_value': activity.get('standard_value'),
                            'standard_units': activity.get('standard_units'),
                            'pchembl_value': activity.get('pchembl_value'),
                            'activity_comment': activity.get('activity_comment'),
                            'assay_type': activity.get('assay_type'),
                            'confidence_score': activity.get('confidence_score')
                        }
                        bioactivity_records.append(record)
            
            logger.debug(f"Found {len(bioactivity_records)} bioactivity records for {chembl_id}")
            return bioactivity_records
            
        except Exception as e:
            logger.error(f"Error fetching bioactivity for {chembl_id}: {e}")
            return []
    
    def enrich_chembl_dataset(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich existing ChEMBL dataset with additional API data
        
        Args:
            base_df: Base ChEMBL DataFrame
            
        Returns:
            Enriched DataFrame with additional ChEMBL data
        """
        logger.info("🔬 Enriching ChEMBL dataset with API data")
        
        enriched_records = []
        
        for idx, row in base_df.iterrows():
            try:
                chembl_id = row['chembl_id']
                
                # Get additional molecule details
                molecule_details = self.fetch_molecule_details(chembl_id)
                
                # Start with base record
                enriched_record = row.to_dict()
                
                # Add API enrichment
                if molecule_details:
                    # Add temporal information
                    enriched_record.update({
                        'first_approval': molecule_details.get('first_approval'),
                        'oral_availability': molecule_details.get('oral'),
                        'parenteral_availability': molecule_details.get('parenteral'),
                        'topical_availability': molecule_details.get('topical'),
                        'black_box_warning': molecule_details.get('black_box_warning'),
                        'availability_type': molecule_details.get('availability_type'),
                        'chirality': molecule_details.get('chirality')
                    })
                
                # Fetch bioactivity for toxicity surrogates
                bioactivity = self.fetch_bioactivity_data(chembl_id)
                
                # Aggregate bioactivity data
                if bioactivity:
                    enriched_record.update(self._aggregate_bioactivity(bioactivity))
                
                # Standardize structure if needed
                if 'smiles' in enriched_record:
                    canonical_smiles, inchi, inchikey, descriptors = canonicalize(enriched_record['smiles'])
                    
                    if canonical_smiles:
                        enriched_record.update({
                            'canonical_smiles': canonical_smiles,
                            'inchi': inchi,
                            'inchikey': inchikey,
                            'structure_standardized': True
                        })
                        
                        # Update descriptors if computed
                        if descriptors:
                            enriched_record.update(descriptors)
                
                # Add temporal metadata
                enriched_record.update({
                    'first_seen_date': datetime.now(),  # Will be updated with actual data
                    'source_first_seen': 'chembl',
                    'created_at': datetime.now(),
                    'last_updated': datetime.now()
                })
                
                enriched_records.append(enriched_record)
                
                # Progress logging
                if (idx + 1) % 100 == 0:
                    logger.info(f"Enriched {idx + 1:,}/{len(base_df):,} compounds")
                
            except Exception as e:
                logger.error(f"Error enriching compound {row.get('chembl_id', 'unknown')}: {e}")
                # Include original record even if enrichment fails
                enriched_records.append(row.to_dict())
                continue
        
        enriched_df = pd.DataFrame(enriched_records)
        logger.info(f"✅ ChEMBL enrichment complete: {len(enriched_df):,} compounds")
        
        return enriched_df
    
    def _aggregate_bioactivity(self, bioactivity_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate bioactivity data for toxicity surrogates
        
        Args:
            bioactivity_records: List of bioactivity records
            
        Returns:
            Aggregated bioactivity data
        """
        aggregated = {
            'bioactivity_count': len(bioactivity_records),
            'target_count': len(set(r.get('target_chembl_id') for r in bioactivity_records)),
        }
        
        # Aggregate by target type
        target_data = {}
        
        for record in bioactivity_records:
            target_type = record.get('target_type')
            standard_type = record.get('standard_type')
            standard_value = record.get('standard_value')
            
            if target_type and standard_value:
                key = f"{target_type}_{standard_type}".lower()
                
                if key not in target_data:
                    target_data[key] = []
                
                try:
                    target_data[key].append(float(standard_value))
                except (ValueError, TypeError):
                    continue
        
        # Calculate statistics for each target
        for key, values in target_data.items():
            if values:
                aggregated.update({
                    f'{key}_min': min(values),
                    f'{key}_max': max(values),
                    f'{key}_mean': sum(values) / len(values),
                    f'{key}_count': len(values)
                })
        
        return aggregated
    
    def build_master_table(self, chembl_file: str, output_file: str) -> pd.DataFrame:
        """
        Build master compound table from ChEMBL data
        
        Args:
            chembl_file: Path to existing ChEMBL CSV file
            output_file: Path for output parquet file
            
        Returns:
            Master compound DataFrame
        """
        logger.info("🏗️ Building master compound table from ChEMBL")
        
        # Load existing data
        base_df = self.load_existing_chembl_data(chembl_file)
        
        if base_df.empty:
            logger.error("No valid ChEMBL data to process")
            return pd.DataFrame()
        
        # Enrich with API data
        enriched_df = self.enrich_chembl_dataset(base_df)
        
        # Save as parquet
        try:
            enriched_df.to_parquet(output_file, compression='snappy')
            logger.info(f"💾 Master table saved: {output_file}")
        except Exception as e:
            logger.error(f"Error saving master table: {e}")
        
        return enriched_df