"""
Quality Control for Clinical Trial Dataset
Validates data quality, removes outliers, and prepares final ML-ready dataset
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class QualityController:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_stats = {}
        self.removed_records = []
        self.feature_stats = {}
        
    def validate_required_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records missing critical fields"""
        self.logger.info("Validating required fields...")
        
        initial_count = len(df)
        
        # Define required fields for ML
        required_fields = {
            'smiles': 'SMILES string required for molecular features',
            'primary_drug': 'Drug name required',
            'primary_condition': 'Condition required',
            'primary_phase': 'Phase information required',
            'lead_sponsor': 'Sponsor information required',
            'binary_outcome': 'Outcome label required'
        }
        
        valid_df = df.copy()
        
        for field, description in required_fields.items():
            if field not in valid_df.columns:
                self.logger.warning(f"Missing column: {field}")
                continue
                
            # Count missing values
            missing_count = valid_df[field].isna().sum()
            if missing_count > 0:
                self.logger.info(f"Removing {missing_count} records missing {field}")
                
                # Store removed records
                removed = valid_df[valid_df[field].isna()].copy()
                removed['removal_reason'] = f'Missing {field}: {description}'
                self.removed_records.extend(removed.to_dict('records'))
                
                # Remove missing records
                valid_df = valid_df[valid_df[field].notna()]
        
        # Remove records with unknown outcomes (binary_outcome = -1)
        unknown_outcomes = valid_df['binary_outcome'] == -1
        if unknown_outcomes.sum() > 0:
            self.logger.info(f"Removing {unknown_outcomes.sum()} records with unknown outcomes")
            
            removed = valid_df[unknown_outcomes].copy()
            removed['removal_reason'] = 'Unknown outcome'
            self.removed_records.extend(removed.to_dict('records'))
            
            valid_df = valid_df[~unknown_outcomes]
        
        final_count = len(valid_df)
        removed_count = initial_count - final_count
        
        removal_rate = removed_count/initial_count*100 if initial_count > 0 else 0
        self.logger.info(f"Required fields validation: {initial_count} → {final_count} "
                        f"({removed_count} removed, {removal_rate:.1f}%)")
        
        return valid_df
    
    def validate_smiles_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate SMILES strings and molecular features"""
        self.logger.info("Validating SMILES quality...")
        
        from rdkit import Chem
        
        initial_count = len(df)
        valid_df = df.copy()
        
        def is_valid_smiles(smiles):
            if not smiles or pd.isna(smiles):
                return False
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    return False
                # Additional checks
                if mol.GetNumAtoms() < 5:  # Too small
                    return False
                if mol.GetNumAtoms() > 200:  # Too large
                    return False
                return True
            except:
                return False
        
        # Validate SMILES
        valid_smiles = valid_df['smiles'].apply(is_valid_smiles)
        invalid_count = (~valid_smiles).sum()
        
        if invalid_count > 0:
            self.logger.info(f"Removing {invalid_count} records with invalid SMILES")
            
            removed = valid_df[~valid_smiles].copy()
            removed['removal_reason'] = 'Invalid SMILES string'
            self.removed_records.extend(removed.to_dict('records'))
            
            valid_df = valid_df[valid_smiles]
        
        # Validate molecular weight (reasonable drug-like range)
        if 'mol_molecular_weight' in valid_df.columns:
            mw_mask = (valid_df['mol_molecular_weight'] >= 100) & (valid_df['mol_molecular_weight'] <= 2000)
            mw_outliers = (~mw_mask).sum()
            
            if mw_outliers > 0:
                self.logger.info(f"Removing {mw_outliers} records with extreme molecular weights")
                
                removed = valid_df[~mw_mask].copy()
                removed['removal_reason'] = 'Molecular weight out of range (100-2000)'
                self.removed_records.extend(removed.to_dict('records'))
                
                valid_df = valid_df[mw_mask]
        
        final_count = len(valid_df)
        removed_count = initial_count - final_count
        
        removal_rate = removed_count/initial_count*100 if initial_count > 0 else 0
        self.logger.info(f"SMILES validation: {initial_count} → {final_count} "
                        f"({removed_count} removed, {removal_rate:.1f}%)")
        
        return valid_df
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers from numeric features"""
        self.logger.info("Removing outliers...")
        
        initial_count = len(df)
        valid_df = df.copy()
        
        # Define numeric columns to check for outliers
        numeric_columns = [
            'enrollment_count', 'trial_duration_days', 'sponsor_experience',
            'mol_molecular_weight', 'mol_logp', 'mol_tpsa'
        ]
        
        # Remove extreme outliers using IQR method
        for col in numeric_columns:
            if col not in valid_df.columns:
                continue
                
            # Skip if no data
            if valid_df[col].isna().all():
                continue
                
            # Calculate IQR
            Q1 = valid_df[col].quantile(0.25)
            Q3 = valid_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # No variance
                continue
            
            # Define outlier bounds (3x IQR)
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outlier_mask = (valid_df[col] < lower_bound) | (valid_df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                self.logger.info(f"Removing {outlier_count} outliers from {col}")
                
                removed = valid_df[outlier_mask].copy()
                removed['removal_reason'] = f'Outlier in {col}'
                self.removed_records.extend(removed.to_dict('records'))
                
                valid_df = valid_df[~outlier_mask]
        
        final_count = len(valid_df)
        removed_count = initial_count - final_count
        
        removal_rate = removed_count/initial_count*100 if initial_count > 0 else 0
        self.logger.info(f"Outlier removal: {initial_count} → {final_count} "
                        f"({removed_count} removed, {removal_rate:.1f}%)")
        
        return valid_df
    
    def check_data_balance(self, df: pd.DataFrame) -> Dict:
        """Check class balance and data distribution"""
        self.logger.info("Checking data balance...")
        
        balance_stats = {}
        
        # Overall binary outcome balance
        outcome_counts = df['binary_outcome'].value_counts()
        success_rate = outcome_counts.get(1, 0) / len(df) * 100
        balance_stats['overall_success_rate'] = success_rate
        balance_stats['total_samples'] = len(df)
        balance_stats['success_samples'] = outcome_counts.get(1, 0)
        balance_stats['failure_samples'] = outcome_counts.get(0, 0)
        
        self.logger.info(f"Overall balance: {success_rate:.1f}% success rate")
        self.logger.info(f"Success: {outcome_counts.get(1, 0)}, Failure: {outcome_counts.get(0, 0)}")
        
        # Phase-specific balance
        phase_balance = {}
        for phase in ['PHASE1', 'PHASE2', 'PHASE3']:
            phase_data = df[df['primary_phase'] == phase]
            if len(phase_data) > 0:
                phase_success = len(phase_data[phase_data['binary_outcome'] == 1])
                phase_total = len(phase_data)
                phase_rate = phase_success / phase_total * 100
                phase_balance[phase] = {
                    'total': phase_total,
                    'success': phase_success,
                    'success_rate': phase_rate
                }
                self.logger.info(f"{phase}: {phase_rate:.1f}% success ({phase_success}/{phase_total})")
        
        balance_stats['phase_balance'] = phase_balance
        
        # Disease category balance
        disease_balance = {}
        if 'disease_category' in df.columns:
            for disease in df['disease_category'].unique():
                disease_data = df[df['disease_category'] == disease]
                if len(disease_data) >= 1:  # Lower threshold for small datasets
                    disease_success = len(disease_data[disease_data['binary_outcome'] == 1])
                    disease_total = len(disease_data)
                    disease_rate = disease_success / disease_total * 100 if disease_total > 0 else 0
                    disease_balance[disease] = {
                        'total': disease_total,
                        'success': disease_success,
                        'success_rate': disease_rate
                    }
        
        balance_stats['disease_balance'] = disease_balance
        
        return balance_stats
    
    def prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning"""
        self.logger.info("Preparing ML features...")
        
        ml_df = df.copy()
        
        # Fill missing values
        numeric_columns = ml_df.select_dtypes(include=[np.number]).columns
        categorical_columns = ml_df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            if ml_df[col].isna().any():
                median_val = ml_df[col].median()
                ml_df[col].fillna(median_val, inplace=True)
                
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if ml_df[col].isna().any():
                mode_val = ml_df[col].mode()[0] if len(ml_df[col].mode()) > 0 else 'unknown'
                ml_df[col].fillna(mode_val, inplace=True)
        
        # Encode categorical variables
        categorical_to_encode = [
            'primary_phase', 'disease_category', 'sponsor_class', 
            'sponsor_experience_category', 'enrollment_size_category',
            'start_season', 'allocation', 'primary_purpose'
        ]
        
        for col in categorical_to_encode:
            if col in ml_df.columns:
                try:
                    le = LabelEncoder()
                    ml_df[f'{col}_encoded'] = le.fit_transform(ml_df[col].astype(str))
                    self.feature_stats[f'{col}_encoder'] = le
                except Exception as e:
                    self.logger.warning(f"Error encoding {col}: {e}")
        
        # Create feature importance scores based on correlation with outcome
        numeric_features = ml_df.select_dtypes(include=[np.number]).columns
        feature_correlations = {}
        
        for feature in numeric_features:
            if feature != 'binary_outcome':
                try:
                    corr = ml_df[feature].corr(ml_df['binary_outcome'])
                    if not pd.isna(corr):
                        feature_correlations[feature] = abs(corr)
                except:
                    pass
        
        self.feature_stats['feature_correlations'] = feature_correlations
        
        # Log feature statistics
        total_features = len(ml_df.columns)
        numeric_features_count = len(ml_df.select_dtypes(include=[np.number]).columns)
        
        self.logger.info(f"ML features prepared: {total_features} total features")
        self.logger.info(f"Numeric features: {numeric_features_count}")
        
        if feature_correlations:
            top_features = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)[:5]
            self.logger.info(f"Top correlated features: {top_features}")
        
        return ml_df
    
    def create_train_test_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits with temporal and stratified splitting"""
        self.logger.info("Creating train/test splits...")
        
        # Sort by start date to ensure temporal consistency
        df_sorted = df.sort_values('start_date_parsed', na_position='last')
        
        # Strategy 1: Temporal split (70% train, 15% val, 15% test)
        n_total = len(df_sorted)
        train_end = int(0.7 * n_total)
        val_end = int(0.85 * n_total)
        
        train_df = df_sorted[:train_end].copy()
        val_df = df_sorted[train_end:val_end].copy()
        test_df = df_sorted[val_end:].copy()
        
        # Ensure we have at least 1 record in each split
        if len(val_df) == 0 and len(test_df) == 0:
            # Very small dataset - just split in half
            mid_point = n_total // 2
            train_df = df_sorted[:mid_point].copy()
            val_df = df_sorted[mid_point:].copy()
            test_df = pd.DataFrame(columns=df_sorted.columns)  # Empty test set
        elif len(test_df) == 0:
            # Split val in half
            mid_val = len(val_df) // 2
            test_df = val_df[mid_val:].copy()
            val_df = val_df[:mid_val].copy()
        
        # Calculate success rates
        train_success_rate = train_df['binary_outcome'].mean() * 100 if len(train_df) > 0 else 0
        val_success_rate = val_df['binary_outcome'].mean() * 100 if len(val_df) > 0 else 0
        test_success_rate = test_df['binary_outcome'].mean() * 100 if len(test_df) > 0 else 0
        
        self.logger.info(f"Train set: {len(train_df)} samples, {train_success_rate:.1f}% success")
        self.logger.info(f"Validation set: {len(val_df)} samples, {val_success_rate:.1f}% success")
        self.logger.info(f"Test set: {len(test_df)} samples, {test_success_rate:.1f}% success")
        
        return train_df, val_df, test_df
    
    def apply_quality_control(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Apply all quality control steps"""
        self.logger.info(f"Starting quality control on {len(df)} records...")
        
        # Step 1: Validate required fields
        clean_df = self.validate_required_fields(df)
        
        # Check if we have any data left
        if len(clean_df) == 0:
            self.logger.warning("No records passed required field validation!")
            # Return empty dataset with basic stats
            quality_stats = {
                'initial_records': len(df),
                'final_records': 0,
                'removed_records': len(df),
                'removal_rate': 100.0,
                'balance_stats': {'overall_success_rate': 0, 'phase_balance': {}, 'disease_balance': {}},
                'train_size': 0, 'val_size': 0, 'test_size': 0,
                'total_features': len(df.columns),
                'feature_stats': {}
            }
            return clean_df, quality_stats
        
        # Step 2: Validate SMILES quality
        clean_df = self.validate_smiles_quality(clean_df)
        
        # Step 3: Remove outliers
        clean_df = self.remove_outliers(clean_df)
        
        # Step 4: Check data balance
        balance_stats = self.check_data_balance(clean_df)
        
        # Step 5: Prepare ML features
        ml_ready_df = self.prepare_ml_features(clean_df)
        
        # Step 6: Create splits
        train_df, val_df, test_df = self.create_train_test_splits(ml_ready_df)
        
        # Compile quality statistics
        quality_stats = {
            'initial_records': len(df),
            'final_records': len(ml_ready_df),
            'removed_records': len(self.removed_records),
            'removal_rate': len(self.removed_records) / len(df) * 100 if len(df) > 0 else 0,
            'balance_stats': balance_stats,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'total_features': len(ml_ready_df.columns),
            'feature_stats': self.feature_stats
        }
        
        self.quality_stats = quality_stats
        
        # Save splits
        train_df.to_parquet('data/final/train_set.parquet')
        val_df.to_parquet('data/final/val_set.parquet')
        test_df.to_parquet('data/final/test_set.parquet')
        ml_ready_df.to_parquet('data/final/complete_dataset.parquet')
        
        self.logger.info("Quality control completed successfully!")
        self.logger.info(f"Final dataset: {len(ml_ready_df)} records with {len(ml_ready_df.columns)} features")
        
        return ml_ready_df, quality_stats

# Test function
def test_quality_controller():
    # Create sample data with some quality issues
    sample_data = pd.DataFrame([
        {
            'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',  # Valid aspirin
            'primary_drug': 'Aspirin', 'primary_condition': 'Pain',
            'primary_phase': 'PHASE2', 'lead_sponsor': 'Pharma Corp',
            'binary_outcome': 1, 'enrollment_count': 200,
            'mol_molecular_weight': 180.2, 'start_date_parsed': '2020-01-01',
            'disease_category': 'other'
        },
        {
            'smiles': 'invalid_smiles',  # Invalid SMILES
            'primary_drug': 'Bad Drug', 'primary_condition': 'Disease',
            'primary_phase': 'PHASE1', 'lead_sponsor': 'Bad Corp',
            'binary_outcome': 0, 'enrollment_count': 50,
            'mol_molecular_weight': 150.0, 'start_date_parsed': '2019-01-01',
            'disease_category': 'other'
        },
        {
            'smiles': None,  # Missing SMILES
            'primary_drug': 'Missing SMILES', 'primary_condition': 'Condition',
            'primary_phase': 'PHASE3', 'lead_sponsor': 'Sponsor',
            'binary_outcome': 1, 'enrollment_count': 1000,
            'mol_molecular_weight': 300.0, 'start_date_parsed': '2021-01-01',
            'disease_category': 'other'
        }
    ])
    
    controller = QualityController()
    clean_data, stats = controller.apply_quality_control(sample_data)
    
    print("Quality control test results:")
    print(f"Initial records: {stats['initial_records']}")
    print(f"Final records: {stats['final_records']}")
    print(f"Removal rate: {stats['removal_rate']:.1f}%")
    
    return clean_data, stats

if __name__ == "__main__":
    test_quality_controller()