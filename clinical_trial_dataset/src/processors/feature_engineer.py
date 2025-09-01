"""
Feature Engineering for Clinical Trial Prediction
Creates additional features for machine learning models
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.label_encoders = {}
        self.tfidf_vectorizers = {}
        
    def extract_sponsor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features about sponsors"""
        result_df = df.copy()
        
        # Sponsor experience (number of trials)
        sponsor_counts = df['lead_sponsor'].value_counts()
        result_df['sponsor_experience'] = result_df['lead_sponsor'].map(sponsor_counts).fillna(1)
        
        # Sponsor experience categories
        result_df['sponsor_experience_category'] = pd.cut(
            result_df['sponsor_experience'],
            bins=[0, 1, 5, 20, 100, float('inf')],
            labels=['first_time', 'novice', 'experienced', 'veteran', 'major_pharma']
        )
        
        # Sponsor class encoding
        sponsor_class_map = {
            'INDUSTRY': 1,
            'OTHER_GOV': 2,
            'FED': 3,
            'ACADEMIC': 4,
            'OTHER': 5
        }
        result_df['sponsor_class_encoded'] = result_df['sponsor_class'].map(sponsor_class_map).fillna(0)
        
        # Sponsor success rate (historical) - only if binary_outcome exists
        if 'binary_outcome' in df.columns:
            sponsor_success_rates = df.groupby('lead_sponsor')['binary_outcome'].apply(
                lambda x: x[x != -1].mean() if len(x[x != -1]) > 0 else 0.5
            )
            result_df['sponsor_historical_success_rate'] = result_df['lead_sponsor'].map(sponsor_success_rates).fillna(0.5)
        else:
            result_df['sponsor_historical_success_rate'] = 0.5
        
        return result_df
    
    def extract_disease_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features about diseases/conditions"""
        result_df = df.copy()
        
        # Disease frequency
        condition_counts = df['primary_condition'].value_counts()
        result_df['condition_frequency'] = result_df['primary_condition'].map(condition_counts).fillna(1)
        
        # Disease categories based on common patterns
        def categorize_disease(condition):
            if not condition or pd.isna(condition):
                return 'unknown'
            
            condition = str(condition).lower()
            
            if any(cancer_term in condition for cancer_term in 
                  ['cancer', 'carcinoma', 'tumor', 'oncology', 'malignant', 'lymphoma', 'leukemia']):
                return 'oncology'
            elif any(cardio_term in condition for cardio_term in 
                    ['heart', 'cardiac', 'cardiovascular', 'hypertension', 'coronary']):
                return 'cardiovascular'
            elif any(neuro_term in condition for neuro_term in 
                    ['alzheimer', 'parkinson', 'dementia', 'depression', 'anxiety', 'schizophrenia']):
                return 'neurology'
            elif any(diabetes_term in condition for diabetes_term in 
                    ['diabetes', 'diabetic']):
                return 'endocrine'
            elif any(infectious_term in condition for infectious_term in 
                    ['infection', 'viral', 'bacterial', 'hepatitis', 'hiv']):
                return 'infectious_disease'
            else:
                return 'other'
        
        result_df['disease_category'] = result_df['primary_condition'].apply(categorize_disease)
        
        # Disease category success rates - only if binary_outcome exists
        if 'binary_outcome' in df.columns:
            category_success_rates = df.groupby(df['primary_condition'].apply(categorize_disease))['binary_outcome'].apply(
                lambda x: x[x != -1].mean() if len(x[x != -1]) > 0 else 0.5
            )
            result_df['disease_category_success_rate'] = result_df['disease_category'].map(category_success_rates).fillna(0.5)
        else:
            result_df['disease_category_success_rate'] = 0.5
        
        return result_df
    
    def extract_protocol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from trial protocols"""
        result_df = df.copy()
        
        # Enrollment features
        result_df['enrollment_log'] = np.log1p(result_df['enrollment_count'].fillna(0))
        
        # Enrollment categories
        result_df['enrollment_size_category'] = pd.cut(
            result_df['enrollment_count'],
            bins=[0, 50, 200, 1000, float('inf')],
            labels=['small', 'medium', 'large', 'very_large']
        )
        
        # Study design features
        result_df['is_randomized'] = result_df['allocation'].astype(str).str.contains('RANDOMIZED', na=False).astype(int)
        result_df['is_blinded'] = result_df['masking'].astype(str).str.contains('DOUBLE|SINGLE|TRIPLE', na=False).astype(int)
        
        # Primary purpose encoding
        purpose_map = {
            'TREATMENT': 1,
            'PREVENTION': 2,
            'DIAGNOSTIC': 3,
            'SUPPORTIVE_CARE': 4,
            'SCREENING': 5,
            'OTHER': 6
        }
        result_df['primary_purpose_encoded'] = result_df['primary_purpose'].map(purpose_map).fillna(0)
        
        return result_df
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features"""
        result_df = df.copy()
        
        # Convert dates (check if columns exist)
        if 'start_date' in result_df.columns:
            result_df['start_date_parsed'] = pd.to_datetime(result_df['start_date'], errors='coerce')
        else:
            result_df['start_date_parsed'] = pd.NaT
            
        if 'completion_date' in result_df.columns:
            result_df['completion_date_parsed'] = pd.to_datetime(result_df['completion_date'], errors='coerce')
        else:
            result_df['completion_date_parsed'] = pd.NaT
        
        # Extract year, month, season
        result_df['start_year'] = result_df['start_date_parsed'].dt.year
        result_df['start_month'] = result_df['start_date_parsed'].dt.month
        result_df['start_quarter'] = result_df['start_date_parsed'].dt.quarter
        
        # Season
        def get_season(month):
            if pd.isna(month):
                return 'unknown'
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'fall'
        
        result_df['start_season'] = result_df['start_month'].apply(get_season)
        
        # Time trend (trials started more recently might have different success rates)
        if result_df['start_year'].notna().any():
            min_year = result_df['start_year'].min()
            result_df['years_since_first_trial'] = result_df['start_year'] - min_year
        else:
            result_df['years_since_first_trial'] = 0
        
        return result_df
    
    def extract_text_features(self, df: pd.DataFrame, max_features: int = 100) -> pd.DataFrame:
        """Extract features from text fields using TF-IDF"""
        result_df = df.copy()
        
        # Process eligibility criteria
        if 'eligibility_criteria' in df.columns:
            criteria_text = df['eligibility_criteria'].fillna('').astype(str)
            
            if len(criteria_text) > 0 and not criteria_text.str.strip().eq('').all():
                try:
                    # Simple TF-IDF on eligibility criteria
                    tfidf = TfidfVectorizer(
                        max_features=min(max_features, 50),
                        stop_words='english',
                        min_df=2,
                        ngram_range=(1, 2)
                    )
                    
                    tfidf_matrix = tfidf.fit_transform(criteria_text)
                    
                    # Add top TF-IDF features
                    feature_names = [f'criteria_tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
                    tfidf_df = pd.DataFrame(
                        tfidf_matrix.toarray(),
                        columns=feature_names,
                        index=result_df.index
                    )
                    
                    result_df = pd.concat([result_df, tfidf_df], axis=1)
                    self.tfidf_vectorizers['eligibility_criteria'] = tfidf
                    
                except Exception as e:
                    self.logger.warning(f"Error extracting text features: {e}")
        
        # Extract simple text statistics
        result_df['eligibility_criteria_length'] = result_df['eligibility_criteria'].astype(str).str.len()
        result_df['eligibility_criteria_word_count'] = result_df['eligibility_criteria'].astype(str).str.split().str.len()
        
        return result_df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        result_df = df.copy()
        
        # Phase × Sponsor experience
        result_df['phase_sponsor_interaction'] = (
            result_df['primary_phase'].astype(str) + '_' + 
            result_df['sponsor_experience_category'].astype(str)
        )
        
        # Disease category × Phase
        result_df['disease_phase_interaction'] = (
            result_df['disease_category'].astype(str) + '_' +
            result_df['primary_phase'].astype(str)
        )
        
        # Enrollment size × Phase
        result_df['enrollment_phase_interaction'] = (
            result_df['enrollment_size_category'].astype(str) + '_' +
            result_df['primary_phase'].astype(str)
        )
        
        # Molecular weight × Phase (if molecular features available)
        if 'mol_molecular_weight' in result_df.columns:
            # Discretize molecular weight
            result_df['mol_weight_category'] = pd.cut(
                result_df['mol_molecular_weight'],
                bins=[0, 300, 500, 800, float('inf')],
                labels=['small', 'medium', 'large', 'very_large']
            )
            
            result_df['molweight_phase_interaction'] = (
                result_df['mol_weight_category'].astype(str) + '_' +
                result_df['primary_phase'].astype(str)
            )
        
        return result_df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        self.logger.info(f"Engineering features for {len(df)} trials...")
        
        # Start with input dataframe
        result_df = df.copy()
        
        # Apply each feature engineering step
        result_df = self.extract_sponsor_features(result_df)
        self.logger.info("✓ Sponsor features extracted")
        
        result_df = self.extract_disease_features(result_df)
        self.logger.info("✓ Disease features extracted")
        
        result_df = self.extract_protocol_features(result_df)
        self.logger.info("✓ Protocol features extracted")
        
        result_df = self.extract_temporal_features(result_df)
        self.logger.info("✓ Temporal features extracted")
        
        result_df = self.extract_text_features(result_df)
        self.logger.info("✓ Text features extracted")
        
        result_df = self.create_interaction_features(result_df)
        self.logger.info("✓ Interaction features created")
        
        # Log feature summary
        total_features = len(result_df.columns)
        numeric_features = len(result_df.select_dtypes(include=[np.number]).columns)
        categorical_features = len(result_df.select_dtypes(include=['object', 'category']).columns)
        
        self.logger.info(f"Feature engineering completed:")
        self.logger.info(f"  Total features: {total_features}")
        self.logger.info(f"  Numeric features: {numeric_features}")
        self.logger.info(f"  Categorical features: {categorical_features}")
        
        return result_df

# Test function
def test_feature_engineer():
    # Create sample data
    sample_data = pd.DataFrame([
        {
            'lead_sponsor': 'Pfizer', 'sponsor_class': 'INDUSTRY',
            'primary_condition': 'Breast Cancer', 'enrollment_count': 200,
            'allocation': 'RANDOMIZED', 'masking': 'DOUBLE BLIND',
            'primary_purpose': 'TREATMENT', 'primary_phase': 'PHASE2',
            'start_date': '2020-03-01', 'binary_outcome': 1,
            'eligibility_criteria': 'Adults aged 18-65 with confirmed breast cancer'
        },
        {
            'lead_sponsor': 'Small Biotech', 'sponsor_class': 'INDUSTRY',
            'primary_condition': 'Diabetes Mellitus', 'enrollment_count': 50,
            'allocation': 'NON_RANDOMIZED', 'masking': 'NONE',
            'primary_purpose': 'TREATMENT', 'primary_phase': 'PHASE1',
            'start_date': '2019-06-01', 'binary_outcome': 0,
            'eligibility_criteria': 'Type 2 diabetes patients on metformin'
        }
    ])
    
    engineer = FeatureEngineer()
    engineered_data = engineer.engineer_all_features(sample_data)
    
    print("Sample feature engineering:")
    print(f"Original features: {len(sample_data.columns)}")
    print(f"Engineered features: {len(engineered_data.columns)}")
    print("New features include:", [col for col in engineered_data.columns if col not in sample_data.columns][:10])
    
    return engineered_data

if __name__ == "__main__":
    test_feature_engineer()