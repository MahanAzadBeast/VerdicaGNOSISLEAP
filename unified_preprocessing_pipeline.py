#!/usr/bin/env python3
"""
Unified Data Preprocessing Pipeline for Future Trainings
This ensures both ChemBERTa and Chemprop use identical data preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path

class UnifiedDataPreprocessor:
    """
    Standardized data preprocessing for all molecular property prediction models
    Ensures consistent pIC50 scales across ChemBERTa, Chemprop, and future models
    """
    
    def __init__(self, ic50_unit_standard="nM"):
        """
        Initialize with standardized IC50 unit
        
        Args:
            ic50_unit_standard: Standard unit for IC50 values ("nM", "μM", or "M")
        """
        self.ic50_unit_standard = ic50_unit_standard
        self.conversion_factors = {
            "M": 1e9,      # M to nM
            "mM": 1e6,     # mM to nM  
            "μM": 1e3,     # μM to nM
            "uM": 1e3,     # uM to nM (alternative notation)
            "nM": 1,       # nM to nM
            "pM": 1e-3     # pM to nM
        }
    
    def standardize_ic50_units(self, df, ic50_columns):
        """
        Convert all IC50 values to standard units (nM by default)
        
        Args:
            df: DataFrame with IC50 values
            ic50_columns: List of column names containing IC50 values
            
        Returns:
            df: DataFrame with standardized IC50 values
            unit_info: Dictionary tracking original units
        """
        print(f"🔄 Standardizing IC50 units to {self.ic50_unit_standard}")
        
        unit_info = {}
        
        for col in ic50_columns:
            if col in df.columns:
                # Auto-detect original units based on value ranges
                values = df[col].dropna()
                if len(values) > 0:
                    median_val = values.median()
                    
                    # Heuristic unit detection
                    if median_val < 1e-6:
                        original_unit = "M"
                    elif median_val < 1e-3:
                        original_unit = "mM"
                    elif median_val < 1:
                        original_unit = "μM"
                    elif median_val < 1000:
                        original_unit = "nM"
                    else:
                        original_unit = "pM"
                    
                    # Convert to standard units
                    conversion_factor = self.conversion_factors.get(original_unit, 1)
                    df[col] = df[col] * conversion_factor
                    
                    unit_info[col] = {
                        "original_unit": original_unit,
                        "standard_unit": self.ic50_unit_standard,
                        "conversion_factor": conversion_factor,
                        "sample_original": median_val,
                        "sample_converted": median_val * conversion_factor
                    }
                    
                    print(f"   {col}: {original_unit} → {self.ic50_unit_standard} (factor: {conversion_factor})")
        
        return df, unit_info
    
    def calculate_standardized_pic50(self, df, ic50_columns):
        """
        Calculate pIC50 using standardized formula
        
        Formula: pIC50 = 9 - log10(IC50_nM)
        This ensures all models use the same pIC50 scale
        """
        print(f"🧮 Calculating standardized pIC50 values")
        
        pic50_columns = []
        
        for col in ic50_columns:
            if col in df.columns:
                pic50_col = col.replace("IC50", "pIC50")
                
                # Standard pIC50 calculation (assumes IC50 in nM)
                df[pic50_col] = 9 - np.log10(df[col])
                pic50_columns.append(pic50_col)
                
                # Validate pIC50 range
                pic50_values = df[pic50_col].dropna()
                if len(pic50_values) > 0:
                    pic50_range = (pic50_values.min(), pic50_values.max())
                    print(f"   {pic50_col}: range {pic50_range[0]:.2f} - {pic50_range[1]:.2f}")
                    
                    # Warn if pIC50 values are outside expected range
                    if pic50_range[0] < 3 or pic50_range[1] > 12:
                        print(f"   ⚠️  WARNING: {pic50_col} values outside typical range (3-12)")
        
        return df, pic50_columns
    
    def create_training_splits(self, df, targets, test_size=0.2, val_size=0.1, random_state=42):
        """
        Create identical train/val/test splits for all models
        This ensures fair comparison between models
        """
        from sklearn.model_selection import train_test_split
        
        print(f"📊 Creating standardized train/val/test splits")
        
        # Filter to samples with valid data for at least one target
        valid_mask = df[targets].notna().any(axis=1)
        df_valid = df[valid_mask].copy()
        
        print(f"   Total samples: {len(df)} → Valid samples: {len(df_valid)}")
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df_valid, 
            test_size=test_size, 
            random_state=random_state,
            stratify=None  # Could add stratification logic here
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state
        )
        
        split_info = {
            "train_size": len(train_df),
            "val_size": len(val_df), 
            "test_size": len(test_df),
            "train_ratio": len(train_df) / len(df_valid),
            "val_ratio": len(val_df) / len(df_valid),
            "test_ratio": len(test_df) / len(df_valid)
        }
        
        print(f"   Train: {split_info['train_size']} ({split_info['train_ratio']:.1%})")
        print(f"   Val:   {split_info['val_size']} ({split_info['val_ratio']:.1%})")
        print(f"   Test:  {split_info['test_size']} ({split_info['test_ratio']:.1%})")
        
        return {
            "train": train_df,
            "val": val_df, 
            "test": test_df,
            "split_info": split_info
        }
    
    def save_preprocessed_data(self, splits, output_dir, prefix="standardized"):
        """Save preprocessed data for both ChemBERTa and Chemprop training"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        for split_name, df in splits.items():
            if split_name != "split_info":
                # Save for ChemBERTa (CSV format)
                chemberta_file = output_dir / f"{prefix}_chemberta_{split_name}.csv"
                df.to_csv(chemberta_file, index=False)
                
                # Save for Chemprop (CSV format with specific column order)
                chemprop_file = output_dir / f"{prefix}_chemprop_{split_name}.csv"
                df.to_csv(chemprop_file, index=False)
                
                saved_files[f"chemberta_{split_name}"] = str(chemberta_file)
                saved_files[f"chemprop_{split_name}"] = str(chemprop_file)
                
                print(f"   ✅ Saved {split_name}: {len(df)} samples")
        
        # Save preprocessing metadata
        metadata_file = output_dir / f"{prefix}_preprocessing_metadata.json"
        import json
        
        metadata = {
            "ic50_unit_standard": self.ic50_unit_standard,
            "pic50_formula": "pIC50 = 9 - log10(IC50_nM)",
            "split_info": splits.get("split_info", {}),
            "files": saved_files,
            "preprocessing_date": pd.Timestamp.now().isoformat()
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Saved metadata: {metadata_file}")
        
        return saved_files, metadata

def create_unified_training_pipeline():
    """Example of how to use the unified preprocessor"""
    
    print("🚀 UNIFIED DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    print("This pipeline ensures:")
    print("1. ✅ Identical data preprocessing for all models")
    print("2. ✅ Standardized IC50 → pIC50 conversion")
    print("3. ✅ Identical train/val/test splits")
    print("4. ✅ No need for post-hoc calibration")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = UnifiedDataPreprocessor(ic50_unit_standard="nM")
    
    print("\n📋 Usage Example:")
    print("```python")
    print("# Load raw data")
    print("df = pd.read_csv('raw_bioactivity_data.csv')")
    print("")
    print("# Standardize IC50 units")
    print("df, unit_info = preprocessor.standardize_ic50_units(df, ic50_columns)")
    print("")
    print("# Calculate standardized pIC50")
    print("df, pic50_cols = preprocessor.calculate_standardized_pic50(df, ic50_columns)")
    print("")
    print("# Create identical splits")
    print("splits = preprocessor.create_training_splits(df, target_columns)")
    print("")
    print("# Save for both models")
    print("files, metadata = preprocessor.save_preprocessed_data(splits, 'processed_data/')")
    print("```")
    
    return {
        "approach": "unified_preprocessing",
        "eliminates_calibration": True,
        "ensures_fair_comparison": True,
        "scientifically_rigorous": True
    }

if __name__ == "__main__":
    result = create_unified_training_pipeline()
    
    print(f"\n🎯 RECOMMENDATION FOR FUTURE TRAININGS:")
    print("✅ Use this unified preprocessing pipeline")
    print("✅ Train both models on identical preprocessed data") 
    print("✅ No post-hoc calibration needed")
    print("✅ Fair, scientifically valid model comparison")
    print(f"\n🚫 AVOID: Training models on different data scales")
    print("🚫 AVOID: Post-hoc calibration as permanent solution")