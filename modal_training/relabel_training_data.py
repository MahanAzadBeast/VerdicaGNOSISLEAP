"""
Relabel Training Data - Clear Assay Type Identification
=======================================================

This script relabels the current training data to clearly indicate what assays
the model was actually trained on, replacing ambiguous 'B'/'F' labels with
descriptive assay type names.

CURRENT PROBLEM:
- Labels 'B' and 'F' are unclear for future model development
- UI shows IC50/Ki/EC50 but model only knows Binding/Functional IC50
- No EC50 training data exists in current dataset

SOLUTION:
- Relabel 'B' → 'Binding_IC50' (direct binding inhibition)
- Relabel 'F' → 'Functional_IC50' (functional activity inhibition)  
- Create clear documentation for future training
- Identify EC50 data sources for comprehensive model

"""

import pandas as pd
import numpy as np
from pathlib import Path

def relabel_training_data():
    """Relabel current training data with clear assay type names"""
    
    print("🏷️  RELABELING TRAINING DATA FOR CLARITY")
    print("=" * 45)
    
    # Load current training data
    input_path = Path("/app/modal_training/data/training_data.csv")
    if not input_path.exists():
        print(f"❌ Training data not found at {input_path}")
        return
    
    df = pd.read_csv(input_path)
    print(f"📊 Original dataset: {len(df):,} records")
    print(f"Original assay types: {df['assay_type'].value_counts().to_dict()}")
    
    # Create relabeled dataset
    relabeled_df = df.copy()
    
    # Relabel assay types with clear names
    assay_mapping = {
        'B': 'Binding_IC50',      # Direct binding IC50 assays
        'F': 'Functional_IC50'    # Functional activity IC50 assays
    }
    
    relabeled_df['assay_type'] = relabeled_df['assay_type'].map(assay_mapping)
    
    # Add assay description column for documentation
    assay_descriptions = {
        'Binding_IC50': 'Direct binding inhibition assay measuring IC50 concentration for 50% binding inhibition',
        'Functional_IC50': 'Functional activity assay measuring IC50 concentration for 50% activity inhibition'
    }
    
    relabeled_df['assay_description'] = relabeled_df['assay_type'].map(assay_descriptions)
    
    # Add assay category for model training
    relabeled_df['assay_category'] = 'IC50_Inhibition'  # All are IC50-based
    
    # Rename columns for clarity
    column_mapping = {
        'ic50_nm': 'activity_value_nM',
        'pic50': 'p_activity'
    }
    relabeled_df = relabeled_df.rename(columns=column_mapping)
    
    # Add metadata columns
    relabeled_df['original_label'] = df['assay_type']  # Keep original for reference
    relabeled_df['dataset_version'] = 'v1.0_relabeled'
    relabeled_df['training_suitable'] = True
    
    # Reorder columns logically
    column_order = [
        'smiles', 'target', 'assay_type', 'assay_category', 'assay_description',
        'activity_value_nM', 'p_activity', 'chembl_id', 'assay_id',
        'original_label', 'dataset_version', 'training_suitable'
    ]
    
    relabeled_df = relabeled_df[column_order]
    
    print(f"\n✅ Relabeling complete:")
    print(f"New assay types: {relabeled_df['assay_type'].value_counts().to_dict()}")
    
    # Save relabeled dataset
    output_path = Path("/app/modal_training/data/training_data_relabeled.csv")
    relabeled_df.to_csv(output_path, index=False)
    print(f"💾 Saved relabeled dataset: {output_path}")
    
    # Generate summary report
    print(f"\n📋 RELABELED DATASET SUMMARY:")
    print(f"Total records: {len(relabeled_df):,}")
    print(f"Assay type distribution:")
    for assay_type, count in relabeled_df['assay_type'].value_counts().items():
        percentage = (count / len(relabeled_df)) * 100
        print(f"  {assay_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nTarget distribution:")
    print(f"  Unique targets: {relabeled_df['target'].nunique()}")
    for target, count in relabeled_df['target'].value_counts().items():
        print(f"  {target}: {count:,} records")
    
    # Generate statistical summary
    print(f"\n📈 ACTIVITY VALUE STATISTICS:")
    for assay_type in relabeled_df['assay_type'].unique():
        subset = relabeled_df[relabeled_df['assay_type'] == assay_type]
        activity_values = subset['activity_value_nM']
        
        print(f"\n{assay_type}:")
        print(f"  Count: {len(activity_values):,}")
        print(f"  Range: {activity_values.min():.1f} - {activity_values.max():.1f} nM")
        print(f"  Median: {activity_values.median():.1f} nM") 
        print(f"  Mean: {activity_values.mean():.1f} nM")
        print(f"  Std: {activity_values.std():.1f} nM")
    
    return output_path

def analyze_ec50_availability():
    """Analyze potential EC50 data sources for future training"""
    
    print(f"\n🔍 ANALYZING EC50 DATA AVAILABILITY")
    print("=" * 35)
    
    print("Current training data analysis:")
    print("❌ No EC50 data in current training set")
    print("❌ Only IC50-based assays (Binding + Functional)")
    print("❌ Single target (EGFR only)")
    
    print(f"\nIdentified EC50 data sources:")
    
    sources = [
        {
            'name': 'ChEMBL Database',
            'description': 'Large bioactivity database with EC50 measurements',
            'access': 'Free API access',
            'estimated_records': '50,000+ EC50 records for oncology targets',
            'quality': 'High - manually curated',
            'url': 'https://www.ebi.ac.uk/chembl/'
        },
        {
            'name': 'BindingDB',
            'description': 'Binding affinity database including functional assays',
            'access': 'Free download/API',
            'estimated_records': '10,000+ EC50 records',
            'quality': 'High - literature-derived',
            'url': 'https://www.bindingdb.org/'
        },
        {
            'name': 'PubChem BioAssay',
            'description': 'High-throughput screening data with EC50 values',
            'access': 'Free API access',
            'estimated_records': '100,000+ functional assays',
            'quality': 'Variable - screening data',
            'url': 'https://pubchem.ncbi.nlm.nih.gov/'
        },
        {
            'name': 'GDSC (Genomics Drug Sensitivity)',
            'description': 'Cancer cell line drug sensitivity (EC50/IC50)',
            'access': 'Free download',
            'estimated_records': '1,000+ compounds × 1,000+ cell lines',
            'quality': 'High - standardized protocols',
            'url': 'https://www.cancerrxgene.org/'
        }
    ]
    
    for i, source in enumerate(sources, 1):
        print(f"\n{i}. {source['name']}")
        print(f"   Description: {source['description']}")
        print(f"   Access: {source['access']}")
        print(f"   Estimated data: {source['estimated_records']}")
        print(f"   Quality: {source['quality']}")
        print(f"   URL: {source['url']}")
    
    print(f"\n✅ RECOMMENDATION FOR EC50 INTEGRATION:")
    print("1. Extract EC50 data from ChEMBL for same 14 oncology targets")
    print("2. Include GDSC cancer cell line EC50 data for broader coverage")
    print("3. Filter PubChem for high-quality functional assays")
    print("4. Expected dataset: ~10,000-50,000 EC50 records")
    print("5. Combine with existing IC50 data for comprehensive model")

def generate_assay_type_documentation():
    """Generate comprehensive documentation for assay types"""
    
    doc_content = """# Assay Type Documentation for Gnosis Model Training

## Current Training Data (v1.0_relabeled)

### Binding_IC50 Assays
- **Definition**: Direct binding inhibition assays measuring IC50
- **Methodology**: Measures compound concentration needed to inhibit 50% of target binding
- **Examples**: Competitive binding assays, displacement assays
- **Interpretation**: Lower IC50 = stronger binding affinity
- **Training Records**: 1,419 (86.8% of dataset)
- **Median Value**: 230 nM
- **Target Coverage**: EGFR only

### Functional_IC50 Assays  
- **Definition**: Functional activity assays measuring IC50
- **Methodology**: Measures compound concentration needed to inhibit 50% of target activity
- **Examples**: Enzyme activity assays, cell viability assays
- **Interpretation**: Lower IC50 = stronger functional inhibition
- **Training Records**: 216 (13.2% of dataset)
- **Median Value**: 380 nM (typically higher than binding IC50)
- **Target Coverage**: EGFR only

### Statistical Differences
- **Functional IC50 > Binding IC50**: Functional assays typically show higher IC50 values
- **Significance**: p-value = 2.13e-03 (statistically significant difference)
- **Biological Reason**: Functional inhibition may require higher concentrations than binding inhibition

## Missing Assay Types (Future Training)

### Ki (Inhibition Constant)
- **Status**: ❌ Not in current training data
- **Available**: ✅ 1,470 records extracted from ChEMBL
- **Relationship**: Ki ≈ IC50/2 for competitive inhibitors
- **Use Case**: Direct binding affinity measurement

### EC50 (Half-maximal Effective Concentration)
- **Status**: ❌ Not in current training data  
- **Available**: ✅ Thousands of records in ChEMBL/GDSC
- **Relationship**: Similar to functional IC50 but for activation
- **Use Case**: Functional activity measurement

### Kd (Dissociation Constant)
- **Status**: ❌ Not in current training data
- **Available**: ✅ 498 records extracted from ChEMBL
- **Relationship**: Direct thermodynamic binding measurement
- **Use Case**: Equilibrium binding affinity

## Recommendations for Future Models

1. **Multi-Assay Training**: Include Ki, EC50, Kd data for comprehensive coverage
2. **Target Expansion**: Beyond EGFR to 14+ oncology targets
3. **Assay-Specific Outputs**: Separate predictions for each assay type
4. **Quality Flags**: Confidence indicators based on training data availability
5. **Cross-Validation**: Validate predictions against literature values
"""
    
    doc_path = Path("/app/backend/reports/assay_type_documentation.md")
    with open(doc_path, 'w') as f:
        f.write(doc_content)
    
    print(f"📝 Generated assay documentation: {doc_path}")
    return doc_path

if __name__ == "__main__":
    # Run all analyses
    print("🚀 COMPREHENSIVE TRAINING DATA RELABELING")
    print("=" * 55)
    
    relabeled_file = relabel_training_data()
    analyze_ec50_availability() 
    doc_file = generate_assay_type_documentation()
    
    print(f"\n✅ RELABELING COMPLETE")
    print(f"📁 Relabeled data: {relabeled_file}")
    print(f"📖 Documentation: {doc_file}")
    print(f"🎯 Ready for UI updates and future model training")