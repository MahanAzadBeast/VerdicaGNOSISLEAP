"""
Verify dataset authenticity and Ki value quality
"""

import modal
import pandas as pd
import numpy as np
from pathlib import Path

app = modal.App("verify-data-quality")

# Setup volume
expanded_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=False)

@app.function(
    image=modal.Image.debian_slim().pip_install(["pandas", "numpy"]),
    volumes={"/vol/expanded": expanded_volume}
)
def verify_data_quality():
    """Verify if dataset is real and Ki values are reasonable"""
    
    print("🔍 DATA QUALITY VERIFICATION")
    print("=" * 80)
    
    # Load the comprehensive training dataset
    dataset_path = Path("/vol/expanded/gnosis_model1_binding_training.csv")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    df = pd.read_csv(dataset_path)
    print(f"📊 Dataset: {len(df):,} records")
    
    # 1. DATASET AUTHENTICITY CHECK
    print(f"\n🔬 DATASET AUTHENTICITY CHECK")
    print("-" * 60)
    
    # Check for realistic SMILES
    smiles_sample = df['SMILES'].dropna().sample(10).tolist()
    print(f"Sample SMILES (authentic molecules?):")
    for i, smiles in enumerate(smiles_sample[:5]):
        print(f"  {i+1}. {smiles}")
    
    # Check data sources
    data_sources = df['data_source'].value_counts()
    print(f"\nData sources: {dict(data_sources)}")
    
    # Check target names (real proteins?)
    target_sample = df['target_name'].value_counts().head(10)
    print(f"\nTop targets (real proteins?): {list(target_sample.index)}")
    
    # 2. Ki VALUE QUALITY ANALYSIS
    print(f"\n⚗️ Ki VALUE QUALITY ANALYSIS")
    print("-" * 60)
    
    # Focus on Ki data
    ki_data = df[df['assay_type'] == 'KI'].copy()
    print(f"Ki records: {len(ki_data):,}")
    
    if len(ki_data) > 0:
        # Ki value distribution
        ki_values_nm = ki_data['affinity_nm']
        
        print(f"\nKi value statistics (nM):")
        print(f"  Count: {len(ki_values_nm):,}")
        print(f"  Min: {ki_values_nm.min():.3f} nM")
        print(f"  Max: {ki_values_nm.max():.1f} nM") 
        print(f"  Median: {ki_values_nm.median():.1f} nM")
        print(f"  Mean: {ki_values_nm.mean():.1f} nM")
        print(f"  Std: {ki_values_nm.std():.1f} nM")
        
        # Check for unrealistic values
        very_high_ki = ki_data[ki_values_nm > 1e6]  # > 1 mM
        very_low_ki = ki_data[ki_values_nm < 0.1]   # < 0.1 nM (sub-picomolar)
        
        print(f"\n🚨 Potentially unrealistic Ki values:")
        print(f"  Very high (>1mM): {len(very_high_ki)} ({len(very_high_ki)/len(ki_data)*100:.1f}%)")
        print(f"  Very low (<0.1nM): {len(very_low_ki)} ({len(very_low_ki)/len(ki_data)*100:.1f}%)")
        
        # Show examples of high Ki values
        if len(very_high_ki) > 0:
            print(f"\nExamples of high Ki values:")
            high_examples = very_high_ki[['target_name', 'SMILES', 'affinity_nm']].head(5)
            for idx, row in high_examples.iterrows():
                print(f"  {row['target_name']}: {row['affinity_nm']:.1f} nM, SMILES: {row['SMILES'][:50]}...")
        
        # pKi values check
        if 'pKi' in ki_data.columns:
            pki_values = ki_data['pKi'].dropna()
            if len(pki_values) > 0:
                print(f"\npKi value statistics:")
                print(f"  Count: {len(pki_values):,}")
                print(f"  Min: {pki_values.min():.2f}")
                print(f"  Max: {pki_values.max():.2f}")
                print(f"  Median: {pki_values.median():.2f}")
                
                # Reasonable pKi range is ~4-12
                reasonable_pki = pki_values[(pki_values >= 4) & (pki_values <= 12)]
                print(f"  Reasonable range (4-12): {len(reasonable_pki)} ({len(reasonable_pki)/len(pki_values)*100:.1f}%)")
    
    # 3. COMPARE WITH IC50 VALUES
    print(f"\n📊 COMPARE Ki vs IC50 VALUES")
    print("-" * 60)
    
    ic50_data = df[df['assay_type'] == 'IC50'].copy()
    
    if len(ic50_data) > 0:
        ic50_values_nm = ic50_data['affinity_nm']
        
        print(f"IC50 value statistics (nM):")
        print(f"  Count: {len(ic50_values_nm):,}")
        print(f"  Min: {ic50_values_nm.min():.3f} nM")
        print(f"  Max: {ic50_values_nm.max():.1f} nM")
        print(f"  Median: {ic50_values_nm.median():.1f} nM")
        
        if len(ki_data) > 0:
            print(f"\nKi vs IC50 comparison:")
            print(f"  Ki median: {ki_values_nm.median():.1f} nM")
            print(f"  IC50 median: {ic50_values_nm.median():.1f} nM")
            print(f"  Ratio (Ki/IC50): {ki_values_nm.median()/ic50_values_nm.median():.2f}")
            
            # Typically Ki < IC50 for competitive inhibition
            if ki_values_nm.median() < ic50_values_nm.median():
                print("  ✅ Ki < IC50 (expected for competitive inhibition)")
            else:
                print("  ⚠️ Ki > IC50 (unusual, needs investigation)")
    
    # 4. TARGET-SPECIFIC Ki ANALYSIS
    print(f"\n🎯 TARGET-SPECIFIC Ki ANALYSIS")
    print("-" * 60)
    
    if len(ki_data) > 0:
        # Top targets with Ki data
        ki_by_target = ki_data.groupby('target_name')['affinity_nm'].agg(['count', 'median', 'std']).round(2)
        ki_by_target = ki_by_target.sort_values('count', ascending=False).head(10)
        
        print("Top targets with Ki data:")
        print(f"{'Target':<15} {'Count':<6} {'Median nM':<10} {'Std nM':<10}")
        print("-" * 50)
        for target, stats in ki_by_target.iterrows():
            print(f"{target:<15} {stats['count']:<6} {stats['median']:<10} {stats['std']:<10}")
    
    # 5. REALISTIC RANGE CHECK
    print(f"\n✅ DATA QUALITY ASSESSMENT")
    print("-" * 60)
    
    quality_flags = []
    
    # Check overall value distributions
    all_affinities = df['affinity_nm']
    reasonable_range = all_affinities[(all_affinities >= 0.1) & (all_affinities <= 1e6)]
    reasonable_pct = len(reasonable_range) / len(all_affinities) * 100
    
    print(f"Overall affinity assessment:")
    print(f"  Total records: {len(all_affinities):,}")
    print(f"  Reasonable range (0.1nM - 1μM): {len(reasonable_range):,} ({reasonable_pct:.1f}%)")
    
    if reasonable_pct > 90:
        quality_flags.append("✅ High-quality affinity data")
    elif reasonable_pct > 75:
        quality_flags.append("⚠️ Mostly good affinity data") 
    else:
        quality_flags.append("❌ Poor affinity data quality")
    
    # Check for synthetic patterns
    duplicate_smiles_pct = (1 - df['SMILES'].nunique() / len(df)) * 100
    if duplicate_smiles_pct < 50:
        quality_flags.append("✅ Diverse compound library")
    else:
        quality_flags.append("⚠️ High compound duplication")
    
    print(f"\nFinal assessment:")
    for flag in quality_flags:
        print(f"  {flag}")
    
    # Recommendation
    if len(ki_data) > 5000 and reasonable_pct > 80:
        print(f"\n🚀 RECOMMENDATION: DATASET IS SUITABLE FOR Ki PREDICTIONS")
        print(f"   - {len(ki_data):,} Ki records available")
        print(f"   - {reasonable_pct:.1f}% values in reasonable range")
        print(f"   - Ready to unlock Ki predictions")
    else:
        print(f"\n⚠️ RECOMMENDATION: Ki DATA NEEDS REVIEW")
        print(f"   - Only {len(ki_data):,} Ki records")
        print(f"   - {reasonable_pct:.1f}% reasonable values")
        
    return {
        'total_records': len(df),
        'ki_records': len(ki_data) if len(ki_data) > 0 else 0,
        'reasonable_pct': reasonable_pct,
        'recommendation': 'suitable' if len(ki_data) > 5000 and reasonable_pct > 80 else 'needs_review'
    }

@app.local_entrypoint()
def main():
    result = verify_data_quality.remote()
    return result