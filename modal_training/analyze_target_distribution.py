"""
Analyze target distribution in the comprehensive Gnosis Model 1 training dataset
"""

import modal
import pandas as pd
import numpy as np
from pathlib import Path

app = modal.App("analyze-target-distribution")

# Setup volume
expanded_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=False)

@app.function(
    image=modal.Image.debian_slim().pip_install(["pandas", "numpy"]),
    volumes={"/vol/expanded": expanded_volume}
)
def analyze_target_distribution():
    """Analyze target distribution in comprehensive training dataset"""
    
    print("🎯 ANALYZING TARGET DISTRIBUTION")
    print("=" * 80)
    
    # Load the comprehensive training dataset
    dataset_path = Path("/vol/expanded/gnosis_model1_binding_training.csv")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    df = pd.read_csv(dataset_path)
    print(f"📊 Total dataset: {len(df):,} records")
    print(f"📊 Dataset columns: {list(df.columns)}")
    
    # Analyze target distribution
    print(f"\n🎯 TARGET ANALYSIS")
    print("-" * 60)
    
    # Overall target distribution
    target_counts = df['target_name'].value_counts()
    print(f"Total unique targets: {len(target_counts)}")
    print(f"Records per target - Min: {target_counts.min()}, Max: {target_counts.max()}, Median: {target_counts.median():.1f}")
    
    # Define training suitability thresholds
    thresholds = {
        'minimal': 50,      # Absolute minimum for any training
        'basic': 100,       # Basic training possible
        'good': 500,        # Good training performance expected
        'excellent': 1000   # Excellent training performance expected
    }
    
    print(f"\n📈 TRAINING SUITABILITY BREAKDOWN:")
    for category, threshold in thresholds.items():
        suitable_targets = target_counts[target_counts >= threshold]
        percentage = (len(suitable_targets) / len(target_counts)) * 100
        print(f"  {category.upper()} (≥{threshold}): {len(suitable_targets)} targets ({percentage:.1f}%)")
    
    # Detailed breakdown
    print(f"\n📋 DETAILED TARGET DISTRIBUTION:")
    print(f"{'Target':<20} {'Count':<8} {'Suitability':<12} {'Assay Types'}")
    print("-" * 70)
    
    # Analyze each target
    target_analysis = []
    
    for target, count in target_counts.head(50).items():  # Show top 50
        # Determine suitability
        if count >= 1000:
            suitability = "EXCELLENT"
        elif count >= 500:
            suitability = "GOOD"
        elif count >= 100:
            suitability = "BASIC"
        elif count >= 50:
            suitability = "MINIMAL"
        else:
            suitability = "POOR"
        
        # Check assay type distribution for this target
        target_data = df[df['target_name'] == target]
        assay_types = target_data['assay_type'].value_counts()
        assay_summary = f"{len(assay_types)} types"
        
        print(f"{target:<20} {count:<8} {suitability:<12} {assay_summary}")
        
        target_analysis.append({
            'target': target,
            'count': count,
            'suitability': suitability,
            'assay_types': len(assay_types)
        })
    
    if len(target_counts) > 50:
        print(f"... and {len(target_counts) - 50} more targets")
    
    # Assay type analysis per target
    print(f"\n🧪 ASSAY TYPE DISTRIBUTION ANALYSIS:")
    print("-" * 60)
    
    # Overall assay distribution
    overall_assay_dist = df['assay_type'].value_counts()
    print(f"Overall assay distribution:")
    for assay_type, count in overall_assay_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {assay_type}: {count:,} ({percentage:.1f}%)")
    
    # Multi-assay targets analysis
    print(f"\n🔬 MULTI-ASSAY TARGET ANALYSIS:")
    print("-" * 60)
    
    multi_assay_targets = []
    for target in target_counts.head(20).index:  # Top 20 targets
        target_data = df[df['target_name'] == target]
        assay_dist = target_data['assay_type'].value_counts()
        
        if len(assay_dist) >= 2:  # Multi-assay target
            multi_assay_targets.append({
                'target': target,
                'total_count': len(target_data),
                'assay_types': len(assay_dist),
                'assay_breakdown': dict(assay_dist)
            })
    
    print(f"Targets with multiple assay types: {len(multi_assay_targets)}")
    
    for target_info in multi_assay_targets[:15]:  # Show top 15
        print(f"\n  🎯 {target_info['target']} ({target_info['total_count']} total):")
        for assay, count in target_info['assay_breakdown'].items():
            percentage = (count / target_info['total_count']) * 100
            print(f"    {assay}: {count} ({percentage:.1f}%)")
    
    # Final training recommendations
    print(f"\n🚀 TRAINING RECOMMENDATIONS:")
    print("-" * 60)
    
    excellent_targets = target_counts[target_counts >= 1000]
    good_targets = target_counts[(target_counts >= 500) & (target_counts < 1000)]
    basic_targets = target_counts[(target_counts >= 100) & (target_counts < 500)]
    
    print(f"🌟 TIER 1 - EXCELLENT ({len(excellent_targets)} targets):")
    print(f"  Targets: {list(excellent_targets.index)}")
    print(f"  Total samples: {excellent_targets.sum():,}")
    
    print(f"\n⭐ TIER 2 - GOOD ({len(good_targets)} targets):")
    print(f"  Targets: {list(good_targets.index[:10])}{'...' if len(good_targets) > 10 else ''}")
    print(f"  Total samples: {good_targets.sum():,}")
    
    print(f"\n🔸 TIER 3 - BASIC ({len(basic_targets)} targets):")
    print(f"  Targets: {list(basic_targets.index[:10])}{'...' if len(basic_targets) > 10 else ''}")
    print(f"  Total samples: {basic_targets.sum():,}")
    
    # Calculate reliable prediction coverage
    reliable_targets = target_counts[target_counts >= 100]  # Basic threshold
    coverage_percentage = (reliable_targets.sum() / len(df)) * 100
    
    print(f"\n📊 RELIABLE PREDICTION COVERAGE:")
    print(f"  Targets suitable for training (≥100 samples): {len(reliable_targets)} out of {len(target_counts)}")
    print(f"  Coverage: {coverage_percentage:.1f}% of all training data")
    print(f"  Recommended model targets: {len(reliable_targets)} targets")
    
    return {
        'total_targets': len(target_counts),
        'excellent_targets': len(excellent_targets),
        'good_targets': len(good_targets),
        'basic_targets': len(basic_targets),
        'reliable_targets': len(reliable_targets),
        'coverage_percentage': coverage_percentage
    }

@app.local_entrypoint()
def main():
    result = analyze_target_distribution.remote()
    return result