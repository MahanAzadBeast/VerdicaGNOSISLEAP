"""
Debug Step 1: Sanity-check the labels (pIC50 distribution)
"""

import modal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

app = modal.App("debug-labels")

image = modal.Image.debian_slim().pip_install([
    "pandas==2.1.0",
    "numpy==1.24.3",
    "matplotlib==3.7.0"
])

data_volume = modal.Volume.from_name("expanded-datasets")

@app.function(
    image=image,
    volumes={"/vol": data_volume}
)
def check_label_sanity():
    """Sanity check pIC50 labels - are they reasonable?"""
    
    print("🔍 STEP 1: SANITY-CHECKING pIC50 LABELS")
    print("=" * 60)
    
    # Load the real GDSC data
    gdsc_path = "/vol/gdsc_comprehensive_training_data.csv"
    df = pd.read_csv(gdsc_path)
    
    print(f"📊 Raw data: {len(df):,} records")
    
    # Check pIC50 distribution
    pic50_values = df['pIC50'].dropna()
    
    print(f"\n📈 pIC50 DISTRIBUTION ANALYSIS:")
    print(f"   Count: {len(pic50_values):,}")
    print(f"   Mean: {pic50_values.mean():.3f}")
    print(f"   Median: {pic50_values.median():.3f}")
    print(f"   Std: {pic50_values.std():.3f}")
    print(f"   Min: {pic50_values.min():.3f}")
    print(f"   Max: {pic50_values.max():.3f}")
    
    # Check percentiles
    percentiles = [5, 25, 50, 75, 95]
    print(f"\n📊 PERCENTILES:")
    for p in percentiles:
        val = np.percentile(pic50_values, p)
        print(f"   {p}th: {val:.3f}")
    
    # Flag potential issues
    print(f"\n🚨 SANITY CHECKS:")
    
    # Check 1: Reasonable range (should be ~5-7 for cell viability)
    if pic50_values.mean() < 3 or pic50_values.mean() > 10:
        print(f"   ⚠️ SUSPICIOUS MEAN: {pic50_values.mean():.3f} (expected ~5-7)")
    else:
        print(f"   ✅ Mean looks reasonable: {pic50_values.mean():.3f}")
    
    # Check 2: Mode near 0 or >10 is red flag
    # Create histogram to find mode
    hist, bin_edges = np.histogram(pic50_values, bins=50)
    mode_bin_idx = np.argmax(hist)
    mode_value = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2
    
    if mode_value < 1 or mode_value > 10:
        print(f"   ⚠️ SUSPICIOUS MODE: {mode_value:.3f} (expected ~5-7)")
    else:
        print(f"   ✅ Mode looks reasonable: {mode_value:.3f}")
    
    # Check 3: Standard deviation should be reasonable
    if pic50_values.std() < 0.5 or pic50_values.std() > 3:
        print(f"   ⚠️ SUSPICIOUS STD: {pic50_values.std():.3f} (expected ~1-2)")
    else:
        print(f"   ✅ Standard deviation reasonable: {pic50_values.std():.3f}")
    
    # Check 4: Look for signs of double transformation or sign flip
    negative_count = (pic50_values < 0).sum()
    very_high_count = (pic50_values > 15).sum()
    
    if negative_count > 0:
        print(f"   ⚠️ NEGATIVE VALUES: {negative_count:,} (should be 0)")
    else:
        print(f"   ✅ No negative values")
        
    if very_high_count > 0:
        print(f"   ⚠️ VERY HIGH VALUES (>15): {very_high_count:,} (suspicious)")
    else:
        print(f"   ✅ No extremely high values")
    
    # Convert back to IC50 to check if it makes sense
    print(f"\n🔄 REVERSE CHECK (pIC50 → IC50):")
    
    # Sample a few values
    sample_pic50 = pic50_values.head(10)
    sample_ic50_uM = 10**(-sample_pic50 + 6)  # Convert to μM
    
    print(f"   Sample pIC50 → IC50 (μM) conversion:")
    for i, (pic50, ic50) in enumerate(zip(sample_pic50, sample_ic50_uM)):
        print(f"     pIC50={pic50:.2f} → IC50={ic50:.2f} μM")
        if i >= 4:  # Show first 5
            break
    
    # Check if IC50 range is reasonable (0.001 - 1000 μM typical)
    median_ic50 = 10**(-pic50_values.median() + 6)
    print(f"   Median IC50: {median_ic50:.3f} μM")
    
    if median_ic50 < 0.001 or median_ic50 > 10000:
        print(f"   ⚠️ SUSPICIOUS IC50 RANGE")
    else:
        print(f"   ✅ IC50 range looks reasonable")
    
    return {
        'mean': float(pic50_values.mean()),
        'median': float(pic50_values.median()),
        'std': float(pic50_values.std()),
        'min': float(pic50_values.min()),
        'max': float(pic50_values.max()),
        'mode': float(mode_value),
        'negative_count': int(negative_count),
        'very_high_count': int(very_high_count),
        'median_ic50_uM': float(median_ic50)
    }

if __name__ == "__main__":
    with app.run():
        result = check_label_sanity.remote()
        
        print(f"\n📊 SUMMARY:")
        print(f"Mean pIC50: {result['mean']:.3f}")
        print(f"Median IC50: {result['median_ic50_uM']:.3f} μM")
        print(f"Issues found: {result['negative_count']} negative + {result['very_high_count']} very high")