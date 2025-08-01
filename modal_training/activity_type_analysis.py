"""
Activity Type Analysis - Check what each database actually contains
"""

import modal
import json
from datetime import datetime

# Modal setup with required packages
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas", 
    "numpy"
])

app = modal.App("activity-type-analysis")
datasets_volume = modal.Volume.from_name("expanded-datasets", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/vol/datasets": datasets_volume},
    cpu=2.0,
    memory=8192,
    timeout=300
)
def analyze_database_activity_types():
    """
    Analyze what activity types each database actually contains
    """
    
    import pandas as pd
    from pathlib import Path
    
    print("🔍 DETAILED ACTIVITY TYPE ANALYSIS")
    print("=" * 80)
    
    try:
        datasets_dir = Path("/vol/datasets")
        
        # Load integrated dataset
        integrated_path = datasets_dir / "final_integrated_raw_data.csv"
        if not integrated_path.exists():
            return {"error": "Integrated dataset not found"}
        
        df = pd.read_csv(integrated_path)
        
        print("📊 ACTIVITY TYPE BREAKDOWN BY DATABASE:")
        print("-" * 60)
        
        # Cross-tabulation analysis
        cross_tab = pd.crosstab(df['data_source'], df['activity_type'])
        
        activity_analysis = {}
        
        for source in ['ChEMBL', 'PubChem', 'BindingDB', 'DTC']:
            if source in cross_tab.index:
                source_row = cross_tab.loc[source]
                total_records = source_row.sum()
                
                print(f"\n🏷️ {source.upper()}:")
                print(f"   Total records: {total_records:,}")
                
                source_analysis = {
                    'total_records': int(total_records),
                    'activity_types': {}
                }
                
                for activity_type in cross_tab.columns:
                    count = source_row.get(activity_type, 0)
                    percentage = (count / total_records * 100) if total_records > 0 else 0
                    
                    source_analysis['activity_types'][activity_type] = {
                        'count': int(count),
                        'percentage': float(percentage)
                    }
                    
                    if count > 0:
                        print(f"   ✅ {activity_type}: {count:,} records ({percentage:.1f}%)")
                    else:
                        print(f"   ❌ {activity_type}: 0 records (0.0%)")
                
                activity_analysis[source] = source_analysis
        
        # Overall activity type distribution
        print(f"\n📊 OVERALL ACTIVITY TYPE DISTRIBUTION:")
        print("-" * 60)
        
        activity_totals = df['activity_type'].value_counts()
        total_records = len(df)
        
        overall_analysis = {}
        
        for activity_type in activity_totals.index:
            count = activity_totals[activity_type]
            percentage = (count / total_records * 100)
            
            overall_analysis[activity_type] = {
                'count': int(count),
                'percentage': float(percentage)
            }
            
            print(f"   {activity_type}: {count:,} records ({percentage:.1f}%)")
        
        print(f"\nTotal integrated records: {total_records:,}")
        
        # Database design explanation
        print(f"\n🎯 DATABASE DESIGN RATIONALE:")
        print("-" * 60)
        print("Each database was designed with specific activity type focus:")
        print("   🧬 ChEMBL: IC50 (inhibitory concentration - cell/enzyme assays)")
        print("   🧪 PubChem: IC50 (bioassay data - similar to ChEMBL)")
        print("   🔗 BindingDB: Ki (binding affinity - direct protein binding)")
        print("   🔬 DTC: EC50 (effective concentration - clinical/drug data)")
        print()
        print("This differentiation provides complementary data types:")
        print("   • IC50: How much compound inhibits biological activity")
        print("   • Ki: How tightly compound binds to target protein")
        print("   • EC50: Effective concentration causing response")
        
        # Check if this is realistic
        print(f"\n❓ REALISTIC DATABASE BEHAVIOR:")
        print("-" * 60)
        print("In reality, databases often contain multiple activity types:")
        print("   • PubChem BioAssay: Should have IC50, EC50, Ki, and others")
        print("   • BindingDB: Should have both Ki and IC50 data")
        print("   • DTC: Should have IC50, EC50, and other clinical data")
        print("   • ChEMBL: Contains all activity types")
        print()
        print("🔧 Current implementation artificially separates by activity type")
        print("   to ensure data complementarity and avoid over-representation")
        
        result = {
            'database_analysis': activity_analysis,
            'overall_analysis': overall_analysis,
            'total_records': int(total_records),
            'cross_tabulation': cross_tab.to_dict(),
            'realistic_assessment': {
                'current_design': 'artificially_separated_by_activity_type',
                'realistic_behavior': 'databases_should_have_multiple_activity_types',
                'recommendation': 'consider_mixed_activity_types_per_database'
            }
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    print("🔍 Activity Type Analysis")