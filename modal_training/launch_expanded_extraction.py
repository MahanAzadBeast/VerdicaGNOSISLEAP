"""
Launch Expanded Multi-Source Data Extraction
"""

import modal
from expanded_multisource_extractor import app

def launch_expanded_extraction():
    """Launch the expanded multi-source data extraction"""
    
    print("🚀 Launching Expanded Multi-Source Data Extraction")
    print("=" * 60)
    print("📋 Target Categories:")
    print("   • Oncoproteins: EGFR, HER2, VEGFR2, BRAF, MET, CDK4, CDK6, ALK, MDM2, PI3KCA")
    print("   • Tumor Suppressors: TP53, RB1, PTEN, APC, BRCA1, BRCA2, VHL") 
    print("   • Metastasis Suppressors: NDRG1, KAI1, KISS1, NM23H1, RKIP, CASP8")
    print("\n📊 Activity Types: IC50, EC50, Ki, Inhibition %, Activity %")
    print("🔗 Data Sources: ChEMBL, PubChem, BindingDB, DTC")
    print("\n🔍 Quality Control:")
    print("   • Experimental assays only (no docking/simulation)")
    print("   • Median aggregation for duplicate compound-target pairs")
    print("   • Discard if >100-fold variance between sources")
    print("   • Valid SMILES validation with RDKit")
    
    print("\n⏳ Starting extraction (estimated time: 2-4 hours)...")
    
    # Execute the extraction
    with app.run() as app_run:
        result = app_run.extract_expanded_multisource_dataset.remote()
    
    if result['status'] == 'success':
        print("\n✅ EXTRACTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Results Summary:")
        print(f"   • Total records: {result['total_records']:,}")
        print(f"   • Unique targets: {result['total_targets']}")
        print(f"   • Unique compounds: {result['total_compounds']:,}")
        print(f"   • Activity types: {len(result['activity_types'])}")
        
        print(f"\n📁 Generated Files:")
        print(f"   • Raw data: {result['raw_data_path']}")
        for activity_type, path in result['matrix_paths'].items():
            print(f"   • {activity_type} matrix: {path}")
        print(f"   • Metadata: {result['metadata_path']}")
        
        print(f"\n🎯 Ready for training: {result['ready_for_training']}")
        
    else:
        print(f"\n❌ EXTRACTION FAILED: {result.get('error', 'Unknown error')}")
    
    return result

if __name__ == "__main__":
    result = launch_expanded_extraction()