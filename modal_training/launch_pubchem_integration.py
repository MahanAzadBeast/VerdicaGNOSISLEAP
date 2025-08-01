"""
Launch PubChem BioAssay integration pipeline
Orchestrates PubChem extraction and integration with existing ChEMBL data
"""

import modal
import json
import time
from datetime import datetime
from pathlib import Path

def launch_pubchem_integration():
    """
    Launch the complete PubChem BioAssay integration pipeline
    """
    
    print("🚀 LAUNCHING PUBCHEM BIOASSAY INTEGRATION PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S AEST')}")
    print()
    
    try:
        # Step 1: Extract PubChem BioAssay data
        print("📊 STEP 1: Extracting PubChem BioAssay data...")
        print("-" * 60)
        
        from enhanced_pubchem_extractor import app as pubchem_app, extract_pubchem_bioassay_data
        
        start_time = datetime.now()
        
        with pubchem_app.run() as app_run:
            pubchem_result = extract_pubchem_bioassay_data.remote()
        
        extraction_duration = datetime.now() - start_time
        
        if pubchem_result['status'] == 'success':
            print(f"✅ PubChem extraction completed in {extraction_duration}")
            print(f"   📈 Records extracted: {pubchem_result['total_records']:,}")
            print(f"   🎯 Targets covered: {pubchem_result['total_targets']}")
            print(f"   🧪 Unique compounds: {pubchem_result['total_compounds']:,}")
        else:
            raise Exception(f"PubChem extraction failed: {pubchem_result.get('error', 'Unknown error')}")
        
        # Step 2: Integrate with existing ChEMBL data
        print(f"\n📊 STEP 2: Integrating with existing ChEMBL data...")
        print("-" * 60)
        
        from integrate_pubchem_with_chembl import app as integration_app, integrate_pubchem_with_existing_data
        
        integration_start = datetime.now()
        
        with integration_app.run() as app_run:
            integration_result = integrate_pubchem_with_existing_data.remote()
        
        integration_duration = datetime.now() - integration_start
        
        if integration_result['status'] == 'success':
            print(f"✅ Integration completed in {integration_duration}")
            print(f"   📈 Original ChEMBL: {integration_result['original_chembl_records']:,} records")
            print(f"   📈 Original PubChem: {integration_result['original_pubchem_records']:,} records")
            print(f"   📈 Integrated total: {integration_result['integrated_total_records']:,} records")
            print(f"   📊 Dataset boost: {integration_result['boost_percentage']:+.1f}%")
            print(f"   🎯 Final targets: {integration_result['total_targets']}")
            print(f"   🧪 Final compounds: {integration_result['total_compounds']:,}")
            print(f"   📋 Matrix shape: {integration_result['matrix_shape']}")
        else:
            raise Exception(f"Integration failed: {integration_result.get('error', 'Unknown error')}")
        
        # Save pipeline summary
        total_duration = datetime.now() - start_time
        
        pipeline_summary = {
            'pipeline_status': 'completed',
            'start_time': start_time.isoformat(),
            'completion_time': datetime.now().isoformat(),
            'total_duration': str(total_duration),
            'steps': {
                'pubchem_extraction': {
                    'status': pubchem_result['status'],
                    'duration': str(extraction_duration),
                    'records': pubchem_result['total_records'],
                    'targets': pubchem_result['total_targets'],
                    'compounds': pubchem_result['total_compounds']
                },
                'integration': {
                    'status': integration_result['status'],
                    'duration': str(integration_duration),
                    'original_chembl': integration_result['original_chembl_records'],
                    'original_pubchem': integration_result['original_pubchem_records'],
                    'integrated_total': integration_result['integrated_total_records'],
                    'boost_percentage': integration_result['boost_percentage'],
                    'final_targets': integration_result['total_targets'],
                    'final_compounds': integration_result['total_compounds'],
                    'matrix_shape': integration_result['matrix_shape']
                }
            },
            'files': {
                'integrated_raw_data': integration_result['integrated_raw_path'],
                'integrated_matrix': integration_result['integrated_matrix_path'],
                'metadata': integration_result['metadata_path']
            }
        }
        
        summary_path = Path("/app/modal_training/pubchem_integration_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(pipeline_summary, f, indent=2, default=str)
        
        # Generate final report
        print(f"\n🎉 PUBCHEM INTEGRATION PIPELINE COMPLETED!")
        print("=" * 80)
        print(f"📁 Pipeline summary: {summary_path}")
        print(f"⏱️ Total duration: {total_duration}")
        
        print(f"\n📊 Final dataset statistics:")
        print(f"  • Dataset size boost: {integration_result['boost_percentage']:+.1f}%")
        print(f"  • Total records: {integration_result['integrated_total_records']:,}")
        print(f"  • Total targets: {integration_result['total_targets']}")
        print(f"  • Total compounds: {integration_result['total_compounds']:,}")
        print(f"  • Training matrix: {integration_result['matrix_shape']}")
        
        print(f"\n🚀 Ready for enhanced model training with expanded dataset!")
        
        return pipeline_summary
        
    except Exception as e:
        print(f"❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        error_summary = {
            'pipeline_status': 'failed',
            'start_time': start_time.isoformat() if 'start_time' in locals() else datetime.now().isoformat(),
            'error_time': datetime.now().isoformat(),
            'error': str(e)
        }
        
        error_path = Path("/app/modal_training/pubchem_integration_error.json")
        with open(error_path, 'w') as f:
            json.dump(error_summary, f, indent=2, default=str)
        
        return error_summary

if __name__ == "__main__":
    try:
        result = launch_pubchem_integration()
        if result.get('pipeline_status') == 'completed':
            print("\n✅ PubChem integration pipeline completed successfully")
        else:
            print(f"\n❌ PubChem integration pipeline failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"\n❌ Pipeline launcher error: {e}")