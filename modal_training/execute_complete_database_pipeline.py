"""
Complete Database Extraction and Standardization Pipeline
Executes extraction for all databases and creates standardized integrated dataset
"""

import modal
import json
import time
from datetime import datetime
from pathlib import Path

def execute_complete_database_pipeline():
    """
    Execute complete database extraction, preparation, and standardization
    """
    
    print("🚀 LAUNCHING COMPLETE DATABASE EXTRACTION & STANDARDIZATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S AEST')}")
    print("🎯 Extracting: PubChem + BindingDB + DTC → Standardizing → Integrating")
    print()
    
    pipeline_start = datetime.now()
    results = {}
    
    try:
        # Step 1: Extract PubChem BioAssay data (fixed version)
        print("📊 STEP 1: Extracting PubChem BioAssay data...")
        print("-" * 60)
        
        from enhanced_pubchem_fixed import app as pubchem_app, extract_enhanced_pubchem_data
        
        print("🧪 Launching PubChem extraction...")
        start_time = datetime.now()
        
        with pubchem_app.run() as app_run:
            pubchem_result = extract_enhanced_pubchem_data.remote()
        
        duration = datetime.now() - start_time
        
        if pubchem_result['status'] == 'success':
            print(f"✅ PubChem extraction completed in {duration}")
            print(f"   📈 Records: {pubchem_result['total_records']:,}")
            print(f"   🎯 Targets: {pubchem_result['total_targets']}")
            print(f"   🧪 Compounds: {pubchem_result['total_compounds']:,}")
            results['pubchem'] = pubchem_result
        else:
            print(f"❌ PubChem extraction failed: {pubchem_result.get('error')}")
            results['pubchem'] = pubchem_result
        
        # Step 2: Extract BindingDB data
        print(f"\n📊 STEP 2: Extracting BindingDB data...")
        print("-" * 60)
        
        from bindingdb_extractor import app as bindingdb_app, extract_bindingdb_data
        
        print("🔗 Launching BindingDB extraction...")
        start_time = datetime.now()
        
        with bindingdb_app.run() as app_run:
            bindingdb_result = extract_bindingdb_data.remote()
        
        duration = datetime.now() - start_time
        
        if bindingdb_result['status'] == 'success':
            print(f"✅ BindingDB extraction completed in {duration}")
            print(f"   📈 Records: {bindingdb_result['total_records']:,}")
            print(f"   🎯 Targets: {bindingdb_result['total_targets']}")
            print(f"   🧪 Compounds: {bindingdb_result['total_compounds']:,}")
            results['bindingdb'] = bindingdb_result
        else:
            print(f"❌ BindingDB extraction failed: {bindingdb_result.get('error')}")
            results['bindingdb'] = bindingdb_result
        
        # Step 3: Extract DTC data
        print(f"\n📊 STEP 3: Extracting DTC data...")
        print("-" * 60)
        
        from dtc_extractor import app as dtc_app, extract_dtc_data
        
        print("🔬 Launching DTC extraction...")
        start_time = datetime.now()
        
        with dtc_app.run() as app_run:
            dtc_result = extract_dtc_data.remote()
        
        duration = datetime.now() - start_time
        
        if dtc_result['status'] == 'success':
            print(f"✅ DTC extraction completed in {duration}")
            print(f"   📈 Records: {dtc_result['total_records']:,}")
            print(f"   🎯 Targets: {dtc_result['total_targets']}")
            print(f"   🧪 Compounds: {dtc_result['total_compounds']:,}")
            results['dtc'] = dtc_result
        else:
            print(f"❌ DTC extraction failed: {dtc_result.get('error')}")
            results['dtc'] = dtc_result
        
        # Step 4: Integrate all databases
        print(f"\n📊 STEP 4: Integrating all databases...")
        print("-" * 60)
        
        from comprehensive_database_integration import app as integration_app, integrate_all_databases
        
        print("🔗 Launching comprehensive integration...")
        start_time = datetime.now()
        
        with integration_app.run() as app_run:
            integration_result = integrate_all_databases.remote()
        
        duration = datetime.now() - start_time
        
        if integration_result['status'] == 'success':
            print(f"✅ Database integration completed in {duration}")
            print(f"   📈 Original total: {integration_result['original_total_records']:,}")
            print(f"   📈 Integrated: {integration_result['integrated_total_records']:,}")
            print(f"   📊 Efficiency: {integration_result['deduplication_efficiency']:.1f}% overlap removed")
            print(f"   🎯 Final targets: {integration_result['total_targets']}")
            print(f"   🧪 Final compounds: {integration_result['total_compounds']:,}")
            print(f"   📋 Matrix shape: {integration_result['comprehensive_matrix_shape']}")
            results['integration'] = integration_result
        else:
            print(f"❌ Database integration failed: {integration_result.get('error')}")
            results['integration'] = integration_result
        
        # Calculate pipeline totals
        pipeline_duration = datetime.now() - pipeline_start
        
        # Count successful extractions
        successful_extractions = sum(1 for result in results.values() if result.get('status') == 'success')
        
        # Create pipeline summary
        pipeline_summary = {
            'pipeline_status': 'completed',
            'start_time': pipeline_start.isoformat(),
            'completion_time': datetime.now().isoformat(),
            'total_duration': str(pipeline_duration),
            'successful_extractions': successful_extractions,
            'total_extractions': len(results),
            'extraction_results': results,
            'databases_ready': successful_extractions >= 3,  # At least 3 of 4 databases
            'training_ready': results.get('integration', {}).get('status') == 'success'
        }
        
        summary_path = Path("/app/modal_training/complete_database_pipeline_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(pipeline_summary, f, indent=2, default=str)
        
        # Generate final report
        print(f"\n🎉 COMPLETE DATABASE PIPELINE FINISHED!")
        print("=" * 80)
        print(f"📁 Pipeline summary: {summary_path}")
        print(f"⏱️ Total duration: {pipeline_duration}")
        
        print(f"\n📊 Extraction Results:")
        print(f"   🧪 PubChem: {'✅' if results.get('pubchem', {}).get('status') == 'success' else '❌'}")
        print(f"   🔗 BindingDB: {'✅' if results.get('bindingdb', {}).get('status') == 'success' else '❌'}")
        print(f"   🔬 DTC: {'✅' if results.get('dtc', {}).get('status') == 'success' else '❌'}")
        print(f"   🔗 Integration: {'✅' if results.get('integration', {}).get('status') == 'success' else '❌'}")
        
        if results.get('integration', {}).get('status') == 'success':
            integration_data = results['integration']
            print(f"\n📈 Final Integrated Dataset:")
            print(f"   • Total records: {integration_data['integrated_total_records']:,}")
            print(f"   • Total targets: {integration_data['total_targets']}")
            print(f"   • Total compounds: {integration_data['total_compounds']:,}")
            print(f"   • Training matrix: {integration_data['comprehensive_matrix_shape']}")
            print(f"   • Databases: {', '.join(integration_data['databases_integrated'])}")
            
            # Show improvement over ChEMBL alone
            chembl_records = 24783  # Known ChEMBL size
            improvement = ((integration_data['integrated_total_records'] - chembl_records) / chembl_records) * 100
            print(f"   • Improvement over ChEMBL alone: +{improvement:.1f}%")
            
            print(f"\n🚀 DATASETS NOW STANDARDIZED AND READY FOR ENHANCED TRAINING!")
        else:
            print(f"\n⚠️ Integration failed, but individual databases may be available")
        
        return pipeline_summary
        
    except Exception as e:
        print(f"❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        error_summary = {
            'pipeline_status': 'failed',
            'start_time': pipeline_start.isoformat(),
            'error_time': datetime.now().isoformat(),
            'error': str(e),
            'partial_results': results
        }
        
        error_path = Path("/app/modal_training/complete_database_pipeline_error.json")
        with open(error_path, 'w') as f:
            json.dump(error_summary, f, indent=2, default=str)
        
        return error_summary

if __name__ == "__main__":
    try:
        result = execute_complete_database_pipeline()
        if result.get('pipeline_status') == 'completed':
            if result.get('training_ready'):
                print("\n✅ Complete database pipeline completed successfully")
                print("🎯 All datasets standardized and integrated")
                print("🚀 Ready for enhanced model training!")
            else:
                print("\n⚠️ Pipeline completed with some issues")
                print("📊 Check individual extraction results")
        else:
            print(f"\n❌ Database pipeline failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"\n❌ Pipeline launcher error: {e}")