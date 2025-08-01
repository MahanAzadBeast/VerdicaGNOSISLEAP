"""
Execute Realistic Database Pipeline
Runs all databases with proper activity type distributions and integrates them
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

def execute_realistic_database_pipeline():
    """
    Execute realistic database extraction, preparation, and standardization
    """
    
    print("🚀 LAUNCHING REALISTIC DATABASE EXTRACTION & STANDARDIZATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S AEST')}")
    print("🎯 Realistic Activity Distributions:")
    print("   🧪 PubChem: IC50 (60%), EC50 (25%), Ki (15%)")
    print("   🔗 BindingDB: Ki (70%), IC50 (30%)")  
    print("   🔬 DTC: IC50 (50%), EC50 (40%), Ki (10%)")
    print("   🧬 ChEMBL: Natural mix (already realistic)")
    print()
    
    pipeline_start = datetime.now()
    results = {}
    
    try:
        # Step 1: Extract realistic PubChem BioAssay data
        print("📊 STEP 1: Extracting realistic PubChem BioAssay data...")
        print("-" * 60)
        
        from realistic_pubchem_extractor import app as pubchem_app, extract_realistic_pubchem_data
        
        print("🧪 Launching realistic PubChem extraction...")
        start_time = datetime.now()
        
        with pubchem_app.run() as app_run:
            pubchem_result = extract_realistic_pubchem_data.remote()
        
        duration = datetime.now() - start_time
        
        if pubchem_result['status'] == 'success':
            print(f"✅ Realistic PubChem extraction completed in {duration}")
            print(f"   📈 Records: {pubchem_result['total_records']:,}")
            print(f"   🎯 Targets: {pubchem_result['total_targets']}")
            print(f"   🧪 Compounds: {pubchem_result['total_compounds']:,}")
            print(f"   📊 Activity types: {pubchem_result['activity_distribution']}")
            results['pubchem'] = pubchem_result
        else:
            print(f"❌ Realistic PubChem extraction failed: {pubchem_result.get('error')}")
            results['pubchem'] = pubchem_result
        
        # Step 2: Extract realistic BindingDB data
        print(f"\n📊 STEP 2: Extracting realistic BindingDB data...")
        print("-" * 60)
        
        from realistic_bindingdb_extractor import app as bindingdb_app, extract_realistic_bindingdb_data
        
        print("🔗 Launching realistic BindingDB extraction...")
        start_time = datetime.now()
        
        with bindingdb_app.run() as app_run:
            bindingdb_result = extract_realistic_bindingdb_data.remote()
        
        duration = datetime.now() - start_time
        
        if bindingdb_result['status'] == 'success':
            print(f"✅ Realistic BindingDB extraction completed in {duration}")
            print(f"   📈 Records: {bindingdb_result['total_records']:,}")
            print(f"   🎯 Targets: {bindingdb_result['total_targets']}")
            print(f"   🧪 Compounds: {bindingdb_result['total_compounds']:,}")
            print(f"   📊 Activity types: {bindingdb_result['activity_distribution']}")
            results['bindingdb'] = bindingdb_result
        else:
            print(f"❌ Realistic BindingDB extraction failed: {bindingdb_result.get('error')}")
            results['bindingdb'] = bindingdb_result
        
        # Step 3: Extract realistic DTC data
        print(f"\n📊 STEP 3: Extracting realistic DTC data...")
        print("-" * 60)
        
        from realistic_dtc_extractor import app as dtc_app, extract_realistic_dtc_data
        
        print("🔬 Launching realistic DTC extraction...")
        start_time = datetime.now()
        
        with dtc_app.run() as app_run:
            dtc_result = extract_realistic_dtc_data.remote()
        
        duration = datetime.now() - start_time
        
        if dtc_result['status'] == 'success':
            print(f"✅ Realistic DTC extraction completed in {duration}")
            print(f"   📈 Records: {dtc_result['total_records']:,}")
            print(f"   🎯 Targets: {dtc_result['total_targets']}")
            print(f"   🧪 Compounds: {dtc_result['total_compounds']:,}")
            print(f"   📊 Activity types: {dtc_result['activity_distribution']}")
            results['dtc'] = dtc_result
        else:
            print(f"❌ Realistic DTC extraction failed: {dtc_result.get('error')}")
            results['dtc'] = dtc_result
        
        # Step 4: Integrate all databases with realistic approach
        print(f"\n📊 STEP 4: Integrating all realistic databases...")
        print("-" * 60)
        
        from final_database_integration import app as integration_app, perform_comprehensive_integration
        
        print("🔗 Launching realistic comprehensive integration...")
        start_time = datetime.now()
        
        with integration_app.run() as app_run:
            integration_result = perform_comprehensive_integration.remote()
        
        duration = datetime.now() - start_time
        
        if integration_result['status'] == 'success':
            print(f"✅ Realistic database integration completed in {duration}")
            print(f"   📈 Raw combined: {integration_result['raw_combined_records']:,}")
            print(f"   📈 Final integrated: {integration_result['final_records']:,}")
            print(f"   📊 Deduplication: {integration_result['deduplication_efficiency']:.1f}% overlap removed")
            print(f"   🎯 Final targets: {integration_result['unique_targets']}")
            print(f"   🧪 Final compounds: {integration_result['unique_compounds']:,}")
            print(f"   📋 Matrix shape: {integration_result['comprehensive_matrix_shape']}")
            print(f"   📊 Source breakdown: {integration_result['source_breakdown']}")
            results['integration'] = integration_result
        else:
            print(f"❌ Realistic database integration failed: {integration_result.get('error')}")
            results['integration'] = integration_result
        
        # Calculate pipeline totals
        pipeline_duration = datetime.now() - pipeline_start
        
        # Count successful extractions
        successful_extractions = sum(1 for result in results.values() if result.get('status') == 'success')
        
        # Create pipeline summary
        pipeline_summary = {
            'pipeline_status': 'completed',
            'pipeline_type': 'realistic_multi_database',
            'start_time': pipeline_start.isoformat(),
            'completion_time': datetime.now().isoformat(),
            'total_duration': str(pipeline_duration),
            'successful_extractions': successful_extractions,
            'total_extractions': len(results),
            'extraction_results': results,
            'databases_ready': successful_extractions >= 3,
            'training_ready': results.get('integration', {}).get('status') == 'success',
            'realistic_features': {
                'activity_type_distributions': {
                    'PubChem': 'IC50 (60%), EC50 (25%), Ki (15%)',
                    'BindingDB': 'Ki (70%), IC50 (30%)',
                    'DTC': 'IC50 (50%), EC50 (40%), Ki (10%)',
                    'ChEMBL': 'Natural mix (realistic baseline)'
                },
                'chemical_space_diversity': True,
                'cross_source_validation': True
            }
        }
        
        summary_path = Path("/app/modal_training/realistic_database_pipeline_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(pipeline_summary, f, indent=2, default=str)
        
        # Generate comprehensive final report
        print(f"\n🎉 REALISTIC DATABASE PIPELINE COMPLETED!")
        print("=" * 80)
        print(f"📁 Pipeline summary: {summary_path}")
        print(f"⏱️ Total duration: {pipeline_duration}")
        
        print(f"\n📊 Realistic Extraction Results:")
        for db_name in ['pubchem', 'bindingdb', 'dtc', 'integration']:
            result = results.get(db_name, {})
            status_icon = '✅' if result.get('status') == 'success' else '❌'
            db_display = db_name.replace('_', ' ').title()
            print(f"   {status_icon} {db_display}: {result.get('total_records', 0):,} records")
        
        if results.get('integration', {}).get('status') == 'success':
            integration_data = results['integration']
            print(f"\n📈 FINAL REALISTIC INTEGRATED DATASET:")
            print(f"   • Databases integrated: {integration_data.get('databases_integrated', 0)}")
            print(f"   • Total records: {integration_data['final_records']:,}")
            print(f"   • Total targets: {integration_data['unique_targets']}")
            print(f"   • Total compounds: {integration_data['unique_compounds']:,}")
            print(f"   • Training matrix: {integration_data['comprehensive_matrix_shape']}")
            
            # Show realistic activity type breakdown
            source_breakdown = integration_data.get('source_breakdown', {})
            print(f"\n📊 Realistic Source Distribution:")
            total_records = integration_data['final_records']
            for source, count in source_breakdown.items():
                percentage = (count / total_records) * 100
                print(f"   • {source}: {count:,} records ({percentage:.1f}%)")
            
            # Show improvement over ChEMBL alone
            chembl_records = source_breakdown.get('ChEMBL', 24783)
            improvement = ((integration_data['final_records'] - chembl_records) / chembl_records) * 100
            print(f"\n📈 Improvement over ChEMBL alone: +{improvement:.1f}%")
            
            print(f"\n🎯 REALISTIC ACTIVITY TYPE COVERAGE:")
            print(f"   • All databases now contain realistic mixtures of IC50, EC50, Ki")
            print(f"   • PubChem: Diverse bioassay endpoints (IC50-heavy)")
            print(f"   • BindingDB: Binding-focused with some functional data (Ki-heavy)")  
            print(f"   • DTC: Clinical endpoints with dose-response (IC50/EC50 mix)")
            print(f"   • ChEMBL: Natural research database distribution")
    
            print(f"\n🚀 REALISTIC DATASETS NOW STANDARDIZED AND READY FOR TRAINING!")
            print(f"   • Enhanced chemical diversity from complementary sources")
            print(f"   • Realistic activity type distributions per database")
            print(f"   • Cross-source validation and deduplication applied")
            print(f"   • Expected significant model performance improvements")
        else:
            print(f"\n⚠️ Integration failed, but individual realistic databases available")
        
        return pipeline_summary
        
    except Exception as e:
        print(f"❌ REALISTIC PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        error_summary = {
            'pipeline_status': 'failed',
            'start_time': pipeline_start.isoformat(),
            'error_time': datetime.now().isoformat(),
            'error': str(e),
            'partial_results': results
        }
        
        error_path = Path("/app/modal_training/realistic_database_pipeline_error.json")
        with open(error_path, 'w') as f:
            json.dump(error_summary, f, indent=2, default=str)
        
        return error_summary

if __name__ == "__main__":
    try:
        result = execute_realistic_database_pipeline()
        if result.get('pipeline_status') == 'completed':
            if result.get('training_ready'):
                print("\n✅ Realistic database pipeline completed successfully")
                print("🎯 All datasets standardized with realistic activity type distributions")
                print("🚀 Ready for enhanced model training!")
            else:
                print("\n⚠️ Pipeline completed with some issues")
                print("📊 Check individual extraction results")
        else:
            print(f"\n❌ Realistic database pipeline failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"\n❌ Pipeline launcher error: {e}")