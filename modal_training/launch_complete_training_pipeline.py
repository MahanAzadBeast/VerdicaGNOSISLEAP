"""
Complete Model Training Pipeline with Enhanced Dataset
Executes PubChem integration followed by ChemBERTa and Chemprop training
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

def launch_complete_training_pipeline():
    """
    Launch the complete training pipeline:
    1. Integrate PubChem BioAssay data with existing ChEMBL
    2. Train ChemBERTa on enhanced dataset
    3. Train Chemprop on enhanced dataset
    """
    
    print("🚀 LAUNCHING COMPLETE MODEL TRAINING PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S AEST')}")
    print()
    
    pipeline_start = datetime.now()
    results = {}
    
    try:
        # Step 1: Launch PubChem integration (run in background)
        print("📊 STEP 1: Launching PubChem BioAssay integration...")
        print("-" * 60)
        
        pubchem_cmd = """
cd /app/modal_training && nohup python -c "
import modal
from enhanced_pubchem_extractor import app as pubchem_app, extract_pubchem_bioassay_data
from integrate_pubchem_with_chembl import app as integration_app, integrate_pubchem_with_existing_data
import json
from datetime import datetime

print('🧪 STARTING PUBCHEM INTEGRATION PIPELINE')
print('Expected boost: 25K → 75K+ records')

try:
    # Extract PubChem data
    print('Step 1: Extracting PubChem BioAssay data...')
    with pubchem_app.run() as app_run:
        pubchem_result = extract_pubchem_bioassay_data.remote()
    
    if pubchem_result['status'] == 'success':
        print(f'✅ PubChem extraction: {pubchem_result[\"total_records\"]} records')
        
        # Integrate with ChEMBL
        print('Step 2: Integrating with ChEMBL data...')
        with integration_app.run() as app_run:
            integration_result = integrate_pubchem_with_existing_data.remote()
        
        if integration_result['status'] == 'success':
            print(f'✅ Integration complete: {integration_result[\"integrated_total_records\"]} total records')
            print(f'Dataset boost: +{integration_result[\"boost_percentage\"]:.1f}%')
            
            final_result = {
                'status': 'success',
                'integration_complete': True,
                'final_records': integration_result['integrated_total_records'],
                'boost_percentage': integration_result['boost_percentage'],
                'ready_for_training': True
            }
        else:
            final_result = {'status': 'integration_failed', 'error': integration_result.get('error')}
    else:
        final_result = {'status': 'extraction_failed', 'error': pubchem_result.get('error')}
    
    with open('/app/modal_training/pubchem_integration_results.json', 'w') as f:
        json.dump(final_result, f, indent=2, default=str)
    
    print('✅ PUBCHEM INTEGRATION COMPLETED!')
    
except Exception as e:
    error_result = {
        'status': 'failed',
        'error': str(e),
        'timestamp': datetime.now().isoformat()
    }
    with open('/app/modal_training/pubchem_integration_results.json', 'w') as f:
        json.dump(error_result, f, indent=2)
    print(f'❌ PUBCHEM INTEGRATION FAILED: {e}')
    import traceback
    traceback.print_exc()
" > /app/modal_training/pubchem_integration_log.txt 2>&1 &
"""
        
        subprocess.run(pubchem_cmd, shell=True)
        print("✅ PubChem integration launched in background")
        print("📊 Expected completion: 30-60 minutes")
        
        # Give it some time to start
        time.sleep(10)
        
        # Step 2: Launch enhanced ChemBERTa training
        print(f"\n🧠 STEP 2: Launching enhanced ChemBERTa training...")
        print("-" * 60)
        
        chemberta_cmd = """
cd /app/modal_training && nohup python -c "
import modal
from train_expanded_chemberta import app, train_expanded_chemberta
import json
from datetime import datetime

print('🧠 STARTING ENHANCED CHEMBERTA TRAINING')
print('Dataset: Integrated ChEMBL + PubChem (expected 60K+ records)')
print('Targets: 23 (oncoproteins + tumor suppressors + metastasis suppressors)')
print('Epochs: 30')
print('Architecture: Multi-task with category-wise analysis')

try:
    with app.run() as app_run:
        result = train_expanded_chemberta.remote(
            epochs=30,
            batch_size=16,
            learning_rate=2e-5,
            run_name='enhanced_integrated_chemberta'
        )
    
    with open('/app/modal_training/chemberta_enhanced_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print('✅ ENHANCED CHEMBERTA TRAINING COMPLETED!')
    print(f'Results saved to: /app/modal_training/chemberta_enhanced_results.json')
    
except Exception as e:
    error_result = {
        'status': 'failed',
        'error': str(e),
        'timestamp': datetime.now().isoformat()
    }
    with open('/app/modal_training/chemberta_enhanced_results.json', 'w') as f:
        json.dump(error_result, f, indent=2)
    print(f'❌ ENHANCED CHEMBERTA TRAINING FAILED: {e}')
    import traceback
    traceback.print_exc()
" > /app/modal_training/chemberta_enhanced_log.txt 2>&1 &
"""
        
        subprocess.run(chemberta_cmd, shell=True)
        print("✅ Enhanced ChemBERTa training launched")
        print("📊 Expected completion: 2-3 hours on A100 GPU")
        
        # Step 3: Launch enhanced Chemprop training  
        print(f"\n🕸️ STEP 3: Launching enhanced Chemprop training...")
        print("-" * 60)
        
        chemprop_cmd = """
cd /app/modal_training && nohup python -c "
import modal
from train_expanded_chemprop import app, train_expanded_chemprop
import json
from datetime import datetime

print('🕸️ STARTING ENHANCED CHEMPROP TRAINING')
print('Dataset: Integrated ChEMBL + PubChem (expected 60K+ records)')
print('Targets: 23 (oncoproteins + tumor suppressors + metastasis suppressors)')
print('Epochs: 40')
print('Architecture: Enhanced GNN (512 hidden, 5-layer depth)')

try:
    with app.run() as app_run:
        result = train_expanded_chemprop.remote(
            epochs=40,
            batch_size=64,
            learning_rate=1e-3,
            run_name='enhanced_integrated_chemprop'
        )
    
    with open('/app/modal_training/chemprop_enhanced_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print('✅ ENHANCED CHEMPROP TRAINING COMPLETED!')
    print(f'Results saved to: /app/modal_training/chemprop_enhanced_results.json')
    
except Exception as e:
    error_result = {
        'status': 'failed',
        'error': str(e),
        'timestamp': datetime.now().isoformat()
    }
    with open('/app/modal_training/chemprop_enhanced_results.json', 'w') as f:
        json.dump(error_result, f, indent=2)
    print(f'❌ ENHANCED CHEMPROP TRAINING FAILED: {e}')
    import traceback
    traceback.print_exc()
" > /app/modal_training/chemprop_enhanced_log.txt 2>&1 &
"""
        
        subprocess.run(chemprop_cmd, shell=True)
        print("✅ Enhanced Chemprop training launched")
        print("📊 Expected completion: 3-4 hours on A100 GPU")
        
        # Create pipeline summary
        pipeline_summary = {
            'pipeline_status': 'launched',
            'start_time': pipeline_start.isoformat(),
            'stages': {
                'pubchem_integration': {
                    'status': 'launched',
                    'expected_duration': '30-60 minutes',
                    'expected_boost': '25K → 75K+ records'
                },
                'chemberta_training': {
                    'status': 'launched',
                    'expected_duration': '2-3 hours',
                    'architecture': 'Multi-task (23 targets)',
                    'epochs': 30
                },
                'chemprop_training': {
                    'status': 'launched', 
                    'expected_duration': '3-4 hours',
                    'architecture': 'Enhanced GNN (512 hidden, 5-layer)',
                    'epochs': 40
                }
            },
            'total_expected_duration': '4-5 hours',
            'monitoring_files': {
                'pubchem_integration': '/app/modal_training/pubchem_integration_results.json',
                'chemberta_results': '/app/modal_training/chemberta_enhanced_results.json',
                'chemprop_results': '/app/modal_training/chemprop_enhanced_results.json'
            }
        }
        
        summary_path = Path("/app/modal_training/complete_training_pipeline_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(pipeline_summary, f, indent=2, default=str)
        
        print(f"\n🎉 COMPLETE TRAINING PIPELINE LAUNCHED!")
        print("=" * 80)
        print(f"📁 Pipeline summary: {summary_path}")
        print(f"⏱️ Total expected duration: 4-5 hours")
        
        print(f"\n📊 Pipeline stages:")
        print(f"  1. 🧪 PubChem Integration: 30-60 min → Dataset boost to 75K+ records")
        print(f"  2. 🧠 ChemBERTa Training: 2-3 hrs → Multi-task learning on 23 targets")
        print(f"  3. 🕸️ Chemprop Training: 3-4 hrs → Enhanced GNN architecture")
        
        print(f"\n🔍 Monitoring:")
        print(f"  • Use 'python pipeline_status.py' to check progress")
        print(f"  • Check individual log files for detailed progress")
        print(f"  • Results will be saved to respective JSON files")
        
        print(f"\n🚀 Training will run on Modal.com A100 GPUs with W&B logging")
        
        return pipeline_summary
        
    except Exception as e:
        print(f"❌ PIPELINE LAUNCH FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        error_summary = {
            'pipeline_status': 'launch_failed',
            'start_time': pipeline_start.isoformat(),
            'error_time': datetime.now().isoformat(),
            'error': str(e)
        }
        
        error_path = Path("/app/modal_training/complete_training_pipeline_error.json")
        with open(error_path, 'w') as f:
            json.dump(error_summary, f, indent=2, default=str)
        
        return error_summary

if __name__ == "__main__":
    try:
        result = launch_complete_training_pipeline()
        if result.get('pipeline_status') == 'launched':
            print("\n✅ Complete training pipeline launched successfully")
            print("🎯 All models will train on enhanced multi-database dataset")
        else:
            print(f"\n❌ Training pipeline launch failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"\n❌ Pipeline launcher error: {e}")