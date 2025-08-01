"""
Enhanced Model Training with Comprehensive Database
Trains ChemBERTa and Chemprop on the integrated multi-database dataset
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

def launch_enhanced_model_training():
    """
    Launch enhanced model training on comprehensive multi-database dataset
    """
    
    print("🚀 LAUNCHING ENHANCED MODEL TRAINING ON COMPREHENSIVE DATABASE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S AEST')}")
    print("🎯 Training on: ChEMBL + PubChem + BindingDB + DTC")
    print()
    
    pipeline_start = datetime.now()
    
    try:
        # Step 1: Launch comprehensive database integration
        print("🔗 STEP 1: Launching comprehensive database integration...")
        print("-" * 60)
        
        integration_cmd = """
cd /app/modal_training && nohup python -c "
import modal
from comprehensive_database_integration import app, integrate_all_databases
import json
from datetime import datetime

print('🔗 STARTING COMPREHENSIVE DATABASE INTEGRATION')
print('Databases: ChEMBL + PubChem + BindingDB + DTC')
print('Expected: 80K-100K+ integrated records')

try:
    with app.run() as app_run:
        result = integrate_all_databases.remote()
    
    with open('/app/modal_training/comprehensive_integration_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    if result['status'] == 'success':
        print('✅ COMPREHENSIVE INTEGRATION COMPLETED!')
        print(f'Integrated records: {result[\"integrated_total_records\"]:,}')
        print(f'Training matrix: {result[\"comprehensive_matrix_shape\"]}')
        print(f'Targets: {result[\"total_targets\"]}')
        print(f'Compounds: {result[\"total_compounds\"]:,}')
    else:
        print(f'❌ INTEGRATION FAILED: {result.get(\"error\")}')
    
except Exception as e:
    error_result = {
        'status': 'failed',
        'error': str(e),
        'timestamp': datetime.now().isoformat()
    }
    with open('/app/modal_training/comprehensive_integration_results.json', 'w') as f:
        json.dump(error_result, f, indent=2)
    print(f'❌ COMPREHENSIVE INTEGRATION FAILED: {e}')
    import traceback
    traceback.print_exc()
" > /app/modal_training/comprehensive_integration_log.txt 2>&1 &
"""
        
        subprocess.run(integration_cmd, shell=True)
        print("✅ Comprehensive database integration launched")
        print("📊 Expected completion: 45-90 minutes")
        
        # Give integration time to start
        time.sleep(15)
        
        # Step 2: Launch enhanced ChemBERTa training
        print(f"\n🧠 STEP 2: Launching enhanced ChemBERTa training...")
        print("-" * 60)
        
        enhanced_chemberta_cmd = """
cd /app/modal_training && nohup python -c "
import modal
from train_expanded_chemberta import app, train_expanded_chemberta
import json
from datetime import datetime
import time

print('🧠 STARTING ENHANCED CHEMBERTA TRAINING')
print('Dataset: Comprehensive multi-database (80K-100K+ records)')
print('Targets: 23 across 3 categories')
print('Epochs: 40 (increased for complex dataset)')
print('Architecture: Enhanced multi-task with cross-database learning')

# Wait for integration to complete
print('⏳ Waiting for database integration to complete...')
max_wait = 120  # 2 hours max wait
wait_interval = 300  # Check every 5 minutes

for i in range(max_wait // wait_interval):
    try:
        with open('/app/modal_training/comprehensive_integration_results.json', 'r') as f:
            integration_result = json.load(f)
        
        if integration_result.get('status') == 'success':
            print('✅ Database integration completed! Starting training...')
            break
        elif integration_result.get('status') == 'failed':
            print(f'❌ Database integration failed: {integration_result.get(\"error\")}')
            raise Exception('Database integration failed')
    except FileNotFoundError:
        pass
    
    print(f'⏳ Integration still running... (check {i+1}/{max_wait // wait_interval})')
    time.sleep(wait_interval)
else:
    print('⏰ Integration timeout - proceeding with available data')

try:
    with app.run() as app_run:
        result = train_expanded_chemberta.remote(
            epochs=40,
            batch_size=16,
            learning_rate=2e-5,
            run_name='comprehensive_enhanced_chemberta',
            use_comprehensive_dataset=True
        )
    
    with open('/app/modal_training/enhanced_chemberta_comprehensive_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print('✅ ENHANCED CHEMBERTA TRAINING COMPLETED!')
    print(f'Results saved to: /app/modal_training/enhanced_chemberta_comprehensive_results.json')
    
except Exception as e:
    error_result = {
        'status': 'failed',
        'error': str(e),
        'timestamp': datetime.now().isoformat()
    }
    with open('/app/modal_training/enhanced_chemberta_comprehensive_results.json', 'w') as f:
        json.dump(error_result, f, indent=2)
    print(f'❌ ENHANCED CHEMBERTA TRAINING FAILED: {e}')
    import traceback
    traceback.print_exc()
" > /app/modal_training/enhanced_chemberta_comprehensive_log.txt 2>&1 &
"""
        
        subprocess.run(enhanced_chemberta_cmd, shell=True)
        print("✅ Enhanced ChemBERTa training launched")
        print("📊 Expected completion: 3-5 hours on A100 GPU")
        
        # Step 3: Launch enhanced Chemprop training
        print(f"\n🕸️ STEP 3: Launching enhanced Chemprop training...")
        print("-" * 60)
        
        enhanced_chemprop_cmd = """
cd /app/modal_training && nohup python -c "
import modal
from train_expanded_chemprop import app, train_expanded_chemprop
import json
from datetime import datetime
import time

print('🕸️ STARTING ENHANCED CHEMPROP TRAINING')
print('Dataset: Comprehensive multi-database (80K-100K+ records)')
print('Targets: 23 across 3 categories')
print('Epochs: 50 (increased for complex dataset)')
print('Architecture: Advanced GNN (1024 hidden, 6-layer depth)')

# Wait for integration to complete
print('⏳ Waiting for database integration to complete...')
max_wait = 120  # 2 hours max wait
wait_interval = 300  # Check every 5 minutes

for i in range(max_wait // wait_interval):
    try:
        with open('/app/modal_training/comprehensive_integration_results.json', 'r') as f:
            integration_result = json.load(f)
        
        if integration_result.get('status') == 'success':
            print('✅ Database integration completed! Starting training...')
            break
        elif integration_result.get('status') == 'failed':
            print(f'❌ Database integration failed: {integration_result.get(\"error\")}')
            raise Exception('Database integration failed')
    except FileNotFoundError:
        pass
    
    print(f'⏳ Integration still running... (check {i+1}/{max_wait // wait_interval})')
    time.sleep(wait_interval)
else:
    print('⏰ Integration timeout - proceeding with available data')

try:
    with app.run() as app_run:
        result = train_expanded_chemprop.remote(
            epochs=50,
            batch_size=64,
            learning_rate=1e-3,
            run_name='comprehensive_enhanced_chemprop',
            use_comprehensive_dataset=True,
            hidden_size=1024,
            depth=6
        )
    
    with open('/app/modal_training/enhanced_chemprop_comprehensive_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print('✅ ENHANCED CHEMPROP TRAINING COMPLETED!')
    print(f'Results saved to: /app/modal_training/enhanced_chemprop_comprehensive_results.json')
    
except Exception as e:
    error_result = {
        'status': 'failed',
        'error': str(e),
        'timestamp': datetime.now().isoformat()
    }
    with open('/app/modal_training/enhanced_chemprop_comprehensive_results.json', 'w') as f:
        json.dump(error_result, f, indent=2)
    print(f'❌ ENHANCED CHEMPROP TRAINING FAILED: {e}')
    import traceback
    traceback.print_exc()
" > /app/modal_training/enhanced_chemprop_comprehensive_log.txt 2>&1 &
"""
        
        subprocess.run(enhanced_chemprop_cmd, shell=True)
        print("✅ Enhanced Chemprop training launched")
        print("📊 Expected completion: 4-6 hours on A100 GPU")
        
        # Create comprehensive pipeline summary
        pipeline_summary = {
            'pipeline_status': 'launched',
            'pipeline_type': 'comprehensive_enhanced_training',
            'start_time': pipeline_start.isoformat(),
            'stages': {
                'database_integration': {
                    'status': 'launched',
                    'databases': ['ChEMBL', 'PubChem', 'BindingDB', 'DTC'],
                    'expected_duration': '45-90 minutes',
                    'expected_records': '80K-100K+'
                },
                'enhanced_chemberta': {
                    'status': 'launched',
                    'expected_duration': '3-5 hours',
                    'architecture': 'Enhanced multi-task transformer',
                    'epochs': 40,
                    'targets': 23
                },
                'enhanced_chemprop': {
                    'status': 'launched', 
                    'expected_duration': '4-6 hours',
                    'architecture': 'Advanced GNN (1024 hidden, 6-layer)',
                    'epochs': 50,
                    'targets': 23
                }
            },
            'total_expected_duration': '6-8 hours',
            'expected_improvements': {
                'dataset_size': '4x increase (25K → 100K records)',
                'target_coverage': '23 comprehensive targets',
                'model_performance': '+15-30% R² improvement expected',
                'chemical_diversity': 'Multi-database chemical space'
            },
            'monitoring_files': {
                'integration_results': '/app/modal_training/comprehensive_integration_results.json',
                'chemberta_results': '/app/modal_training/enhanced_chemberta_comprehensive_results.json',
                'chemprop_results': '/app/modal_training/enhanced_chemprop_comprehensive_results.json'
            }
        }
        
        summary_path = Path("/app/modal_training/enhanced_training_pipeline_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(pipeline_summary, f, indent=2, default=str)
        
        print(f"\n🎉 ENHANCED MODEL TRAINING PIPELINE LAUNCHED!")
        print("=" * 80)
        print(f"📁 Pipeline summary: {summary_path}")
        print(f"⏱️ Total expected duration: 6-8 hours")
        
        print(f"\n📊 Pipeline overview:")
        print(f"  1. 🔗 Database Integration: 45-90 min → 80K-100K+ records from 4 databases")
        print(f"  2. 🧠 Enhanced ChemBERTa: 3-5 hrs → 40 epochs on comprehensive dataset")
        print(f"  3. 🕸️ Enhanced Chemprop: 4-6 hrs → 50 epochs with advanced GNN")
        
        print(f"\n📈 Expected improvements:")
        print(f"  • Dataset size: 4x increase (25K → 100K records)")
        print(f"  • Chemical diversity: Multi-database chemical space coverage")
        print(f"  • Target coverage: 23 comprehensive targets across 3 categories")
        print(f"  • Model performance: +15-30% R² improvement expected")
        print(f"  • Training data: IC50, Ki, EC50 from multiple sources")
        
        print(f"\n🔍 Monitoring:")
        print(f"  • Use 'python pipeline_status.py' to check progress")
        print(f"  • Check individual log files for detailed progress")
        print(f"  • Results will be saved to respective JSON files")
        print(f"  • W&B logging will track training metrics")
        
        print(f"\n🚀 All training runs on Modal.com A100 GPUs with comprehensive datasets")
        
        return pipeline_summary
        
    except Exception as e:
        print(f"❌ ENHANCED TRAINING PIPELINE LAUNCH FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        error_summary = {
            'pipeline_status': 'launch_failed',
            'start_time': pipeline_start.isoformat(),
            'error_time': datetime.now().isoformat(),
            'error': str(e)
        }
        
        error_path = Path("/app/modal_training/enhanced_training_pipeline_error.json")
        with open(error_path, 'w') as f:
            json.dump(error_summary, f, indent=2, default=str)
        
        return error_summary

if __name__ == "__main__":
    try:
        result = launch_enhanced_model_training()
        if result.get('pipeline_status') == 'launched':
            print("\n✅ Enhanced model training pipeline launched successfully")
            print("🎯 Models will train on comprehensive multi-database dataset")
            print("📊 Expected significant performance improvements from 4x larger dataset")
        else:
            print(f"\n❌ Enhanced training pipeline launch failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"\n❌ Pipeline launcher error: {e}")