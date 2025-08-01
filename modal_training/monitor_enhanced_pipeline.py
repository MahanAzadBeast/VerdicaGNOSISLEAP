"""
Enhanced Pipeline Monitor
Monitors the comprehensive database integration and enhanced model training
"""

import json
import time
from pathlib import Path
from datetime import datetime

def monitor_enhanced_pipeline():
    """Monitor the enhanced training pipeline progress"""
    
    print("🔍 ENHANCED PIPELINE MONITORING")
    print("=" * 80)
    print(f"Monitoring started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S AEST')}")
    print()
    
    # Monitoring files
    integration_results = Path("/app/modal_training/comprehensive_integration_results.json")
    chemberta_results = Path("/app/modal_training/enhanced_chemberta_comprehensive_results.json")  
    chemprop_results = Path("/app/modal_training/enhanced_chemprop_comprehensive_results.json")
    
    while True:
        print(f"\n📊 Pipeline Status - {datetime.now().strftime('%H:%M:%S AEST')}")
        print("-" * 60)
        
        # Check database integration
        if integration_results.exists():
            try:
                with open(integration_results, 'r') as f:
                    data = json.load(f)
                
                if data.get('status') == 'success':
                    print(f"🔗 Database Integration: ✅ COMPLETED")
                    print(f"   📈 Integrated records: {data.get('integrated_total_records', 0):,}")
                    print(f"   🎯 Targets: {data.get('total_targets', 0)}")
                    print(f"   🧪 Compounds: {data.get('total_compounds', 0):,}")
                    print(f"   📋 Matrix shape: {data.get('comprehensive_matrix_shape', 'unknown')}")
                    print(f"   🔗 Databases: {', '.join(data.get('databases_integrated', []))}")
                else:
                    print(f"🔗 Database Integration: ❌ FAILED")
                    print(f"   Error: {data.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"🔗 Database Integration: ❓ ERROR reading results - {e}")
        else:
            print(f"🔗 Database Integration: 🔄 RUNNING...")
        
        # Check ChemBERTa training
        if chemberta_results.exists():
            try:
                with open(chemberta_results, 'r') as f:
                    data = json.load(f)
                
                if data.get('status') == 'success':
                    print(f"🧠 Enhanced ChemBERTa: ✅ COMPLETED")
                    r2_scores = data.get('r2_scores', {})
                    if r2_scores:
                        mean_r2 = sum(r2_scores.values()) / len(r2_scores)
                        print(f"   📈 Mean R²: {mean_r2:.3f}")
                        print(f"   🎯 Targets trained: {len(r2_scores)}")
                else:
                    print(f"🧠 Enhanced ChemBERTa: ❌ FAILED")
                    print(f"   Error: {data.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"🧠 Enhanced ChemBERTa: ❓ ERROR reading results - {e}")
        else:
            print(f"🧠 Enhanced ChemBERTa: 🔄 QUEUED/RUNNING...")
        
        # Check Chemprop training
        if chemprop_results.exists():
            try:
                with open(chemprop_results, 'r') as f:
                    data = json.load(f)
                
                if data.get('status') == 'success':
                    print(f"🕸️ Enhanced Chemprop: ✅ COMPLETED")
                    r2_scores = data.get('r2_scores', {})
                    if r2_scores:
                        mean_r2 = sum(r2_scores.values()) / len(r2_scores)
                        print(f"   📈 Mean R²: {mean_r2:.3f}")
                        print(f"   🎯 Targets trained: {len(r2_scores)}")
                else:
                    print(f"🕸️ Enhanced Chemprop: ❌ FAILED")
                    print(f"   Error: {data.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"🕸️ Enhanced Chemprop: ❓ ERROR reading results - {e}")
        else:
            print(f"🕸️ Enhanced Chemprop: 🔄 QUEUED/RUNNING...")
        
        # Check if all completed
        all_completed = (
            integration_results.exists() and 
            chemberta_results.exists() and 
            chemprop_results.exists()
        )
        
        if all_completed:
            try:
                # Load all results
                with open(integration_results, 'r') as f:
                    integration_data = json.load(f)
                with open(chemberta_results, 'r') as f:
                    chemberta_data = json.load(f)
                with open(chemprop_results, 'r') as f:
                    chemprop_data = json.load(f)
                
                # Check if all successful
                if all(data.get('status') == 'success' for data in [integration_data, chemberta_data, chemprop_data]):
                    print(f"\n🎉 ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
                    print("=" * 80)
                    
                    print(f"📊 Final Results:")
                    print(f"   🔗 Database: {integration_data.get('integrated_total_records', 0):,} records integrated")
                    print(f"   🧠 ChemBERTa: Mean R² = {sum(chemberta_data.get('r2_scores', {}).values()) / len(chemberta_data.get('r2_scores', {})):.3f}")
                    print(f"   🕸️ Chemprop: Mean R² = {sum(chemprop_data.get('r2_scores', {}).values()) / len(chemprop_data.get('r2_scores', {})):.3f}")
                    
                    break
                else:
                    print(f"\n❌ Pipeline completed with errors - check individual results")
                    break
            except Exception as e:
                print(f"\n❓ Error checking final results: {e}")
        
        print(f"\n⏳ Next check in 10 minutes...")
        time.sleep(600)  # Check every 10 minutes

if __name__ == "__main__":
    try:
        monitor_enhanced_pipeline()
    except KeyboardInterrupt:
        print(f"\n🛑 Pipeline monitoring interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline monitoring error: {e}")