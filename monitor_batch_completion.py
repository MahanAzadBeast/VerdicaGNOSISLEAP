#!/usr/bin/env python3
"""
Monitor Batch Completion - Ensure ALL Batches Complete
Continuously monitors the SMILES search to ensure all batches finish
and provides real-time progress updates
"""

import os
import json
import subprocess
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchCompletionMonitor:
    """Monitors batch completion to ensure all batches finish"""
    
    def __init__(self):
        self.progress_dir = Path("clinical_trial_dataset/data/smiles_progress")
        self.github_dir = Path("clinical_trial_dataset/data/github_final")
        
    def get_total_batches_needed(self):
        """Calculate total batches needed based on unique drugs"""
        try:
            # Count unique drugs from all trial parts
            all_drugs = set()
            
            for part_num in range(1, 5):
                part_file = self.github_dir / f"trials_part_{part_num}.csv"
                if part_file.exists():
                    df = pd.read_csv(part_file)
                    for drug in df['primary_drug'].dropna():
                        if isinstance(drug, str) and len(drug.strip()) > 2:
                            all_drugs.add(drug.strip())
            
            total_drugs = len(all_drugs)
            batch_size = 20  # From the script
            total_batches = (total_drugs - 1) // batch_size + 1
            
            return total_drugs, total_batches
            
        except Exception as e:
            logger.error(f"Error calculating total batches: {e}")
            return 0, 0
    
    def check_process_status(self):
        """Check if SMILES process is still running"""
        try:
            result = subprocess.run(['pgrep', '-f', 'get_all_real_smiles'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                # Get process details
                ps_result = subprocess.run(['ps', '-eo', 'pid,etime,pcpu,pmem'], 
                                         capture_output=True, text=True)
                for line in ps_result.stdout.split('\n'):
                    if result.stdout.strip() in line:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            return {
                                'running': True,
                                'pid': parts[0],
                                'runtime': parts[1],
                                'cpu': parts[2],
                                'memory': parts[3]
                            }
                
                return {'running': True, 'details': 'unknown'}
            else:
                return {'running': False}
                
        except Exception as e:
            logger.error(f"Error checking process: {e}")
            return {'running': False, 'error': str(e)}
    
    def get_current_progress(self):
        """Get current progress from saved files"""
        progress = {
            'drugs_processed': 0,
            'smiles_found': 0,
            'batches_completed': 0,
            'success_rate': 0.0,
            'last_update': 'unknown'
        }
        
        # Check drug results
        drug_results_file = self.progress_dir / 'all_drug_smiles_results.json'
        if drug_results_file.exists():
            try:
                with open(drug_results_file, 'r') as f:
                    drug_results = json.load(f)
                
                progress['drugs_processed'] = len(drug_results)
                progress['smiles_found'] = sum(1 for v in drug_results.values() 
                                             if isinstance(v, dict) and v.get('smiles'))
                progress['success_rate'] = (progress['smiles_found'] / progress['drugs_processed'] * 100) if progress['drugs_processed'] > 0 else 0
                
            except Exception as e:
                logger.debug(f"Error reading drug results: {e}")
        
        # Check batch progress
        batch_file = self.progress_dir / 'batch_progress.json'
        if batch_file.exists():
            try:
                with open(batch_file, 'r') as f:
                    batch_info = json.load(f)
                
                progress['batches_completed'] = batch_info.get('processed_batches', 0)
                progress['last_update'] = batch_info.get('last_update', 'unknown')
                
            except Exception as e:
                logger.debug(f"Error reading batch progress: {e}")
        
        return progress
    
    def monitor_until_completion(self, check_interval=60):
        """Monitor process until all batches complete"""
        logger.info("🔍 MONITORING BATCH COMPLETION")
        logger.info("🎯 Ensuring ALL batches complete successfully")
        logger.info("=" * 70)
        
        # Get total batches needed
        total_drugs, total_batches = self.get_total_batches_needed()
        logger.info(f"📊 Target: {total_drugs:,} drugs in {total_batches} batches")
        
        monitoring_round = 0
        last_batch_count = 0
        stalled_checks = 0
        max_stalled_checks = 5
        
        while True:
            monitoring_round += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n📊 MONITORING ROUND {monitoring_round} - {current_time}")
            print("-" * 50)
            
            # Check process status
            process_status = self.check_process_status()
            
            if process_status['running']:
                print(f"🔄 Process ACTIVE: PID {process_status.get('pid', 'unknown')}")
                print(f"   Runtime: {process_status.get('runtime', 'unknown')}")
                print(f"   CPU: {process_status.get('cpu', 'unknown')}%")
                print(f"   Memory: {process_status.get('memory', 'unknown')}%")
            else:
                print("✅ Process COMPLETED or STOPPED")
            
            # Check progress
            progress = self.get_current_progress()
            
            print(f"📈 Progress Status:")
            print(f"   Drugs processed: {progress['drugs_processed']:,}/{total_drugs:,}")
            print(f"   Batches completed: {progress['batches_completed']}/{total_batches}")
            print(f"   SMILES found: {progress['smiles_found']:,}")
            print(f"   Success rate: {progress['success_rate']:.1f}%")
            print(f"   Last update: {progress['last_update']}")
            
            # Calculate completion percentage
            batch_completion = (progress['batches_completed'] / total_batches * 100) if total_batches > 0 else 0
            drug_completion = (progress['drugs_processed'] / total_drugs * 100) if total_drugs > 0 else 0
            
            print(f"📊 Completion Status:")
            print(f"   Batch completion: {batch_completion:.1f}%")
            print(f"   Drug completion: {drug_completion:.1f}%")
            
            # Check if making progress
            if progress['batches_completed'] == last_batch_count:
                stalled_checks += 1
                print(f"⚠️ No progress in last {stalled_checks} checks")
                
                if stalled_checks >= max_stalled_checks and process_status['running']:
                    print(f"🔧 Process may be stalled - consider intervention")
            else:
                stalled_checks = 0
                batch_progress = progress['batches_completed'] - last_batch_count
                print(f"✅ Progress made: +{batch_progress} batches since last check")
            
            last_batch_count = progress['batches_completed']
            
            # Check completion
            if progress['batches_completed'] >= total_batches:
                print(f"\n🎉 ALL BATCHES COMPLETED!")
                print(f"📊 Final results: {progress['smiles_found']:,} SMILES found")
                break
            elif not process_status['running']:
                if progress['batches_completed'] < total_batches:
                    print(f"\n⚠️ PROCESS STOPPED BEFORE COMPLETION")
                    print(f"📊 Completed: {progress['batches_completed']}/{total_batches} batches")
                    print(f"🔄 Process may need to be restarted")
                else:
                    print(f"\n✅ PROCESS COMPLETED SUCCESSFULLY")
                break
            
            # Show estimated completion time
            if progress['batches_completed'] > 0 and monitoring_round > 1:
                batches_per_check = (progress['batches_completed'] - (progress['batches_completed'] - batch_progress)) / 1 if batch_progress > 0 else 0
                if batches_per_check > 0:
                    remaining_batches = total_batches - progress['batches_completed']
                    estimated_checks = remaining_batches / batches_per_check
                    estimated_minutes = estimated_checks * (check_interval / 60)
                    print(f"⏱️ Estimated completion: {estimated_minutes:.1f} minutes")
            
            # Wait before next check
            print(f"\n⏳ Next check in {check_interval} seconds...")
            time.sleep(check_interval)
        
        return progress
    
    def verify_final_results(self):
        """Verify final results when process completes"""
        logger.info("🔍 VERIFYING FINAL RESULTS")
        
        # Check for final output files
        final_files = [
            'clinical_trials_all_real_smiles.csv',
            'train_all_real_smiles.csv',
            'val_all_real_smiles.csv',
            'test_all_real_smiles.csv'
        ]
        
        results_summary = {}
        
        for filename in final_files:
            file_path = self.github_dir / filename
            
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024*1024)
                
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    
                    # Count SMILES
                    if 'has_real_smiles' in df.columns:
                        smiles_count = df['has_real_smiles'].sum()
                    elif 'smiles' in df.columns:
                        smiles_count = df['smiles'].notna().sum()
                    else:
                        smiles_count = 0
                    
                    coverage = (smiles_count / len(df) * 100) if len(df) > 0 else 0
                    
                    results_summary[filename] = {
                        'total_trials': len(df),
                        'trials_with_smiles': smiles_count,
                        'coverage_percent': coverage,
                        'file_size_mb': size_mb
                    }
                    
                    logger.info(f"✅ {filename}:")
                    logger.info(f"   Trials: {len(df):,}")
                    logger.info(f"   SMILES: {smiles_count:,} ({coverage:.1f}%)")
                    logger.info(f"   Size: {size_mb:.1f} MB")
                    
                    # Check NCT02688101
                    if 'nct_id' in df.columns:
                        nct_check = df[df['nct_id'] == 'NCT02688101']
                        if len(nct_check) > 0:
                            has_smiles = False
                            if 'has_real_smiles' in df.columns:
                                has_smiles = nct_check.iloc[0].get('has_real_smiles', False)
                            elif 'smiles' in df.columns:
                                has_smiles = pd.notna(nct_check.iloc[0].get('smiles'))
                            
                            logger.info(f"   🎯 NCT02688101: {'✅ INCLUDED with SMILES' if has_smiles else '❌ Missing SMILES'}")
                
                except Exception as e:
                    logger.error(f"❌ Error verifying {filename}: {e}")
            else:
                logger.warning(f"⚠️ {filename}: Not found")
        
        return results_summary

def main():
    """Main monitoring execution"""
    logger.info("🔍 BATCH COMPLETION MONITOR")
    logger.info("🎯 Ensuring ALL batches complete successfully")
    logger.info("💾 Monitoring incremental saves")
    logger.info("=" * 70)
    
    monitor = BatchCompletionMonitor()
    
    # Monitor until completion
    final_progress = monitor.monitor_until_completion(check_interval=60)
    
    # Verify final results
    results = monitor.verify_final_results()
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("🎉 BATCH COMPLETION MONITORING FINISHED")
    logger.info("=" * 70)
    
    if results:
        # Find the complete dataset
        complete_file = results.get('clinical_trials_all_real_smiles.csv')
        if complete_file:
            logger.info(f"📊 Final Complete Dataset:")
            logger.info(f"   Total trials: {complete_file['total_trials']:,}")
            logger.info(f"   Trials with SMILES: {complete_file['trials_with_smiles']:,}")
            logger.info(f"   Final coverage: {complete_file['coverage_percent']:.1f}%")
            logger.info(f"   File size: {complete_file['file_size_mb']:.1f} MB")
        
        logger.info(f"✅ ALL BATCHES MONITORING COMPLETE")
        logger.info(f"📊 Final progress: {final_progress}")
    else:
        logger.warning("⚠️ No final results found - process may not have completed")
    
    return results

if __name__ == "__main__":
    main()