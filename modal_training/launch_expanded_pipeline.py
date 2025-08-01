"""
Master Script for Expanded Database Training Pipeline
Orchestrates the complete workflow:
1. Multi-source data extraction
2. Expanded ChemBERTa training
3. Expanded Chemprop training
4. Model comparison and analysis
"""

import modal
from pathlib import Path
import json
import logging
from datetime import datetime
import time

# Import our modal apps
from expanded_multisource_extractor import app as extractor_app
from train_expanded_chemberta import app as chemberta_app
from train_expanded_chemprop import app as chemprop_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_expanded_pipeline(
    extract_data: bool = True,
    train_chemberta: bool = True,
    train_chemprop: bool = True,
    activity_type: str = 'IC50',
    chemberta_epochs: int = 30,
    chemprop_epochs: int = 40,
    run_name: str = None
):
    """
    Run the complete expanded database training pipeline
    
    Args:
        extract_data: Whether to run data extraction (skip if already done)
        train_chemberta: Whether to train expanded ChemBERTa model
        train_chemprop: Whether to train expanded Chemprop model
        activity_type: Primary activity type to focus on ('IC50', 'EC50', 'Ki')
        chemberta_epochs: Number of epochs for ChemBERTa training
        chemprop_epochs: Number of epochs for Chemprop training
        run_name: Optional name for this training run
    """
    
    if not run_name:
        run_name = f"expanded_{activity_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("🚀 EXPANDED DATABASE TRAINING PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info(f"📋 Pipeline Configuration:")
    logger.info(f"   • Run name: {run_name}")
    logger.info(f"   • Extract data: {extract_data}")
    logger.info(f"   • Train ChemBERTa: {train_chemberta}")
    logger.info(f"   • Train Chemprop: {train_chemprop}")
    logger.info(f"   • Activity type: {activity_type}")
    logger.info(f"   • ChemBERTa epochs: {chemberta_epochs}")
    logger.info(f"   • Chemprop epochs: {chemprop_epochs}")
    
    pipeline_results = {
        'run_name': run_name,
        'activity_type': activity_type,
        'start_time': datetime.now().isoformat(),
        'stages': {}
    }
    
    try:
        # Stage 1: Data Extraction
        if extract_data:
            logger.info("\n" + "="*80)
            logger.info("🔍 STAGE 1: MULTI-SOURCE DATA EXTRACTION")
            logger.info("="*80)
            logger.info("📊 Extracting data from:")
            logger.info("   • ChEMBL: Primary bioactivity database")
            logger.info("   • PubChem: Supplementary bioassay data")
            logger.info("   • BindingDB: Binding affinity data")
            logger.info("   • DTC: Drug target commons")
            logger.info("\n⏳ Starting extraction (estimated time: 2-4 hours)...")
            
            start_time = time.time()
            
            with extractor_app.run() as app_run:
                extraction_result = app_run.extract_expanded_multisource_dataset.remote()
            
            extraction_time = time.time() - start_time
            
            if extraction_result['status'] == 'success':
                logger.info("✅ DATA EXTRACTION COMPLETED SUCCESSFULLY!")
                logger.info(f"   ⏱️ Time taken: {extraction_time/3600:.1f} hours")
                logger.info(f"   📊 Total records: {extraction_result['total_records']:,}")
                logger.info(f"   🎯 Unique targets: {extraction_result['total_targets']}")
                logger.info(f"   🧪 Unique compounds: {extraction_result['total_compounds']:,}")
                
                pipeline_results['stages']['extraction'] = {
                    'status': 'success',
                    'duration_hours': extraction_time / 3600,
                    'total_records': extraction_result['total_records'],
                    'total_targets': extraction_result['total_targets'],
                    'total_compounds': extraction_result['total_compounds']
                }
            else:
                logger.error(f"❌ DATA EXTRACTION FAILED: {extraction_result.get('error', 'Unknown error')}")
                pipeline_results['stages']['extraction'] = {
                    'status': 'failed',
                    'error': extraction_result.get('error', 'Unknown error')
                }
                return pipeline_results
        else:
            logger.info("\n🔄 SKIPPING DATA EXTRACTION (using existing data)")
            pipeline_results['stages']['extraction'] = {
                'status': 'skipped',
                'reason': 'Using existing extracted data'
            }
        
        # Stage 2: ChemBERTa Training
        if train_chemberta:
            logger.info("\n" + "="*80)
            logger.info("🧠 STAGE 2: EXPANDED CHEMBERTA TRAINING")
            logger.info("="*80)
            logger.info("📊 Training Configuration:")
            logger.info(f"   • Model: ChemBERTa Transformer (ZINC pre-trained)")
            logger.info(f"   • Activity type: {activity_type}")
            logger.info(f"   • Epochs: {chemberta_epochs}")
            logger.info(f"   • Multi-task targets: Oncoproteins + Tumor Suppressors + Metastasis Suppressors")
            logger.info("\n⏳ Starting ChemBERTa training (estimated time: 3-5 hours)...")
            
            start_time = time.time()
            
            with chemberta_app.run() as app_run:
                chemberta_result = app_run.train_expanded_chemberta.remote(
                    activity_type=activity_type,
                    num_epochs=chemberta_epochs,
                    run_name=f"{run_name}_chemberta"
                )
            
            training_time = time.time() - start_time
            
            if chemberta_result['status'] == 'success':
                logger.info("✅ CHEMBERTA TRAINING COMPLETED SUCCESSFULLY!")
                logger.info(f"   ⏱️ Time taken: {training_time/3600:.1f} hours")
                logger.info(f"   📈 Overall Mean R²: {chemberta_result['overall_mean_r2']:.3f}")
                logger.info(f"   🎯 Available targets: {len(chemberta_result['available_targets'])}")
                logger.info(f"   💾 Model saved: {chemberta_result['model_path']}")
                
                pipeline_results['stages']['chemberta'] = {
                    'status': 'success',
                    'duration_hours': training_time / 3600,
                    'overall_mean_r2': chemberta_result['overall_mean_r2'],
                    'available_targets': len(chemberta_result['available_targets']),
                    'model_path': chemberta_result['model_path'],
                    'wandb_run_id': chemberta_result['wandb_run_id']
                }
            else:
                logger.error(f"❌ CHEMBERTA TRAINING FAILED: {chemberta_result.get('error', 'Unknown error')}")
                pipeline_results['stages']['chemberta'] = {
                    'status': 'failed',
                    'error': chemberta_result.get('error', 'Unknown error')
                }
        else:
            logger.info("\n🔄 SKIPPING CHEMBERTA TRAINING")
            pipeline_results['stages']['chemberta'] = {
                'status': 'skipped',
                'reason': 'Training disabled'
            }
        
        # Stage 3: Chemprop Training
        if train_chemprop:
            logger.info("\n" + "="*80)
            logger.info("🕸️ STAGE 3: EXPANDED CHEMPROP TRAINING")
            logger.info("="*80)
            logger.info("📊 Training Configuration:")
            logger.info(f"   • Model: Chemprop Graph Neural Network")
            logger.info(f"   • Activity type: {activity_type}")
            logger.info(f"   • Epochs: {chemprop_epochs}")
            logger.info(f"   • Multi-task targets: Oncoproteins + Tumor Suppressors + Metastasis Suppressors")
            logger.info("\n⏳ Starting Chemprop training (estimated time: 2-3 hours)...")
            
            start_time = time.time()
            
            with chemprop_app.run() as app_run:
                chemprop_result = app_run.train_expanded_chemprop.remote(
                    activity_type=activity_type,
                    epochs=chemprop_epochs,
                    run_name=f"{run_name}_chemprop"
                )
            
            training_time = time.time() - start_time
            
            if chemprop_result['status'] == 'success':
                logger.info("✅ CHEMPROP TRAINING COMPLETED SUCCESSFULLY!")
                logger.info(f"   ⏱️ Time taken: {training_time/3600:.1f} hours")
                logger.info(f"   📈 Mean R²: {chemprop_result['mean_r2']:.3f}")
                logger.info(f"   🎯 Available targets: {len(chemprop_result['available_targets'])}")
                logger.info(f"   💾 Model saved: {chemprop_result['model_path']}")
                
                # Category performance breakdown
                if 'category_performance' in chemprop_result:
                    logger.info("   📊 Category Performance:")
                    for category, performance in chemprop_result['category_performance'].items():
                        logger.info(f"     • {category.replace('_', ' ').title()}: {performance:.3f}")
                
                pipeline_results['stages']['chemprop'] = {
                    'status': 'success',
                    'duration_hours': training_time / 3600,
                    'mean_r2': chemprop_result['mean_r2'],
                    'available_targets': len(chemprop_result['available_targets']),
                    'category_performance': chemprop_result.get('category_performance', {}),
                    'model_path': chemprop_result['model_path'],
                    'wandb_run_id': chemprop_result['wandb_run_id']
                }
            else:
                logger.error(f"❌ CHEMPROP TRAINING FAILED: {chemprop_result.get('error', 'Unknown error')}")
                pipeline_results['stages']['chemprop'] = {
                    'status': 'failed',
                    'error': chemprop_result.get('error', 'Unknown error')
                }
        else:
            logger.info("\n🔄 SKIPPING CHEMPROP TRAINING")
            pipeline_results['stages']['chemprop'] = {
                'status': 'skipped',
                'reason': 'Training disabled'
            }
        
        # Pipeline Summary
        pipeline_results['end_time'] = datetime.now().isoformat()
        total_duration = sum([
            stage.get('duration_hours', 0) 
            for stage in pipeline_results['stages'].values() 
            if 'duration_hours' in stage
        ])
        pipeline_results['total_duration_hours'] = total_duration
        
        logger.info("\n" + "="*80)
        logger.info("🎉 EXPANDED DATABASE TRAINING PIPELINE COMPLETED!")
        logger.info("="*80)
        logger.info(f"📊 Pipeline Summary:")
        logger.info(f"   • Run name: {run_name}")
        logger.info(f"   • Total duration: {total_duration:.1f} hours")
        logger.info(f"   • Activity type: {activity_type}")
        
        # Stage summaries
        for stage_name, stage_info in pipeline_results['stages'].items():
            status_emoji = "✅" if stage_info['status'] == 'success' else "❌" if stage_info['status'] == 'failed' else "🔄"
            logger.info(f"   • {stage_name.title()}: {status_emoji} {stage_info['status']}")
            
            if stage_info['status'] == 'success':
                if 'overall_mean_r2' in stage_info:
                    logger.info(f"     R²: {stage_info['overall_mean_r2']:.3f}")
                elif 'mean_r2' in stage_info:
                    logger.info(f"     R²: {stage_info['mean_r2']:.3f}")
                
                if 'duration_hours' in stage_info:
                    logger.info(f"     Duration: {stage_info['duration_hours']:.1f}h")
        
        # Model comparison (if both models were trained)
        if (train_chemberta and train_chemprop and 
            pipeline_results['stages']['chemberta']['status'] == 'success' and
            pipeline_results['stages']['chemprop']['status'] == 'success'):
            
            chemberta_r2 = pipeline_results['stages']['chemberta'].get('overall_mean_r2', 0)
            chemprop_r2 = pipeline_results['stages']['chemprop'].get('mean_r2', 0)
            
            logger.info(f"\n🏆 MODEL COMPARISON ({activity_type}):")
            logger.info(f"   • ChemBERTa Transformer: {chemberta_r2:.3f}")
            logger.info(f"   • Chemprop GNN: {chemprop_r2:.3f}")
            
            if chemberta_r2 > chemprop_r2:
                winner = "ChemBERTa"
                diff = chemberta_r2 - chemprop_r2
            else:
                winner = "Chemprop"
                diff = chemprop_r2 - chemberta_r2
            
            logger.info(f"   🥇 Better model: {winner} (+{diff:.3f})")
            
            pipeline_results['model_comparison'] = {
                'chemberta_r2': chemberta_r2,
                'chemprop_r2': chemprop_r2,
                'winner': winner,
                'difference': diff
            }
        
        logger.info("\n📁 Results saved to pipeline logs")
        logger.info("🔗 Check W&B dashboard for detailed training metrics")
        logger.info("=" * 80)
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        pipeline_results['status'] = 'failed'
        pipeline_results['error'] = str(e)
        pipeline_results['end_time'] = datetime.now().isoformat()
        
        return pipeline_results

def launch_pipeline():
    """Launch the complete expanded training pipeline"""
    
    print("🚀 VERIDICA AI - EXPANDED DATABASE TRAINING PIPELINE")
    print("=" * 60)
    print("📋 This pipeline will:")
    print("   1. Extract data from multiple sources (ChEMBL, PubChem, BindingDB, DTC)")
    print("   2. Train expanded ChemBERTa model on new targets")
    print("   3. Train expanded Chemprop model on new targets") 
    print("   4. Compare model performance across target categories")
    print("\n🎯 Target Categories:")
    print("   • Oncoproteins (10): EGFR, HER2, VEGFR2, BRAF, MET, CDK4, CDK6, ALK, MDM2, PI3KCA")
    print("   • Tumor Suppressors (7): TP53, RB1, PTEN, APC, BRCA1, BRCA2, VHL")
    print("   • Metastasis Suppressors (6): NDRG1, KAI1, KISS1, NM23H1, RIKP, CASP8")
    print("\n⏳ Estimated total time: 6-12 hours")
    print("💰 Estimated cost: $50-100 (Modal A100 GPU time)")
    
    # Pipeline configuration
    config = {
        'extract_data': True,
        'train_chemberta': True, 
        'train_chemprop': True,
        'activity_type': 'IC50',
        'chemberta_epochs': 30,
        'chemprop_epochs': 40,
        'run_name': f"expanded_multisource_{datetime.now().strftime('%Y%m%d_%H%M')}"
    }
    
    print(f"\n📊 Configuration:")
    for key, value in config.items():
        print(f"   • {key}: {value}")
    
    print(f"\n🚀 Starting pipeline...")
    
    # Run the pipeline
    results = run_expanded_pipeline(**config)
    
    # Display final results
    if results.get('status') != 'failed':
        print(f"\n✅ PIPELINE COMPLETED!")
        print(f"   • Duration: {results.get('total_duration_hours', 0):.1f} hours")
        print(f"   • Run name: {results['run_name']}")
        
        if 'model_comparison' in results:
            comparison = results['model_comparison']
            print(f"\n🏆 FINAL MODEL COMPARISON:")
            print(f"   • ChemBERTa: {comparison['chemberta_r2']:.3f}")
            print(f"   • Chemprop: {comparison['chemprop_r2']:.3f}")
            print(f"   • Winner: {comparison['winner']} (+{comparison['difference']:.3f})")
    else:
        print(f"\n❌ PIPELINE FAILED: {results.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    results = launch_pipeline()