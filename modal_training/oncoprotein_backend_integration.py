"""
Multi-Task ChemBERTa Oncoprotein Backend Integration
Provides API endpoints for monitoring and controlling the oncoprotein training pipeline
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import logging
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Pydantic models for API
class OncoproteincPipelineStatus(BaseModel):
    status: str
    current_stage: str
    progress_percent: float
    stages: Dict[str, Dict[str, Any]]
    error: Optional[str] = None
    last_update: str

class OncoproteincPipelineRequest(BaseModel):
    action: str  # 'start', 'stop', 'resume'
    stage: Optional[str] = None  # 'download', 'extract', 'train', 'report', 'inference'

class OncoproteincPredictionRequest(BaseModel):
    smiles: str
    targets: Optional[List[str]] = None  # If None, predict for all 14 targets

class OncoproteincPredictionResponse(BaseModel):
    smiles: str
    predictions: Dict[str, Dict[str, float]]  # target -> {pIC50, IC50_nM}
    model_info: Dict[str, Any]
    prediction_time: str

# API Router
oncoprotein_router = APIRouter(prefix="/api/oncoprotein")

@oncoprotein_router.get("/status", response_model=OncoproteincPipelineStatus)
async def get_pipeline_status():
    """Get current status of the multi-task ChemBERTa pipeline"""
    
    try:
        # Check for log files to determine current status
        log_files = {
            'pipeline_execution_fixed.log': 'pipeline',
            'oncoprotein_deploy_fixed_v2.log': 'deployment'
        }
        
        stages = {
            'deployment': {'completed': False, 'progress': 0, 'message': 'Not started'},
            'download': {'completed': False, 'progress': 0, 'message': 'Not started'},
            'extraction': {'completed': False, 'progress': 0, 'message': 'Not started'},
            'training': {'completed': False, 'progress': 0, 'message': 'Not started'},
            'report': {'completed': False, 'progress': 0, 'message': 'Not started'},
            'inference': {'completed': False, 'progress': 0, 'message': 'Not started'}
        }
        
        current_stage = 'deployment'
        progress_percent = 0.0
        status = 'not_started'
        error = None
        
        # Check deployment status
        deploy_log = Path('/app/modal_training/oncoprotein_deploy_fixed_v2.log')
        if deploy_log.exists():
            with open(deploy_log, 'r') as f:
                deploy_content = f.read()
            
            if '✓ App deployed' in deploy_content:
                stages['deployment']['completed'] = True
                stages['deployment']['progress'] = 100
                stages['deployment']['message'] = 'App deployed successfully'
                progress_percent = 16.7  # 1/6 stages
                current_stage = 'download'
                status = 'running'
        
        # Check pipeline execution status
        pipeline_log = Path('/app/modal_training/pipeline_execution_fixed.log')
        if pipeline_log.exists():
            with open(pipeline_log, 'r') as f:
                pipeline_content = f.read()
            
            # Parse download progress
            if 'Downloaded' in pipeline_content:
                # Extract latest download progress
                lines = pipeline_content.split('\n')
                download_lines = [l for l in lines if 'Downloaded' in l and 'GB' in l]
                if download_lines:
                    latest_download = download_lines[-1]
                    # Extract current and total GB from log like "Downloaded 1.8 GB..."
                    try:
                        if 'Downloading' in pipeline_content and 'GB...' in pipeline_content:
                            total_line = [l for l in lines if 'Downloading' in l and 'GB...' in l][0]
                            total_gb = float(total_line.split('Downloading ')[1].split(' GB')[0])
                            
                            current_gb = float(latest_download.split('Downloaded ')[1].split(' GB')[0])
                            download_progress = min(100, (current_gb / total_gb) * 100)
                            
                            stages['download']['progress'] = download_progress
                            stages['download']['message'] = f'Downloaded {current_gb:.1f} GB / {total_gb:.1f} GB'
                            
                            if download_progress >= 100:
                                stages['download']['completed'] = True
                                current_stage = 'extraction'
                                progress_percent = 33.3  # 2/6 stages
                            else:
                                progress_percent = 16.7 + (download_progress / 100 * 16.7)
                    except:
                        stages['download']['progress'] = 50
                        stages['download']['message'] = 'Download in progress...'
                        progress_percent = 25
            
            # Check for extraction completion
            if '✅ ChEMBL downloaded' in pipeline_content:
                stages['download']['completed'] = True
                stages['download']['progress'] = 100
                current_stage = 'extraction'
                progress_percent = 33.3
            
            if '✅ Data extracted' in pipeline_content:
                stages['extraction']['completed'] = True
                stages['extraction']['progress'] = 100
                current_stage = 'training'
                progress_percent = 50.0
            
            if '✅ Training completed' in pipeline_content:
                stages['training']['completed'] = True
                stages['training']['progress'] = 100
                current_stage = 'report'
                progress_percent = 66.7
            
            if '✅ Report generated' in pipeline_content:
                stages['report']['completed'] = True
                stages['report']['progress'] = 100
                current_stage = 'inference'
                progress_percent = 83.3
            
            if '✅ Inference script created' in pipeline_content:
                stages['inference']['completed'] = True
                stages['inference']['progress'] = 100
                current_stage = 'completed'
                progress_percent = 100.0
                status = 'completed'
            
            # Check for errors
            if 'ERROR:' in pipeline_content or 'Failed' in pipeline_content:
                error_lines = [l for l in pipeline_content.split('\n') if 'ERROR:' in l or 'Failed' in l]
                if error_lines:
                    error = error_lines[-1]  # Get latest error
                    status = 'error'
        
        return OncoproteincPipelineStatus(
            status=status,
            current_stage=current_stage,
            progress_percent=progress_percent,
            stages=stages,
            error=error,
            last_update=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get pipeline status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline status: {str(e)}")

@oncoprotein_router.post("/pipeline", response_model=Dict[str, Any])
async def control_pipeline(request: OncoproteincPipelineRequest):
    """Control the multi-task ChemBERTa pipeline"""
    
    try:
        if request.action == 'start':
            logger.info("🚀 Starting multi-task ChemBERTa pipeline...")
            
            # Check if already running
            log_file = Path('/app/modal_training/pipeline_execution_fixed.log')
            if log_file.exists():
                # Check if process is still running
                try:
                    result = subprocess.run(['pgrep', '-f', 'oncoprotein_chemberta'], 
                                          capture_output=True, text=True)
                    if result.stdout.strip():
                        return {
                            "status": "already_running",
                            "message": "Pipeline is already running",
                            "log_file": str(log_file)
                        }
                except:
                    pass
            
            # Start the pipeline
            cmd = [
                'nohup', 'modal', 'run', 
                '/app/modal_training/oncoprotein_chemberta.py::run_complete_pipeline'
            ]
            
            # Run in background
            process = subprocess.Popen(
                cmd,
                stdout=open('/app/modal_training/pipeline_execution_new.log', 'w'),
                stderr=subprocess.STDOUT,
                cwd='/app/modal_training'
            )
            
            return {
                "status": "started",
                "message": "Multi-task ChemBERTa pipeline started",
                "process_id": process.pid,
                "log_file": "/app/modal_training/pipeline_execution_new.log"
            }
            
        elif request.action == 'stop':
            logger.info("⏹️ Stopping multi-task ChemBERTa pipeline...")
            
            # Find and kill the process
            try:
                result = subprocess.run(['pgrep', '-f', 'oncoprotein_chemberta'], 
                                      capture_output=True, text=True)
                pids = result.stdout.strip().split('\n')
                
                for pid in pids:
                    if pid:
                        subprocess.run(['kill', pid])
                
                return {
                    "status": "stopped",
                    "message": f"Stopped {len([p for p in pids if p])} pipeline processes"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to stop pipeline: {str(e)}"
                }
        
        elif request.action == 'resume':
            # Resume from a specific stage
            if request.stage:
                logger.info(f"🔄 Resuming pipeline from stage: {request.stage}")
                
                stage_functions = {
                    'download': 'download_chembl_database',
                    'extract': 'extract_oncoprotein_data', 
                    'train': 'train_multitask_chemberta',
                    'report': 'generate_dataset_report',
                    'inference': 'create_inference_script'
                }
                
                if request.stage in stage_functions:
                    func_name = stage_functions[request.stage]
                    cmd = [
                        'nohup', 'modal', 'run', 
                        f'/app/modal_training/oncoprotein_chemberta.py::{func_name}'
                    ]
                    
                    process = subprocess.Popen(
                        cmd,
                        stdout=open(f'/app/modal_training/pipeline_{request.stage}_resume.log', 'w'),
                        stderr=subprocess.STDOUT,
                        cwd='/app/modal_training'
                    )
                    
                    return {
                        "status": "resumed",
                        "message": f"Pipeline resumed from {request.stage} stage",
                        "process_id": process.pid,
                        "log_file": f"/app/modal_training/pipeline_{request.stage}_resume.log"
                    }
                else:
                    raise HTTPException(status_code=400, detail=f"Invalid stage: {request.stage}")
            else:
                raise HTTPException(status_code=400, detail="Stage required for resume action")
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
            
    except Exception as e:
        logger.error(f"❌ Pipeline control failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline control failed: {str(e)}")

@oncoprotein_router.get("/targets")
async def get_oncoprotein_targets():
    """Get list of oncoprotein targets supported by the model"""
    
    targets = {
        "EGFR": {"chembl_id": "CHEMBL203", "name": "Epidermal Growth Factor Receptor"},
        "HER2": {"chembl_id": "CHEMBL1824", "name": "Human Epidermal Growth Factor Receptor 2"},
        "VEGFR2": {"chembl_id": "CHEMBL279", "name": "Vascular Endothelial Growth Factor Receptor 2"},
        "ALK": {"chembl_id": "CHEMBL3565", "name": "Anaplastic Lymphoma Kinase"},
        "BRAF": {"chembl_id": "CHEMBL1823", "name": "B-Raf Proto-Oncogene"},
        "MET": {"chembl_id": "CHEMBL3717", "name": "MET Proto-Oncogene"},
        "MDM2": {"chembl_id": "CHEMBL5023", "name": "MDM2 Proto-Oncogene"},
        "STAT3": {"chembl_id": "CHEMBL5407", "name": "Signal Transducer And Activator Of Transcription 3"},
        "RRM2": {"chembl_id": "CHEMBL3352", "name": "Ribonucleotide Reductase Regulatory Subunit M2"},
        "CTNNB1": {"chembl_id": "CHEMBL6132", "name": "β-Catenin"},
        "MYC": {"chembl_id": "CHEMBL6130", "name": "MYC Proto-Oncogene"},
        "PI3KCA": {"chembl_id": "CHEMBL4040", "name": "Phosphatidylinositol-4,5-Bisphosphate 3-Kinase Catalytic Subunit Alpha"},
        "CDK4": {"chembl_id": "CHEMBL331", "name": "Cyclin Dependent Kinase 4"},
        "CDK6": {"chembl_id": "CHEMBL3974", "name": "Cyclin Dependent Kinase 6"}
    }
    
    return {
        "total_targets": len(targets),
        "targets": targets,
        "description": "14 key oncoproteins for multi-task ChemBERTa training"
    }

@oncoprotein_router.post("/predict", response_model=OncoproteincPredictionResponse)
async def predict_oncoprotein_activity(request: OncoproteincPredictionRequest):
    """Make predictions using the trained multi-task ChemBERTa model"""
    
    try:
        # Check if model is trained and available
        model_dir = Path('/vol/models/chemberta_multitask')  # Modal volume path
        if not model_dir.exists():
            # Try local fallback path
            model_dir = Path('/app/models/chemberta_multitask')
            if not model_dir.exists():
                raise HTTPException(
                    status_code=503, 
                    detail="Multi-task ChemBERTa model not available. Please complete training first."
                )
        
        # Validate SMILES
        from rdkit import Chem
        mol = Chem.MolFromSmiles(request.smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES string")
        
        # For now, return simulated predictions until actual model is ready
        # This will be replaced with actual model inference once training completes
        targets = request.targets if request.targets else [
            'EGFR', 'HER2', 'VEGFR2', 'ALK', 'BRAF', 'MET', 'MDM2', 
            'STAT3', 'RRM2', 'CTNNB1', 'MYC', 'PI3KCA', 'CDK4', 'CDK6'
        ]
        
        predictions = {}
        import random
        import hashlib
        
        # Generate consistent predictions based on SMILES hash for demo
        seed = int(hashlib.md5(request.smiles.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        for target in targets:
            # Generate realistic pIC50 values (typically 4-10)
            pIC50 = random.uniform(4.5, 9.5)
            IC50_nM = 10 ** (9 - pIC50)  # Convert to nM
            
            predictions[target] = {
                "pIC50": round(pIC50, 3),
                "IC50_nM": round(IC50_nM, 1)
            }
        
        return OncoproteincPredictionResponse(
            smiles=request.smiles,
            predictions=predictions,
            model_info={
                "model_type": "multi_task_chemberta",
                "training_stage": "simulated_predictions", 
                "note": "Actual model predictions will be available after training completion"
            },
            prediction_time=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@oncoprotein_router.get("/logs/{log_type}")
async def get_pipeline_logs(log_type: str, lines: int = 50):
    """Get pipeline execution logs"""
    
    log_files = {
        'pipeline': '/app/modal_training/pipeline_execution_fixed.log',
        'deployment': '/app/modal_training/oncoprotein_deploy_fixed_v2.log',
        'download': '/app/modal_training/pipeline_download_resume.log',
        'extract': '/app/modal_training/pipeline_extract_resume.log',
        'train': '/app/modal_training/pipeline_train_resume.log'
    }
    
    if log_type not in log_files:
        raise HTTPException(status_code=400, detail=f"Invalid log type. Choose from: {list(log_files.keys())}")
    
    log_file = Path(log_files[log_type])
    if not log_file.exists():
        return {
            "log_type": log_type,
            "lines": 0,
            "content": "Log file not found",
            "file_path": str(log_file)
        }
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Get last N lines
        lines_list = content.split('\n')
        if len(lines_list) > lines:
            lines_list = lines_list[-lines:]
        
        return {
            "log_type": log_type,
            "lines": len(lines_list),
            "content": '\n'.join(lines_list),
            "file_path": str(log_file),
            "last_updated": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to read log file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read log file: {str(e)}")