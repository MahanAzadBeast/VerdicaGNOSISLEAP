"""
Training Progress Monitoring API
Receives progress updates from remote GPU training
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter()

# Store training progress in memory (could use Redis/DB in production)
training_progress = {}

class TrainingProgress(BaseModel):
    status: str  # "loading_data", "training_started", "training", "completed", "failed"
    message: str
    progress: float  # 0-100 or -1 for error
    epoch: Optional[int] = None
    loss: Optional[float] = None
    r2_score: Optional[float] = None
    results: Optional[Dict[Any, Any]] = None
    training_time_hours: Optional[float] = None

@router.post("/training-progress")
async def receive_training_progress(progress: TrainingProgress):
    """Receive training progress updates from remote GPU training"""
    
    session_id = "molbert_gpu_training"  # Could be dynamic
    timestamp = datetime.now().isoformat()
    
    # Store progress
    if session_id not in training_progress:
        training_progress[session_id] = []
    
    progress_data = progress.dict()
    progress_data['timestamp'] = timestamp
    training_progress[session_id].append(progress_data)
    
    logger.info(f"📊 Training Progress: {progress.status} - {progress.message} ({progress.progress}%)")
    
    # Save to file for persistence
    progress_file = Path("/app/backend/gpu_training_progress.json")
    with open(progress_file, 'w') as f:
        json.dump(training_progress, f, indent=2)
    
    return {"status": "received", "timestamp": timestamp}

@router.get("/training-progress")
async def get_training_progress():
    """Get current training progress"""
    session_id = "molbert_gpu_training"
    
    if session_id in training_progress and training_progress[session_id]:
        latest = training_progress[session_id][-1]
        return {
            "current_status": latest,
            "history": training_progress[session_id][-10:],  # Last 10 updates
            "total_updates": len(training_progress[session_id])
        }
    else:
        return {"current_status": None, "message": "No training in progress"}

@router.get("/training-results")
async def get_training_results():
    """Get final training results if available"""
    session_id = "molbert_gpu_training"
    
    if session_id not in training_progress:
        return {"status": "no_training", "results": None}
    
    # Find completed training
    for update in reversed(training_progress[session_id]):
        if update['status'] == 'completed' and update.get('results'):
            return {
                "status": "completed",
                "results": update['results'],
                "training_time": update.get('training_time_hours'),
                "completion_time": update['timestamp']
            }
    
    return {"status": "not_completed", "results": None}