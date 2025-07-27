# Remote GPU Training Setup Guide

## 🚀 Quick Start with RunPod

### 1. **Prepare Your Environment**
```bash
# Export your RunPod API key
export RUNPOD_API_KEY="your-api-key-here"

# Build and deploy
cd /app/gpu_training
chmod +x deploy_to_runpod.sh
./deploy_to_runpod.sh
```

### 2. **Alternative: Manual RunPod Setup**
1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Click "Deploy" → "New Pod"
3. Select GPU: **RTX 4090** or **A100** 
4. Template: **PyTorch 2.1.0**
5. Container Disk: **50GB**
6. Upload this folder as zip

### 3. **Monitor Training Progress**
The training will send updates to your webhook URL. Add this to your main backend:

```python
# Add to /app/backend/server.py
from gpu_training.progress_monitor import router as gpu_router
app.include_router(gpu_router, prefix="/api/gpu")
```

## 📊 **Expected Performance**

| GPU Type | Training Time | Cost/Hour | Total Cost |
|----------|---------------|-----------|------------|
| RTX 4090 | 2-4 hours | $0.40 | $0.80-$1.60 |
| A100 40GB | 1-2 hours | $1.20 | $1.20-$2.40 |
| H100 80GB | 30-60 min | $2.50 | $1.25-$2.50 |

## 🔧 **Configuration Options**

Edit `training_config.json`:
- `max_epochs`: Target epochs (default: 50)
- `batch_size`: GPU batch size (32-64 for RTX 4090)
- `progress_webhook`: Your app's progress endpoint
- `mixed_precision`: Enable FP16 for 2x speed boost

## 📥 **Retrieving Results**

After training completes:
1. Check progress: `GET /api/gpu/training-progress`
2. Download models via RunPod file browser
3. Results auto-uploaded to configured cloud storage

## 🎯 **Multiple Target Training**

To train all targets (EGFR, BRAF, CDK2, etc.):
```json
{
  "targets": ["EGFR", "BRAF", "CDK2", "PARP1", "BCL2", "VEGFR2"],
  "parallel_training": true,
  "max_epochs_per_target": 50
}
```

## 💡 **Cost Optimization Tips**

1. **Spot Instances**: Use RunPod spot pricing (50% cheaper)
2. **Batch Training**: Train multiple targets in one session
3. **Early Stopping**: Stop when R² plateaus
4. **Mixed Precision**: 2x speed with same accuracy

## 🚨 **Important Notes**

- **Data Transfer**: ChEMBL data (~100MB) downloads automatically
- **Checkpoints**: Saved every 5 epochs + final model
- **Monitoring**: Real-time progress via webhook
- **Auto-shutdown**: Instance stops after training completes