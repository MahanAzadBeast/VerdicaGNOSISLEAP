# Enhanced Modal MolBERT Setup - Complete Guide

## 🎯 Overview

You now have a complete Modal.com infrastructure for MolBERT with:

- **Persistent model storage** using `molbert-cache` volume
- **Automatic model download** of `seyonec/ChemBERTa-zinc-base-v1` from Hugging Face
- **Hugging Face cache mounting** for efficient model reuse
- **Fine-tuning capabilities** with A100 GPU support
- **Backend API integration** with fallback mechanisms

## 📁 Files Created

### Core Modal Infrastructure
- `/app/modal_training/modal_molbert_enhanced.py` - Enhanced Modal app with caching
- `/app/modal_training/enhanced_backend_integration.py` - Backend integration client
- `/app/modal_training/deploy_enhanced.sh` - Deployment script
- `/app/setup_modal_molbert.py` - Complete setup helper

### Backend Integration
- Updated `/app/backend/server.py` with new Modal endpoints
- Updated `/app/backend/requirements.txt` with `modal` package

## 🚀 API Endpoints Added

### Modal MolBERT Endpoints

1. **`GET /api/modal/molbert/status`**
   - Get Modal MolBERT setup status
   - Returns: modal availability, credentials status, model info

2. **`POST /api/modal/molbert/setup`**
   - Setup Modal MolBERT (download pretrained model)
   - One-time setup - downloads and caches the model

3. **`POST /api/modal/molbert/train/{target}`**
   - Start fine-tuning for specific target (EGFR, BRAF, etc.)
   - Uses A100 GPU with cached pretrained model
   - Parameters: webhook_url (optional)

4. **`POST /api/modal/molbert/predict`**
   - Run predictions using Modal-hosted MolBERT
   - Parameters: smiles, target, use_finetuned
   - Auto-fallback to local prediction if Modal unavailable

## 🔧 Setup Instructions

### Step 1: Modal CLI Setup
```bash
# Install Modal CLI
pip install modal

# Authenticate (get token from https://modal.com/settings/tokens)
modal token new
```

### Step 2: Deploy Modal Infrastructure
```bash
# Option A: Automated setup
cd /app
python setup_modal_molbert.py

# Option B: Manual setup
cd /app/modal_training
./deploy_enhanced.sh
```

### Step 3: Configure Backend
Add to `/app/backend/.env`:
```
MODAL_TOKEN_ID=your_token_id
MODAL_TOKEN_SECRET=your_token_secret
```

### Step 4: Restart Backend
```bash
sudo supervisorctl restart backend
```

## 📊 Modal Volumes Created

### `molbert-cache`
- **Purpose**: Model storage + Hugging Face cache
- **Contents**: 
  - `/cache/huggingface/` - HF model cache
  - `/cache/model_info.json` - Model metadata
- **Persistence**: Permanent storage across deployments

### `molbert-training`
- **Purpose**: Fine-tuned model storage
- **Contents**: Target-specific fine-tuned models
- **Structure**: `/training/{target}_molbert_finetuned/`

## 🧪 Testing the Setup

### Test Modal Status
```bash
curl http://localhost:8001/api/modal/molbert/status
```

### Test Model Setup (if not done automatically)
```bash
curl -X POST http://localhost:8001/api/modal/molbert/setup
```

### Test Prediction
```bash
curl -X POST "http://localhost:8001/api/modal/molbert/predict" \
     -H "Content-Type: application/json" \
     -d '{"smiles": "CCO", "target": "EGFR", "use_finetuned": false}'
```

### Start Fine-tuning
```bash
curl -X POST "http://localhost:8001/api/modal/molbert/train/EGFR"
```

## 🔄 Workflow

### First Time Setup
1. Modal CLI authentication
2. Deploy Modal app (`modal_molbert_enhanced.py`)
3. Download pretrained model (creates `molbert-cache`)
4. Set backend environment variables
5. Restart backend

### Regular Usage
1. **Predictions**: Use `/api/modal/molbert/predict`
   - Automatically uses cached pretrained model
   - Falls back to local prediction if Modal unavailable

2. **Fine-tuning**: Use `/api/modal/molbert/train/{target}`
   - Uses A100 GPU for fast training
   - Saves to `molbert-training` volume
   - Future predictions can use fine-tuned model

### Model Lifecycle
- **Pretrained**: Downloaded once, cached permanently
- **Fine-tuned**: Created per target, stored in training volume
- **Predictions**: Auto-select best available model

## 💡 Key Features

### Intelligent Fallback
- Modal unavailable → Local heuristic prediction
- Fine-tuned model missing → Use pretrained model
- All errors gracefully handled

### Cost Optimization
- Models cached permanently (no re-downloading)
- GPU usage only during training/prediction
- Efficient resource management

### Scalability
- Multiple targets supported
- Parallel training possible
- Volume-based persistence

## 🔗 Monitoring

- **Modal Dashboard**: https://modal.com/apps
- **App Name**: `molbert-enhanced`
- **Functions**: 
  - `download_pretrained_molbert`
  - `train_molbert_with_cache`
  - `predict_with_cached_model`
  - `get_model_info`

## 🚨 Troubleshooting

### Common Issues

1. **"Modal credentials not available"**
   - Set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in backend/.env
   - Restart backend

2. **"Model not cached"**
   - Run setup: `POST /api/modal/molbert/setup`
   - Or manually: `modal run modal_molbert_enhanced.py::download_pretrained_molbert`

3. **Import errors**
   - Ensure `modal` package installed (not `modal-client`)
   - Check `/app/modal_training` in Python path

4. **Volume not found**
   - Re-run deployment script
   - Volumes are created automatically on first use

### Debug Commands
```bash
# Check Modal authentication
modal token verify

# List Modal apps
modal app list

# Check volumes
modal volume list

# Get model info
modal run modal_molbert_enhanced.py::get_model_info
```

## 🎉 Success Indicators

✅ **Backend logs show**: "Enhanced Modal MolBERT integration loaded"  
✅ **Status endpoint returns**: `modal_available: true` (when credentials set)  
✅ **Volumes exist**: `molbert-cache`, `molbert-training`  
✅ **Model cached**: `model_info.json` exists with pretrained model details  
✅ **Predictions work**: Both Modal and fallback predictions functional  

---

**Your Modal MolBERT infrastructure is now ready for production use! 🚀**