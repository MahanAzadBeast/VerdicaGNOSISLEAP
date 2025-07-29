#!/bin/bash

# Enhanced Modal MolBERT Deployment Script
# Sets up molbert-cache volume and deploys enhanced Modal functions

set -e  # Exit on any error

echo "🚀 Enhanced Modal MolBERT Deployment"
echo "===================================="
echo ""

# Check if Modal CLI is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found!"
    echo "💡 Install with: pip install modal-client"
    exit 1
fi

# Check if Modal is authenticated
if ! modal token verify &> /dev/null; then
    echo "❌ Modal not authenticated!"
    echo "💡 Run: modal token new"
    echo "🔗 Get token at: https://modal.com/settings/tokens"
    exit 1
fi

echo "✅ Modal CLI ready"
echo ""

# Deploy the enhanced Modal app
echo "📦 Deploying Enhanced MolBERT App..."
echo "-----------------------------------"

cd /app/modal_training

if [ -f "modal_molbert_enhanced.py" ]; then
    echo "🔧 Deploying modal_molbert_enhanced.py..."
    
    # Deploy the app
    modal deploy modal_molbert_enhanced.py
    
    if [ $? -eq 0 ]; then
        echo "✅ App deployed successfully!"
    else
        echo "❌ App deployment failed!"
        exit 1
    fi
else
    echo "❌ modal_molbert_enhanced.py not found!"
    exit 1
fi

echo ""
echo "🎯 Setting up pretrained model cache..."
echo "-------------------------------------"

# Download and cache the pretrained model
echo "📥 Downloading seyonec/ChemBERTa-zinc-base-v1..."
modal run modal_molbert_enhanced.py::download_pretrained_molbert

if [ $? -eq 0 ]; then
    echo "✅ Pretrained model cached successfully!"
else
    echo "❌ Model caching failed!"
    exit 1
fi

echo ""
echo "🧪 Testing model setup..."
echo "------------------------"

# Test model info
echo "📊 Getting model information..."
modal run modal_molbert_enhanced.py::get_model_info

echo ""
echo "🎉 Enhanced Modal MolBERT Setup Complete!"
echo "========================================"
echo ""
echo "📋 Next steps:"
echo "  1. Set Modal credentials in backend/.env:"
echo "     MODAL_TOKEN_ID=your_token_id"
echo "     MODAL_TOKEN_SECRET=your_token_secret"
echo ""
echo "  2. Test prediction:"
echo "     modal run modal_molbert_enhanced.py::predict_with_cached_model --smiles 'CCO' --target 'EGFR'"
echo ""
echo "  3. Start fine-tuning:"
echo "     modal run modal_molbert_enhanced.py::train_molbert_with_cache --target 'EGFR'"
echo ""
echo "🔗 Monitor at: https://modal.com/apps"
echo "📊 Volumes created:"
echo "  • molbert-cache (model storage + HF cache)"
echo "  • molbert-training (fine-tuned models)"
echo ""