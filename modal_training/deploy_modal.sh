#!/bin/bash

# Modal MolBERT Training Deployment Script
# Usage: ./deploy_modal.sh [target] [webhook_url]

set -e

echo "🚀 Deploying MolBERT Training to Modal.com"

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    pip install modal
fi

# Check authentication
if ! modal token list &> /dev/null; then
    echo "🔑 Setting up Modal authentication..."
    echo "Please authenticate with Modal:"
    modal token new
fi

# Configuration
TARGET=${1:-"EGFR"}
WEBHOOK_URL=${2:-""}
ALL_TARGETS=${3:-false}

echo "📊 Configuration:"
echo "  Target: $TARGET"
echo "  Webhook: $WEBHOOK_URL"  
echo "  All Targets: $ALL_TARGETS"

# Deploy the app
echo "📦 Deploying Modal app..."
if [ "$ALL_TARGETS" = "true" ]; then
    echo "🎯 Training all targets (EGFR, BRAF, CDK2, PARP1, BCL2, VEGFR2)"
    if [ -n "$WEBHOOK_URL" ]; then
        modal run modal_molbert.py --all-targets --webhook-url "$WEBHOOK_URL"
    else
        modal run modal_molbert.py --all-targets
    fi
else
    echo "🎯 Training single target: $TARGET"
    if [ -n "$WEBHOOK_URL" ]; then
        modal run modal_molbert.py --target "$TARGET" --webhook-url "$WEBHOOK_URL"
    else
        modal run modal_molbert.py --target "$TARGET"
    fi
fi

echo ""
echo "✅ Modal deployment completed!"
echo "🔗 Monitor your jobs at: https://modal.com/apps"
echo "📊 Training progress will be sent to webhook if configured"
echo "💾 Models will be saved to Modal volume and available for download"

# Instructions for downloading models
echo ""
echo "📥 To download trained models later:"
echo "modal run modal_molbert.py::download_trained_model --target $TARGET"