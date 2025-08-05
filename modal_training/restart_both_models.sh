#!/bin/bash

echo "🚀 RESTARTING BOTH MODELS WITH IMPROVED CONFIGURATIONS"
echo "=" * 80

echo "📋 Current Model Status:"
modal app list | grep -E "(model1|model2)" | tail -5

echo ""
echo "🔧 Starting Model 2 (Cancer-Only) - Target >0.6 R²..."
modal run --detach model2_cancer_only_training.py &
MODEL2_PID=$!

echo "🔧 Starting Model 1 (Checkpointed) - Resume from 0.5994 R²..."  
modal run --detach model1_checkpointed_training.py &
MODEL1_PID=$!

echo ""
echo "⏳ Waiting for initial startup (30 seconds)..."
sleep 30

echo ""
echo "📊 Updated Model Status:"
modal app list | grep -E "(model1|model2)" | tail -5

echo ""
echo "✅ Both models started in detached mode with:"
echo "  • Model 2: Cancer-only IC50 prediction (simplified architecture)"
echo "  • Model 1: Checkpointed training (proven architecture that reached R²=0.5994)"
echo "  • Epoch-by-epoch checkpointing for both models"
echo "  • --detach flag to prevent client disconnection"

echo ""
echo "🎯 TRAINING TARGETS:"
echo "  • Model 2: Cancer IC50 R² > 0.6 (removing normal cells complexity)"
echo "  • Model 1: Ligand Activity R² > 0.6 (building on previous success)"

echo ""
echo "📈 Monitor progress with:"
echo "  modal app list"
echo "  modal app logs <APP_ID>"