#!/bin/bash

# RunPod Deployment Script for MolBERT Training
# Usage: ./deploy_to_runpod.sh

set -e

echo "🚀 Deploying MolBERT Training to RunPod"

# Configuration
RUNPOD_API_KEY="${RUNPOD_API_KEY}"
IMAGE_NAME="veridica-ai/molbert-training"
POD_TYPE="NVIDIA RTX 4090"  # or "NVIDIA A100" for faster training

# Check if API key is set
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "❌ Error: RUNPOD_API_KEY environment variable not set"
    echo "Get your API key from: https://www.runpod.io/console/user/settings"
    exit 1
fi

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -t $IMAGE_NAME .

echo "⬆️ Pushing to Docker Hub..."
docker push $IMAGE_NAME

# Create RunPod instance via API
echo "🏗️ Creating RunPod instance..."
curl -X POST \
  https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mutation { podRentInterruptable(input: { bidPerGpu: 0.15, cloudType: SECURE, containerDiskInGb: 50, dockerArgs: \"\", env: [{ key: \"PYTHONPATH\", value: \"/app\" }], gpuCount: 1, gpuTypeId: \"NVIDIA RTX 4090\", imageName: \"'$IMAGE_NAME'\", name: \"molbert-training\", ports: \"8888/http\", templateId: \"runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04\", volumeInGb: 0, volumeMountPath: \"/workspace\" }) { id costPerHr machine { podHostId } } }"
  }'

echo ""
echo "✅ Deployment initiated!"
echo "🔗 Check your pods at: https://www.runpod.io/console/pods"
echo "📊 Monitor training progress via the webhook URL configured in training_config.json"