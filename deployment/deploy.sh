#!/bin/bash

# Stop the deployment if any command fails.
set -e

# Deployment settings.
APP_DIR="/opt/mlops/loan-approval-prediction"
APP_NAME="loan-approval-app"
IMAGE_NAME="loan-approval-app:latest"

# This controls which versioned model package the Flask app loads.
# The drift/retrain workflow can update this to model-v2-candidate after approval.
MODEL_VERSION="model-v2-candidate"

echo "Starting deployment for Loan Approval Prediction API..."
echo "App directory: ${APP_DIR}"
echo "Docker image: ${IMAGE_NAME}"
echo "Docker container: ${APP_NAME}"
echo "Selected model version: ${MODEL_VERSION}"

# Move into the app repository on the VM.
cd "${APP_DIR}"

# Pull latest app code, model packages and deployment config from GitHub.
echo "Pulling latest code from GitHub..."
git pull origin main

# Build a fresh Docker image from the current repository state.
echo "Building Docker image..."
docker build -t "${IMAGE_NAME}" .

# Stop and remove the previous container if it exists.
echo "Stopping old container if it exists..."
docker stop "${APP_NAME}" || true

echo "Removing old container if it exists..."
docker rm "${APP_NAME}" || true

# Start the new container.
# Port 80 on the VM maps to Flask port 5000 inside the container.
# The logs directory is mounted so prediction logs survive redeploys.
# MODEL_VERSION tells app.py to load models/<MODEL_VERSION>/.
echo "Starting new Docker container..."
docker run -d \
  --name "${APP_NAME}" \
  --restart unless-stopped \
  -p 80:5000 \
  -v /opt/mlops/loan-approval-prediction/logs:/app/logs \
  -e PORT=5000 \
  -e APP_VERSION=manual-deploy-script \
  -e MODEL_VERSION="${MODEL_VERSION}" \
  -e GIT_COMMIT=$(git rev-parse --short HEAD) \
  -e BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  -e DEPLOYMENT_ENVIRONMENT=google-vm-docker-bridge \
  "${IMAGE_NAME}"

# Give the Flask app time to start.
echo "Waiting for app to start..."
sleep 10

# Health check proves the container is running.
echo "Running health check..."
curl -f http://localhost/health

echo ""

# Version check proves which app build and model package are deployed.
echo "Running version check..."
curl -f http://localhost/version

echo ""
echo "Deployment completed successfully."