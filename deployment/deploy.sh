#!/bin/bash

# Stop the deployment if any command fails.
set -e

# Deployment settings.
APP_DIR="/opt/mlops/loan-approval-prediction"
APP_NAME="loan-approval-app"
IMAGE_NAME="loan-approval-app:latest"

# This controls which versioned model package app.py loads.
# The drift/retrain workflow can update this after the evaluation gate approves the candidate.
MODEL_VERSION="model-v2-candidate"

echo "Starting deployment for Loan Approval Prediction API..."
echo "App directory: ${APP_DIR}"
echo "Docker image: ${IMAGE_NAME}"
echo "Docker container/service: ${APP_NAME}"
echo "Expected model version: ${MODEL_VERSION}"

# Move into the app repository on the VM.
cd "${APP_DIR}"

# Force the VM repository to match origin/main.
# This avoids stale files or manual VM edits affecting deployment.
echo "Synchronising VM repository with origin/main..."
git fetch origin main
git reset --hard origin/main

echo "Current deployment model setting:"
grep "MODEL_VERSION" deployment/deploy.sh

# Export variables so docker-compose.yml can read them.
export MODEL_VERSION="${MODEL_VERSION}"
export GIT_COMMIT="$(git rev-parse --short HEAD)"
export BUILD_TIME="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

echo "Deployment environment variables:"
echo "MODEL_VERSION=${MODEL_VERSION}"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "BUILD_TIME=${BUILD_TIME}"

# Stop the existing Compose service if it exists.
echo "Stopping existing Docker Compose service if it exists..."
docker compose down || true

# Build and start the application using Docker Compose.
echo "Building and starting application using Docker Compose..."
docker compose up -d --build

# Give the Flask app time to start.
echo "Waiting for app to start..."
sleep 10

# Show running containers.
echo "Showing running containers..."
docker ps

# Check that the Docker container was started with the expected model version.
echo "Checking Docker container MODEL_VERSION..."
CONTAINER_MODEL_VERSION=$(docker inspect "${APP_NAME}" --format='{{range .Config.Env}}{{println .}}{{end}}' | grep '^MODEL_VERSION=' | cut -d '=' -f2)

echo "Container MODEL_VERSION: ${CONTAINER_MODEL_VERSION}"

if [ "${CONTAINER_MODEL_VERSION}" != "${MODEL_VERSION}" ]; then
  echo "ERROR: Container MODEL_VERSION does not match expected MODEL_VERSION."
  echo "Expected: ${MODEL_VERSION}"
  echo "Actual: ${CONTAINER_MODEL_VERSION}"
  exit 1
fi

# Health check proves the container is running.
echo "Running health check..."
curl -f http://localhost/health

echo ""

# Version check proves which app build and model package are deployed.
echo "Running version check..."
curl -f http://localhost/version

echo ""

# Check that the live /version endpoint reports the expected model version.
echo "Checking /version model_version..."
LIVE_MODEL_VERSION=$(python - <<EOF
import json
import urllib.request

with urllib.request.urlopen("http://localhost/version") as response:
    data = json.load(response)

print(data.get("model_version"))
EOF
)

echo "Live /version model_version: ${LIVE_MODEL_VERSION}"

if [ "${LIVE_MODEL_VERSION}" != "${MODEL_VERSION}" ]; then
  echo "ERROR: /version model_version does not match expected MODEL_VERSION."
  echo "Expected: ${MODEL_VERSION}"
  echo "Actual: ${LIVE_MODEL_VERSION}"
  exit 1
fi

echo ""
echo "Deployment completed successfully with MODEL_VERSION=${MODEL_VERSION}."