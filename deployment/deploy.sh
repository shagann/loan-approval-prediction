#!/bin/bash

set -e

echo "Starting deployment for Loan Approval Prediction API..."

cd /opt/mlops/loan-approval-prediction

echo "Pulling latest code from GitHub..."
git pull origin main

echo "Building Docker image..."
docker build --network=host -t loan-approval-app .

echo "Stopping old container if it exists..."
docker stop loan-approval-app || true

echo "Removing old container if it exists..."
docker rm loan-approval-app || true

echo "Starting new Docker container..."
docker run -d \
  --name loan-approval-app \
  --restart unless-stopped \
  -p 80:5000 \
  -e APP_VERSION=manual-deploy-script \
  -e GIT_COMMIT=$(git rev-parse --short HEAD) \
  -e BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  -e DEPLOYMENT_ENVIRONMENT=google-vm-docker-bridge \
  loan-approval-app

echo "Waiting for app to start..."
sleep 10

echo "Running health check..."
curl -f http://localhost/health

echo ""
echo "Running version check..."
curl -f http://localhost/version

echo ""
echo "Deployment completed successfully."