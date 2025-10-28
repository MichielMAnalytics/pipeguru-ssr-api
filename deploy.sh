#!/bin/bash
set -e

# Pipeguru SSR API Cloud Run Deployment Script
# This script deploys the FastAPI backend to GCP Cloud Run using .env.production

echo "🚀 Starting pipeguru-ssr-api deployment to Cloud Run..."

# Load environment variables from .env.production
if [ ! -f .env.production ]; then
    echo "❌ Error: .env.production file not found!"
    exit 1
fi

# Parse .env.production and prepare for Cloud Run
# This extracts all non-commented, non-empty lines
ENV_VARS=$(grep -v '^#' .env.production | grep -v '^$' | sed 's/^/--set-env-vars /' | tr '\n' ' ')

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"advertising-475814"}
REGION=${GCP_REGION:-"europe-west1"}
SERVICE_NAME="pipeguru-ssr-api"
SERVICE_ACCOUNT="pipeguru-storage@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "📦 Configuration:"
echo "  Project: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Service: ${SERVICE_NAME}"
echo "  Service Account: ${SERVICE_ACCOUNT}"
echo "  Image: ${IMAGE_NAME}"
echo ""

# Build and push Docker image with BuildKit for better caching
echo "🏗️  Building Docker image for linux/amd64 with caching..."
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -t ${IMAGE_NAME}:latest .

echo "📤 Pushing image to Google Container Registry..."
docker push ${IMAGE_NAME}:latest

# Deploy to Cloud Run
echo "🚢 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --project ${PROJECT_ID} \
  --service-account ${SERVICE_ACCOUNT} \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 600 \
  --concurrency 80 \
  ${ENV_VARS}

echo "✅ Deployment complete!"
echo ""
echo "🌐 Service URL:"
gcloud run services describe ${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --project ${PROJECT_ID} \
  --format 'value(status.url)'
echo ""
echo "📝 Next steps:"
echo "1. Update the main dashboard's .env.production with SSR_API_URL pointing to the service URL above"
echo "2. Redeploy the main dashboard to use the production SSR API"
