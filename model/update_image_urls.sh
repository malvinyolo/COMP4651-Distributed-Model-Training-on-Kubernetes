#!/bin/bash
# Script to update Kubernetes manifests with your ECR image URL

# Get AWS account ID and region
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)

if [ -z "$REGION" ]; then
    REGION="us-east-1"
fi

echo "AWS Account ID: $ACCOUNT_ID"
echo "AWS Region: $REGION"
echo ""

# ECR image URL
ECR_IMAGE="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/stock-trainer:latest"
echo "ECR Image URL: $ECR_IMAGE"
echo ""

# Update all YAML files to use ECR image
echo "Updating Kubernetes manifests..."

# Update hyperparam-search-job.yaml
sed -i.bak "s|image: stock-trainer:latest|image: $ECR_IMAGE|g" k8s/hyperparam-search-job.yaml
sed -i.bak "s|imagePullPolicy: IfNotPresent|imagePullPolicy: Always|g" k8s/hyperparam-search-job.yaml

# Update large-dataset-job.yaml
sed -i.bak "s|image: stock-trainer:latest|image: $ECR_IMAGE|g" k8s/large-dataset-job.yaml
sed -i.bak "s|imagePullPolicy: IfNotPresent|imagePullPolicy: Always|g" k8s/large-dataset-job.yaml

# Remove backup files
rm k8s/*.bak 2>/dev/null

echo "✅ Manifests updated to use ECR image"
echo ""
echo "Next steps:"
echo "1. Build and push Docker image:"
echo "   docker build -t stock-trainer:latest ."
echo "   docker tag stock-trainer:latest $ECR_IMAGE"
echo "   aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
echo "   docker push $ECR_IMAGE"
echo ""
echo "2. Deploy to Kubernetes:"
echo "   ./deploy.sh"
