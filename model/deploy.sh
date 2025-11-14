#!/bin/bash
# Deploy Stock Training to Kubernetes
# This script sets up the complete Kubernetes environment for distributed training

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Stock Training - Kubernetes Deployment"
echo "=========================================="

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}✗ kubectl not found. Please install kubectl first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl found${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ docker not found. Please install docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ docker found${NC}"

# Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}✗ Cannot connect to Kubernetes cluster${NC}"
    echo "Please configure kubectl to connect to your cluster"
    exit 1
fi
echo -e "${GREEN}✓ Connected to Kubernetes cluster${NC}"

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
K8S_DIR="$SCRIPT_DIR/k8s"

# Step 1: Create namespace
echo -e "\n${YELLOW}Step 1: Creating namespace...${NC}"
kubectl apply -f "$K8S_DIR/namespace.yaml"
echo -e "${GREEN}✓ Namespace created${NC}"

# Step 2: Create PVC
echo -e "\n${YELLOW}Step 2: Creating persistent volume claim...${NC}"
kubectl apply -f "$K8S_DIR/pvc.yaml"
echo -e "${GREEN}✓ PVC created${NC}"

# Wait for PVC to be bound
echo -e "${YELLOW}Waiting for PVC to be bound...${NC}"
kubectl wait --for=condition=bound pvc/stock-data-pvc -n stock-training --timeout=120s
echo -e "${GREEN}✓ PVC bound${NC}"

# Step 3: Build and push Docker image (if needed)
echo -e "\n${YELLOW}Step 3: Checking Docker image...${NC}"
IMAGE_NAME="stock-trainer:latest"

if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker build -t $IMAGE_NAME "$SCRIPT_DIR"
    echo -e "${GREEN}✓ Image built${NC}"
else
    echo -e "${GREEN}✓ Image already exists${NC}"
fi

# If using a remote registry, push the image
# Uncomment and modify if you're using a remote registry
# REGISTRY="your-registry.com"
# docker tag $IMAGE_NAME $REGISTRY/$IMAGE_NAME
# docker push $REGISTRY/$IMAGE_NAME

# Step 4: Upload data to PVC
echo -e "\n${YELLOW}Step 4: Uploading data to PVC...${NC}"

# Create a temporary pod to upload data
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: data-uploader
  namespace: stock-training
spec:
  containers:
  - name: uploader
    image: busybox
    command: ["sleep", "3600"]
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: stock-data-pvc
EOF

# Wait for pod to be ready
kubectl wait --for=condition=ready pod/data-uploader -n stock-training --timeout=120s

# Copy data files
echo -e "${YELLOW}Copying dataset files...${NC}"
kubectl cp "$SCRIPT_DIR/../data-pipeline/data/processed/" stock-training/data-uploader:/data/

echo -e "${YELLOW}Copying config files...${NC}"
kubectl cp "$SCRIPT_DIR/config_test.yaml" stock-training/data-uploader:/data/config.yaml

# Delete uploader pod
kubectl delete pod data-uploader -n stock-training
echo -e "${GREEN}✓ Data uploaded${NC}"

# Step 5: Install Kubeflow Training Operator (if not already installed)
echo -e "\n${YELLOW}Step 5: Checking Kubeflow Training Operator...${NC}"

if ! kubectl get crd pytorchjobs.kubeflow.org &> /dev/null; then
    echo -e "${YELLOW}Installing Kubeflow Training Operator...${NC}"
    kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"
    
    echo -e "${YELLOW}Waiting for operator to be ready...${NC}"
    kubectl wait --for=condition=available deployment/training-operator -n kubeflow --timeout=300s
    echo -e "${GREEN}✓ Kubeflow Training Operator installed${NC}"
else
    echo -e "${GREEN}✓ Kubeflow Training Operator already installed${NC}"
fi

# Print summary
echo -e "\n=========================================="
echo -e "${GREEN}✓ Deployment Complete!${NC}"
echo -e "=========================================="
echo ""
echo "Your Kubernetes environment is ready for distributed training."
echo ""
echo "Next steps:"
echo "  1. Run hyperparameter search:"
echo "     kubectl apply -f $K8S_DIR/hyperparam-search-job.yaml"
echo ""
echo "  2. Run large dataset training:"
echo "     kubectl apply -f $K8S_DIR/large-dataset-job.yaml"
echo ""
echo "Monitor jobs:"
echo "  kubectl get pytorchjobs -n stock-training"
echo "  kubectl get pods -n stock-training"
echo "  kubectl logs <pod-name> -n stock-training"
echo ""
echo "Clean up:"
echo "  kubectl delete namespace stock-training"
echo "=========================================="
