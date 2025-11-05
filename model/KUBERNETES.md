# Kubernetes Deployment Guide

This guide explains how to prepare and deploy the model training for distributed training on Kubernetes using PyTorch DDP (Distributed Data Parallel).

## Current State

✅ **Ready for Kubernetes:**
- Code is DDP-friendly (stateless functions, clean loops)
- All state passed explicitly (no globals)
- Deterministic training with seeds
- Config-driven design

❌ **Not Yet Implemented:**
- PyTorch DDP wrapper code
- Kubernetes manifests (Job, Service, ConfigMap)
- Docker containerization
- Distributed storage setup

---

## What Needs to Be Added for Kubernetes

### 1. Add Distributed Training Support

Create `src/distributed.py`:

```python
"""
Distributed training utilities for PyTorch DDP
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed():
    """Initialize distributed training environment."""
    if 'WORLD_SIZE' in os.environ:
        # Running in distributed mode
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU/CPU training
        return 0, 1, 0

def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    """Check if current process is main."""
    return rank == 0

def wrap_model_ddp(model, device, local_rank):
    """Wrap model with DDP."""
    if dist.is_initialized():
        return DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    return model

def get_distributed_sampler(dataset, world_size, rank, shuffle=True):
    """Create distributed sampler for dataset."""
    if dist.is_initialized():
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
    return None
```

**Modify `src/train.py`:**
- Add distributed sampler support
- Only save checkpoints on rank 0
- Synchronize metrics across processes

**Modify `src/cli.py`:**
- Call `setup_distributed()` at start
- Wrap model with DDP
- Only main process saves artifacts

---

### 2. Create Dockerfile

Create `Dockerfile` in model directory:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app

# Default command (override in k8s)
CMD ["python", "src/cli.py"]
```

**Build and push:**
```bash
docker build -t your-registry/sp500-trainer:latest .
docker push your-registry/sp500-trainer:latest
```

---

### 3. Create Kubernetes Manifests

Create `k8s/` directory with manifests:

#### a) ConfigMap for data and config (`k8s/configmap.yaml`)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-config
data:
  config.yaml: |
    data:
      npz_path: "/mnt/data/sp500_classification.npz"
      valid_from_train: 0.1
      shuffle_train: false
      norm: "zscore"
    
    model:
      kind: "lstm"
      hidden: 64
      layers: 1
      dropout: 0.1
    
    train:
      epochs: 25
      batch_size: 64
      lr: 0.001
      early_stop_patience: 5
      seed: 42
      device: "auto"
    
    eval:
      threshold: 0.5
      save_cm: true
    
    io:
      save_dir: "/mnt/output"
      run_name: null
```

#### b) Persistent Volume for data (`k8s/pvc.yaml`)

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-pvc
spec:
  accessModes:
    - ReadOnlyMany  # Multiple pods read same data
  resources:
    requests:
      storage: 1Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-output-pvc
spec:
  accessModes:
    - ReadWriteMany  # Multiple pods write outputs
  resources:
    requests:
      storage: 5Gi
```

#### c) PyTorch Distributed Training Job (`k8s/pytorch-job.yaml`)

Using Kubeflow PyTorchJob:

```yaml
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: sp500-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: your-registry/sp500-trainer:latest
              args:
                - --config
                - /config/config.yaml
              env:
                - name: NCCL_DEBUG
                  value: "INFO"
              volumeMounts:
                - name: training-data
                  mountPath: /mnt/data
                  readOnly: true
                - name: training-output
                  mountPath: /mnt/output
                - name: config
                  mountPath: /config
              resources:
                limits:
                  nvidia.com/gpu: 1  # 1 GPU per pod
                  memory: "8Gi"
                  cpu: "4"
          volumes:
            - name: training-data
              persistentVolumeClaim:
                claimName: training-data-pvc
            - name: training-output
              persistentVolumeClaim:
                claimName: training-output-pvc
            - name: config
              configMap:
                name: training-config
    
    Worker:
      replicas: 3  # 3 worker nodes
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: your-registry/sp500-trainer:latest
              args:
                - --config
                - /config/config.yaml
              env:
                - name: NCCL_DEBUG
                  value: "INFO"
              volumeMounts:
                - name: training-data
                  mountPath: /mnt/data
                  readOnly: true
                - name: training-output
                  mountPath: /mnt/output
                - name: config
                  mountPath: /config
              resources:
                limits:
                  nvidia.com/gpu: 1
                  memory: "8Gi"
                  cpu: "4"
          volumes:
            - name: training-data
              persistentVolumeClaim:
                claimName: training-data-pvc
            - name: training-output
              persistentVolumeClaim:
                claimName: training-output-pvc
            - name: config
              configMap:
                name: training-config
```

---

### 4. Data Upload to Kubernetes

Create a script to upload data (`scripts/upload_data.sh`):

```bash
#!/bin/bash
# Upload training data to Kubernetes PVC

DATA_FILE="../data-pipeline/data/processed/sp500_classification.npz"
PVC_NAME="training-data-pvc"
NAMESPACE="default"

# Create a temporary pod to upload data
kubectl run data-uploader \
  --image=busybox \
  --restart=Never \
  --namespace=$NAMESPACE \
  --overrides='
{
  "spec": {
    "containers": [{
      "name": "uploader",
      "image": "busybox",
      "command": ["sleep", "3600"],
      "volumeMounts": [{
        "name": "data",
        "mountPath": "/data"
      }]
    }],
    "volumes": [{
      "name": "data",
      "persistentVolumeClaim": {
        "claimName": "'$PVC_NAME'"
      }
    }]
  }
}'

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/data-uploader --timeout=60s

# Copy data file
kubectl cp $DATA_FILE data-uploader:/data/sp500_classification.npz

# Cleanup
kubectl delete pod data-uploader

echo "Data uploaded to PVC: $PVC_NAME"
```

---

### 5. Deployment Steps

1. **Install Kubeflow Training Operator** (for PyTorchJob):
```bash
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone"
```

2. **Create namespace**:
```bash
kubectl create namespace ml-training
```

3. **Apply manifests**:
```bash
kubectl apply -f k8s/configmap.yaml -n ml-training
kubectl apply -f k8s/pvc.yaml -n ml-training
```

4. **Upload data**:
```bash
bash scripts/upload_data.sh
```

5. **Submit training job**:
```bash
kubectl apply -f k8s/pytorch-job.yaml -n ml-training
```

6. **Monitor training**:
```bash
# Watch job status
kubectl get pytorchjob sp500-training -n ml-training -w

# View logs from master
kubectl logs -f sp500-training-master-0 -n ml-training

# View logs from workers
kubectl logs -f sp500-training-worker-0 -n ml-training
```

7. **Retrieve results**:
```bash
# Create pod to download results
kubectl run result-downloader \
  --image=busybox \
  --restart=Never \
  -n ml-training \
  --overrides='...'  # Similar to upload

kubectl cp result-downloader:/mnt/output ./results/
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 Kubernetes Cluster                   │
│                                                       │
│  ┌──────────────┐      ┌──────────────┐            │
│  │   Master-0   │◄────►│  Worker-0    │            │
│  │   (Rank 0)   │      │  (Rank 1)    │            │
│  └──────┬───────┘      └──────┬───────┘            │
│         │                     │                      │
│         ├─────────────────────┼──────────┐          │
│         │                     │          │           │
│  ┌──────▼───────┐      ┌─────▼──────┐ ┌▼────────┐ │
│  │  Worker-1    │      │ Worker-2   │ │  ...    │ │
│  │  (Rank 2)    │      │ (Rank 3)   │ │         │ │
│  └──────┬───────┘      └─────┬──────┘ └─────────┘ │
│         │                     │                      │
│         └─────────┬───────────┘                     │
│                   │                                  │
│         ┌─────────▼─────────┐                       │
│         │   Shared Storage   │                       │
│         │  (PVC - Data/Out)  │                       │
│         └────────────────────┘                       │
└─────────────────────────────────────────────────────┘
```

---

## Scaling Considerations

**Current batch size:** 64
**With 4 GPUs:** Effective batch size = 64 × 4 = 256

Adjust learning rate when scaling:
- **Linear scaling rule:** `lr_new = lr_base × num_gpus`
- Example: `lr: 0.001` → `lr: 0.004` for 4 GPUs

**In ConfigMap:**
```yaml
train:
  batch_size: 64    # Per-GPU batch size
  lr: 0.004         # Scaled for 4 GPUs
```

---

## Testing Locally Before K8s

Test multi-GPU on single machine:
```bash
# 2 GPUs
torchrun --nproc_per_node=2 src/cli.py --config config.yaml

# CPU simulation (2 processes)
torchrun --nproc_per_node=2 --master_port=29500 src/cli.py --config config.yaml
```

---

## Summary of Required Changes

### Code Changes:
1. ✅ Add `src/distributed.py` (DDP utilities)
2. ✅ Modify `src/train.py` (distributed training loop)
3. ✅ Modify `src/cli.py` (DDP initialization)
4. ✅ Modify `src/datamod.py` (distributed sampler)

### Infrastructure:
1. ✅ Create `Dockerfile`
2. ✅ Create `k8s/configmap.yaml`
3. ✅ Create `k8s/pvc.yaml`
4. ✅ Create `k8s/pytorch-job.yaml`
5. ✅ Create `scripts/upload_data.sh`

### Deployment:
1. ✅ Build and push Docker image
2. ✅ Install Kubeflow Training Operator
3. ✅ Create PVCs and upload data
4. ✅ Submit PyTorchJob
5. ✅ Monitor and retrieve results

---

## Next Steps

Would you like me to implement:
1. **Distributed training code** (`src/distributed.py` + modifications)?
2. **Docker and K8s manifests** (complete k8s deployment files)?
3. **Both** (full K8s-ready implementation)?

Let me know and I'll generate the complete implementation!
