# COMP4651: Distributed Model Training on Kubernetes

Demonstrating cloud capabilities that enable workloads **impossible on a single local machine**.

## 🎯 Project Overview

This project demonstrates **capability enablement through cloud computing** rather than just performance improvement. We show how Kubernetes/cloud enables two key scenarios impossible or impractical on local machines:

1. **Dataset Scale**: Training on datasets exceeding local memory limits
2. **Parallel Experimentation**: Running multiple hyperparameter searches simultaneously

### Key Components:
- **Data Pipeline**: ETL for S&P 500 historical data + dataset scaling utilities
- **Model**: Simple MLP regressor for time-series prediction
- **Experiments**: Local vs cloud capability comparisons
- **Goal**: Show cloud enables what's impossible locally (not just faster)

## 📁 Repository Structure

```
├── data-pipeline/       # Data extraction, processing, and dataset creation
│   ├── src/            # Pipeline modules
│   ├── data/           # Raw and processed datasets
│   ├── create_large_dataset.py  # Scale datasets for testing
│   └── requirements.txt
│
├── model/              # Regression model training and experiments
│   ├── src/            # Training code (single-machine, distributed)
│   ├── experiments/    # Comparison scripts (local vs cloud)
│   ├── k8s/            # Kubernetes manifests for deployment
│   ├── outputs/        # Training artifacts and logs
│   ├── deploy.sh       # One-command Kubernetes deployment
│   ├── COMPARISON_GUIDE.md  # Detailed methodology
│   └── requirements.txt
│
└── README.md           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (for Kubernetes deployment)
- kubectl (for Kubernetes deployment)
- Access to a Kubernetes cluster (for cloud experiments)

### 1. Setup Data Pipeline

```bash
cd data-pipeline
pip install -r requirements.txt

# Run pipeline to generate base dataset
python -m src.run_pipeline

# Create scaled datasets for testing
python create_large_dataset.py --multiplier 10   # 10x larger
python create_large_dataset.py --multiplier 50   # 50x larger  
python create_large_dataset.py --multiplier 200  # 200x larger
```

### 2. Run Local Experiments (Baseline)

```bash
cd ../model
pip install -r requirements.txt

# Test local memory limits with progressively larger datasets
cd experiments
python test_scale_local.py

# Run sequential hyperparameter search (baseline)
python hyperparam_search_local.py
```

**Expected Results**:
- Scale test: Local succeeds up to ~10x-50x, then OOM
- Hyperparameter search: ~6 minutes for 18 configurations

### 3. Deploy to Kubernetes

```bash
cd ..  # Back to model/ directory

# One-command deployment (creates namespace, PVC, uploads data)
chmod +x deploy.sh
./deploy.sh
```

### 4. Run Cloud Experiments

```bash
# Run large dataset training (200x multiplier)
kubectl apply -f k8s/large-dataset-job.yaml

# Run parallel hyperparameter search
python experiments/hyperparam_search_cloud.py
```

**Expected Results**:
- Large dataset: Success where local failed (OOM)
- Hyperparameter search: ~20 seconds (18x speedup)

### 5. Compare Results

```bash
# Generate comparison report
python experiments/compare_results.py
```

This shows:
- Dataset scale: Local OOM vs cloud success
- Hyperparameter search: Sequential 6min vs parallel 20s
- Cloud capability summary

## 📊 What This Demonstrates

### Comparison 1: Dataset Scale (Memory Limits)

**Local Machine** (Single Node):
- ✓ 1x dataset (860 samples, ~300KB): Success
- ✓ 10x dataset (8,600 samples, ~3MB): Success
- ✗ 50x dataset (43,000 samples, ~15MB): **Out of Memory**

**Cloud** (Kubernetes, 4 workers):
- ✓ 1x, 10x, 50x datasets: Success
- ✓ 200x dataset (172,000 samples, ~60MB): **Success**
- Data distributed across workers

**Key Insight**: Cloud enables training on datasets **4x-20x larger** than local capacity through distributed memory.

### Comparison 2: Hyperparameter Search (Parallelism)

**Local Machine** (Sequential):
- 18 hyperparameter combinations
- Run one at a time: ~20s each
- Total time: **~6 minutes**

**Cloud** (Kubernetes, Parallel):
- 18 hyperparameter combinations  
- Run all simultaneously: 18 workers
- Total time: **~20 seconds** (18x speedup)

**Key Insight**: Cloud enables **rapid experimentation** through parallel execution, transforming research workflow.

## 🎓 Course Information

**Course**: COMP 4651 - Cloud Computing and Big Data Systems  
**Semester**: Fall 2025  
**Focus**: Demonstrating cloud capability enablement (what's impossible locally but possible in cloud)

## 📝 Technical Details

### Why These Comparisons Matter

**Traditional Performance Benchmarking**:
- Compares same workload on different hardware
- Shows X% speedup with more resources
- Not meaningful for small datasets on CPU

**Our Approach - Capability Enablement**:
- Shows workloads that **cannot run locally** but **succeed in cloud**
- Demonstrates qualitative differences, not just quantitative
- Highlights cloud's true value: enabling impossible workloads

### Architecture

**Local Training**:
```
[Single Machine]
├── Limited RAM (8-16GB typical)
├── Single CPU/GPU
└── Sequential execution

Constraints:
- Dataset must fit in memory
- One experiment at a time
```

**Distributed Training (Kubernetes)**:
```
[Kubernetes Cluster]
├── [Master Node] - Coordinates
├── [Worker 1] - Data shard 1
├── [Worker 2] - Data shard 2
└── [Worker N] - Data shard N

Capabilities:
- Pooled memory across workers
- Data parallelism
- Multiple experiments in parallel
```

### Technologies Used

- **PyTorch**: Deep learning framework with DDP support
- **Kubernetes**: Container orchestration for distributed training
- **Kubeflow Training Operator**: PyTorchJob CRD for distributed training
- **Docker**: Containerization of training environment
- **yfinance**: S&P 500 data collection

## 📚 Documentation

- **[COMPARISON_GUIDE.md](model/COMPARISON_GUIDE.md)**: Detailed methodology and step-by-step guide
- **[experiments/README.md](model/experiments/README.md)**: Experiment scripts documentation
- **[KUBERNETES.md](model/KUBERNETES.md)**: Kubernetes deployment details (if exists)

## 🔧 Development

### Data Pipeline
Located in `data-pipeline/`:
- `src/data_collector.py`: Fetch S&P 500 data via yfinance
- `src/preprocess.py`: Normalize and create sequences
- `src/run_pipeline.py`: End-to-end pipeline
- `create_large_dataset.py`: Scale datasets with noise injection

### Model Training  
Located in `model/src/`:
- `single/`: Single-machine training implementation
- `distributed/`: DDP (DistributedDataParallel) implementation
- `models.py`: MLP regressor architecture
- `datamod.py`: Data loading and preprocessing
- `metrics.py`: Evaluation metrics (MSE, MAE, R²)
- `artifacts.py`: Save checkpoints, configs, metrics

### Experiments
Located in `model/experiments/`:
- `test_scale_local.py`: Test local memory limits
- `hyperparam_search_local.py`: Sequential baseline
- `hyperparam_search_cloud.py`: Parallel cloud execution
- `compare_results.py`: Generate comparison reports

## 🐛 Troubleshooting

### Local Experiments

**OOM too early**: Reduce batch size in scripts
**Takes too long**: Normal for CPU, reduce epochs or use GPU
**Import errors**: Check `requirements.txt` installed

### Kubernetes Deployment

**PVC not binding**: Check storage class with `kubectl get sc`
**Image pull errors**: Ensure image built locally or pushed to registry
**Jobs pending**: Check resources with `kubectl describe node`
**Pod failures**: Check logs with `kubectl logs <pod-name> -n stock-training`

See [COMPARISON_GUIDE.md](model/COMPARISON_GUIDE.md) for detailed troubleshooting.

## 🔮 Future Work

- [x] Implement PyTorch DDP training
- [x] Create dataset scaling utilities
- [x] Kubernetes deployment automation
- [x] Capability comparison experiments
- [ ] Larger hyperparameter search spaces (100+ configs)
- [ ] Multi-dataset experiments (different stocks, timeframes)
- [ ] Cost analysis (cloud resources vs local hardware)
- [ ] Advanced architectures (LSTM, Transformer)
- [ ] Real-time prediction deployment

## 📄 License

Educational project for COMP 4651.