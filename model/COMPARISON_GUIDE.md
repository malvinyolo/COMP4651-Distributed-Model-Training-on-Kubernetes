# Cloud Capability Demonstration: Local vs Kubernetes

This document describes how to demonstrate the capabilities that cloud/Kubernetes enables which are **impossible or impractical on a single local machine**.

## Project Focus

This project demonstrates **capability enablement** rather than just performance improvement. We show two key scenarios:

1. **Dataset Scale**: Training on datasets that exceed local machine memory limits
2. **Hyperparameter Search**: Parallel execution of multiple experiments simultaneously

## Two Key Comparisons

### Comparison 1: Dataset Scale (Memory Limits)

**Hypothesis**: Local machines will fail (OOM) with large datasets, while Kubernetes can handle them through distributed training.

**Local Baseline**:
- AAPL stock dataset: ~860 samples base
- Sequence length: 60 timesteps × 5 features
- Base dataset size: ~300KB
- Expected local limit: ~10x-50x multiplier before OOM

**Cloud Capability**:
- Distribute dataset across multiple nodes
- Each node handles a portion of the data
- Can scale to 100x-200x multipliers

**How to Run**:

```bash
# 1. Test local limits
cd model/experiments
python test_scale_local.py

# This will:
# - Try progressively larger datasets (1x, 10x, 50x, 100x, 200x)
# - Track memory usage and time
# - Stop when OOM or timeout occurs
# - Save results to scale_test_local/results.json

# 2. Create large datasets
cd ../../data-pipeline
python create_large_dataset.py --multiplier 10
python create_large_dataset.py --multiplier 50
python create_large_dataset.py --multiplier 200

# 3. Deploy to Kubernetes
cd ../model
chmod +x deploy.sh
./deploy.sh

# 4. Run large dataset training on cloud
kubectl apply -f k8s/large-dataset-job.yaml

# 5. Monitor progress
kubectl get pytorchjobs -n stock-training
kubectl logs -f <pod-name> -n stock-training

# 6. Compare results
python experiments/compare_results.py
```

**Expected Results**:
- Local: Success up to ~10x-50x, then OOM
- Cloud: Success at 200x+ with distributed training
- Demonstration: Cloud enables 4x-20x larger datasets

### Comparison 2: Hyperparameter Search (Parallelism)

**Hypothesis**: Local machines must run experiments sequentially (~6 minutes for 18 configs), while Kubernetes can run all in parallel (~20 seconds).

**Local Baseline**:
- 18 hyperparameter combinations:
  - learning_rate: [1e-4, 5e-4, 1e-3]
  - hidden_size: [64, 128, 256]
  - num_layers: [2, 3]
- Sequential execution: ~20s per config = ~6 minutes total
- No parallelism: 1 config at a time

**Cloud Capability**:
- Launch all 18 configs simultaneously
- Each runs on separate worker
- Wall-clock time: ~20s (single config duration)
- Speedup: 18x (perfect parallelism)

**How to Run**:

```bash
# 1. Run local baseline (sequential)
cd model/experiments
python hyperparam_search_local.py

# This will:
# - Run all 18 configs one by one
# - Track total time and individual times
# - Save results to hp_search_local/summary.json
# Expected: ~6 minutes total

# 2. Deploy to Kubernetes (if not already done)
cd ..
./deploy.sh

# 3. Run parallel search on cloud
python experiments/hyperparam_search_cloud.py

# This will:
# - Generate 18 separate config files
# - Create 18 Kubernetes jobs
# - Launch all simultaneously
# - Wait for completion
# - Save results to hp_search_cloud/summary.json
# Expected: ~20-30 seconds total

# 4. Compare results
python experiments/compare_results.py
```

**Expected Results**:
- Local: ~6 minutes (sequential, 18 × 20s)
- Cloud: ~20 seconds (parallel, max(20s))
- Speedup: 18x (linear with parallelism)
- Demonstration: Cloud enables rapid experimentation

## Understanding the Results

### Why This Matters

**Dataset Scale Comparison**:
- Shows cloud enables training on datasets that **cannot fit** on local machine
- Not about speed, but about **capability**: local fails, cloud succeeds
- Demonstrates distributed training's value for large-scale data

**Hyperparameter Search Comparison**:
- Shows cloud enables **rapid experimentation** through parallelism
- Local: Limited by single machine → slow iteration
- Cloud: Multiple machines → fast iteration → better models
- Demonstrates value of parallel computing resources

### Reading the Output

When you run `compare_results.py`, you'll see:

```
COMPARISON 1: Dataset Scale
============================
LOCAL MACHINE RESULTS:
1x: ✓ Success (2.3s, 150MB)
10x: ✓ Success (23.1s, 1.2GB)
50x: ✗ Failed (OOM after 45s)

CLOUD RESULTS:
1x: ✓ Success (3.1s, 4 nodes)
10x: ✓ Success (8.2s, 4 nodes)
50x: ✓ Success (28.3s, 4 nodes)
200x: ✓ Success (95.1s, 4 nodes)

🎯 CLOUD ENABLES 20X LARGER DATASETS

COMPARISON 2: Hyperparameter Search
====================================
LOCAL (Sequential): 18 configs, 6.2 minutes
CLOUD (Parallel): 18 configs, 21.3 seconds

🎯 SPEEDUP: 17.5x faster
🎯 TIME SAVED: 5.9 minutes
```

## Architecture

### Local Training
```
[Single Machine]
├── Load full dataset into memory
├── Train model (single GPU/CPU)
└── Limited by: RAM, storage, single processor

Constraints:
- Dataset must fit in memory
- One experiment at a time
- Total time = sum of all experiments
```

### Distributed Training
```
[Kubernetes Cluster]
├── [Master Node]
│   └── Coordinates training
├── [Worker 1]
│   └── Trains on data shard 1
├── [Worker 2]
│   └── Trains on data shard 2
└── [Worker N]
    └── Trains on data shard N

Advantages:
- Dataset sharded across workers
- Parallel gradient computation
- Combined memory pool
- Multiple experiments in parallel
```

## Technical Implementation

### Dataset Scaling
- Base dataset: 860 samples
- Multiplier: Repeats dataset with added noise
- Noise: Gaussian (0, 0.01) to prevent exact duplicates
- Sizes: 1x, 10x, 50x, 100x, 200x, 1000x

### Hyperparameter Search
- Grid search over 3 dimensions
- Total: 18 combinations
- Local: Python loop (sequential)
- Cloud: 18 separate PyTorchJobs (parallel)

### Resource Allocation
```yaml
# Per worker resources
resources:
  requests:
    cpu: "1"
    memory: "2Gi"
  limits:
    cpu: "2"
    memory: "4Gi"

# Large dataset job: 4 workers = 8-16Gi total
# Hyperparameter search: 18 workers = 36-72Gi total
```

## Troubleshooting

### Local OOM Issues
If local tests OOM too early:
- Reduce batch size in config
- Start with smaller multipliers
- Monitor with `htop` or Activity Monitor

### Kubernetes Issues
```bash
# Check pod status
kubectl get pods -n stock-training

# Check logs
kubectl logs <pod-name> -n stock-training

# Check resource usage
kubectl top pods -n stock-training

# Debug PVC
kubectl describe pvc stock-data-pvc -n stock-training
```

### Common Problems

1. **PVC not binding**: Check storage class availability
2. **Image pull errors**: Ensure image is available (use local or push to registry)
3. **Jobs stuck pending**: Check resource availability (`kubectl describe node`)
4. **OOM in cloud**: Increase worker memory limits

## Measuring Success

### Key Metrics

**Dataset Scale**:
- ✅ Local fails at Nx multiplier (OOM)
- ✅ Cloud succeeds at Mx multiplier where M >> N
- ✅ Ratio M/N shows capability improvement

**Hyperparameter Search**:
- ✅ Local time: T_local = num_configs × avg_time
- ✅ Cloud time: T_cloud ≈ avg_time (parallel)
- ✅ Speedup: T_local / T_cloud ≈ num_configs

### What to Report

1. Maximum dataset size:
   - Local: X samples (OOM)
   - Cloud: Y samples (success)
   - Ratio: Y/X

2. Hyperparameter search time:
   - Local: T_local minutes (sequential)
   - Cloud: T_cloud seconds (parallel)
   - Speedup: T_local / T_cloud

3. Resource utilization:
   - Local: Single machine, limited RAM
   - Cloud: N nodes, pooled resources

## Next Steps

After demonstrating these capabilities:

1. **Scale Further**: Try 1000x dataset with more workers
2. **Larger Search**: Run 100+ hyperparameter combinations
3. **Real Workload**: Use actual large dataset (full S&P 500 history)
4. **Cost Analysis**: Compare cloud costs vs local hardware investment

## References

- [PyTorch Distributed Training](https://pytorch.org/docs/stable/distributed.html)
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/)
- [Kubernetes Resource Management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
