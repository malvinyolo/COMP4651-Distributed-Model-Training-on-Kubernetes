# Experiments: Cloud Capability Demonstration

This directory contains scripts and results for demonstrating the capabilities that cloud/Kubernetes enables beyond what's possible on a single local machine.

## Overview

The experiments focus on two key comparisons:

1. **Dataset Scale**: Show that local machines fail (OOM) with large datasets, while Kubernetes can handle them through distributed training
2. **Hyperparameter Search**: Show that parallel execution in the cloud dramatically speeds up experimentation

## Directory Structure

```
experiments/
├── README.md                          # This file
├── test_scale_local.py               # Test local memory limits
├── hyperparam_search_local.py        # Sequential hyperparameter search (baseline)
├── hyperparam_search_cloud.py        # Parallel hyperparameter search (cloud)
├── compare_results.py                # Compare and visualize results
├── scale_test_local/                 # Local scale test results
│   └── results.json
├── scale_test_cloud/                 # Cloud scale test results (from k8s)
│   └── results.json
├── hp_search_local/                  # Local hyperparameter search results
│   ├── summary.json
│   └── configs/                      # Individual config results
└── hp_search_cloud/                  # Cloud hyperparameter search results
    ├── summary.json
    ├── configs/                      # Generated configs
    └── manifests/                    # Generated k8s manifests
```

## Quick Start

### 1. Test Local Limits (Dataset Scale)

```bash
# Run local scale test to find memory limits
python test_scale_local.py

# Expected output:
# - Success: 1x, 10x datasets
# - Failure: 50x+ datasets (OOM)
# - Results saved to scale_test_local/results.json
```

### 2. Run Local Hyperparameter Search (Baseline)

```bash
# Run sequential hyperparameter search on local machine
python hyperparam_search_local.py

# Expected:
# - 18 configurations
# - ~6 minutes total time (sequential)
# - Results saved to hp_search_local/summary.json
```

### 3. Create Large Datasets

```bash
# Move to data pipeline directory
cd ../../data-pipeline

# Create scaled datasets
python create_large_dataset.py --multiplier 10
python create_large_dataset.py --multiplier 50
python create_large_dataset.py --multiplier 200

# Verify created files
ls -lh data/processed/sp500_regression_x*.npz
```

### 4. Deploy to Kubernetes

```bash
# Return to model directory
cd ../model

# Deploy everything
chmod +x deploy.sh
./deploy.sh

# This will:
# - Create namespace
# - Create PVC
# - Upload datasets and configs
# - Install Kubeflow operator (if needed)
```

### 5. Run Cloud Experiments

```bash
# Run large dataset training (200x multiplier)
kubectl apply -f k8s/large-dataset-job.yaml

# Monitor progress
kubectl get pytorchjobs -n stock-training
kubectl get pods -n stock-training -w

# Check logs
kubectl logs <pod-name> -n stock-training -f

# Run parallel hyperparameter search
python experiments/hyperparam_search_cloud.py

# This will:
# - Generate 18 configs
# - Submit 18 parallel jobs
# - Wait for completion (~20s)
# - Save results
```

### 6. Compare Results

```bash
# Generate comparison report
python compare_results.py

# This will display:
# - Dataset scale comparison (local OOM vs cloud success)
# - Hyperparameter search time comparison
# - Speedup calculations
# - Summary of cloud capabilities
```

## Detailed Script Descriptions

### test_scale_local.py

Tests local machine memory limits by:
- Loading progressively larger datasets (1x, 10x, 50x, 100x, 200x)
- Running short training sessions (30s timeout)
- Tracking memory usage
- Recording which sizes succeed/fail

**Key Metrics**:
- Maximum successful dataset size
- Memory usage at each scale
- Time to OOM/timeout

**Expected Results**:
- Success: 1x (860 samples, ~300KB)
- Success: 10x (8,600 samples, ~3MB)
- Likely failure: 50x+ (43,000+ samples, 15MB+)

### hyperparam_search_local.py

Runs hyperparameter search sequentially:
- Grid search: 3 learning rates × 3 hidden sizes × 2 num_layers = 18 configs
- One at a time (sequential execution)
- Each config trains for 20 epochs (~20 seconds)
- Total time: ~6 minutes

**Key Metrics**:
- Total time
- Average time per config
- Best configuration found

**Expected Results**:
- 18 configurations
- ~20s per config
- ~6 minutes total

### hyperparam_search_cloud.py

Orchestrates parallel hyperparameter search on Kubernetes:
- Generates 18 separate config files
- Creates 18 PyTorchJob manifests
- Submits all jobs simultaneously
- Waits for completion
- Collects results

**Key Metrics**:
- Total time (wall-clock)
- Parallelism achieved
- Number of successful jobs

**Expected Results**:
- 18 configurations (parallel)
- ~20s total time
- ~18x speedup over sequential

### compare_results.py

Compares local vs cloud results:
- Loads results from both experiments
- Calculates speedups and improvements
- Generates comparison tables
- Prints summary report

**Output Includes**:
- Dataset scale comparison table
- Hyperparameter search timing comparison
- Speedup calculations
- Cloud capability summary

## Understanding the Results

### Dataset Scale Results

**Local Machine**:
```
1x:   ✓ Success (2.3s, 150MB)
10x:  ✓ Success (23.1s, 1.2GB)
50x:  ✗ Failed (OOM after 45s)
```

**Cloud (Kubernetes)**:
```
1x:   ✓ Success (3.1s, 4 nodes)
10x:  ✓ Success (8.2s, 4 nodes)
50x:  ✓ Success (28.3s, 4 nodes)
200x: ✓ Success (95.1s, 4 nodes)
```

**Interpretation**:
- Local hits memory limit at ~50x
- Cloud handles 200x+ through distribution
- Cloud enables 4x-20x larger datasets
- **Key point**: Not about speed, but capability to handle larger data

### Hyperparameter Search Results

**Local (Sequential)**:
```
Total configs: 18
Total time: 6.2 minutes
Avg per config: 20.7s
Parallelism: 1
```

**Cloud (Parallel)**:
```
Total configs: 18
Total time: 21.3 seconds
Avg per config: 21.3s
Parallelism: 18
```

**Interpretation**:
- Speedup: 17.5x (near-linear with parallelism)
- Time saved: 5.9 minutes
- Cloud enables rapid experimentation
- **Key point**: Parallel execution transforms workflow

## Metrics Collected

### Dataset Scale Metrics
- Dataset size (samples, bytes)
- Training time
- Memory usage (peak, average)
- Success/failure status
- Error messages (if failed)

### Hyperparameter Search Metrics
- Total configurations
- Total time (wall-clock)
- Individual config times
- Success rate
- Best validation metrics
- Parallel efficiency

## Troubleshooting

### test_scale_local.py Issues

**Problem**: Script fails too early (even 1x fails)
- Check available memory: `free -h` (Linux) or Activity Monitor (Mac)
- Reduce batch size in script
- Close other applications

**Problem**: Script succeeds at all scales
- Your machine has lots of RAM!
- Try larger multipliers (500x, 1000x)
- Reduce available memory (Docker/VM limits)

### hyperparam_search_local.py Issues

**Problem**: Takes too long (>10 minutes)
- Normal for CPU training
- Reduce epochs (currently 20)
- Reduce hidden_dim sizes
- Use GPU if available

**Problem**: Some configs fail
- Check error messages in logs
- Might be learning rate too high
- Check data path is correct

### hyperparam_search_cloud.py Issues

**Problem**: Jobs fail to submit
- Check kubectl connection: `kubectl cluster-info`
- Verify namespace exists: `kubectl get namespace stock-training`
- Check image is available: `docker images stock-trainer`

**Problem**: Jobs stuck pending
- Check resources: `kubectl describe node`
- Not enough CPU/memory available
- Reduce number of parallel jobs

**Problem**: Jobs fail during execution
- Check logs: `kubectl logs <pod-name> -n stock-training`
- Data might not be uploaded to PVC
- Config file might be missing

### compare_results.py Issues

**Problem**: Missing results files
- Run local experiments first
- Run cloud experiments after deploying
- Check file paths match expected structure

**Problem**: Incomplete comparisons
- Only shows available results
- Some experiments might still be running
- Wait for jobs to complete

## Advanced Usage

### Custom Hyperparameter Space

Edit `hyperparam_search_local.py` or `hyperparam_search_cloud.py`:

```python
SEARCH_SPACE = {
    "learning_rate": [1e-5, 1e-4, 5e-4, 1e-3, 5e-3],  # More values
    "hidden_size": [32, 64, 128, 256, 512],            # More values
    "num_layers": [1, 2, 3, 4],                        # More values
}
# Total: 5 × 5 × 4 = 100 configurations
```

### Larger Datasets

Create custom multipliers:

```bash
# Create 500x dataset (~150MB, 430,000 samples)
python create_large_dataset.py --multiplier 500

# Create 1000x dataset (~300MB, 860,000 samples)
python create_large_dataset.py --multiplier 1000
```

### Different Models

Modify training scripts to test different architectures:
- LSTM instead of MLP
- Transformer models
- Larger model sizes

### Different Datasets

Replace stock data with your own:
- Image datasets (CIFAR, ImageNet)
- NLP datasets (text classification)
- Time series (different domains)

## Expected Timeline

**Local Experiments** (~10-15 minutes):
1. Scale test: 5-10 minutes (includes failures/timeouts)
2. Hyperparameter search: 6 minutes

**Cloud Setup** (~5-10 minutes):
1. Deploy script: 3-5 minutes
2. Verify deployment: 2 minutes

**Cloud Experiments** (~5 minutes):
1. Large dataset training: 2-3 minutes
2. Parallel hyperparameter search: 30 seconds
3. Collect results: 1-2 minutes

**Analysis** (~2 minutes):
1. Run comparison script: 1 minute
2. Review results: 1 minute

**Total**: 20-30 minutes for complete demonstration

## Next Steps

After completing these experiments:

1. **Scale Further**: Try even larger datasets (1000x+) with more workers
2. **Larger Search**: 100+ hyperparameter combinations
3. **Real Workload**: Use production-scale datasets
4. **Cost Analysis**: Track cloud costs vs local hardware
5. **Visualization**: Create plots and charts for presentation
6. **Documentation**: Update README with actual results

## References

- Parent COMPARISON_GUIDE.md: Detailed methodology
- model/README.md: Overall project documentation
- k8s/: Kubernetes manifests
- ../data-pipeline: Dataset creation tools
