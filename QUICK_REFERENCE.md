# Quick Reference Card

## 🎯 Project Goal
Demonstrate cloud enables workloads **impossible locally** (not just faster)

## 📊 Two Key Comparisons

### 1. Dataset Scale (Memory Limits)
**Local**: OOM at ~50x multiplier  
**Cloud**: Success at 200x+ multiplier  
**Demonstrates**: Distributed memory pooling

### 2. Hyperparameter Search (Parallelism)
**Local**: 6 minutes (sequential)  
**Cloud**: 20 seconds (parallel)  
**Demonstrates**: Parallel execution capability

## 🚀 Quick Start Commands

### Phase 1: Local Baseline (15 min)
```bash
cd model
chmod +x run_local_experiments.sh
./run_local_experiments.sh

# This runs:
# 1. Create scaled datasets (10x, 50x, 200x)
# 2. Test local memory limits
# 3. Sequential hyperparameter search
```

### Phase 2: Cloud Deployment (10 min)
```bash
cd model
chmod +x deploy.sh
./deploy.sh

# This sets up:
# - Kubernetes namespace
# - Storage (PVC)
# - Uploads datasets and configs
# - Installs Kubeflow operator
```

### Phase 3: Cloud Experiments (5 min)
```bash
# Large dataset training
kubectl apply -f k8s/large-dataset-job.yaml

# Parallel hyperparameter search
python experiments/hyperparam_search_cloud.py

# Monitor
kubectl get pytorchjobs -n stock-training -w
```

### Phase 4: Compare Results (2 min)
```bash
python experiments/compare_results.py
```

## 📁 Key Files

### Scripts You Run
- `run_local_experiments.sh` - One command for all local tests
- `deploy.sh` - One command for Kubernetes setup
- `experiments/compare_results.py` - Generate comparison report

### Results Files
- `experiments/scale_test_local/results.json` - Local memory test
- `experiments/hp_search_local/summary.json` - Sequential search
- `experiments/hp_search_cloud/summary.json` - Parallel search

### Documentation
- `README.md` - Project overview
- `COMPARISON_GUIDE.md` - Detailed methodology
- `experiments/README.md` - Experiment scripts guide
- `IMPLEMENTATION_SUMMARY.md` - What was implemented

## 🔍 Expected Results

### Local Experiments
```
Dataset Scale Test:
  1x (860 samples):     ✓ Success
  10x (8,600 samples):  ✓ Success
  50x (43,000 samples): ✗ OOM or timeout

Hyperparameter Search:
  18 configurations: ~6 minutes total
```

### Cloud Experiments
```
Dataset Scale Test:
  200x (172,000 samples): ✓ Success (distributed)

Hyperparameter Search:
  18 configurations: ~20 seconds total (18x speedup)
```

## 🐛 Quick Troubleshooting

### Local Issues
**OOM too early**: Reduce batch_size in scripts  
**Takes too long**: Normal for CPU, or reduce epochs  
**Import errors**: `pip install -r requirements.txt`

### Kubernetes Issues
**PVC not binding**: Check `kubectl get sc`  
**Image pull**: Build with `docker build -t stock-trainer:latest .`  
**Jobs pending**: Check `kubectl describe node`  
**Pod failures**: Check `kubectl logs <pod-name> -n stock-training`

## 📋 Full Workflow

```bash
# 1. Setup data pipeline
cd data-pipeline
pip install -r requirements.txt
python -m src.run_pipeline

# 2. Run local baseline
cd ../model
pip install -r requirements.txt
./run_local_experiments.sh

# 3. Deploy to cloud
./deploy.sh

# 4. Run cloud experiments
kubectl apply -f k8s/large-dataset-job.yaml
python experiments/hyperparam_search_cloud.py

# 5. Compare
python experiments/compare_results.py
```

## 📊 What to Report

### Dataset Scale Comparison
- Local maximum: Nx (before OOM)
- Cloud maximum: Mx (success)
- Improvement: M/N ratio

### Hyperparameter Search Comparison
- Local time: T_local minutes
- Cloud time: T_cloud seconds
- Speedup: T_local / T_cloud

### Key Insight
Cloud doesn't just make things faster - it makes previously **impossible** workloads **possible**.

## 🔗 Useful Commands

### Check local results
```bash
# Scale test
cat experiments/scale_test_local/results.json | python -m json.tool

# Hyperparameter search
cat experiments/hp_search_local/summary.json | python -m json.tool
```

### Monitor Kubernetes
```bash
# Jobs
kubectl get pytorchjobs -n stock-training

# Pods
kubectl get pods -n stock-training

# Logs
kubectl logs <pod-name> -n stock-training -f

# Describe
kubectl describe pod <pod-name> -n stock-training
```

### Clean up
```bash
# Delete namespace (removes everything)
kubectl delete namespace stock-training

# Or delete specific jobs
kubectl delete pytorchjob <job-name> -n stock-training
```

## 📚 Documentation Hierarchy

1. **README.md** - Start here (project overview)
2. **THIS FILE** - Quick commands and reference
3. **COMPARISON_GUIDE.md** - Detailed methodology
4. **experiments/README.md** - Script documentation
5. **IMPLEMENTATION_SUMMARY.md** - Implementation details

## ⏱️ Time Estimates

| Phase | Task | Time |
|-------|------|------|
| Setup | Data pipeline + deps | 5 min |
| Local | Scaled datasets | 2 min |
| Local | Scale test | 5-10 min |
| Local | Hyperparameter search | 6 min |
| Cloud | Deploy to K8s | 5-10 min |
| Cloud | Large dataset training | 2-3 min |
| Cloud | Parallel search | 30 sec |
| Analysis | Compare results | 2 min |
| **TOTAL** | | **25-35 min** |

## 🎓 Key Takeaways

1. **Small datasets on CPU**: DDP doesn't show speedup
2. **Large datasets**: Cloud enables what local cannot (OOM)
3. **Multiple experiments**: Cloud parallelism transforms workflow
4. **Focus**: Capability enablement, not just performance

## 💡 Tips

- Run local experiments first (establish baseline)
- Document actual OOM point on your machine
- Take screenshots of kubectl output
- Save comparison report for presentation
- Emphasize "impossible → possible" narrative

---
**Need help?** See COMPARISON_GUIDE.md for detailed troubleshooting.
