# COMP4651: Distributed Model Training on Kubernetes

Benchmarking single-machine vs distributed deep learning training performance on Kubernetes.

## 🎯 Project Overview

This project implements a **regression model** to predict next-day normalized S&P 500 values and benchmarks training performance across different distributed configurations.

### Key Components:
- **Data Pipeline**: ETL for S&P 500 historical data
- **Model**: Simple MLP regressor with performance benchmarking
- **Goal**: Compare single-machine vs DDP/Kubernetes distributed training

## 📁 Repository Structure

```
├── data-pipeline/       # Data extraction, processing, and dataset creation
│   ├── src/            # Pipeline modules
│   ├── data/           # Raw and processed datasets
│   └── requirements.txt
│
├── model/              # Regression model training and benchmarking
│   ├── src/            # Training, models, metrics, utils
│   ├── outputs/        # Training artifacts and logs
│   └── requirements.txt
│
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Setup Data Pipeline

```bash
cd data-pipeline
pip install -r requirements.txt

# Run pipeline to generate sp500_regression.npz
python -m src.run_pipeline
```

### 2. Train Baseline Model

```bash
cd ../model
pip install -r requirements.txt

# Train with default settings
python -m src.cli --npz_path ../data-pipeline/data/processed/sp500_regression.npz

# Train with custom hyperparameters
python -m src.cli \
    --npz_path ../data-pipeline/data/processed/sp500_regression.npz \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --hidden_dim 128
```

## 📊 Model Performance

The baseline MLP model predicts next-day normalized S&P 500 values using 60 previous timesteps.

**Example Results:**
- VAL: MSE=0.000532, MAE=0.0184, R²=0.61
- TEST: MSE=0.007773, MAE=0.0798, R²=0.37
- Time/epoch: ~0.17s

## 🎓 Course Information

**Course**: COMP 4651 - Cloud Computing and Big Data Systems  
**Semester**: Fall 2025  
**Focus**: Distributed training performance analysis on Kubernetes

## 📝 Development

### Data Pipeline
Located in `data-pipeline/`, this component:
- Fetches S&P 500 historical data using yfinance
- Processes and normalizes time-series data
- Creates sequence datasets for regression tasks
- Outputs train/test splits in NPZ format

### Model Training
Located in `model/`, this component:
- Implements simple MLP regressor
- Provides clean, modular training loops
- Tracks performance metrics (MSE, MAE, R²)
- Records timing statistics for benchmarking
- Saves all artifacts for reproducibility

## 🔮 Future Work

- [ ] Implement PyTorch DDP (Distributed Data Parallel)
- [ ] Deploy on Kubernetes cluster
- [ ] Multi-node scaling experiments
- [ ] Performance comparison reports
- [ ] Advanced model architectures (LSTM, Transformer)

## 📄 License

Educational project for COMP 4651.