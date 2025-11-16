# Model: Stock Time-Series Regression

Simple MLP regressor for benchmarking training performance on time-series regression for individual stocks and market indices.

## 🎯 Purpose

Train a baseline regression model to predict next-day normalized stock values. This serves as a benchmark for comparing:
- **Single-machine training** performance
- Cloud-based distributed training scalability
- Kubernetes cluster efficiency
- Performance across different stocks

## 🏗️ Architecture

**MLPRegressor**: Simple feedforward network
- Input: 60 timesteps × F features (normalized prices/indicators)
- Architecture:
  1. Temporal average pooling
  2. Linear(F, hidden_dim) → ReLU → Dropout
  3. Linear(hidden_dim, 1) → Output
- Loss: MSE (Mean Squared Error)
- Optimizer: Adam

## 📂 Project Structure

```
model/
├── src/
│   ├── single/              # Single-machine training
│   │   ├── cli.py          # CLI entry point
│   │   ├── train.py        # Training loops
│   │   └── datamod.py      # Data loading (standard DataLoader)
│   │
│   ├── models.py           # Shared: MLPRegressor architecture
│   ├── metrics.py          # Shared: MSE, MAE, R² computation
│   ├── artifacts.py        # Shared: Save/load checkpoints
│   └── utils.py            # Shared: Utilities (seeding, timing, device)
│
├── train_single.py         # Entry point: single-machine training
├── train_all_stocks_single.py  # Batch: train all stocks (single-machine)
│
├── outputs/                # Training artifacts (auto-created)
│   └── run_*/              # Timestamped run directories
│
├── TRAINING_GUIDE.md       # Comprehensive training guide
├── ARCHITECTURE.md         # System architecture diagrams
├── REORGANIZATION_SUMMARY.md  # Code reorganization details
├── requirements.txt
└── README.md               # This file
```

## 🚀 Quick Start

### Installation

```bash
cd model
pip install -r requirements.txt
```

### Single-Machine Training

Train on a specific stock:
```bash
# Train on Apple (AAPL)
python train_single.py --stock AAPL

# Train on Microsoft with custom hyperparameters
python train_single.py \
    --stock MSFT \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --hidden_dim 128
```

**Available Stocks:** AAPL, AMZN, JNJ, JPM, MSFT, TSLA, XOM

Train on S&P 500:
```bash
python train_single.py --npz_path ../data-pipeline/data/processed/sp500_regression.npz
```

Train all stocks sequentially:
```bash
python train_all_stocks_single.py --epochs 50
```

## Usage

### Basic Training

```

```bash

## 🚀 Usagecd src

python cli.py --npz_path /path/to/sp500_classification.npz

### Installation```



```bash### With Custom Parameters

cd model

pip install -r requirements.txt```bash

```python cli.py \

  --npz_path /path/to/data.npz \

### Training  --model lstm \

  --epochs 50 \

```bash  --batch_size 128 \

# Basic training  --lr 0.001 \

python -m src.cli --npz_path ../data-pipeline/data/processed/sp500_regression.npz  --hidden 128 \

  --layers 2

# Custom hyperparameters```

python -m src.cli \

    --npz_path ../data-pipeline/data/processed/sp500_regression.npz \### With Config File

    --epochs 50 \

    --batch_size 128 \Create `config.yaml`:

    --lr 1e-3 \```yaml

    --hidden_dim 128 \data:

    --dropout 0.2 \  npz_path: "/path/to/data.npz"

    --device cuda \  valid_from_train: 0.1

    --seed 42  norm: "zscore"

```

model:

### Command-Line Arguments  kind: "gru"

  hidden: 128

| Argument | Default | Description |  layers: 2

|----------|---------|-------------|  dropout: 0.2

| `--npz_path` | `/mnt/data/sp500_regression.npz` | Path to dataset |

| `--epochs` | 50 | Maximum training epochs |train:

| `--batch_size` | 64 | Batch size |  epochs: 50

| `--lr` | 1e-3 | Learning rate |  batch_size: 128

| `--hidden_dim` | 64 | Hidden layer size |  lr: 0.0005

| `--dropout` | 0.1 | Dropout probability |  early_stop_patience: 10

| `--patience` | 5 | Early stopping patience |```

| `--device` | auto | Device (auto/cpu/cuda/mps) |

| `--seed` | 42 | Random seed |Run:

| `--save_dir` | ./outputs | Output directory |```bash

| `--valid_ratio` | 0.1 | Validation split ratio |python cli.py --config config.yaml

| `--shuffle_train` | False | Shuffle training data |```



## 📊 Output Artifacts## Configuration



Each training run creates a timestamped directory in `outputs/` with:### Data

- `npz_path`: Path to NPZ data file

### 1. **best.ckpt**- `valid_from_train`: Fraction of training data for validation (default: 0.1)

PyTorch state_dict of best model (lowest validation loss)- `shuffle_train`: Whether to shuffle training data (default: false)

- `norm`: Normalization method ("zscore" or "none")

### 2. **metrics_valid.json**

```json### Model

{- `kind`: Model type ("lstm" or "gru")

  "mse": 0.000532,- `input_dim`: Input feature dimension (auto-inferred if null)

  "mae": 0.0184,- `hidden`: Hidden layer size (default: 64)

  "r2": 0.6120- `layers`: Number of recurrent layers (default: 1)

}- `dropout`: Dropout rate (default: 0.1)

```

### Training

### 3. **metrics_test.json**- `epochs`: Maximum training epochs (default: 25)

```json- `batch_size`: Batch size (default: 64)

{- `lr`: Learning rate (default: 0.001)

  "mse": 0.007773,- `weight_decay`: L2 regularization (default: 0.0)

  "mae": 0.0798,- `early_stop_metric`: Metric for early stopping ("auc" or "val_loss")

  "r2": 0.3734- `early_stop_patience`: Patience for early stopping (default: 5)

}- `seed`: Random seed (default: 42)

```- `device`: Device ("cpu", "cuda", or "auto")



### 4. **config.yaml**### Evaluation

Complete training configuration (hyperparameters, paths, etc.)- `threshold`: Classification threshold (default: 0.5)

- `save_cm`: Save confusion matrix plots (default: true)

### 5. **norm_stats.json**

Normalization statistics (mean, std) from training data### I/O

```json- `save_dir`: Output directory (default: "./outputs")

{- `run_name`: Custom run name (auto-generated if null)

  "mu": [0.0123],

  "sd": [0.0456]## Outputs

}

```Each run creates a directory `outputs/run_YYYYMMDD_HHMMSS/` containing:

- `best.ckpt`: Best model checkpoint

### 6. **timing.json**- `config.yaml`: Resolved configuration

```json- `norm_stats.json`: Normalization statistics (if using z-score)

{- `metrics_valid.json`: Validation metrics

  "total_epochs": 32,- `metrics_test.json`: Test metrics

  "best_epoch": 27,- `confusion_matrix_valid.png`: Validation confusion matrix

  "total_time_sec": 5.3,- `confusion_matrix_test.png`: Test confusion matrix

  "mean_time_per_epoch": 0.17,

  "std_time_per_epoch": 0.51,## Metrics

  "epoch_times": [3.01, 0.09, 0.08, ...]

}- **Accuracy**: Overall classification accuracy

```- **AUC**: Area under ROC curve

- **Precision**: Positive predictive value

## 🎯 Design Principles- **Recall**: True positive rate



### DDP-Ready Architecture## Features

- **Stateless training loops**: No global state

- **Device-agnostic**: Explicit device arguments- ✅ DDP-friendly (stateless loops, clean functions)

- **Deterministic**: Seed control for reproducibility- ✅ Deterministic training (fixed seeds)

- ✅ Early stopping with patience

### Performance Benchmarking- ✅ Automatic validation split (chronological)

- Per-epoch timing with statistics- ✅ Z-score normalization (computed on train only)

- Total training time tracking- ✅ Binary classification with BCE loss

- Early stopping to measure convergence speed- ✅ LSTM and GRU architectures

- ✅ Comprehensive metrics and artifacts

### Data Integrity

- Chronological validation/test splits (no shuffle)## Example Output

- Normalization using train stats only

- No data leakage between splits```

[2025-11-05 14:30:15] Set random seed: 42

## 📈 Typical Performance[2025-11-05 14:30:15] Using device: cuda

[2025-11-05 14:30:15] Run directory: outputs/run_20251105_143015

**Hardware**: Single machine (CPU/GPU/MPS)[2025-11-05 14:30:15] Loading data and building dataloaders...

- **Throughput**: ~600 samples/sec[2025-11-05 14:30:16] Data loading took 0.82s

- **Time/epoch**: 0.1-0.5s (depending on hardware)[2025-11-05 14:30:16]   Train batches: 28

- **Convergence**: 20-40 epochs (early stopping)[2025-11-05 14:30:16]   Val batches: 4

[2025-11-05 14:30:16]   Test batches: 8

**Metrics** (varies by data split):[2025-11-05 14:30:16]   Input dim: 1

- Validation R²: 0.5-0.7[2025-11-05 14:30:16] Building LSTM model...

- Test R²: 0.3-0.5[2025-11-05 14:30:16] Model parameters: 12,801

- Test MAE: 0.05-0.10 (normalized scale)

============================================================

## 🔜 Future Work

- [ ] Kubernetes deployment
- [ ] Multi-node scaling experiments
- [ ] Advanced architectures (LSTM, Transformer)

...

## 📝 Notes

VAL:  acc=0.56 auc=0.60 prec=0.57 rec=0.55

This is a **baseline model** designed for performance benchmarking, not maximum prediction accuracy. The simple architecture allows for:TEST: acc=0.55 auc=0.58 prec=0.56 rec=0.54

- Fast training iterationsSaved → outputs/run_20251105_143015

- Clear performance comparisons

- Easy distributed training adaptation✓ Done!

```

## 🧪 Testing

Verify the setup works correctly:

```bash
# Quick test run (5 epochs)
python -m src.cli \
    --npz_path ../data-pipeline/data/processed/sp500_regression.npz \
    --epochs 5 \
    --batch_size 64

# Expected output structure:
# outputs/
#   └── run_YYYYMMDD_HHMMSS/
#       ├── best.ckpt
#       ├── config.yaml
#       ├── metrics_test.json
#       ├── metrics_valid.json
#       ├── norm_stats.json
#       └── timing.json
```

## 📚 Module Documentation

### `datamod.py`
- `SeqDataset`: PyTorch Dataset for sequence data
- `build_dataloaders()`: Load NPZ, normalize, create train/val/test loaders

### `models.py`
- `MLPRegressor`: Simple feedforward network with temporal pooling

### `train.py`
- `train_one_epoch()`: Single epoch training
- `evaluate()`: Validation/test evaluation
- `fit()`: Full training loop with early stopping
- `test()`: Load checkpoint and evaluate

### `metrics.py`
- `regression_metrics()`: Compute MSE, MAE, R²

### `artifacts.py`
- `make_run_dir()`: Create timestamped output directory
- `save_state_dict()`: Save model checkpoint
- `save_json()` / `save_yaml()`: Save configs and metrics
- `save_all_artifacts()`: Save complete training run

### `utils.py`
- `seed_everything()`: Set random seeds
- `get_device()`: Device selection (auto/cpu/cuda/mps)
- `Timer`: Context manager for timing
- `log()`: Timestamped logging

### `cli.py`
- Command-line interface
- Argument parsing
- End-to-end training pipeline
