# Model: S&P 500 Regression# Model Training Package



Simple MLP regressor for benchmarking training performance on time-series regression.Binary classification model for S&P 500 T+1 direction prediction using pre-sequenced data.



## 🎯 Purpose## Structure



Train a baseline regression model to predict next-day normalized S&P 500 values. This serves as a benchmark for comparing:```

- Single-machine training performancemodel/

- Distributed training (DDP) scalability├── src/

- Kubernetes cluster efficiency│   ├── cli.py          # Main entrypoint

│   ├── datamod.py      # Data loading and preprocessing

## 🏗️ Architecture│   ├── models.py       # LSTM and GRU classifiers

│   ├── train.py        # Training and evaluation loops

**MLPRegressor**: Simple feedforward network│   ├── metrics.py      # Metrics computation

- Input: 60 timesteps × 1 feature (normalized prices)│   ├── artifacts.py    # Checkpoint and artifact management

- Architecture:│   └── utils.py        # Utilities (seeding, device, logging)

  1. Temporal average pooling├── outputs/            # Training run outputs

  2. Linear(1, hidden_dim) → ReLU → Dropout└── requirements.txt    # Dependencies

  3. Linear(hidden_dim, 1) → Output```

- Loss: MSE (Mean Squared Error)

- Optimizer: Adam## Data Contract



## 📂 Project StructureExpects NPZ file with:

- `X_train`: shape `(N_train, T, F)` float32

```- `y_train`: shape `(N_train,)` int {0,1}

model/- `X_test`: shape `(N_test, T, F)` float32

├── src/- `y_test`: shape `(N_test,)` int {0,1}

│   ├── datamod.py      # Data loading, normalization, DataLoaders

│   ├── models.py       # MLPRegressor architectureWhere:

│   ├── train.py        # Training/validation/test loops- `N`: number of samples

│   ├── metrics.py      # MSE, MAE, R² computation- `T`: sequence length (e.g., 60)

│   ├── artifacts.py    # Save checkpoints and configs- `F`: number of features (e.g., 1)

│   ├── utils.py        # Utilities (seeding, timing, device)

│   └── cli.py          # Command-line interface## Installation

├── outputs/            # Training artifacts (auto-created)

│   └── run_*/          # Timestamped run directories```bash

│       ├── best.ckptcd model

│       ├── metrics_valid.jsonpip install -r requirements.txt

│       ├── metrics_test.json```

│       ├── config.yaml

│       ├── norm_stats.json## Usage

│       └── timing.json

└── requirements.txt### Basic Training

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

## 🔜 Future WorkTRAINING

============================================================

- [ ] Distributed training (PyTorch DDP)[2025-11-05 14:30:16] Starting training for 25 epochs...

- [ ] Kubernetes deployment[2025-11-05 14:30:16] Early stopping: metric=auc, patience=5

- [ ] Multi-node scaling experiments[2025-11-05 14:30:17] Epoch 1/25 | train_loss=0.6845 | val_loss=0.6732 | val_acc=0.560 | val_auc=0.603

- [ ] Advanced architectures (LSTM, Transformer)[2025-11-05 14:30:17]   ✓ New best auc=0.6034, saved checkpoint

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
