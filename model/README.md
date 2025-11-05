# Model Training Package

Binary classification model for S&P 500 T+1 direction prediction using pre-sequenced data.

## Structure

```
model/
├── src/
│   ├── cli.py          # Main entrypoint
│   ├── datamod.py      # Data loading and preprocessing
│   ├── models.py       # LSTM and GRU classifiers
│   ├── train.py        # Training and evaluation loops
│   ├── metrics.py      # Metrics computation
│   ├── artifacts.py    # Checkpoint and artifact management
│   └── utils.py        # Utilities (seeding, device, logging)
├── outputs/            # Training run outputs
└── requirements.txt    # Dependencies
```

## Data Contract

Expects NPZ file with:
- `X_train`: shape `(N_train, T, F)` float32
- `y_train`: shape `(N_train,)` int {0,1}
- `X_test`: shape `(N_test, T, F)` float32
- `y_test`: shape `(N_test,)` int {0,1}

Where:
- `N`: number of samples
- `T`: sequence length (e.g., 60)
- `F`: number of features (e.g., 1)

## Installation

```bash
cd model
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
cd src
python cli.py --npz_path /path/to/sp500_classification.npz
```

### With Custom Parameters

```bash
python cli.py \
  --npz_path /path/to/data.npz \
  --model lstm \
  --epochs 50 \
  --batch_size 128 \
  --lr 0.001 \
  --hidden 128 \
  --layers 2
```

### With Config File

Create `config.yaml`:
```yaml
data:
  npz_path: "/path/to/data.npz"
  valid_from_train: 0.1
  norm: "zscore"

model:
  kind: "gru"
  hidden: 128
  layers: 2
  dropout: 0.2

train:
  epochs: 50
  batch_size: 128
  lr: 0.0005
  early_stop_patience: 10
```

Run:
```bash
python cli.py --config config.yaml
```

## Configuration

### Data
- `npz_path`: Path to NPZ data file
- `valid_from_train`: Fraction of training data for validation (default: 0.1)
- `shuffle_train`: Whether to shuffle training data (default: false)
- `norm`: Normalization method ("zscore" or "none")

### Model
- `kind`: Model type ("lstm" or "gru")
- `input_dim`: Input feature dimension (auto-inferred if null)
- `hidden`: Hidden layer size (default: 64)
- `layers`: Number of recurrent layers (default: 1)
- `dropout`: Dropout rate (default: 0.1)

### Training
- `epochs`: Maximum training epochs (default: 25)
- `batch_size`: Batch size (default: 64)
- `lr`: Learning rate (default: 0.001)
- `weight_decay`: L2 regularization (default: 0.0)
- `early_stop_metric`: Metric for early stopping ("auc" or "val_loss")
- `early_stop_patience`: Patience for early stopping (default: 5)
- `seed`: Random seed (default: 42)
- `device`: Device ("cpu", "cuda", or "auto")

### Evaluation
- `threshold`: Classification threshold (default: 0.5)
- `save_cm`: Save confusion matrix plots (default: true)

### I/O
- `save_dir`: Output directory (default: "./outputs")
- `run_name`: Custom run name (auto-generated if null)

## Outputs

Each run creates a directory `outputs/run_YYYYMMDD_HHMMSS/` containing:
- `best.ckpt`: Best model checkpoint
- `config.yaml`: Resolved configuration
- `norm_stats.json`: Normalization statistics (if using z-score)
- `metrics_valid.json`: Validation metrics
- `metrics_test.json`: Test metrics
- `confusion_matrix_valid.png`: Validation confusion matrix
- `confusion_matrix_test.png`: Test confusion matrix

## Metrics

- **Accuracy**: Overall classification accuracy
- **AUC**: Area under ROC curve
- **Precision**: Positive predictive value
- **Recall**: True positive rate

## Features

- ✅ DDP-friendly (stateless loops, clean functions)
- ✅ Deterministic training (fixed seeds)
- ✅ Early stopping with patience
- ✅ Automatic validation split (chronological)
- ✅ Z-score normalization (computed on train only)
- ✅ Binary classification with BCE loss
- ✅ LSTM and GRU architectures
- ✅ Comprehensive metrics and artifacts

## Example Output

```
[2025-11-05 14:30:15] Set random seed: 42
[2025-11-05 14:30:15] Using device: cuda
[2025-11-05 14:30:15] Run directory: outputs/run_20251105_143015
[2025-11-05 14:30:15] Loading data and building dataloaders...
[2025-11-05 14:30:16] Data loading took 0.82s
[2025-11-05 14:30:16]   Train batches: 28
[2025-11-05 14:30:16]   Val batches: 4
[2025-11-05 14:30:16]   Test batches: 8
[2025-11-05 14:30:16]   Input dim: 1
[2025-11-05 14:30:16] Building LSTM model...
[2025-11-05 14:30:16] Model parameters: 12,801

============================================================
TRAINING
============================================================
[2025-11-05 14:30:16] Starting training for 25 epochs...
[2025-11-05 14:30:16] Early stopping: metric=auc, patience=5
[2025-11-05 14:30:17] Epoch 1/25 | train_loss=0.6845 | val_loss=0.6732 | val_acc=0.560 | val_auc=0.603
[2025-11-05 14:30:17]   ✓ New best auc=0.6034, saved checkpoint
...

VAL:  acc=0.56 auc=0.60 prec=0.57 rec=0.55
TEST: acc=0.55 auc=0.58 prec=0.56 rec=0.54
Saved → outputs/run_20251105_143015

✓ Done!
```
