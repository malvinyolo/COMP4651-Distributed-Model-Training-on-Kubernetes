# Training System Organization

This directory contains a **single-machine** training implementation for time-series stock regression.

## Directory Structure

```
model/
├── src/
│   ├── single/              # Single-machine training code
│   │   ├── cli.py          # CLI entry point
│   │   ├── train.py        # Training loops (fit, evaluate)
│   │   └── datamod.py      # Data loading (standard DataLoader)
│   │
│   ├── models.py           # Shared: MLPRegressor architecture
│   ├── metrics.py          # Shared: Evaluation metrics (MSE, MAE, R²)
│   ├── artifacts.py        # Shared: Model saving/loading
│   └── utils.py            # Shared: Utilities (Timer, device detection)
│
├── train_single.py          # Entry point: single-machine training
└── train_all_stocks_single.py      # Batch: train all 7 stocks (single-machine)
```

## Quick Start

### Single-Machine Training

Train on a specific stock:
```bash
python train_single.py --stock AAPL --epochs 50
```

Train on S&P 500 index:
```bash
python train_single.py --npz_path ../data-pipeline/data/processed/sp500_regression.npz --epochs 50
```

Train on all 7 stocks:
```bash
python train_all_stocks_single.py --epochs 50 --batch_size 64
```

## Available Stocks

The system supports 7 individual stocks (5 features each):
- **AAPL** - Apple Inc.
- **AMZN** - Amazon.com Inc.
- **JNJ** - Johnson & Johnson
- **JPM** - JPMorgan Chase & Co.
- **MSFT** - Microsoft Corporation
- **TSLA** - Tesla Inc.
- **XOM** - Exxon Mobil Corporation

Plus the S&P 500 index (1 feature).

## Training Features

The single-machine implementation includes:
- Standard PyTorch DataLoader
- Single process, single GPU (or CPU)
- Easy to debug and prototype
- Suitable for: prototyping, development, single GPU machines

## Configuration Options

The training script supports:
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 1e-3)
- `--hidden_dim`: MLP hidden dimension (default: 64)
- `--dropout`: Dropout probability (default: 0.1)
- `--patience`: Early stopping patience (default: 5)
- `--seed`: Random seed (default: 42)

## Backend Support

- **CUDA**: NVIDIA GPUs
- **MPS**: Apple Silicon GPUs
- **CPU**: Fallback

## Output Structure

Training outputs are saved to `outputs/<run_name>/`:
```
outputs/
└── AAPL_20240115_123456/
    ├── best.pt              # Best model checkpoint
    ├── config.json          # Training configuration
    ├── metrics.json         # Training metrics
    └── norm_stats.json      # Normalization statistics
```

## Testing

Quick test (3 epochs):
```bash
python train_single.py --stock AAPL --epochs 3
```

## Notes

1. **Model checkpoints** - Best model is saved automatically based on validation loss
2. **Normalization statistics** - Saved with the model for inference
3. **Metrics tracking** - MSE, MAE, and R² are computed on validation and test sets

## Troubleshooting

**Q: Out of memory errors**
- Reduce `--batch_size`
- Reduce `--hidden_dim`

**Q: Import errors**
- Run from the `model/` directory
- Ensure Python can find the `src/` package

## Development

To add new features:
- **Shared functionality** → `src/models.py`, `src/metrics.py`, `src/utils.py`
- **Training code** → `src/single/`

This separation keeps code maintainable and testable.
