# Training System Organization

This directory contains both **single-machine** and **distributed (DDP)** training implementations for time-series stock regression.

## Directory Structure

```
model/
├── src/
│   ├── single/              # Single-machine training code
│   │   ├── cli.py          # CLI entry point
│   │   ├── train.py        # Training loops (fit, evaluate)
│   │   └── datamod.py      # Data loading (standard DataLoader)
│   │
│   ├── distributed/         # DDP (Distributed Data Parallel) code
│   │   ├── cli.py          # DDP CLI entry point
│   │   ├── train.py        # DDP training loops with gradient sync
│   │   └── datamod.py      # DDP data loading (DistributedSampler)
│   │
│   ├── models.py           # Shared: MLPRegressor architecture
│   ├── metrics.py          # Shared: Evaluation metrics (MSE, MAE, R²)
│   ├── artifacts.py        # Shared: Model saving/loading
│   └── utils.py            # Shared: Utilities (Timer, device detection)
│
├── train_single.py          # Entry point: single-machine training
├── train_ddp.py             # Entry point: DDP training
├── train_all_stocks.py      # Batch: train all 7 stocks (single-machine)
└── train_all_stocks_ddp.py  # Batch: train all 7 stocks (DDP)
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
python train_all_stocks.py --epochs 50 --batch_size 64
```

### Distributed Training (DDP)

Train on a specific stock with 2 GPUs:
```bash
python train_ddp.py --stock AAPL --epochs 50 --world_size 2
```

Train on all 7 stocks with DDP:
```bash
python train_all_stocks_ddp.py --epochs 50 --world_size 2
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

## Key Differences: Single vs DDP

### Single-Machine (`src/single/`)
- Standard PyTorch DataLoader
- Single process, single GPU (or CPU)
- Simpler code, easier to debug
- Best for: prototyping, small datasets, single GPU machines

### DDP (`src/distributed/`)
- DistributedSampler for data parallelism
- Multi-process, multi-GPU training
- Gradient synchronization via AllReduce
- Scales training across multiple GPUs
- Best for: large datasets, multi-GPU machines, production

## Configuration Options

Both training modes support:
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 64; per GPU for DDP)
- `--lr`: Learning rate (default: 1e-3)
- `--hidden_dim`: MLP hidden dimension (default: 64)
- `--dropout`: Dropout probability (default: 0.1)
- `--patience`: Early stopping patience (default: 5)
- `--seed`: Random seed (default: 42)

DDP-specific:
- `--world_size`: Number of processes/GPUs (default: auto-detect)
- `--num_workers`: DataLoader workers per process (default: 0)

## Backend Support

- **CUDA**: NVIDIA GPUs (uses NCCL backend for DDP)
- **MPS**: Apple Silicon GPUs (single-machine only)
- **CPU**: Fallback (uses Gloo backend for DDP)

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
# Single-machine
python train_single.py --stock AAPL --epochs 3

# DDP (if you have GPUs)
python train_ddp.py --stock AAPL --epochs 3 --world_size 2
```

## Notes

1. **DDP requires GPUs for optimal performance** - CPU DDP is supported but slower than single-machine CPU training
2. **Batch size is per GPU** - effective batch size = `batch_size × world_size`
3. **Each process uses a subset of data** - DistributedSampler ensures no overlap
4. **Only rank 0 saves outputs** - prevents file conflicts across processes
5. **Gradients are synchronized** - ensures all processes have identical model weights

## Troubleshooting

**Q: DDP hangs during initialization**
- Check that all processes can communicate (firewall, network)
- Try reducing `--world_size`
- Set `NCCL_DEBUG=INFO` for verbose logging

**Q: Out of memory errors**
- Reduce `--batch_size`
- Reduce `--hidden_dim`
- Use fewer GPUs (`--world_size`)

**Q: Import errors**
- Run from the `model/` directory
- Ensure Python can find the `src/` package

## Development

To add new features:
- **Shared functionality** → `src/models.py`, `src/metrics.py`, `src/utils.py`
- **Single-machine only** → `src/single/`
- **DDP only** → `src/distributed/`

This separation keeps code maintainable and testable.
