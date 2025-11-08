# Quick Start Guide: Multi-Stock Training# Quick Start Guide



## 🎯 Available Training Options## Local Testing (Recommended First Step!)



### 1. Train on Individual StockBefore training on full data, test with a small synthetic dataset:

```bash

# Train on specific stock (AAPL, AMZN, JNJ, JPM, MSFT, TSLA, XOM)### Option 1: Automated Quick Test

python -m src.cli --stock AAPL

```bash

# With custom settingscd model

python -m src.cli --stock MSFT --epochs 50 --batch_size 128 --lr 1e-3./test_local.sh

``````



### 2. Train on S&P 500 IndexThis will:

```bash1. Create synthetic data (200 train, 50 test samples)

python -m src.cli --npz_path ../data-pipeline/data/processed/sp500_regression.npz2. Validate the data format

```3. Run a quick 10-epoch training (~30 seconds)

4. Save results to `outputs/test_run/`

### 3. Train All Stocks at Once

```bash### Option 2: Manual Quick Test

python train_all_stocks.py

``````bash

cd model

## 📊 Available Stocks

# 1. Create small test data

| Ticker | Company | Features |python create_test_data.py

|--------|---------|----------|

| AAPL | Apple | 5 |# 2. Train on it

| AMZN | Amazon | 5 |cd src

| JNJ | Johnson & Johnson | 5 |python cli.py --config ../config_test.yaml

| JPM | JPMorgan Chase | 5 |```

| MSFT | Microsoft | 5 |

| TSLA | Tesla | 5 |### Option 3: Even Smaller Test (Ultra Fast)

| XOM | Exxon Mobil | 5 |

```bash

*5 features: Open, High, Low, Close, Volume (normalized)*# Tiny dataset: 50 train, 10 test, 5 epochs

python create_test_data.py --n_train 50 --n_test 10 --output tiny.npz

## 🔧 Common Commandscd src

python cli.py --npz_path ../tiny.npz --epochs 5 --batch_size 16 --hidden 16

### Quick Test (3 epochs)```

```bash

python -m src.cli --stock AAPL --epochs 3---

```

## Training on Real Data

### Production Training

```bashOnce you've tested locally, train on the full S&P 500 dataset:

python -m src.cli --stock AAPL \

    --epochs 50 \### Step 1: Ensure Data Exists

    --batch_size 64 \

    --lr 1e-3 \```bash

    --hidden_dim 128 \# Check if classification data exists

    --dropout 0.1 \cd model

    --patience 5 \python test_data.py

    --device auto```

```

If data doesn't exist, generate it:

### CPU-Only Training```bash

```bashcd ../data-pipeline/src

python -m src.cli --stock AAPL --device cpupython run_pipeline.py

``````



### With Different Seed### Step 2: Train with Defaults

```bash

python -m src.cli --stock AAPL --seed 123```bash

```cd model/src

python cli.py --npz_path ../data-pipeline/data/processed/sp500_classification.npz

## 📁 Output Structure```



After training, outputs are saved in:### Step 3: Train with Custom Parameters

```

outputs/**LSTM with larger model:**

└── run_YYYYMMDD_HHMMSS/```bash

    ├── best.ckpt              # Best model checkpointpython cli.py \

    ├── config.yaml            # Training configuration  --npz_path ../data-pipeline/data/processed/sp500_classification.npz \

    ├── metrics_valid.json     # Validation metrics  --model lstm \

    ├── metrics_test.json      # Test metrics  --hidden 128 \

    ├── norm_stats.json        # Normalization statistics  --layers 2 \

    └── timing.json            # Timing information  --epochs 50 \

```  --batch_size 128

```

## 🎨 Example Output

**GRU with different learning rate:**

``````bash

[2025-11-08 17:07:46] Set random seed to 42python cli.py \

[2025-11-08 17:07:46] Using device: mps  --npz_path ../data-pipeline/data/processed/sp500_classification.npz \

[2025-11-08 17:07:46] Loading data for stock: AAPL...  --model gru \

[2025-11-08 17:07:46] Data loaded. Input dim: 5, Train batches: 14, Val batches: 2, Test batches: 4  --lr 0.0005 \

...  --epochs 100 \

============================================================  --early_stop_patience 10

RESULTS SUMMARY```

============================================================

Data: AAPL stock | Input features: 5---

VAL:  MSE=0.000532  MAE=0.0184  R²=0.6120

TEST: MSE=0.007773  MAE=0.0798  R²=0.3734## Quick Reference

Time/epoch: 0.17s (±0.51)

Total time: 5.3s (32 epochs)```bash

Saved → ./outputs/run_20251108_170746# Test installation

============================================================./test_local.sh

```

# Train with defaults

## 🚀 Benchmark All Stockscd src && python cli.py --npz_path <path>



To train on all 7 stocks and get a comprehensive comparison:# Common options

--model lstm|gru          # Model type

```bash--epochs 50               # Training epochs

python train_all_stocks.py--batch_size 64           # Batch size

```--lr 0.001                # Learning rate

--hidden 128              # Hidden size

This will:--layers 2                # Number of layers

- Train each stock sequentially--device cpu|cuda|auto    # Device

- Save individual results for each--run_name my_run         # Custom name

- Generate a summary JSON with all results--config config.yaml      # Config file

- Report success/failure for each stock```



## 🔍 Check Available StocksSee README.md for complete documentation.


```bash
ls -1 ../data-pipeline/data/processed/classification/*.npz
```
