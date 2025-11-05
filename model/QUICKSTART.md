# Quick Start Guide

## Local Testing (Recommended First Step!)

Before training on full data, test with a small synthetic dataset:

### Option 1: Automated Quick Test

```bash
cd model
./test_local.sh
```

This will:
1. Create synthetic data (200 train, 50 test samples)
2. Validate the data format
3. Run a quick 10-epoch training (~30 seconds)
4. Save results to `outputs/test_run/`

### Option 2: Manual Quick Test

```bash
cd model

# 1. Create small test data
python create_test_data.py

# 2. Train on it
cd src
python cli.py --config ../config_test.yaml
```

### Option 3: Even Smaller Test (Ultra Fast)

```bash
# Tiny dataset: 50 train, 10 test, 5 epochs
python create_test_data.py --n_train 50 --n_test 10 --output tiny.npz
cd src
python cli.py --npz_path ../tiny.npz --epochs 5 --batch_size 16 --hidden 16
```

---

## Training on Real Data

Once you've tested locally, train on the full S&P 500 dataset:

### Step 1: Ensure Data Exists

```bash
# Check if classification data exists
cd model
python test_data.py
```

If data doesn't exist, generate it:
```bash
cd ../data-pipeline/src
python run_pipeline.py
```

### Step 2: Train with Defaults

```bash
cd model/src
python cli.py --npz_path ../data-pipeline/data/processed/sp500_classification.npz
```

### Step 3: Train with Custom Parameters

**LSTM with larger model:**
```bash
python cli.py \
  --npz_path ../data-pipeline/data/processed/sp500_classification.npz \
  --model lstm \
  --hidden 128 \
  --layers 2 \
  --epochs 50 \
  --batch_size 128
```

**GRU with different learning rate:**
```bash
python cli.py \
  --npz_path ../data-pipeline/data/processed/sp500_classification.npz \
  --model gru \
  --lr 0.0005 \
  --epochs 100 \
  --early_stop_patience 10
```

---

## Quick Reference

```bash
# Test installation
./test_local.sh

# Train with defaults
cd src && python cli.py --npz_path <path>

# Common options
--model lstm|gru          # Model type
--epochs 50               # Training epochs
--batch_size 64           # Batch size
--lr 0.001                # Learning rate
--hidden 128              # Hidden size
--layers 2                # Number of layers
--device cpu|cuda|auto    # Device
--run_name my_run         # Custom name
--config config.yaml      # Config file
```

See README.md for complete documentation.
