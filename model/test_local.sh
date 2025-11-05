#!/bin/bash
# Quick local test - creates small data and runs fast training

set -e  # Exit on error

echo "=========================================="
echo "Quick Local Test"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

# Step 1: Create small synthetic dataset
echo "Step 1: Creating small test dataset..."
python create_test_data.py --n_train 200 --n_test 50 --seq_len 60
echo ""

# Step 2: Test data loading
echo "Step 2: Validating data format..."
cat > test_loader.py << 'EOF'
import sys
sys.path.insert(0, 'src')
from datamod import load_npz
data = load_npz('test_data_small.npz')
print(f"✓ Data loaded successfully")
print(f"  X_train: {data['X_train'].shape}")
print(f"  y_train: {data['y_train'].shape}")
EOF

python test_loader.py
rm test_loader.py
echo ""

# Step 3: Run quick training
echo "Step 3: Running quick training (10 epochs)..."
echo ""
cd src
python cli.py --npz_path ../test_data_small.npz --epochs 10 --batch_size 32 --device cpu --run_name test_run

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Check outputs/test_run/ for results"
echo ""
echo "To run a full training on real data:"
echo "  cd src"
echo "  python cli.py --npz_path ../data-pipeline/data/processed/sp500_classification.npz"
