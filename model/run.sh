#!/bin/bash
# Quick start script for training

echo "=========================================="
echo "Binary Classifier Training - Quick Start"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the model/ directory"
    exit 1
fi

# Check if data exists
echo "Checking data..."
python test_data.py
if [ $? -ne 0 ]; then
    echo ""
    echo "Please generate the classification data first:"
    echo "  cd ../data-pipeline/src"
    echo "  python run_pipeline.py"
    exit 1
fi

echo ""
echo "=========================================="
echo "Installing dependencies..."
echo "=========================================="
pip install -q -r requirements.txt

echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="
cd src

# Run training with default config (reads from ../data-pipeline/data/processed/)
python cli.py --npz_path ../data-pipeline/data/processed/sp500_classification.npz

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo "Check the outputs/ directory for results"
