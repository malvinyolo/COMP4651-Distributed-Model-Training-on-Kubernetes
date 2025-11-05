#!/usr/bin/env python
"""
Quick test to verify data loading
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

def test_data_format():
    """Test if the NPZ file has the correct format."""
    
    # Path to the data file
    data_path = "../data-pipeline/data/processed/sp500_classification.npz"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("\nPlease run the data pipeline first:")
        print("  cd ../data-pipeline/src")
        print("  python run_pipeline.py")
        return False
    
    print(f"✓ Found data file: {data_path}")
    
    # Load data
    try:
        data = np.load(data_path, allow_pickle=False)
        print(f"✓ Loaded NPZ file")
    except Exception as e:
        print(f"❌ Failed to load NPZ: {e}")
        return False
    
    # Check keys
    required_keys = ['X_train', 'y_train', 'X_test', 'y_test']
    for key in required_keys:
        if key not in data:
            print(f"❌ Missing key: {key}")
            return False
    print(f"✓ All required keys present: {required_keys}")
    
    # Extract arrays
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Check shapes
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape} ({X_train.dtype})")
    print(f"  y_train: {y_train.shape} ({y_train.dtype})")
    print(f"  X_test:  {X_test.shape} ({X_test.dtype})")
    print(f"  y_test:  {y_test.shape} ({y_test.dtype})")
    
    # Validate dtypes
    if X_train.dtype != np.float32:
        print(f"⚠️  X_train should be float32, got {X_train.dtype}")
    if X_test.dtype != np.float32:
        print(f"⚠️  X_test should be float32, got {X_test.dtype}")
    
    # Validate dimensions
    if X_train.ndim != 3:
        print(f"❌ X_train should be 3D (N,T,F), got {X_train.ndim}D")
        return False
    if X_test.ndim != 3:
        print(f"❌ X_test should be 3D (N,T,F), got {X_test.ndim}D")
        return False
    if y_train.ndim != 1:
        print(f"❌ y_train should be 1D (N,), got {y_train.ndim}D")
        return False
    if y_test.ndim != 1:
        print(f"❌ y_test should be 1D (N,), got {y_test.ndim}D")
        return False
    
    print(f"✓ All dimensions correct")
    
    # Check labels are binary
    unique_train = np.unique(y_train)
    unique_test = np.unique(y_test)
    
    print(f"\nLabel distribution:")
    print(f"  Train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Test:  {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    if not set(unique_train).issubset({0, 1}):
        print(f"❌ y_train should only contain 0 and 1, got {unique_train}")
        return False
    if not set(unique_test).issubset({0, 1}):
        print(f"❌ y_test should only contain 0 and 1, got {unique_test}")
        return False
    
    print(f"✓ Labels are binary (0/1)")
    
    # Check for NaN/Inf
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print(f"⚠️  X_train contains NaN or Inf values")
    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        print(f"⚠️  X_test contains NaN or Inf values")
    
    print(f"\n✅ Data format is valid!")
    print(f"\nSequence length (T): {X_train.shape[1]}")
    print(f"Number of features (F): {X_train.shape[2]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    return True


if __name__ == '__main__':
    success = test_data_format()
    sys.exit(0 if success else 1)
