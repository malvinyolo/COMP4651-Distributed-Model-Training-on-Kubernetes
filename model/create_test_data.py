#!/usr/bin/env python
"""
Create a small synthetic dataset for quick local testing
"""
import numpy as np
import os

def create_synthetic_data(
    n_train=200,
    n_test=50,
    seq_len=60,
    n_features=1,
    output_path="test_data_small.npz"
):
    """
    Create synthetic binary classification sequences for testing.
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        seq_len: Sequence length (T)
        n_features: Number of features (F)
        output_path: Output path for NPZ file
    """
    print(f"Creating synthetic dataset...")
    print(f"  Train samples: {n_train}")
    print(f"  Test samples: {n_test}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Features: {n_features}")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate training data
    # Create sequences with some pattern: positive samples have upward trend
    X_train = np.random.randn(n_train, seq_len, n_features).astype(np.float32)
    y_train = np.random.randint(0, 2, n_train).astype(np.int64)
    
    # Add pattern: positive samples (y=1) have slight upward trend
    for i in range(n_train):
        if y_train[i] == 1:
            trend = np.linspace(0, 0.5, seq_len).reshape(-1, 1)
            X_train[i] += trend
    
    # Generate test data with same pattern
    X_test = np.random.randn(n_test, seq_len, n_features).astype(np.float32)
    y_test = np.random.randint(0, 2, n_test).astype(np.int64)
    
    for i in range(n_test):
        if y_test[i] == 1:
            trend = np.linspace(0, 0.5, seq_len).reshape(-1, 1)
            X_test[i] += trend
    
    # Save to NPZ
    np.savez(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    print(f"\n✓ Created synthetic data: {output_path}")
    print(f"\nData info:")
    print(f"  X_train: {X_train.shape} ({X_train.dtype})")
    print(f"  y_train: {y_train.shape} ({y_train.dtype})")
    print(f"  X_test:  {X_test.shape} ({X_test.dtype})")
    print(f"  y_test:  {y_test.shape} ({y_test.dtype})")
    
    # Print label distribution
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    print(f"\nLabel distribution:")
    print(f"  Train: class 0: {train_counts[0]}, class 1: {train_counts[1]}")
    print(f"  Test:  class 0: {test_counts[0]}, class 1: {test_counts[1]}")
    
    return output_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create synthetic test data')
    parser.add_argument('--n_train', type=int, default=200, help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=50, help='Number of test samples')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length')
    parser.add_argument('--features', type=int, default=1, help='Number of features')
    parser.add_argument('--output', type=str, default='test_data_small.npz', help='Output path')
    
    args = parser.parse_args()
    
    create_synthetic_data(
        n_train=args.n_train,
        n_test=args.n_test,
        seq_len=args.seq_len,
        n_features=args.features,
        output_path=args.output
    )
    
    print(f"\nYou can now test training with:")
    print(f"  cd src")
    print(f"  python cli.py --npz_path ../{args.output} --epochs 10 --batch_size 32")
