"""
Create progressively larger datasets for scale testing.
This script replicates training data with slight noise to simulate larger datasets.
"""
import numpy as np
import sys
from pathlib import Path
import argparse


def create_scaled_dataset(base_path: str, multiplier: int, output_path: str = None):
    """
    Create a larger dataset by replicating and adding noise.
    
    Args:
        base_path: Path to original .npz file
        multiplier: How many times to replicate the data
        output_path: Optional custom output path
    """
    print(f"Loading base dataset: {base_path}")
    data = np.load(base_path)
    
    # Check what keys are available
    available_keys = list(data.keys())
    print(f"Available keys: {available_keys}")
    
    # Original sizes
    orig_train_size = len(data['X_train'])
    orig_test_size = len(data['X_test'])
    has_val = 'X_val' in data
    
    if has_val:
        orig_val_size = len(data['X_val'])
        print(f"Original sizes: train={orig_train_size}, val={orig_val_size}, test={orig_test_size}")
    else:
        print(f"Original sizes: train={orig_train_size}, test={orig_test_size} (no validation set)")
    
    # Replicate training data with noise
    X_train_list = []
    y_train_list = []
    
    np.random.seed(42)  # For reproducibility
    
    for i in range(multiplier):
        # Add different noise for each replication
        noise_scale = 0.02
        X_noisy = data['X_train'] + np.random.normal(0, noise_scale, data['X_train'].shape)
        y_noisy = data['y_train'] + np.random.normal(0, noise_scale * 0.5, data['y_train'].shape)
        
        X_train_list.append(X_noisy)
        y_train_list.append(y_noisy)
    
    X_train_large = np.concatenate(X_train_list, axis=0)
    y_train_large = np.concatenate(y_train_list, axis=0)
    
    # Keep validation and test sets the same (if they exist)
    if has_val:
        X_val = data['X_val']
        y_val = data['y_val']
    else:
        X_val = None
        y_val = None
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Calculate sizes
    new_train_size = len(X_train_large)
    size_mb = (X_train_large.nbytes + y_train_large.nbytes) / (1024 * 1024)
    
    print(f"\nNew training size: {new_train_size} samples ({size_mb:.1f} MB)")
    print(f"Multiplier: {multiplier}x")
    
    # Save
    if output_path is None:
        base = Path(base_path)
        output_path = str(base.parent / f"{base.stem}_x{multiplier}{base.suffix}")
    
    # Build the save dictionary
    save_dict = {
        'X_train': X_train_large,
        'y_train': y_train_large,
        'X_test': X_test,
        'y_test': y_test
    }
    
    # Add validation data if it exists
    if X_val is not None:
        save_dict['X_val'] = X_val
        save_dict['y_val'] = y_val
    
    # Add normalization stats if they exist
    if 'mu' in data:
        save_dict['mu'] = data['mu']
    if 'sd' in data:
        save_dict['sd'] = data['sd']
    
    np.savez(output_path, **save_dict)
    
    print(f"\n✅ Saved to: {output_path}")
    print(f"   Training samples: {new_train_size:,}")
    if X_val is not None:
        print(f"   Validation samples: {len(X_val):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   File size: {size_mb:.1f} MB")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create scaled datasets for testing')
    parser.add_argument('--base-path', type=str, default=None, help='Path to original .npz file')
    parser.add_argument('--multiplier', type=int, required=True, help='How many times to replicate data')
    parser.add_argument('--output', type=str, default=None, help='Custom output path')
    
    args = parser.parse_args()
    
    # Default to sp500_regression.npz if not specified
    base_path = args.base_path
    if base_path is None:
        # Try to find the default dataset
        script_dir = Path(__file__).parent
        default_path = script_dir / 'data' / 'processed' / 'sp500_regression.npz'
        if default_path.exists():
            base_path = str(default_path)
        else:
            print("Error: No base dataset found. Please specify --base-path")
            sys.exit(1)
    
    create_scaled_dataset(base_path, args.multiplier, args.output)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Default behavior for quick testing
        print("Usage: python create_large_dataset.py --multiplier <number> [--base-path <path>] [--output <path>]")
        print("\nExample:")
        print("  python create_large_dataset.py --multiplier 50")
        print("  python create_large_dataset.py --multiplier 50 --base-path data/processed/AAPL_data.npz")
        print("\nThis will create datasets of different sizes:")
        print("  x10   = ~8.6K samples, ~3 MB")
        print("  x50   = ~43K samples, ~15 MB")
        print("  x200  = ~172K samples, ~60 MB")
        print("  x1000 = ~860K samples, ~300 MB")
    else:
        main()