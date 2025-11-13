"""Data loading, normalization, and DataLoader creation for DDP training."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple, Dict, Optional
import os


class SeqDataset(Dataset):
    """Dataset for sequence regression."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: shape (N, T, F) - sequence inputs
            y: shape (N,) - regression targets
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def __len__(self) -> int:
        return len(self.X)


def build_ddp_dataloaders(
    npz_path=None,
    stock_ticker=None,
    rank=0,
    world_size=2,
    batch_size=64,
    valid_ratio=0.1,
    num_workers=0,
    data_dir='../data-pipeline/data/processed'
):
    """Build train, validation, and test data loaders for DDP."""
    from pathlib import Path
    from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
    import torch
    import numpy as np
    
    # Support both local and Docker paths
    if Path(data_dir).exists():
        base_path = Path(data_dir)
    elif Path('/data').exists():
        base_path = Path('/data')
    else:
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Determine data source and load data
    if npz_path:
        data_path = Path(npz_path)
        data_source = f"Direct NPZ: {data_path.name}"
    elif stock_ticker:
        data_path = base_path / 'classification' / f'{stock_ticker}.npz'
        data_source = f"{stock_ticker} stock"
    else:
        raise ValueError("Must provide either npz_path or stock_ticker")
    
    # Load data
    data = np.load(data_path, allow_pickle=False)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Get input dimension
    input_dim = X_train.shape[-1]  # (N, T, F) -> F
    
    # Split train into train and validation
    n_valid = int(len(X_train) * valid_ratio)
    n_train = len(X_train) - n_valid
    
    X_train_split = X_train[:-n_valid]
    y_train_split = y_train[:-n_valid]
    X_valid = X_train[-n_valid:]
    y_valid = y_train[-n_valid:]
    
    # Normalize using training statistics only
    mu = X_train_split.mean(axis=(0, 1), keepdims=True)
    sd = X_train_split.std(axis=(0, 1), keepdims=True) + 1e-8
    
    # Store normalization stats
    norm_stats = {
        'mean': mu,
        'std': sd
    }
    
    X_train_norm = (X_train_split - mu) / sd
    X_valid_norm = (X_valid - mu) / sd
    X_test_norm = (X_test - mu) / sd
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_norm).float(),
        torch.from_numpy(y_train_split).float()
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(X_valid_norm).float(),
        torch.from_numpy(y_valid).float()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test_norm).float(),
        torch.from_numpy(y_test).float()
    )
    
    # Calculate appropriate batch sizes for small datasets
    samples_per_rank_val = len(valid_dataset) // world_size
    samples_per_rank_test = len(test_dataset) // world_size
    
    # Adjust batch size if necessary
    val_batch_size = min(batch_size, max(1, samples_per_rank_val))
    test_batch_size = min(batch_size, max(1, samples_per_rank_test))
    
    if rank == 0:
        if val_batch_size < batch_size:
            print(f"[WARNING] Reduced validation batch_size from {batch_size} to {val_batch_size} "
                  f"(only {samples_per_rank_val} samples per rank)")
        if test_batch_size < batch_size:
            print(f"[WARNING] Reduced test batch_size from {batch_size} to {test_batch_size} "
                  f"(only {samples_per_rank_test} samples per rank)")
    
    # Create DistributedSamplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False  # Keep all training samples
    )
    
    # For validation/test: drop_last=True to ensure consistent batch counts
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True  # Drop incomplete batches
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True  # Drop incomplete batches
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False  # Set to False for CPU
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=val_batch_size,  # Use adjusted size
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,  # Use adjusted size
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # Return all 6 values expected by train.py
    return train_loader, valid_loader, test_loader, norm_stats, input_dim, data_source
