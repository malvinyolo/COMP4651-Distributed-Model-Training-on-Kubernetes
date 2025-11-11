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
    npz_path: Optional[str] = None,
    stock_ticker: Optional[str] = None,
    data_dir: str = "../data-pipeline/data/processed",
    valid_ratio: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, list], int, str]:
    """
    Build dataloaders with DistributedSampler for DDP training.
    
    Each rank gets a different subset of the training data.
    Validation and test data are replicated (not distributed).
    
    Args:
        npz_path: Direct path to NPZ file (takes precedence if provided)
        stock_ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        data_dir: Base directory for data files
        valid_ratio: Fraction of training data to use for validation
        batch_size: Batch size per rank
        num_workers: Number of DataLoader workers per rank
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
    
    Returns:
        train_loader: Training DataLoader with DistributedSampler
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        norm_stats: Normalization statistics
        input_dim: Feature dimension
        data_source: String describing the data source
    """
    # Determine data source
    if npz_path is not None:
        data_path = npz_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"NPZ file not found: {data_path}")
        data_source = os.path.basename(data_path)
    elif stock_ticker is not None:
        stock_ticker = stock_ticker.upper()
        data_path = os.path.join(data_dir, "classification", f"{stock_ticker}.npz")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Stock data not found: {data_path}\n"
                f"Available stocks: Check {os.path.join(data_dir, 'classification')} directory"
            )
        data_source = f"{stock_ticker} stock"
    else:
        raise ValueError("Either npz_path or stock_ticker must be provided")
    
    # Load data (all ranks load the same data)
    data = np.load(data_path, allow_pickle=False)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Split validation from tail of training data
    n_train = len(X_train)
    n_valid = int(n_train * valid_ratio)
    n_train_actual = n_train - n_valid
    
    X_train_split = X_train[:n_train_actual]
    y_train_split = y_train[:n_train_actual]
    X_valid = X_train[n_train_actual:]
    y_valid = y_train[n_train_actual:]
    
    # Compute normalization stats from training data only
    mu = X_train_split.reshape(-1, X_train_split.shape[-1]).mean(axis=0)
    sd = X_train_split.reshape(-1, X_train_split.shape[-1]).std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    
    # Normalize all splits
    X_train_norm = (X_train_split - mu) / sd
    X_valid_norm = (X_valid - mu) / sd
    X_test_norm = (X_test - mu) / sd
    
    # Create datasets
    train_dataset = SeqDataset(X_train_norm, y_train_split)
    val_dataset = SeqDataset(X_valid_norm, y_valid)
    test_dataset = SeqDataset(X_test_norm, y_test)
    
    # Create DistributedSampler for training data
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # Use DistributedSampler
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation and test loaders are NOT distributed
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    norm_stats = {'mu': mu.tolist(), 'sd': sd.tolist()}
    input_dim = X_train.shape[-1]
    
    return train_loader, val_loader, test_loader, norm_stats, input_dim, data_source
