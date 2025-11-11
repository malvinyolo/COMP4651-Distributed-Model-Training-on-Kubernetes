"""Data loading, normalization, and DataLoader creation for single-machine training."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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


def build_dataloaders(
    npz_path: Optional[str] = None,
    stock_ticker: Optional[str] = None,
    data_dir: str = "../data-pipeline/data/processed",
    valid_ratio: float = 0.1,
    batch_size: int = 64,
    shuffle_train: bool = False,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, list], int, str]:
    """
    Load NPZ data, split validation, normalize, and create DataLoaders.
    
    Supports two modes:
    1. Direct path: Use npz_path to load a specific file (e.g., sp500_regression.npz)
    2. Stock ticker: Use stock_ticker to load individual stock data (e.g., 'AAPL')
    
    Args:
        npz_path: Direct path to NPZ file (takes precedence if provided)
        stock_ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT'). Used if npz_path is None.
        data_dir: Base directory for data files (used with stock_ticker)
        valid_ratio: Fraction of training data to use for validation (from tail)
        batch_size: Batch size for all loaders
        shuffle_train: Whether to shuffle training data
        num_workers: Number of workers for DataLoader
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        norm_stats: Dict with 'mu' and 'sd' (lists) computed from train data only
        input_dim: Feature dimension (X.shape[-1])
        data_source: String describing the data source (for logging)
    """
    # Determine data source
    if npz_path is not None:
        # Direct path mode
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        data_source = os.path.basename(npz_path)
    elif stock_ticker is not None:
        # Stock ticker mode
        stock_ticker = stock_ticker.upper()
        npz_path = os.path.join(data_dir, "classification", f"{stock_ticker}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Stock data not found: {npz_path}\n"
                f"Available stocks: Check {os.path.join(data_dir, 'classification')} directory"
            )
        data_source = f"{stock_ticker} stock"
    else:
        raise ValueError("Either npz_path or stock_ticker must be provided")
    
    # Load data
    data = np.load(npz_path, allow_pickle=False)
    X_train = data['X_train']  # (N_train, T, F)
    y_train = data['y_train']  # (N_train,)
    X_test = data['X_test']    # (N_test, T, F)
    y_test = data['y_test']    # (N_test,)
    
    # Split validation from tail of training data (chronological)
    n_train = len(X_train)
    n_valid = int(n_train * valid_ratio)
    n_train_actual = n_train - n_valid
    
    X_train_split = X_train[:n_train_actual]
    y_train_split = y_train[:n_train_actual]
    X_valid = X_train[n_train_actual:]
    y_valid = y_train[n_train_actual:]
    
    # Compute normalization stats from training data only
    # Shape: (N, T, F) -> compute mean/std across N and T
    mu = X_train_split.reshape(-1, X_train_split.shape[-1]).mean(axis=0)
    sd = X_train_split.reshape(-1, X_train_split.shape[-1]).std(axis=0)
    
    # Avoid division by zero
    sd = np.where(sd < 1e-8, 1.0, sd)
    
    # Normalize all splits
    X_train_norm = (X_train_split - mu) / sd
    X_valid_norm = (X_valid - mu) / sd
    X_test_norm = (X_test - mu) / sd
    
    # Create datasets
    train_dataset = SeqDataset(X_train_norm, y_train_split)
    val_dataset = SeqDataset(X_valid_norm, y_valid)
    test_dataset = SeqDataset(X_test_norm, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
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
    
    # Store normalization stats
    norm_stats = {
        'mu': mu.tolist(),
        'sd': sd.tolist()
    }
    
    input_dim = X_train.shape[-1]
    
    return train_loader, val_loader, test_loader, norm_stats, input_dim, data_source
