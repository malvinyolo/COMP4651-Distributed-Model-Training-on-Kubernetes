"""
Data module: NPZ loading, train/val split, normalization, DataLoaders
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SeqDataset(Dataset):
    """
    PyTorch Dataset for sequence data.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Features of shape (N, T, F) - float32
            y: Labels of shape (N,) - int64
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_npz(path: str) -> dict:
    """
    Load NPZ file and validate contents.
    
    Args:
        path: Path to NPZ file
    
    Returns:
        Dictionary with X_train, y_train, X_test, y_test
    """
    data = np.load(path, allow_pickle=False)
    
    # Extract arrays
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Validate dtypes - accept float32 or float64 for X
    assert X_train.dtype in [np.float32, np.float64], f"X_train must be float, got {X_train.dtype}"
    assert X_test.dtype in [np.float32, np.float64], f"X_test must be float, got {X_test.dtype}"
    assert y_train.dtype in [np.int64, np.int32, np.int_], f"y_train must be int, got {y_train.dtype}"
    assert y_test.dtype in [np.int64, np.int32, np.int_], f"y_test must be int, got {y_test.dtype}"
    
    # Convert to float32 and int64
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    
    # Validate shapes
    assert X_train.ndim == 3, f"X_train must be 3D (N,T,F), got shape {X_train.shape}"
    assert X_test.ndim == 3, f"X_test must be 3D (N,T,F), got shape {X_test.shape}"
    assert y_train.ndim == 1, f"y_train must be 1D (N,), got shape {y_train.shape}"
    assert y_test.ndim == 1, f"y_test must be 1D (N,), got shape {y_test.shape}"
    
    # Validate labels are binary
    assert set(np.unique(y_train)).issubset({0, 1}), "y_train must contain only 0 and 1"
    assert set(np.unique(y_test)).issubset({0, 1}), "y_test must contain only 0 and 1"
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


def split_train_valid(X_tr: np.ndarray, y_tr: np.ndarray, valid_ratio: float) -> tuple:
    """
    Split train into train/valid, preserving time order (validation is the tail).
    
    Args:
        X_tr: Training features (N, T, F)
        y_tr: Training labels (N,)
        valid_ratio: Fraction of train to use for validation
    
    Returns:
        (X_train, y_train, X_valid, y_valid)
    """
    n_total = len(X_tr)
    n_valid = int(n_total * valid_ratio)
    n_train = n_total - n_valid
    
    # Split chronologically: train is first n_train, valid is last n_valid
    X_train = X_tr[:n_train]
    y_train = y_tr[:n_train]
    X_valid = X_tr[n_train:]
    y_valid = y_tr[n_train:]
    
    return X_train, y_train, X_valid, y_valid


def compute_norm_stats(X_tr: np.ndarray) -> dict:
    """
    Compute normalization statistics (mean, std) per feature across all timesteps.
    
    Args:
        X_tr: Training data of shape (N, T, F)
    
    Returns:
        Dictionary with 'mu' and 'sd' lists of length F
    """
    # Compute mean and std across samples and time (axis 0 and 1)
    # Shape: (F,)
    mu = np.mean(X_tr, axis=(0, 1), keepdims=False)
    sd = np.std(X_tr, axis=(0, 1), keepdims=False)
    
    # Avoid division by zero
    sd = np.where(sd == 0, 1.0, sd)
    
    return {
        'mu': mu.tolist(),
        'sd': sd.tolist()
    }


def apply_norm(X: np.ndarray, norm_stats: dict) -> np.ndarray:
    """
    Apply z-score normalization using precomputed statistics.
    
    Args:
        X: Data of shape (N, T, F)
        norm_stats: Dictionary with 'mu' and 'sd' lists
    
    Returns:
        Normalized data (same shape)
    """
    mu = np.array(norm_stats['mu'], dtype=np.float32)
    sd = np.array(norm_stats['sd'], dtype=np.float32)
    
    # Broadcast: (N, T, F) - (F,) => (N, T, F)
    X_norm = (X - mu) / sd
    
    return X_norm


def build_dataloaders(cfg: dict) -> tuple:
    """
    Build train, validation, and test dataloaders.
    
    Args:
        cfg: Configuration dictionary
    
    Returns:
        (train_loader, val_loader, test_loader, norm_stats, input_dim)
    """
    # Load data
    data = load_npz(cfg['data']['npz_path'])
    X_train_full = data['X_train']
    y_train_full = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Get input dimension
    input_dim = X_train_full.shape[-1]
    
    # Split train into train/valid
    valid_ratio = cfg['data']['valid_from_train']
    X_train, y_train, X_valid, y_valid = split_train_valid(
        X_train_full, y_train_full, valid_ratio
    )
    
    # Compute normalization stats on train only
    norm_stats = None
    if cfg['data']['norm'] == 'zscore':
        norm_stats = compute_norm_stats(X_train)
        X_train = apply_norm(X_train, norm_stats)
        X_valid = apply_norm(X_valid, norm_stats)
        X_test = apply_norm(X_test, norm_stats)
    
    # Create datasets
    train_dataset = SeqDataset(X_train, y_train)
    val_dataset = SeqDataset(X_valid, y_valid)
    test_dataset = SeqDataset(X_test, y_test)
    
    # Create dataloaders
    batch_size = cfg['train']['batch_size']
    shuffle_train = cfg['data']['shuffle_train']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, norm_stats, input_dim
