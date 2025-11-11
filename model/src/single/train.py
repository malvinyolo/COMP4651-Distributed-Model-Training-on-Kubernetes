"""Training, validation, and testing loops for single-machine training."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict
import os
from ..utils import Timer, log
from ..artifacts import save_state_dict


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        loader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        loader: DataLoader (validation or test)
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        avg_loss: Average loss
        y_true: Ground truth values (numpy array)
        y_pred: Predicted values (numpy array)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    
    return avg_loss, y_true, y_pred


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict,
    save_dir: str,
    device: str
) -> Tuple[str, Dict]:
    """
    Train model with early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        cfg: Configuration dict with 'epochs', 'lr', 'patience'
        save_dir: Directory to save checkpoints
        device: Device to train on
    
    Returns:
        ckpt_path: Path to best checkpoint
        timing_info: Dict with timing statistics
    """
    epochs = cfg['epochs']
    lr = cfg['lr']
    patience = cfg.get('patience', 5)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    ckpt_path = os.path.join(save_dir, 'best.ckpt')
    
    epoch_times = []
    
    log(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        with Timer() as timer:
            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            
            # Validate
            val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        
        epoch_time = timer.elapsed
        epoch_times.append(epoch_time)
        
        # Log progress
        log(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | Time: {epoch_time:.2f}s")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_state_dict(model, ckpt_path)
            log(f"  → New best! Saved checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"Early stopping triggered after {epoch} epochs (patience={patience})")
                break
    
    log(f"Training complete. Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})")
    
    # Compute timing statistics
    timing_info = {
        'total_epochs': len(epoch_times),
        'best_epoch': best_epoch,
        'total_time_sec': sum(epoch_times),
        'mean_time_per_epoch': np.mean(epoch_times),
        'std_time_per_epoch': np.std(epoch_times),
        'epoch_times': epoch_times
    }
    
    return ckpt_path, timing_info


def test(
    model: nn.Module,
    test_loader: DataLoader,
    ckpt_path: str,
    device: str
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Load best checkpoint and evaluate on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test DataLoader
        ckpt_path: Path to checkpoint
        device: Device to evaluate on
    
    Returns:
        test_metrics: Dict with test loss
        y_true: Ground truth values
        y_pred: Predicted values
    """
    # Load best checkpoint
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    log(f"Loaded checkpoint from {ckpt_path}")
    
    criterion = nn.MSELoss()
    test_loss, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    
    test_metrics = {'test_loss': test_loss}
    
    return test_metrics, y_true, y_pred
