"""Training, validation, and testing loops for single-machine training."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict
import os
from pathlib import Path
from ..utils import Timer, log
from ..artifacts import save_state_dict
from ..metrics import regression_metrics


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
) -> Tuple[str, str, Dict]:
    """
    Train model for fixed number of epochs (NO EARLY STOPPING).
    Saves both best (lowest val loss) and final models.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        cfg: Configuration dict with 'epochs', 'lr'
        save_dir: Directory to save checkpoints
        device: Device to train on
    
    Returns:
        best_ckpt_path: Path to best checkpoint
        final_ckpt_path: Path to final checkpoint
        timing_info: Dict with timing and training statistics
    """
    epochs = cfg['epochs']
    lr = cfg['lr']
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Track best model
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    best_epoch = 0
    
    # Paths for checkpoints
    save_dir = Path(save_dir)
    best_ckpt_path = save_dir / 'best.ckpt'
    final_ckpt_path = save_dir / 'final.ckpt'
    
    # Track training history
    epoch_times = []
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    log("="*80)
    log("Starting Single-Machine Training (Fixed Epochs)")
    log("="*80)
    
    for epoch in range(1, epochs + 1):
        with Timer() as timer:
            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            
            # Validate and get predictions
            val_loss, y_val_true, y_val_pred = evaluate(model, val_loader, criterion, device)
        
        epoch_time = timer.elapsed
        epoch_times.append(epoch_time)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Calculate R² score
        val_metrics = regression_metrics(y_val_true, y_val_pred)
        val_r2 = val_metrics['r2']
        val_r2_scores.append(val_r2)
        
        # Log progress
        log(f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val R²: {val_r2:.4f} | "
            f"Time: {epoch_time:.2f}s")
        
        # Save best model (by validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_r2 = val_r2
            best_epoch = epoch
            save_state_dict(model, str(best_ckpt_path))
            log(f"  → New best! Loss={val_loss:.6f}, R²={val_r2:.4f}")
    
    # Save final model
    save_state_dict(model, str(final_ckpt_path))
    log(f"\n✅ Training completed! {epochs} epochs finished.")
    log(f"Best model: epoch {best_epoch} (Val Loss={best_val_loss:.6f}, Val R²={best_val_r2:.4f})")
    
    # Compute timing statistics with training history
    timing_info = {
        'total_epochs': len(epoch_times),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_r2': best_val_r2,
        'total_time_sec': sum(epoch_times),
        'mean_time_per_epoch': float(np.mean(epoch_times)),
        'std_time_per_epoch': float(np.std(epoch_times)),
        'epoch_times': epoch_times,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_r2_scores': val_r2_scores,
        'distributed': False
    }
    
    return str(best_ckpt_path), str(final_ckpt_path), timing_info


def test(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    best_ckpt_path: str,
    final_ckpt_path: str,
    device: str
) -> Tuple[Dict, Dict, Dict, np.ndarray, np.ndarray]:
    """
    Load both best and final checkpoints, evaluate both, and return the better one.
    
    Args:
        model: PyTorch model
        val_loader: Validation DataLoader (to re-evaluate)
        test_loader: Test DataLoader
        best_ckpt_path: Path to best checkpoint
        final_ckpt_path: Path to final checkpoint
        device: Device to evaluate on
    
    Returns:
        test_metrics: Dict with test metrics (from better model)
        test_metrics_best: Dict with test metrics from best model
        test_metrics_final: Dict with test metrics from final model
        y_true: Ground truth values
        y_pred: Predicted values
    """
    criterion = nn.MSELoss()
    
    log("\n" + "="*80)
    log("EVALUATING ON TEST SET")
    log("="*80)
    
    # 1. Evaluate best model (lowest validation loss)
    state_dict = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Re-evaluate on validation set
    val_loss, y_val_true, y_val_pred = evaluate(model, val_loader, criterion, device)
    val_metrics = regression_metrics(y_val_true, y_val_pred)
    
    # Evaluate on test set
    test_loss_best, y_test_true_best, y_test_pred_best = evaluate(model, test_loader, criterion, device)
    test_metrics_best = regression_metrics(y_test_true_best, y_test_pred_best)
    test_metrics_best['loss'] = test_loss_best
    
    log(f"\n[BEST MODEL]")
    log(f"  Val:  Loss={val_loss:.6f}, R²={val_metrics['r2']:.4f}")
    log(f"  Test: Loss={test_loss_best:.6f}, R²={test_metrics_best['r2']:.4f}, MAE={test_metrics_best['mae']:.4f}")
    
    # 2. Evaluate final model (last epoch)
    state_dict = torch.load(final_ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    
    test_loss_final, y_test_true_final, y_test_pred_final = evaluate(model, test_loader, criterion, device)
    test_metrics_final = regression_metrics(y_test_true_final, y_test_pred_final)
    test_metrics_final['loss'] = test_loss_final
    
    log(f"\n[FINAL MODEL]")
    log(f"  Test: Loss={test_loss_final:.6f}, R²={test_metrics_final['r2']:.4f}, MAE={test_metrics_final['mae']:.4f}")
    
    # 3. Choose better model based on test R²
    if test_metrics_final['r2'] > test_metrics_best['r2']:
        log(f"\n✅ Final model performs better! Using final model.")
        test_metrics = test_metrics_final
        y_true = y_test_true_final
        y_pred = y_test_pred_final
        best_model_type = 'final'
    else:
        log(f"\n✅ Best model performs better! Using best model.")
        test_metrics = test_metrics_best
        y_true = y_test_true_best
        y_pred = y_test_pred_best
        best_model_type = 'best'
    
    # Add selection info to metrics
    test_metrics['selected_model'] = best_model_type
    test_metrics_best['model_type'] = 'best'
    test_metrics_final['model_type'] = 'final'
    
    return test_metrics, test_metrics_best, test_metrics_final, y_true, y_pred
