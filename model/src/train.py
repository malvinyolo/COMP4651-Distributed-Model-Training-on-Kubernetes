"""
Training and evaluation loops with early stopping
"""
import torch
import torch.nn as nn
import numpy as np
from utils import log


def train_one_epoch(model: nn.Module, loader, optimizer, device) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        loader: Training data loader
        optimizer: Optimizer
        device: torch device
    
    Returns:
        Dictionary with average loss
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    
    total_loss = 0.0
    n_batches = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float()  # BCEWithLogitsLoss expects float targets
        
        # Forward pass
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    
    return {"loss": avg_loss}


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> dict:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        loader: Data loader
        device: torch device
    
    Returns:
        Dictionary with loss, y_true, y_prob (numpy arrays)
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    total_loss = 0.0
    n_batches = 0
    
    all_logits = []
    all_labels = []
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float()
        
        # Forward pass
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        total_loss += loss.item()
        n_batches += 1
        
        # Collect predictions and labels
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    
    # Concatenate all batches
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Convert logits to probabilities via sigmoid
    y_prob = 1.0 / (1.0 + np.exp(-all_logits))
    y_true = all_labels.astype(np.int64)
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    
    return {
        "loss": avg_loss,
        "y_true": y_true,
        "y_prob": y_prob
    }


def fit(model: nn.Module,
        train_loader,
        val_loader,
        cfg: dict,
        device,
        save_path: str) -> tuple:
    """
    Train model with early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        cfg: Configuration dictionary
        device: torch device
        save_path: Path to save best checkpoint
    
    Returns:
        (best_checkpoint_path, history_dict)
    """
    from metrics import bin_metrics
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay']
    )
    
    epochs = cfg['train']['epochs']
    patience = cfg['train']['early_stop_patience']
    metric_name = cfg['train']['early_stop_metric']
    
    best_metric = -float('inf') if metric_name == 'auc' else float('inf')
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_prec': [],
        'val_rec': []
    }
    
    log(f"Starting training for {epochs} epochs...")
    log(f"Early stopping: metric={metric_name}, patience={patience}")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_results = evaluate(model, val_loader, device)
        val_metrics = bin_metrics(val_results['y_true'], val_results['y_prob'])
        
        # Log
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_prec'].append(val_metrics['prec'])
        history['val_rec'].append(val_metrics['rec'])
        
        log(f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_results['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.3f} | "
            f"val_auc={val_metrics['auc']:.3f}")
        
        # Early stopping check
        if metric_name == 'auc':
            current_metric = val_metrics['auc']
            improved = current_metric > best_metric
        else:  # val_loss
            current_metric = val_results['loss']
            improved = current_metric < best_metric
        
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            
            # Save best checkpoint
            torch.save(model.state_dict(), save_path)
            log(f"  ✓ New best {metric_name}={best_metric:.4f}, saved checkpoint")
        else:
            patience_counter += 1
            log(f"  No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                log(f"Early stopping triggered at epoch {epoch}")
                break
    
    log(f"Training complete. Best epoch: {best_epoch}, best {metric_name}: {best_metric:.4f}")
    
    return save_path, history


def test(model: nn.Module,
         test_loader,
         checkpoint_path: str,
         device,
         threshold: float = 0.5) -> dict:
    """
    Test model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        checkpoint_path: Path to best checkpoint
        device: torch device
        threshold: Classification threshold
    
    Returns:
        Dictionary with metrics and predictions
    """
    from metrics import bin_metrics, confusion
    
    # Load best checkpoint
    log(f"Loading best checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Evaluate on test set
    test_results = evaluate(model, test_loader, device)
    test_metrics = bin_metrics(test_results['y_true'], test_results['y_prob'], threshold)
    cm = confusion(test_results['y_true'], test_results['y_prob'], threshold)
    
    log(f"Test Results:")
    log(f"  Loss: {test_results['loss']:.4f}")
    log(f"  Acc:  {test_metrics['acc']:.3f}")
    log(f"  AUC:  {test_metrics['auc']:.3f}")
    log(f"  Prec: {test_metrics['prec']:.3f}")
    log(f"  Rec:  {test_metrics['rec']:.3f}")
    
    return {
        'loss': test_results['loss'],
        'metrics': test_metrics,
        'confusion_matrix': cm,
        'y_true': test_results['y_true'],
        'y_prob': test_results['y_prob']
    }
