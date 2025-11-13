"""
Distributed Data Parallel (DDP) training implementation.
Trains a single model across multiple GPUs/nodes with gradient synchronization.
"""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time
from typing import Dict

from ..utils import seed_everything, log, Timer
from ..artifacts import save_state_dict, save_json, save_yaml, make_run_dir
from ..metrics import regression_metrics
from .datamod import build_ddp_dataloaders


def setup_ddp(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Initialize the distributed process group.
    
    Args:
        rank: Unique identifier for this process (0 to world_size-1)
        world_size: Total number of processes
        backend: 'nccl' for GPU, 'gloo' for CPU
    """
    # Set environment variables if not already set
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device for this process
    if backend == 'nccl' and torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    if rank == 0:
        log(f"[Rank {rank}] DDP initialized: world_size={world_size}, backend={backend}")


def cleanup_ddp():
    """Clean up the distributed process group."""
    dist.destroy_process_group()


def train_one_epoch_ddp(
    model: DDP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    rank: int
) -> float:
    """
    Train model for one epoch with DDP.
    
    Args:
        model: DDP-wrapped model
        loader: Training DataLoader with DistributedSampler
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        rank: Process rank
    
    Returns:
        Average training loss across all ranks
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()  # Gradients are automatically synchronized across ranks
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    
    # Average loss across all ranks
    loss_tensor = torch.tensor([avg_loss], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    
    return loss_tensor.item()


def evaluate_ddp(
    model: DDP,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    rank: int,
    sync_loss: bool = True
) -> tuple:
    """
    Evaluate model on validation/test set.
    Only rank 0 returns predictions.
    
    Args:
        model: DDP-wrapped model
        loader: DataLoader (not distributed)
        criterion: Loss function
        device: Device
        rank: Process rank
        sync_loss: If True, synchronize loss across all ranks via all_reduce
    
    Returns:
        avg_loss, y_true, y_pred (only rank 0 gets real data)
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
            
            if rank == 0:  # Only rank 0 collects predictions
                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    
    # Optionally average loss across all ranks
    if sync_loss:
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    
    if rank == 0:
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
    else:
        y_true = np.array([])
        y_pred = np.array([])
    
    return avg_loss, y_true, y_pred


def train_ddp(rank: int, world_size: int, cfg: Dict):
    """
    DDP training function executed by each process.
    
    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        cfg: Configuration dictionary with all hyperparameters
    """
    # Setup DDP
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    setup_ddp(rank, world_size, backend)
    
    # Set device
    if backend == 'nccl':
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')
    
    # Seed for reproducibility
    seed_everything(cfg['seed'] + rank)
    
    # Build dataloaders with DistributedSampler
    train_loader, val_loader, test_loader, norm_stats, input_dim, data_source = build_ddp_dataloaders(
        npz_path=cfg.get('npz_path'),
        stock_ticker=cfg.get('stock_ticker'),
        data_dir=cfg.get('data_dir', '../data-pipeline/data/processed'),
        valid_ratio=cfg.get('valid_ratio', 0.1),
        batch_size=cfg['batch_size'],
        num_workers=cfg.get('num_workers', 0),
        rank=rank,
        world_size=world_size
    )
    
    if rank == 0:
        log(f"Data source: {data_source}")
        log(f"Input dimension: {input_dim}")
        log(f"Training samples per rank: {len(train_loader.dataset) // world_size}")
        log(f"Validation samples: {len(val_loader.dataset)}")
        log(f"Test samples: {len(test_loader.dataset)}")
    
    # Import model here to avoid circular imports
    from ..models import MLPRegressor
    
    # Build model
    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dim=cfg['hidden_dim'],
        dropout=cfg['dropout']
    ).to(device)
    
    # Wrap model with DDP
    if backend == 'nccl':
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        log(f"Model parameters: {total_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = cfg.get('patience', 5)
    epoch_times = []
    best_epoch = 0
    
    if rank == 0:
        log("="*80)
        log("Starting DDP Training")
        log("="*80)
    
    for epoch in range(cfg['epochs']):
        with Timer() as timer:
            # Set epoch for DistributedSampler (shuffles differently each epoch)
            train_loader.sampler.set_epoch(epoch)
            
            # Train
            train_loss = train_one_epoch_ddp(model, train_loader, optimizer, criterion, device, rank)
            
            # Validate
            val_loss, _, _ = evaluate_ddp(model, val_loader, criterion, device, rank)
        
        epoch_time = timer.elapsed
        epoch_times.append(epoch_time)
        
        if rank == 0:
            log(f"Epoch {epoch+1}/{cfg['epochs']} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Time: {epoch_time:.2f}s")
        
        # Early stopping (only rank 0 decides)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save checkpoint (only rank 0)
            if rank == 0:
                save_dir = Path(cfg['save_dir'])
                ckpt_path = save_dir / 'best.ckpt'
                save_state_dict(model.module, str(ckpt_path))
                log("  → New best! Saved checkpoint.")
        else:
            patience_counter += 1
        
        # Broadcast early stopping decision to all ranks
        should_stop = torch.tensor([patience_counter >= patience], dtype=torch.bool, device=device)
        dist.broadcast(should_stop, src=0)
        
        if should_stop.item():
            if rank == 0:
                log(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Synchronize all processes before test evaluation
    dist.barrier()
    
    # Final evaluation on test set (only rank 0)
    if rank == 0:
        log("\n" + "="*80)
        log("EVALUATING ON TEST SET")
        log("="*80)
        
        # Load best checkpoint
        ckpt_path = Path(cfg['save_dir']) / 'best.ckpt'
        model.module.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        # Validation metrics (no DDP sync needed, only rank 0 evaluates)
        val_loss, y_val_true, y_val_pred = evaluate_ddp(model, val_loader, criterion, device, rank, sync_loss=False)
        val_metrics = regression_metrics(y_val_true, y_val_pred)
        val_metrics['loss'] = val_loss
        
        # Test metrics (no DDP sync needed, only rank 0 evaluates)
        test_loss, y_test_true, y_test_pred = evaluate_ddp(model, test_loader, criterion, device, rank, sync_loss=False)
        test_metrics = regression_metrics(y_test_true, y_test_pred)
        test_metrics['loss'] = test_loss
        
        # Timing statistics
        timing = {
            'total_epochs': len(epoch_times),
            'best_epoch': best_epoch,
            'total_time_sec': sum(epoch_times),
            'mean_time_per_epoch': float(np.mean(epoch_times)),
            'std_time_per_epoch': float(np.std(epoch_times)),
            'epoch_times': epoch_times,
            'world_size': world_size,
            'distributed': True
        }
        
        # Save artifacts
        save_dir = Path(cfg['save_dir'])
        save_json(val_metrics, str(save_dir / 'metrics_valid.json'))
        save_json(test_metrics, str(save_dir / 'metrics_test.json'))
        save_json(norm_stats, str(save_dir / 'norm_stats.json'))
        save_json(timing, str(save_dir / 'timing.json'))
        
        # Add DDP info to config
        cfg_with_ddp = cfg.copy()
        cfg_with_ddp['world_size'] = world_size
        cfg_with_ddp['distributed'] = True
        cfg_with_ddp['input_dim'] = input_dim
        cfg_with_ddp['data_source'] = data_source
        save_yaml(cfg_with_ddp, str(save_dir / 'config.yaml'))
        
        # Print summary
        log("\n" + "="*80)
        log("RESULTS SUMMARY (DDP)")
        log("="*80)
        log(f"Data: {data_source} | Input features: {input_dim}")
        log(f"VAL:  MSE={val_metrics['mse']:.6f}  MAE={val_metrics['mae']:.4f}  R²={val_metrics['r2']:.4f}")
        log(f"TEST: MSE={test_metrics['mse']:.6f}  MAE={test_metrics['mae']:.4f}  R²={test_metrics['r2']:.4f}")
        log(f"Time/epoch: {timing['mean_time_per_epoch']:.2f}s (±{timing['std_time_per_epoch']:.2f}s)")
        log(f"Total time: {timing['total_time_sec']:.1f}s ({timing['total_epochs']} epochs)")
        log(f"World size: {world_size} GPUs/processes")
        log(f"Saved → {save_dir}")
        log("="*80)
    
    # Ensure all processes wait for rank 0 to finish
    dist.barrier()
    
    # Cleanup
    cleanup_ddp()


def launch_ddp_training(cfg: Dict):
    """
    Launch DDP training with torch.multiprocessing.
    
    Args:
        cfg: Configuration dictionary
    """
    world_size = cfg.get('world_size', torch.cuda.device_count())
    if world_size == 0:
        world_size = 1  # CPU fallback
    
    # Create save directory (before spawning processes)
    save_dir = make_run_dir(cfg.get('save_dir', './outputs'), cfg.get('run_name'))
    cfg['save_dir'] = str(save_dir)
    
    log(f"Launching DDP training with {world_size} processes...")
    
    if world_size == 1:
        # Single process (still use DDP for consistency)
        train_ddp(0, 1, cfg)
    else:
        # Multi-process
        import torch.multiprocessing as mp
        mp.spawn(
            train_ddp,
            args=(world_size, cfg),
            nprocs=world_size,
            join=True
        )
