"""Command-line interface for training and evaluation."""
import argparse
import os
import sys
import torch

from .utils import seed_everything, get_device, log
from .datamod import build_dataloaders
from .models import MLPRegressor
from .train import fit, test
from .metrics import regression_metrics
from .artifacts import make_run_dir, save_all_artifacts


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train simple MLP regressor on SP500 data"
    )
    
    # Data
    parser.add_argument(
        '--npz_path',
        type=str,
        default='/mnt/data/sp500_regression.npz',
        help='Path to sp500_regression.npz'
    )
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # System
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./outputs', help='Output directory')
    
    # Data options
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--shuffle_train', action='store_true', help='Shuffle training data')
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Set seed
    seed_everything(args.seed)
    log(f"Set random seed to {args.seed}")
    
    # Get device
    device = get_device(args.device)
    log(f"Using device: {device}")
    
    # Check if data file exists
    if not os.path.exists(args.npz_path):
        log(f"ERROR: Data file not found: {args.npz_path}")
        sys.exit(1)
    
    # Load data
    log(f"Loading data from {args.npz_path}...")
    train_loader, val_loader, test_loader, norm_stats, input_dim = build_dataloaders(
        npz_path=args.npz_path,
        valid_ratio=args.valid_ratio,
        batch_size=args.batch_size,
        shuffle_train=args.shuffle_train
    )
    log(f"Data loaded. Input dim: {input_dim}, "
        f"Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}, "
        f"Test batches: {len(test_loader)}")
    
    # Create run directory
    run_dir = make_run_dir(args.save_dir)
    log(f"Run directory: {run_dir}")
    
    # Initialize model
    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model initialized. Parameters: {n_params:,}")
    
    # Training configuration
    config = {
        'npz_path': args.npz_path,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'patience': args.patience,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'device': device,
        'seed': args.seed,
        'valid_ratio': args.valid_ratio,
        'shuffle_train': args.shuffle_train,
        'n_params': n_params
    }
    
    # Train model
    log("=" * 60)
    ckpt_path, timing_info = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=config,
        save_dir=run_dir,
        device=device
    )
    log("=" * 60)
    
    # Evaluate on validation set (with best checkpoint)
    log("Evaluating on validation set...")
    _, y_val_true, y_val_pred = test(model, val_loader, ckpt_path, device)
    val_metrics = regression_metrics(y_val_true, y_val_pred)
    
    # Evaluate on test set
    log("Evaluating on test set...")
    _, y_test_true, y_test_pred = test(model, test_loader, ckpt_path, device)
    test_metrics = regression_metrics(y_test_true, y_test_pred)
    
    # Save all artifacts
    save_all_artifacts(
        run_dir=run_dir,
        config=config,
        norm_stats=norm_stats,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        timing_info=timing_info
    )
    
    # Print summary
    log("=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    log(f"VAL:  MSE={val_metrics['mse']:.6f}  MAE={val_metrics['mae']:.4f}  R²={val_metrics['r2']:.4f}")
    log(f"TEST: MSE={test_metrics['mse']:.6f}  MAE={test_metrics['mae']:.4f}  R²={test_metrics['r2']:.4f}")
    log(f"Time/epoch: {timing_info['mean_time_per_epoch']:.2f}s (±{timing_info['std_time_per_epoch']:.2f})")
    log(f"Total time: {timing_info['total_time_sec']:.1f}s ({timing_info['total_epochs']} epochs)")
    log(f"Saved → {run_dir}")
    log("=" * 60)


if __name__ == '__main__':
    main()
