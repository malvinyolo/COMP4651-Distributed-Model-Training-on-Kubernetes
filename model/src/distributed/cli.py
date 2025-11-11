"""Command-line interface for Distributed Data Parallel (DDP) training."""
import argparse
import torch
from pathlib import Path

from .train import launch_ddp_training
from ..utils import log


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='DDP Training for Stock Time-Series Regression',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--npz_path', type=str,
                           help='Direct path to NPZ file')
    data_group.add_argument('--stock', type=str,
                           help='Stock ticker (e.g., AAPL, MSFT, TSLA, JNJ, JPM, AMZN, XOM)')
    
    parser.add_argument('--data_dir', type=str,
                       default='../data-pipeline/data/processed',
                       help='Base directory for stock data files')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size per GPU/process')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout probability')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                       help='Validation split ratio')
    
    # DDP arguments
    parser.add_argument('--world_size', type=int, default=None,
                       help='Number of processes (default: all available GPUs, min 1)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader workers per process')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./outputs',
                       help='Directory to save outputs')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Run name (default: auto-generated timestamp)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    
    log("="*80)
    log("DISTRIBUTED DATA PARALLEL (DDP) TRAINING")
    log("="*80)
    log(f"CUDA Available: {cuda_available}")
    log(f"GPU Count: {gpu_count}")
    
    if not cuda_available:
        log("WARNING: No CUDA GPUs detected. Falling back to CPU.")
        log("DDP will use 'gloo' backend for CPU training (slower).")
    
    # Determine world size
    if args.world_size is None:
        args.world_size = max(1, gpu_count)
    
    log(f"World Size: {args.world_size} processes")
    log("="*80)
    
    # Build configuration
    cfg = {
        'npz_path': args.npz_path,
        'stock_ticker': args.stock,
        'data_dir': args.data_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'patience': args.patience,
        'valid_ratio': args.valid_ratio,
        'world_size': args.world_size,
        'num_workers': args.num_workers,
        'seed': args.seed,
        'save_dir': args.save_dir,
        'run_name': args.run_name,
    }
    
    # Launch DDP training
    try:
        launch_ddp_training(cfg)
    except Exception as e:
        log(f"ERROR: DDP training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
