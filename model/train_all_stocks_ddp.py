#!/usr/bin/env python
"""Train model on all available stocks sequentially using DDP."""
import subprocess
import sys
import json
import os
from datetime import datetime

# List of stocks to train on
STOCKS = ['AAPL', 'AMZN', 'JNJ', 'JPM', 'MSFT', 'TSLA', 'XOM']

# Training configuration (can be overridden with command-line args)
DEFAULT_CONFIG = {
    'epochs': 50,
    'batch_size': 64,
    'lr': 1e-3,
    'hidden_dim': 64,
    'dropout': 0.1,
    'patience': 5,
    'world_size': 2,  # Number of GPUs/processes
    'num_workers': 0,
    'seed': 42
}


def train_stock_ddp(stock: str, config: dict) -> dict:
    """
    Train model on a single stock using DDP.
    
    Args:
        stock: Stock ticker symbol
        config: Training configuration
    
    Returns:
        Dict with training results or error info
    """
    print(f"\n{'='*80}")
    print(f"Training on {stock} with DDP (World Size: {config['world_size']})")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'train_ddp.py',
        '--stock', stock,
        '--epochs', str(config['epochs']),
        '--batch_size', str(config['batch_size']),
        '--lr', str(config['lr']),
        '--hidden_dim', str(config['hidden_dim']),
        '--dropout', str(config['dropout']),
        '--patience', str(config['patience']),
        '--world_size', str(config['world_size']),
        '--num_workers', str(config['num_workers']),
        '--seed', str(config['seed'])
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        return {
            'stock': stock,
            'status': 'success',
            'returncode': 0
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error training {stock}:")
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        
        return {
            'stock': stock,
            'status': 'failed',
            'returncode': e.returncode,
            'error': str(e)
        }


def main():
    """Train on all stocks sequentially."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train on all stocks using DDP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                       help='Batch size per GPU/process')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'],
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=DEFAULT_CONFIG['hidden_dim'],
                       help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=DEFAULT_CONFIG['dropout'],
                       help='Dropout probability')
    parser.add_argument('--patience', type=int, default=DEFAULT_CONFIG['patience'],
                       help='Early stopping patience')
    parser.add_argument('--world_size', type=int, default=DEFAULT_CONFIG['world_size'],
                       help='Number of GPUs/processes for DDP')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_CONFIG['num_workers'],
                       help='DataLoader workers per process')
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'],
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Update config with command-line args
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'patience': args.patience,
        'world_size': args.world_size,
        'num_workers': args.num_workers,
        'seed': args.seed
    }
    
    print("="*80)
    print("BATCH DDP TRAINING ON ALL STOCKS")
    print("="*80)
    print(f"Stocks: {', '.join(STOCKS)}")
    print(f"Configuration:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    print("="*80)
    
    start_time = datetime.now()
    results = []
    
    for stock in STOCKS:
        result = train_stock_ddp(stock, config)
        results.append(result)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count
    
    print(f"Total stocks: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {duration}")
    
    if failed_count > 0:
        print("\nFailed stocks:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {r['stock']}: {r.get('error', 'Unknown error')}")
    
    print("="*80)
    
    # Save results to JSON
    results_file = f'batch_ddp_results_{start_time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
