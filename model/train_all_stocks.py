#!/usr/bin/env python
"""Train model on all available stocks sequentially."""
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
    'device': 'auto',
    'seed': 42
}


def train_stock(stock: str, config: dict) -> dict:
    """
    Train model on a single stock.
    
    Args:
        stock: Stock ticker symbol
        config: Training configuration
    
    Returns:
        Dict with training results or error info
    """
    print(f"\n{'='*80}")
    print(f"Training on {stock}")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', '-m', 'src.cli',
        '--stock', stock,
        '--epochs', str(config['epochs']),
        '--batch_size', str(config['batch_size']),
        '--lr', str(config['lr']),
        '--hidden_dim', str(config['hidden_dim']),
        '--dropout', str(config['dropout']),
        '--patience', str(config['patience']),
        '--device', config['device'],
        '--seed', str(config['seed'])
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        return {'status': 'success', 'stock': stock}
    except subprocess.CalledProcessError as e:
        print(f"ERROR training {stock}:", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return {'status': 'failed', 'stock': stock, 'error': str(e)}


def main():
    """Train all stocks and generate summary report."""
    print(f"\n{'='*80}")
    print(f"TRAINING ALL STOCKS")
    print(f"{'='*80}")
    print(f"Stocks: {', '.join(STOCKS)}")
    print(f"Config: {DEFAULT_CONFIG}")
    print(f"{'='*80}\n")
    
    results = []
    start_time = datetime.now()
    
    for stock in STOCKS:
        result = train_stock(stock, DEFAULT_CONFIG)
        results.append(result)
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"Total stocks: {len(STOCKS)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    if successful:
        print(f"\n✓ Successfully trained: {', '.join([r['stock'] for r in successful])}")
    
    if failed:
        print(f"\n✗ Failed: {', '.join([r['stock'] for r in failed])}")
        for r in failed:
            print(f"  - {r['stock']}: {r.get('error', 'Unknown error')}")
    
    print(f"\n{'='*80}\n")
    
    # Save summary
    summary = {
        'timestamp': start_time.isoformat(),
        'config': DEFAULT_CONFIG,
        'stocks': STOCKS,
        'results': results,
        'total_time_sec': total_time,
        'successful_count': len(successful),
        'failed_count': len(failed)
    }
    
    summary_path = f"outputs/training_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('outputs', exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}\n")
    
    # Exit with error if any failed
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
