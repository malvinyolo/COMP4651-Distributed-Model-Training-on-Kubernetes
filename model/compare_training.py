#!/usr/bin/env python
"""
Compare single-machine vs DDP training for benchmarking.

This script runs both training modes and compares:
- Training time
- Throughput (samples/second)
- Final metrics (MSE, MAE, R²)
- GPU utilization (if available)
"""
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime


def run_single_machine(stock: str, epochs: int = 10, batch_size: int = 64) -> dict:
    """Run single-machine training."""
    print(f"\n{'='*80}")
    print(f"SINGLE-MACHINE TRAINING: {stock}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    cmd = [
        'python', 'train_single.py',
        '--stock', stock,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--run_name', f'{stock}_single_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(result.stdout)
        
        # Parse output for metrics
        lines = result.stdout.split('\n')
        test_metrics = None
        for line in lines:
            if 'TEST:' in line:
                # Parse: TEST: MSE=0.283374  MAE=0.4991  R²=-0.1337
                parts = line.split('TEST:')[1].strip().split()
                test_metrics = {
                    'mse': float(parts[0].split('=')[1]),
                    'mae': float(parts[1].split('=')[1]),
                    'r2': float(parts[2].split('=')[1])
                }
                break
        
        return {
            'mode': 'single-machine',
            'stock': stock,
            'success': True,
            'duration': duration,
            'epochs': epochs,
            'batch_size': batch_size,
            'metrics': test_metrics
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        
        return {
            'mode': 'single-machine',
            'stock': stock,
            'success': False,
            'error': str(e)
        }


def run_ddp(stock: str, epochs: int = 10, batch_size: int = 64, world_size: int = 2) -> dict:
    """Run DDP training."""
    print(f"\n{'='*80}")
    print(f"DDP TRAINING: {stock} (World Size: {world_size})")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    cmd = [
        'python', 'train_ddp.py',
        '--stock', stock,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--world_size', str(world_size),
        '--run_name', f'{stock}_ddp_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(result.stdout)
        
        # Parse output for metrics
        lines = result.stdout.split('\n')
        test_metrics = None
        for line in lines:
            if 'TEST:' in line:
                parts = line.split('TEST:')[1].strip().split()
                test_metrics = {
                    'mse': float(parts[0].split('=')[1]),
                    'mae': float(parts[1].split('=')[1]),
                    'r2': float(parts[2].split('=')[1])
                }
                break
        
        return {
            'mode': 'ddp',
            'stock': stock,
            'world_size': world_size,
            'success': True,
            'duration': duration,
            'epochs': epochs,
            'batch_size': batch_size,
            'effective_batch_size': batch_size * world_size,
            'metrics': test_metrics
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        
        return {
            'mode': 'ddp',
            'stock': stock,
            'world_size': world_size,
            'success': False,
            'error': str(e)
        }


def compare_results(single_result: dict, ddp_result: dict):
    """Print comparison between single-machine and DDP results."""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    if not single_result['success'] or not ddp_result['success']:
        print("Cannot compare - one or both runs failed")
        return
    
    print(f"\nStock: {single_result['stock']}")
    print(f"Epochs: {single_result['epochs']}")
    
    print("\n--- Training Time ---")
    single_time = single_result['duration']
    ddp_time = ddp_result['duration']
    speedup = single_time / ddp_time if ddp_time > 0 else 0
    
    print(f"Single-Machine: {single_time:.2f}s")
    print(f"DDP (world_size={ddp_result['world_size']}): {ddp_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    if speedup > 1:
        print(f"✓ DDP is {speedup:.2f}x FASTER")
    else:
        print(f"✗ DDP is {1/speedup:.2f}x SLOWER (unexpected - check overhead)")
    
    print("\n--- Throughput ---")
    # Assuming similar dataset sizes
    single_throughput = single_result['epochs'] / single_time
    ddp_throughput = ddp_result['epochs'] / ddp_time
    
    print(f"Single-Machine: {single_throughput:.2f} epochs/sec")
    print(f"DDP: {ddp_throughput:.2f} epochs/sec")
    
    print("\n--- Batch Size ---")
    print(f"Single-Machine: {single_result['batch_size']}")
    print(f"DDP: {ddp_result['batch_size']} per GPU × {ddp_result['world_size']} GPUs = {ddp_result['effective_batch_size']} effective")
    
    print("\n--- Test Metrics ---")
    if single_result['metrics'] and ddp_result['metrics']:
        single_mse = single_result['metrics']['mse']
        ddp_mse = ddp_result['metrics']['mse']
        
        print(f"Single-Machine MSE: {single_mse:.6f}")
        print(f"DDP MSE:            {ddp_mse:.6f}")
        print(f"Difference:         {abs(single_mse - ddp_mse):.6f}")
        
        if abs(single_mse - ddp_mse) < 0.01:
            print("✓ Metrics are similar (good)")
        else:
            print("⚠ Metrics differ significantly (check batch size effects)")
    
    print("\n" + "="*80)


def main():
    """Run comparison."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Compare single-machine vs DDP training'
    )
    parser.add_argument('--stock', type=str, default='AAPL',
                       help='Stock ticker to test')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (per GPU for DDP)')
    parser.add_argument('--world_size', type=int, default=2,
                       help='World size for DDP')
    parser.add_argument('--skip_single', action='store_true',
                       help='Skip single-machine training')
    parser.add_argument('--skip_ddp', action='store_true',
                       help='Skip DDP training')
    
    args = parser.parse_args()
    
    results = {}
    
    # Run single-machine
    if not args.skip_single:
        results['single'] = run_single_machine(
            args.stock,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    # Run DDP
    if not args.skip_ddp:
        results['ddp'] = run_ddp(
            args.stock,
            epochs=args.epochs,
            batch_size=args.batch_size,
            world_size=args.world_size
        )
    
    # Compare
    if 'single' in results and 'ddp' in results:
        compare_results(results['single'], results['ddp'])
    
    # Save results
    output_file = f'comparison_{args.stock}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
