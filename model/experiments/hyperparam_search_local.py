"""
Sequential hyperparameter search (LOCAL approach).
This demonstrates the baseline performance for comparison with Kubernetes parallel search.
"""
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime


def generate_configs():
    """Generate hyperparameter grid."""
    configs = []
    
    # Grid search space
    hidden_dims = [64, 128, 256]
    learning_rates = [0.0001, 0.001]
    batch_sizes = [16, 32, 64]
    
    config_id = 0
    for hidden_dim in hidden_dims:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                configs.append({
                    'id': config_id,
                    'hidden_dim': hidden_dim,
                    'lr': lr,
                    'batch_size': batch_size,
                    'dropout': 0.1,
                    'epochs': 20
                })
                config_id += 1
    
    return configs


def train_sequential(configs, dataset='sp500_regression.npz'):
    """Train configs one by one (sequential)."""
    print("="*70)
    print("HYPERPARAMETER SEARCH - SEQUENTIAL (LOCAL)")
    print("="*70)
    print(f"Total configurations: {len(configs)}")
    print(f"Dataset: {dataset}")
    print(f"Expected time: ~{len(configs) * 20}s ({len(configs) * 20 / 60:.1f} minutes)")
    print("="*70)
    print()
    
    start_time = time.time()
    results = []
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path.cwd() / 'hp_search_local' / timestamp
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Get paths
    project_root = Path.cwd().parent.parent
    data_path = project_root / 'data-pipeline' / 'data' / 'processed' / dataset
    
    for i, config in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Config {config['id']}: "
              f"hidden={config['hidden_dim']}, lr={config['lr']}, bs={config['batch_size']}")
        
        config_start = time.time()
        
        # Direct Python execution (no Docker)
        config_output = output_base / f"config_{config['id']}"
        
        cmd = [
            sys.executable,
            '-m', 'src.single.cli',
            '--npz_path', str(data_path),
            '--epochs', str(config['epochs']),
            '--hidden_dim', str(config['hidden_dim']),
            '--lr', str(config['lr']),
            '--batch_size', str(config['batch_size']),
            '--dropout', str(config['dropout']),
            '--save_dir', str(config_output)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=project_root / 'model',
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per config
            )
            
            config_time = time.time() - config_start
            
            if result.returncode == 0:
                results.append({
                    'config': config,
                    'time': config_time,
                    'status': 'success'
                })
                print(f"  ✅ Completed in {config_time:.1f}s")
            else:
                results.append({
                    'config': config,
                    'time': config_time,
                    'status': 'failed',
                    'error': result.stderr[-500:] if result.stderr else 'Unknown error'
                })
                print(f"  ❌ Failed after {config_time:.1f}s")
                if result.stderr:
                    print(f"  Error: {result.stderr[-200:]}")
            
            print(f"  Progress: {i+1}/{len(configs)} ({(i+1)/len(configs)*100:.1f}%)")
            print()
            
        except subprocess.TimeoutExpired:
            config_time = time.time() - config_start
            results.append({
                'config': config,
                'time': config_time,
                'status': 'timeout'
            })
            print(f"  ⏱️ Timeout after {config_time:.1f}s")
            print()
            
        except subprocess.TimeoutExpired:
            config_time = 300
            results.append({
                'config': config,
                'time': config_time,
                'status': 'timeout'
            })
            print(f"  ⏱️  Timeout after {config_time:.1f}s")
            print()
    
    total_time = time.time() - start_time
    
    # Save results
    results_summary = {
        'total_configs': len(configs),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'failed'),
        'timeout': sum(1 for r in results if r['status'] == 'timeout'),
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'avg_time_per_config': total_time / len(configs),
        'dataset': dataset,
        'approach': 'sequential',
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    results_file = output_base / 'summary.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print summary
    print()
    print("="*70)
    print("SEQUENTIAL SEARCH COMPLETE")
    print("="*70)
    print(f"Total configs: {len(configs)}")
    print(f"Successful: {results_summary['successful']}")
    print(f"Failed: {results_summary['failed']}")
    print(f"Timeout: {results_summary['timeout']}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Avg time per config: {results_summary['avg_time_per_config']:.1f}s")
    print(f"Results saved to: {results_file}")
    print("="*70)
    
    return results_summary


if __name__ == '__main__':
    # Generate configs
    configs = generate_configs()
    print(f"Generated {len(configs)} configurations for grid search:\n")
    
    # Show sample configs
    print("Sample configurations:")
    for i in range(min(3, len(configs))):
        print(f"  Config {i}: {configs[i]}")
    print(f"  ... ({len(configs) - 3} more)")
    print()
    
    # Estimate time
    estimated_time = len(configs) * 20  # ~20s per config
    print(f"⏱️  Estimated total time: {estimated_time}s ({estimated_time/60:.1f} minutes)")
    print()
    
    # Check if dataset exists
    project_root = Path.cwd().parent.parent
    data_path = project_root / 'data-pipeline' / 'data' / 'processed' / 'sp500_regression.npz'
    
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        print("   Run the data pipeline first:")
        print("   cd data-pipeline && python -m src.run_pipeline")
        sys.exit(1)
    else:
        print(f"✅ Dataset found: sp500_regression.npz")
    
    # Ask for confirmation
    response = input("\nStart sequential hyperparameter search? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        results = train_sequential(configs)
    else:
        print("Cancelled.")
