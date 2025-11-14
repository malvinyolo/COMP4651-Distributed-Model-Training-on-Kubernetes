"""
Test dataset scale limits on local machine.
This script tests progressively larger datasets to find the memory/performance limits.
"""
import subprocess
import time
import sys
import json
from pathlib import Path
from datetime import datetime


def test_dataset_scale(datasets, timeout=300):
    """
    Test training on datasets of different sizes.
    
    Args:
        datasets: List of dataset filenames to test
        timeout: Max seconds to wait per dataset
    """
    print("="*70)
    print("DATASET SCALE TESTING (LOCAL)")
    print("="*70)
    print(f"Testing {len(datasets)} datasets with {timeout}s timeout each\n")
    
    results = []
    
    for i, dataset in enumerate(datasets):
        print(f"[{i+1}/{len(datasets)}] Testing: {dataset}")
        print("-"*70)
        
        start_time = time.time()
        
        # Use direct Python execution (no Docker needed for local testing)
        # Path from experiments/ directory: ../.. to project root
        project_root = Path.cwd().parent.parent
        data_path = project_root / 'data-pipeline' / 'data' / 'processed' / dataset
        output_dir = project_root / 'model' / 'outputs' / f'scale_test_{dataset.replace(".npz", "")}'
        
        cmd = [
            sys.executable,  # Use current Python interpreter
            '-m', 'src.single.cli',
            '--npz_path', str(data_path),
            '--epochs', '5',  # Reduced for faster testing
            '--batch_size', '32',
            '--hidden_dim', '64',
            '--save_dir', str(output_dir)
        ]
        
        try:
            # Run from model directory
            result = subprocess.run(
                cmd,
                cwd=project_root / 'model',
                timeout=timeout,
                capture_output=True,
                text=True
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                status = '✅ SUCCESS'
                print(f"{status} - Completed in {elapsed:.1f}s")
            else:
                status = '❌ FAILED'
                print(f"{status} - Failed after {elapsed:.1f}s")
                print(f"Error: {result.stderr[-500:]}")  # Last 500 chars of error
                
            results.append({
                'dataset': dataset,
                'status': 'success' if result.returncode == 0 else 'failed',
                'time': elapsed
            })
            
        except subprocess.TimeoutExpired:
            elapsed = timeout
            status = '⏱️ TIMEOUT'
            print(f"{status} - Exceeded {timeout}s limit")
            results.append({
                'dataset': dataset,
                'status': 'timeout',
                'time': elapsed
            })
        
        except Exception as e:
            elapsed = time.time() - start_time
            status = '💥 ERROR'
            print(f"{status} - {str(e)}")
            results.append({
                'dataset': dataset,
                'status': 'error',
                'time': elapsed
            })
        
        print()
    
    # Print summary
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Dataset':<30} {'Status':<15} {'Time (s)':<10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['dataset']:<30} {r['status']:<15} {r['time']:.1f}")
    
    print("="*70)
    
    # Interpretation
    print("\nINTERPRETATION:")
    success_count = sum(1 for r in results if r['status'] == 'success')
    timeout_count = sum(1 for r in results if r['status'] == 'timeout')
    failed_count = sum(1 for r in results if r['status'] in ['failed', 'error'])
    
    print(f"  ✅ Succeeded: {success_count}/{len(results)}")
    print(f"  ⏱️  Timeout: {timeout_count}/{len(results)}")
    print(f"  ❌ Failed/Error: {failed_count}/{len(results)}")
    
    if failed_count > 0 or timeout_count > 0:
        print(f"\n💡 Local machine limit: {results[success_count]['dataset'] if success_count < len(results) else 'Not reached'}")
        print("   Datasets beyond this point require cloud/Kubernetes!")
    
    # Save results to JSON
    output_dir = Path.cwd() / 'scale_test_local'
    output_dir.mkdir(exist_ok=True)
    
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'total_datasets': len(results),
        'successful': success_count,
        'timeout': timeout_count,
        'failed': failed_count,
        'results': results
    }
    
    output_file = output_dir / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    # Define test datasets (progressively larger)
    test_datasets = [
        'sp500_regression.npz',        # ~1963 samples, 1.1 MB (baseline)
        'sp500_regression_x10.npz',    # ~19.6K samples, 9.4 MB
        'sp500_regression_x50.npz',    # ~98K samples, 46 MB
        'sp500_regression_x200.npz',   # ~392K samples, 183 MB
    ]
    
    # Check if datasets exist
    # Get the correct path - go up from experiments/ to project root
    data_dir = Path.cwd().parent.parent / 'data-pipeline' / 'data' / 'processed'
    
    print("Checking for datasets...")
    print(f"Looking in: {data_dir}")
    existing_datasets = []
    missing_datasets = []
    
    for dataset in test_datasets:
        if (data_dir / dataset).exists():
            existing_datasets.append(dataset)
            print(f"  ✅ Found: {dataset}")
        else:
            missing_datasets.append(dataset)
            print(f"  ❌ Missing: {dataset}")
    
    if missing_datasets:
        print("\n⚠️  Some datasets are missing. Create them with:")
        print("  cd ../data-pipeline")
        for dataset in missing_datasets:
            if 'x' in dataset:
                mult = dataset.replace('sp500_regression_x', '').replace('.npz', '')
                print(f"  python create_large_dataset.py --multiplier {mult}")
        print()
    
    if not existing_datasets:
        print("\n❌ No datasets found. Please create them first.")
        sys.exit(1)
    
    # Run tests
    print(f"\nWill test {len(existing_datasets)} datasets:")
    for d in existing_datasets:
        print(f"  - {d}")
    
    input("\nPress Enter to start testing...")
    
    results = test_dataset_scale(existing_datasets)
