#!/usr/bin/env python
"""
Quick test of cli.py with 2 stocks and 3 epochs.
Run: python cli_test.py
"""
import subprocess
import sys

STOCKS = ['AAPL', 'MSFT']  # Just 2 for quick test

def main():
    print("=" * 80)
    print("QUICK TEST: Training 2 stocks with 3 epochs each")
    print("=" * 80)
    
    for i, stock in enumerate(STOCKS, 1):
        print(f"\n[{i}/{len(STOCKS)}] Training {stock}...")
        
        cmd = [
            sys.executable, '-m', 'src.cli',
            '--stock', stock,
            '--epochs', '3',
            '--batch_size', '64',
            '--lr', '1e-3',
            '--hidden_dim', '64'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ {stock} completed")
        except subprocess.CalledProcessError:
            print(f"✗ {stock} failed")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE - Check outputs/ directory")
    print("=" * 80)

if __name__ == '__main__':
    main()
