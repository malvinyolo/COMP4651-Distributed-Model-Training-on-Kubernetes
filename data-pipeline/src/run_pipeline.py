# data-pipeline/src/run_pipeline.py
"""
Main pipeline runner for Daily Classification
"""
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))

from data_collector import main as collect_data
from preprocess import main as preprocess_data
from config import *

def main():
    print("STARTING DAILY CLASSIFICATION DATA PIPELINE")
    print("=" * 50)
    print(f"Stocks: {len(STOCKS)}")
    print(f"Data: {DATA_INTERVAL} over {DATA_PERIOD}")
    print(f"Sequence length: {SEQUENCE_LENGTH} days")
    print(f"Prediction: Next-day Long/Short")
    print(f"Threshold: {CLASSIFICATION_THRESHOLD:.3f}")
    print("=" * 50)
    
    # Step 1: Data Collection
    print("\nSTEP 1: Data Collection")
    print("-" * 30)
    success = collect_data()
    
    if not success:
        print("Data collection failed!")
        return
    
    # Step 2: Data Processing for Classification
    print("\nSTEP 2: Classification Sequence Creation")
    print("-" * 40)
    preprocess_data()
    
    print("\nPIPELINE COMPLETED!")
    print("Classification data ready for distributed training!")

if __name__ == "__main__":
    main()