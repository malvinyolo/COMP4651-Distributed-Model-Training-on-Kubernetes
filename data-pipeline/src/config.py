# data-pipeline/src/config.py
import os

# Data Parameters
SEQUENCE_LENGTH = 60
FEATURES = ['Close']  # Start simple
TARGET = 'Close'
TICKER = '^GSPC'  # S&P 500
TEST_SIZE = 0.2

# File Paths
RAW_DATA_PATH = '../data/raw/sp500_raw.csv'
PROCESSED_DATA_PATH = '../data/processed/sp500_sequences.npz'
SCALER_PATH = '../data/processed/scaler.pkl'

# Create directories
os.makedirs('../data/raw', exist_ok=True)
os.makedirs('../data/processed', exist_ok=True)