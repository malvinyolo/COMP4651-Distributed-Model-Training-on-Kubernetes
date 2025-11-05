# data-pipeline/src/config.py
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This is src folder
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")  # Go up from src to data-pipeline, then to data

# Data Parameters
SEQUENCE_LENGTH = 60
FEATURE_COLUMN = 'Close'  # Start simple
TARGET = 'Close'
TICKER = '^GSPC'  # S&P 500
TEST_SIZE = 0.2
CLASSIFICATION_THRESHOLD = 0.001  # 0.1% minimum return for "Long"

# File Paths
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw/sp500_raw.csv")
REGRESSION_DATA_PATH = os.path.join(DATA_DIR, "processed/sp500_regression.npz")
CLASSIFICATION_DATA_PATH = os.path.join(DATA_DIR, "processed/sp500_classification.npz")
SCALER_PATH = os.path.join(DATA_DIR, "processed/scaler.pkl")

# Create directories
os.makedirs('../data/raw', exist_ok=True)
os.makedirs('../data/processed', exist_ok=True)