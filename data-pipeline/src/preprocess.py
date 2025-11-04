# data-pipeline/src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# CONFIGURATION
SEQUENCE_LENGTH = 60
FEATURE_COLUMN = 'Close'
TARGET_COLUMN = 'Close'
TEST_SIZE = 0.2
RAW_DATA_PATH = "../data/raw/sp500_raw.csv"
PROCESSED_DATA_PATH = "../data/processed/sp500_sequences.npz"
SCALER_PATH = "../data/processed/scaler.pkl"

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def load_and_clean_data(self):
        """Load and clean raw data"""
        print("Loading and cleaning data...")
        
        # First load without date parsing to see raw structure
        data = pd.read_csv(RAW_DATA_PATH)
        print(f"Raw data shape: {data.shape}")
        print(f"Raw columns: {data.columns.tolist()}")
        
        # Clean the data - remove any non-numeric rows
        print("Cleaning data...")
        
        # Convert numeric columns, coercing errors to NaN
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove rows with NaN values (where conversion failed)
        initial_count = len(data)
        data = data.dropna()
        cleaned_count = len(data)
        
        print(f"Data cleaning: {initial_count} → {cleaned_count} records "
              f"(removed {initial_count - cleaned_count} non-numeric rows)")
        
        # Now set the index and parse dates
        data = data.set_index(data.columns[0])  # Set first column as index
        try:
            data.index = pd.to_datetime(data.index)
            print("Successfully parsed dates")
        except:
            print("Could not parse dates, using raw index")
        
        print(f"Final columns: {data.columns.tolist()}")
        print(f"Final data shape: {data.shape}")
        
        return data
    
    def create_sequences(self, data):
        """Create LSTM sequences"""
        print("Creating sequences...")
        
        # Extract the feature column
        feature_data = data[FEATURE_COLUMN].values.reshape(-1, 1)
        print(f"Feature data shape: {feature_data.shape}")
        print(f"Feature data sample: {feature_data[:5].flatten()}")  # Show first 5 values
        
        # Scale the data
        print("Scaling data...")
        scaled_data = self.scaler.fit_transform(feature_data)
        print(f"Scaled data range: [{scaled_data.min():.3f}, {scaled_data.max():.3f}]")
        print(f"Scaled data sample: {scaled_data[:5].flatten()}")  # Show first 5 scaled values
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(scaled_data)):
            X.append(scaled_data[i-SEQUENCE_LENGTH:i, 0])  # Past 60 days
            y.append(scaled_data[i, 0])  # Next day's Close price
            
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X to 3D for LSTM: (samples, timesteps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        print(f"Created {len(X)} sequences")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y):
        """Split data sequentially"""
        split_idx = int(len(X) * (1 - TEST_SIZE))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Split: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data"""
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        
        np.savez(PROCESSED_DATA_PATH, 
                 X_train=X_train, X_test=X_test, 
                 y_train=y_train, y_test=y_test)
        joblib.dump(self.scaler, SCALER_PATH)
        
        print(f"Processed data saved:")
        print(f"   {PROCESSED_DATA_PATH}")
        print(f"   {SCALER_PATH}")

def main():
    processor = DataPreprocessor()
    
    # Load and clean data
    data = processor.load_and_clean_data()
    
    # Create sequences
    X, y = processor.create_sequences(data)
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    # Save results
    processor.save_processed_data(X_train, X_test, y_train, y_test)
    
    print(f"\nFinal Data Shapes:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_test:  {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, processor.scaler

if __name__ == "__main__":
    main()