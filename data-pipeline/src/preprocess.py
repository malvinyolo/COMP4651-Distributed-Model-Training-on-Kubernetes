# data-pipeline/src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

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
    
    def create_regression_sequences(self, data):
        """Create sequences for price prediction (regression)"""
        print("Creating regression sequences...")
        
        # Extract the feature column
        feature_data = data[FEATURE_COLUMN].values.reshape(-1, 1)
        print(f"Feature data shape: {feature_data.shape}")
        
        # Scale the data
        print("Scaling data...")
        scaled_data = self.scaler.fit_transform(feature_data)
        print(f"Scaled data range: [{scaled_data.min():.3f}, {scaled_data.max():.3f}]")
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(scaled_data)):
            X.append(scaled_data[i-SEQUENCE_LENGTH:i, 0])  # Past 60 days
            y.append(scaled_data[i, 0])  # Next day's Close price
            
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X to 3D for LSTM: (samples, timesteps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        print(f"Created {len(X)} regression sequences")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        
        return X, y
    
    def create_classification_sequences(self, data):
        """Create sequences for Long/Short classification"""
        print("Creating classification sequences...")
        
        # Calculate daily returns and create labels
        data = data.copy()
        data['Return'] = data['Close'].pct_change()
        data['Label'] = (data['Return'].shift(-1) > CLASSIFICATION_THRESHOLD).astype(int)
        
        # Remove NaN values created by pct_change and shift
        data = data.dropna()
        
        print(f"Label distribution:")
        print(f"   Long (1): {data['Label'].sum()} samples")
        print(f"   Short (0): {len(data) - data['Label'].sum()} samples")
        print(f"   Threshold: {CLASSIFICATION_THRESHOLD:.3f} ({CLASSIFICATION_THRESHOLD*100:.2f}%)")
        print(f"   Class balance: {data['Label'].mean():.2%} positive")
        
        # Extract and scale the feature column (Close prices)
        feature_data = data['Close'].values.reshape(-1, 1)
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(scaled_features)):
            X.append(scaled_features[i-SEQUENCE_LENGTH:i, 0])  # Past 60 days of prices
            y.append(data['Label'].iloc[i])  # Binary label for next day
            
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X to 3D for LSTM: (samples, timesteps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        print(f"Created {len(X)} classification sequences")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape} (binary labels: 0=Short, 1=Long)")
        
        return X, y
    
    def split_data(self, X, y):
        """Split data sequentially"""
        split_idx = int(len(X) * (1 - TEST_SIZE))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Split: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, file_path):
        """Save processed data"""
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        np.savez(file_path, 
                 X_train=X_train, X_test=X_test, 
                 y_train=y_train, y_test=y_test)
        
        print(f"Data saved to: {file_path}")
    
    def save_scaler(self):
        """Save the scaler for inverse transformations"""
        joblib.dump(self.scaler, SCALER_PATH)
        print(f"Scaler saved to: {SCALER_PATH}")

def main():
    processor = DataPreprocessor()
    
    # Load and clean data
    data = processor.load_and_clean_data()
    
    print("\n" + "="*50)
    print("PROCESSING REGRESSION DATA")
    print("="*50)
    
    # Create regression sequences (price prediction)
    X_reg, y_reg = processor.create_regression_sequences(data)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = processor.split_data(X_reg, y_reg)
    
    # Save regression data
    processor.save_processed_data(X_train_reg, X_test_reg, y_train_reg, y_test_reg, REGRESSION_DATA_PATH)
    processor.save_scaler()
    
    print("\n" + "="*50)
    print("PROCESSING CLASSIFICATION DATA") 
    print("="*50)
    
    # Create classification sequences (Long/Short prediction)
    X_cls, y_cls = processor.create_classification_sequences(data)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = processor.split_data(X_cls, y_cls)
    
    # Save classification data
    processor.save_processed_data(X_train_cls, X_test_cls, y_train_cls, y_test_cls, CLASSIFICATION_DATA_PATH)
    
    # Final summary
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETED!")
    print("="*50)
    print(f"Regression data: {REGRESSION_DATA_PATH}")
    print(f"      - For predicting exact prices")
    print(f"      - X_train: {X_train_reg.shape}, y_train: {y_train_reg.shape}")
    print(f"Classification data: {CLASSIFICATION_DATA_PATH}")
    print(f"      - For Long/Short trading decisions") 
    print(f"      - X_train: {X_train_cls.shape}, y_train: {y_train_cls.shape}")
    print(f"      - Class balance: {np.mean(y_train_cls):.2%} Long positions")
    print(f"Scaler: {SCALER_PATH}")
    print(f"      - For converting predictions back to prices")
    
    return {
        'regression': (X_train_reg, X_test_reg, y_train_reg, y_test_reg),
        'classification': (X_train_cls, X_test_cls, y_train_cls, y_test_cls)
    }

if __name__ == "__main__":
    main()