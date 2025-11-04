# data-pipeline/src/data_collector.py
import yfinance as yf
import pandas as pd
import sys
import os
# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TICKER, RAW_DATA_PATH

class DataCollector:
    def __init__(self):
        self.ticker = TICKER
        
    def fetch_data(self, period="10y"):
        """Fetch S&P 500 data from Yahoo Finance"""
        print("Fetching S&P 500 data...")
        
        try:
            data = yf.download(
                tickers=self.ticker,
                period=period,
                interval="1d",
                progress=True
            )
            
            # Keep essential columns and clean
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data = data.dropna()
            
            print(f"Downloaded {len(data)} trading days")
            return data
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def save_data(self, data):
        """Save data to CSV"""
        data.to_csv(RAW_DATA_PATH)
        print(f"Data saved to {RAW_DATA_PATH}")
        
    def get_data_summary(self, data):
        """Generate data summary"""
        return {
            'start_date': data.index.min(),
            'end_date': data.index.max(),
            'total_records': len(data),
            'columns': list(data.columns),
            'missing_values': data.isnull().sum().to_dict()
        }

def main():
    collector = DataCollector()
    data = collector.fetch_data()
    
    if data is not None:
        collector.save_data(data)
        summary = collector.get_data_summary(data)
        
        print("\nData Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
            
        return data
    return None

if __name__ == "__main__":
    main()