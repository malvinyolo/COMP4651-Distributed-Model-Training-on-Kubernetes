# check_columns.py
import pandas as pd

# Load the data and check columns
data = pd.read_csv('../data/raw/sp500_raw.csv')
print("Data shape:", data.shape)
print("Columns:", data.columns.tolist())
print("Column types:", data.dtypes)
print("\nFirst 3 rows:")
print(data.head(3))