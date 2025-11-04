# src/run_pipeline.py
"""
Main script to run the entire data pipeline
Usage: python run_pipeline.py
"""
from data_collector import main as collect_data
from preprocess import main as preprocess_data

def main():
    print("STARTING COMPLETE DATA PIPELINE")
    print("=" * 50)
    
    # Step 1: Collect data
    print("\nSTEP 1: Data Collection")
    data = collect_data()
    
    if data is None:
        print("Data collection failed!")
        return
    
    # Step 2: Preprocess data
    print("\nSTEP 2: Data Preprocessing")
    preprocess_data()
    
    print("\nPIPELINE COMPLETED SUCCESSFULLY!")
    print("Data is ready for the Modeling Lead!")

if __name__ == "__main__":
    main()