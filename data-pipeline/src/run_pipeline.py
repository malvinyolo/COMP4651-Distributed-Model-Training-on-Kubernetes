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

import argparse
import os
import sys
from google.cloud import storage


def upload_directory_to_gcs(local_dir: str, gcs_uri: str):
    """Recursively upload a local directory to a GCS prefix.

    gcs_uri must be of form gs://bucket/path (path optional)
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")

    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            blob_path = os.path.join(prefix, rel_path).replace('\\', '/')
            blob = bucket.blob(blob_path)
            print(f"Uploading {local_path} -> gs://{bucket_name}/{blob_path}")
            blob.upload_from_filename(local_path)


def run_pipeline(mode: str = "all", out_gcs: str | None = None):
    print("STARTING DAILY CLASSIFICATION DATA PIPELINE")
    print("=" * 50)
    print(f"Stocks: {len(STOCKS)}")
    print(f"Data: {DATA_INTERVAL} over {DATA_PERIOD}")
    print(f"Sequence length: {SEQUENCE_LENGTH} days")
    print(f"Prediction: Next-day Long/Short")
    print(f"Threshold: {CLASSIFICATION_THRESHOLD:.3f}")
    print("=" * 50)

    if mode in ("collect", "all"):
        print("\nSTEP 1: Data Collection")
        print("-" * 30)
        success = collect_data()
        if not success:
            print("Data collection failed!")
            return

    if mode in ("preprocess", "all"):
        print("\nSTEP 2: Classification Sequence Creation")
        print("-" * 40)
        preprocess_data()

        # After preprocessing, optionally upload processed directory to GCS
        if out_gcs:
            processed_dir = PROCESSED_DATA_DIR
            if os.path.exists(processed_dir):
                print(f"Uploading processed data from {processed_dir} to {out_gcs} ...")
                try:
                    upload_directory_to_gcs(processed_dir, out_gcs)
                    print("Upload completed.")
                except Exception as e:
                    print(f"Failed to upload processed data to GCS: {e}")
                    raise
            else:
                print(f"Processed data directory not found: {processed_dir}")

    print("\nPIPELINE COMPLETED!")
    print("Classification data ready for distributed training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data pipeline and optionally upload outputs to GCS")
    parser.add_argument("--mode", choices=["collect", "preprocess", "all"], default="all")
    parser.add_argument("--out-gcs", help="Optional GCS prefix to upload processed outputs, e.g. gs://bucket/path")
    args = parser.parse_args()

    run_pipeline(mode=args.mode, out_gcs=args.out_gcs)