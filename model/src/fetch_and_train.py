"""Utility that downloads preprocessed data from GCS, runs the training entrypoint, and
optionally uploads training outputs back to GCS.

This script is meant to be executed inside your container image on Vertex AI. It
assumes the container has the repository files in the working directory (as in the
project Dockerfile) and that `google-cloud-storage` is available.

Example usage inside the container:
  python model/src/fetch_and_train.py --gcs-preprocessed gs://comp4651-pipeline-bucket/preprocessed --stock AAPL --upload-output gs://comp4651-pipeline-bucket/models/run-001
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from google.cloud import storage


def download_gcs_prefix(gcs_uri: str, local_dir: str):
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    os.makedirs(local_dir, exist_ok=True)
    count = 0
    for blob in blobs:
        # skip "directory" placeholder blobs
        if blob.name.endswith('/'):
            continue
        rel_path = os.path.relpath(blob.name, prefix) if prefix else blob.name
        local_path = os.path.join(local_dir, rel_path)
        local_parent = os.path.dirname(local_path)
        if local_parent:
            os.makedirs(local_parent, exist_ok=True)
        print(f"Downloading gs://{bucket_name}/{blob.name} -> {local_path}")
        blob.download_to_filename(local_path)
        count += 1
    print(f"Downloaded {count} files from {gcs_uri} to {local_dir}")


def upload_directory_to_gcs(local_dir: str, gcs_uri: str):
    # simple upload mirror
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs-preprocessed", required=True)
    parser.add_argument("--stock", default=None, help="Ticker to train on (mutually exclusive with --npz-path)")
    parser.add_argument("--npz-path", default=None, help="Direct NPZ path inside preprocessed data")
    parser.add_argument("--upload-output", default=None, help="GCS prefix to upload training outputs")
    parser.add_argument("--data-subdir", default="classification", help="Subdirectory under the preprocessed prefix containing NPZs")
    parser.add_argument("--extra-args", default="", help="Extra args to forward to training script (comma-separated)")
    args = parser.parse_args(argv)

    work_dir = "/tmp/preprocessed"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    download_gcs_prefix(args.gcs_preprocessed.rstrip('/') + '/', work_dir)

    # Determine train invocation
    train_cmd = [sys.executable, "model/train_single.py"]
    if args.npz_path:
        npz_local = os.path.join(work_dir, args.npz_path.lstrip('/'))
        train_cmd += ["--npz_path", npz_local]
    elif args.stock:
        train_cmd += ["--stock", args.stock, "--data_dir", os.path.join(work_dir, args.data_subdir)]
    else:
        print("ERROR: either --npz-path or --stock must be provided")
        sys.exit(2)

    if args.extra_args:
        extras = [a for a in args.extra_args.split(',') if a]
        train_cmd += extras

    print("Running training command:", " ".join(train_cmd))
    ret = subprocess.call(train_cmd)
    if ret != 0:
        print(f"Training script exited with code {ret}")
        sys.exit(ret)

    # Optionally upload outputs
    if args.upload_output:
        local_outputs = os.path.abspath("./outputs")
        if os.path.exists(local_outputs):
            upload_directory_to_gcs(local_outputs, args.upload_output)
        else:
            print(f"No local outputs found at {local_outputs}; skipping upload")


if __name__ == '__main__':
    main()
