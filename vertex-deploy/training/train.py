import argparse
import os
import numpy as np
import joblib

from sklearn.neural_network import MLPClassifier
from google.cloud import storage
import tempfile


def load_npz_files(gcs_path):
    """
    Loads X_train and y_train from all NPZ files under a GCS prefix.
    """
    from google.cloud import storage
    import tempfile
    
    if not gcs_path.startswith("gs://"):
        raise ValueError("data_dir must be a GCS path starting with gs://")

    bucket_name, prefix = gcs_path.replace("gs://", "").split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    X_list, y_list = [], []

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Downloading NPZ files from: gs://{bucket_name}/{prefix}")

        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            if not blob.name.endswith(".npz"):
                print(f"Skipping non-NPZ file: {blob.name}")
                continue

            local_path = os.path.join(tmpdir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)

            data = np.load(local_path)

            # Your files contain X_train and y_train
            X_train = data["X_train"]
            y_train = data["y_train"]

            print(f"Loaded {blob.name}: X_train={X_train.shape}, y_train={y_train.shape}")

            X_list.append(X_train)
            y_list.append(y_train)

    if not X_list:
        raise RuntimeError("No training data found in NPZ files.")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return X, y

def upload_to_gcs(bucket_name, blob_path, local_path):
    """Upload local file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded model to gs://{bucket_name}/{blob_path}")


def main(args):
    print("Starting training...")

    # Load data
    X, y = load_npz_files(args.data_dir)
    print(f"Loaded dataset: X={X.shape}, y={y.shape}")

    # Flatten sequences if needed and ensure correct dtypes/shapes
    if X.ndim == 3:
        n_samples, seq_len, n_features = X.shape
        X = X.reshape(n_samples, seq_len * n_features)
        print(f"Flattened X to {X.shape} (seq_len={seq_len}, n_features={n_features})")
    elif X.ndim == 2:
        print("Input X already 2D; proceeding without reshape.")
    else:
        raise ValueError(f"Unexpected X shape {X.shape}. Expected 2D or 3D array.")

    y = np.asarray(y).astype(int).ravel()
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    print(f"Class distribution: {class_counts}")

    # Define simple MLP model
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=50,        # Keep small for quick tests
        random_state=42,
        early_stopping=True,
        n_iter_no_change=5,
        tol=1e-4,
    )

    print("Training model...")
    model.fit(X, y)

    # Save locally
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    # Upload to GCS
    upload_to_gcs(
        args.bucket,
        f"{args.model_dir}/model.pkl",
        model_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True,
                        help="GCS prefix (gs://bucket/prefix) containing .npz files saved by preprocessing.")
    parser.add_argument("--bucket", type=str, required=True,
                        help="GCS bucket name.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path inside bucket to store model.")
    parser.add_argument("--output_dir", type=str, default="/tmp/model",
                        help="Local directory to store model before upload.")

    args = parser.parse_args()
    main(args)
