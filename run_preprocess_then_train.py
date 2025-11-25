"""Submit a preprocessing CustomJob and then a training CustomJob on Vertex AI.

Usage examples (after installing google-cloud-aiplatform):

  python run_preprocess_then_train.py

Or override defaults:

  python run_preprocess_then_train.py \
    --project sp500-distributed-ml \
    --region us-central1 \
    --image-uri us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/sp500-processor:v3 \
    --preprocess-out gs://comp4651-pipeline-bucket/preprocessed \
    --train-output gs://comp4651-pipeline-bucket/models/run-001

This script uses the Vertex AI Python SDK to create two CustomJobs sequentially and waits
for the preprocess job to finish before submitting the training job.

Note: ensure you have application default credentials or are running from an environment
with proper gcloud authentication & permissions.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from typing import List, Optional

from google.cloud import aiplatform


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_preprocess_then_train")


def make_worker_pool_spec(
    image_uri: str,
    command: List[str],
    args: List[str],
    machine_type: str = "n1-standard-4",
    replica_count: int = 1,
    boot_disk_size_gb: int = 100,
) -> dict:
    return {
        "machine_spec": {"machine_type": machine_type},
        "replica_count": replica_count,
        "container_spec": {"image_uri": image_uri, "command": command, "args": args},
        "disk_spec": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": boot_disk_size_gb},
    }


def submit_custom_job(
    project: str,
    region: str,
    display_name: str,
    worker_pool_specs: List[dict],
    service_account: Optional[str] = None,
):
    aiplatform.init(project=project, location=region)

    job = aiplatform.CustomJob(display_name=display_name, worker_pool_specs=worker_pool_specs)

    logger.info("Submitting job %s...", display_name)
    try:
        if service_account:
            # run with provided service account
            job.run(sync=True, service_account=service_account)
        else:
            job.run(sync=True)
    except Exception:
        logger.exception("Job %s failed or raised an exception", display_name)
        raise
    logger.info("Job %s finished.", display_name)
    return job


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="sp500-distributed-ml")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument(
        "--image-uri",
        default=(
            "us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/sp500-processor:v3"
        ),
    )
    parser.add_argument("--preprocess-out", default="gs://comp4651-pipeline-bucket/preprocessed")
    parser.add_argument("--train-output", default="gs://comp4651-pipeline-bucket/models/run-auto")
    parser.add_argument("--service-account", default=None, help="Optional service account email for the jobs")
    parser.add_argument("--preprocess-machine", default="n1-standard-4")
    parser.add_argument("--train-machine", default="n1-standard-8")
    parser.add_argument("--timestamped", action="store_true", help="Append a timestamp to output locations")

    args = parser.parse_args(argv)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    preprocess_out = args.preprocess_out
    train_output = args.train_output
    if args.timestamped:
        preprocess_out = preprocess_out.rstrip("/") + f"/{timestamp}"
        train_output = train_output.rstrip("/") + f"/{timestamp}"

    # Preprocess job
    preprocess_display = "sp500-preprocess-job-auto"
    preprocess_worker = make_worker_pool_spec(
        image_uri=args.image_uri,
        command=["python"],
        args=["data-pipeline/src/run_pipeline.py", "--mode", "preprocess", "--out-gcs", preprocess_out],
        machine_type=args.preprocess_machine,
        replica_count=1,
    )

    try:
        submit_custom_job(
            project=args.project,
            region=args.region,
            display_name=preprocess_display,
            worker_pool_specs=[preprocess_worker],
            service_account=args.service_account,
        )
    except Exception:
        logger.error("Preprocess job failed; aborting training.")
        sys.exit(1)

    # Training job uses the preprocess_out path produced above
    train_display = "sp500-train-job-auto"
    train_worker = make_worker_pool_spec(
        image_uri=args.image_uri,
        command=["python"],
        args=[
            "model/src/train_single.py",
            "--data-gcs",
            preprocess_out,
            "--output-gcs",
            train_output,
        ],
        machine_type=args.train_machine,
        replica_count=1,
    )

    try:
        submit_custom_job(
            project=args.project,
            region=args.region,
            display_name=train_display,
            worker_pool_specs=[train_worker],
            service_account=args.service_account,
        )
    except Exception:
        logger.error("Training job failed.")
        sys.exit(1)

    logger.info("Preprocess + Train workflow completed successfully.")


if __name__ == "__main__":
    main()
