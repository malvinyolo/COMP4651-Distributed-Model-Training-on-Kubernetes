from typing import List

import kfp
from kfp import dsl
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

"""
KFP v2 pipeline for distributed preprocessing and training on Vertex AI.

- Preprocess step: ParallelFor over tickers. Each runs the data-pipeline container
  and writes outputs to GCS under output_prefix/{ticker}/ ...
- Train step: Launches a custom job (single or multi-node) using the training container.

Fill in placeholders in pipeline parameters or default values.
"""

@dsl.pipeline(
    name="distributed-preprocess-train",
)
def preprocess_and_train(
    project_id: str,
    region: str = "us-central1",
    artifact_registry_image_prefix: str = "us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline",
    preprocess_image: str = "preprocess:latest",
    train_image: str = "train:latest",
    gcs_output_prefix: str = "gs://comp4651-pipeline-bucket/outputs",
    tickers: List[str] = ["AAPL", "AMZN", "MSFT"],
    # Training parameters (annotated for KFP compiler)
    distributed: bool = False,
    num_nodes: int = 1,
    gpus_per_node: int = 1,
    machine_type: str = "e2-standard-4",  # lightweight CPU default
    accelerator_type: str = "",  # when use_gpu=True supply e.g. "NVIDIA_TESLA_T4"
    accelerator_count: int = 0,   # when use_gpu=True supply >0
    use_gpu: bool = False,        # explicit flag to include accelerator fields; avoids truthiness issues with param channels
):
    # Compose fully qualified image names
    preprocess_image_uri = f"{artifact_registry_image_prefix}/{preprocess_image}"
    train_image_uri = f"{artifact_registry_image_prefix}/{train_image}"

    # 1) Preprocessing: use ParallelFor so 'tickers' can be a runtime parameter.
    # We create a CustomTrainingJobOp inside the loop (no dynamic component base_image).
    preprocess_jobs = []
    with dsl.ParallelFor(tickers) as tk:
        preprocess_worker_pool = [
            {
                "machine_spec": {"machine_type": machine_type},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": preprocess_image_uri,
                    "command": [
                        "python",
                        "entrypoint_preprocess.py",
                        "--tickers",
                        tk,
                        "--output_gcs",
                        f"{gcs_output_prefix}/{tk}",
                    ],
                },
            }
        ]
        job = CustomTrainingJobOp(
            display_name="preprocess-task",
            project=project_id,
            location=region,
            worker_pool_specs=preprocess_worker_pool,
            base_output_directory=f"{gcs_output_prefix}/{tk}",
        )
        preprocess_jobs.append(job)

    # 2) Training step
    # We use dsl.Condition to avoid serializing empty accelerator fields that caused parse errors.
    # Each branch creates a training task; only one executes at runtime.
    with dsl.Condition(use_gpu == True):  # GPU branch
        if distributed:
            gpu_worker_pool = [
                {
                    "machine_spec": {
                        "machine_type": machine_type,
                        "accelerator_type": accelerator_type,
                        "accelerator_count": accelerator_count,
                    },
                    "replica_count": num_nodes,
                    "container_spec": {
                        "image_uri": train_image_uri,
                        "command": [
                            "python",
                            "entrypoint_train.py",
                            "--distributed",
                            "--gpus_per_node",
                            str(gpus_per_node),
                            "--args",
                            "--data_gcs_prefix",
                            gcs_output_prefix,
                        ],
                    },
                }
            ]
        else:
            gpu_worker_pool = [
                {
                    "machine_spec": {
                        "machine_type": machine_type,
                        "accelerator_type": accelerator_type,
                        "accelerator_count": accelerator_count,
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": train_image_uri,
                        "command": [
                            "python",
                            "entrypoint_train.py",
                            "--gpus_per_node",
                            str(gpus_per_node),
                            "--args",
                            "--data_gcs_prefix",
                            gcs_output_prefix,
                        ],
                    },
                }
            ]
        CustomTrainingJobOp(
            display_name="train-model",
            project=project_id,
            location=region,
            worker_pool_specs=gpu_worker_pool,
            base_output_directory=gcs_output_prefix,
        ).after(*preprocess_jobs)

    with dsl.Condition(use_gpu == False):  # CPU branch
        if distributed:
            cpu_worker_pool = [
                {
                    "machine_spec": {
                        "machine_type": machine_type,
                    },
                    "replica_count": num_nodes,
                    "container_spec": {
                        "image_uri": train_image_uri,
                        "command": [
                            "python",
                            "entrypoint_train.py",
                            "--distributed",
                            "--gpus_per_node",
                            str(gpus_per_node),
                            "--args",
                            "--data_gcs_prefix",
                            gcs_output_prefix,
                        ],
                    },
                }
            ]
        else:
            cpu_worker_pool = [
                {
                    "machine_spec": {
                        "machine_type": machine_type,
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": train_image_uri,
                        "command": [
                            "python",
                            "entrypoint_train.py",
                            "--gpus_per_node",
                            str(gpus_per_node),
                            "--args",
                            "--data_gcs_prefix",
                            gcs_output_prefix,
                        ],
                    },
                }
            ]
        CustomTrainingJobOp(
            display_name="train-model",
            project=project_id,
            location=region,
            worker_pool_specs=cpu_worker_pool,
            base_output_directory=gcs_output_prefix,
        ).after(*preprocess_jobs)


if __name__ == "__main__":
    # Compile pipeline to JSON for upload
    kfp.compiler.Compiler().compile(
        pipeline_func=preprocess_and_train,
        package_path="vertex_pipeline.json",
    )
