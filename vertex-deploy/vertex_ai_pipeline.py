from kfp import dsl
from kfp.dsl import component, Output, Input, Dataset
from kfp import compiler
from google.cloud import aiplatform

# Global configuration
PROJECT_ID = 'sp500-distributed-ml'
REGION = 'us-central1'
GCS_OUTPUT_BUCKET = 'gs://comp4651-pipeline-bucket'
STAGING_BUCKET = 'gs://comp4651-pipeline-bucket/staging'

@component(base_image='python:3.10', packages_to_install=['google-cloud-aiplatform'])
def data_preprocess_op(
    project_id: str, 
    region: str, 
    gcs_output_bucket: str,
    staging_bucket: str
) -> str:
    from google.cloud import aiplatform
    
    aiplatform.init(
        project=project_id, 
        location=region,
        staging_bucket=staging_bucket
    )

    job = aiplatform.CustomJob(
        display_name="data-preprocess-job",
        staging_bucket=staging_bucket,
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": "n1-standard-4"},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": "us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/data-preprocessor:latest",
                    "command": [
                        "python", "src/run_pipeline.py"
                    ],
                    "args": [
                        "--mode=all",
                        f"--out-gcs={gcs_output_bucket}/pipeline-output"
                    ],
                },
            }
        ],
    )
    job.run(sync=True)
    return f"{gcs_output_bucket}/pipeline-output"

@component(base_image='python:3.10', packages_to_install=['google-cloud-aiplatform'])
def model_train_op(
    project_id: str, 
    region: str, 
    gcs_output_bucket: str,
    staging_bucket: str,
    preprocessed_data_path: str
) -> str:
    from google.cloud import aiplatform
    
    aiplatform.init(
        project=project_id, 
        location=region,
        staging_bucket=staging_bucket
    )

    job = aiplatform.CustomJob(
        display_name="model-train-job",
        staging_bucket=staging_bucket,
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": "n1-standard-4"},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": "us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/model-train:latest",
                    "args": [
                        f"--data_dir={preprocessed_data_path}/classification",
                        "--bucket=comp4651-pipeline-bucket",
                        "--model_dir=models/mlp_classifier"
                    ],
                },
            }
        ],
    )
    job.run(sync=True)
    return "Model trained successfully!"

@dsl.pipeline(
    name="data-preprocess-train-pipeline",
    description="A pipeline that runs data preprocessing and then model training."
)
def pipeline():
    preprocess_task = data_preprocess_op(
        project_id=PROJECT_ID,
        region=REGION,
        gcs_output_bucket=GCS_OUTPUT_BUCKET,
        staging_bucket=STAGING_BUCKET,
    )
    
    train_task = model_train_op(
        project_id=PROJECT_ID,
        region=REGION,
        gcs_output_bucket=GCS_OUTPUT_BUCKET,
        staging_bucket=STAGING_BUCKET,
        preprocessed_data_path=preprocess_task.output
    )
    train_task.after(preprocess_task)

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.json"
    )

    aiplatform.init(project=PROJECT_ID, location=REGION)
    job = aiplatform.PipelineJob(
        display_name="data-preprocess-train-pipeline",
        template_path="pipeline.json",
        pipeline_root=f"{GCS_OUTPUT_BUCKET}/pipeline_runs",
    )
    job.run()
