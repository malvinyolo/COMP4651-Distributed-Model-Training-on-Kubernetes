# Vertex Deploy — Data Pipeline & Training on Vertex AI

This folder documents how to package and run the project's data preprocessing and model training on Google Cloud (Cloud Build, Artifact Registry, Vertex AI / Cloud Run).


## Quick start

Prerequisites

- Install and authenticate the Google Cloud SDK (`gcloud`) and set your project:
- Make sure you have a valid Google Account to use for login.

```cmd
gcloud auth login
gcloud config set project <PROJECT-NAME>
gcloud config set compute/region <REGION>
```

- Create an Artifact Registry repo and a GCS bucket for artifacts:

```cmd
gsutil mb -l <REGION> gs://<GCS-BUCKET-NAME>
```



## Build & push images (Cloud Build)

Use Cloud Build to build and push images to Artifact Registry (no local Docker required):

Please run these commands within the data-pipeline/ and training/ directories respectively.

```cmd
gcloud builds submit --project=<PROJECT-NAME> \
  --tag=<REGION>/<PROJECT-NAME>/<ARTIFACT-REGISTRY>/<DOCKER-IMAGE>:<TAGS> .

Example: 
1. data-preprocessor
gcloud builds submit --project=sp500-distributed-ml --tag=us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/data-preprocessor:latest .

2. model-trainer
gcloud builds submit --project=sp500-distributed-ml \
  --tag=us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/model-train:latest .
```

Tip: use semantic tags (v1, v2) for reproducibility.



## Run preprocessing

The repository's data pipeline scripts write processed data to a local directory inside the container. Use the `--out-gcs` argument to upload processed outputs to GCS.

Example (Cloud Run job):

```cmd
gcloud run jobs create <JOB-NAME> ^
  --image=<REGION>/<PROJET>/<ARTIIFACT-REGISTRY>/<DOCKER-IMAGE>:<TAGS> ^
  --region=<REGIOM> ^
  --args="--mode=all,--out-gcs=gs://<GCS-BUCKET-NAME>"
  
gcloud run jobs execute <JOB-NAME> --region=<REGION>

Example:
gcloud run jobs create data-all-job ^
  --image=us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/data-preprocessor:latest ^
  --region=us-central1 ^
  --args="--mode=all,--out-gcs=gs://comp4651-pipeline-bucket/preprocessed"

gcloud run jobs execute data-all-job --region=us-central1
```

Alternatively submit as a Vertex AI Custom Job to use VM-based workers with more resources:

```cmd
gcloud ai custom-jobs create ^
  --region=us-central1 ^
  --display-name=data-preprocess-job ^
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/data-preprocessor:latest ^
  --args=--mode,all,--out-gcs,gs://comp4651-pipeline-bucket/pipeline-output
```

After the job completes, verify outputs:

```cmd
gsutil ls "gs://comp4651-pipeline-bucket/preprocessed/"
```



## Run training

You can run training either with Cloud Run (for CPU light runs) or Vertex AI Custom Jobs (for larger CPU/GPU resources).

Example (Cloud Run job):

```cmd
gcloud run jobs create <JOB-NAME> ^
  --image=<REGION>/<PROJET>/<ARTIIFACT-REGISTRY>/<DOCKER-IMAGE>:<TAGS> ^
  --region=<REGIOM> ^
  --args=--data_dir=gs://<PATH/TO/DATA> ^
  --args=--bucket=<GCS-BUCKET-NAME>> ^
  --args=--model_dir=<OUTPUT-GCS-BUCKET>
  
gcloud run jobs execute <JOB-NAME> --region=<REGION>

Example:
gcloud run jobs create model-train-job ^
  --image=us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/model-train:latest ^
  --region=us-central1 ^
  --args=--data_dir=gs://comp4651-pipeline-bucket/pipeline-output/classification ^
  --args=--bucket=comp4651-pipeline-bucket ^
  --args=--model_dir=models/mlp_classifier

gcloud run jobs execute model-train-job --region=us-central1
```

Alternatively submit as a Vertex AI Custom Job to use VM-based workers with more resources:

```cmd
gcloud ai custom-jobs create ^
  --region=us-central1 ^
  --display-name=model-train-job ^
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/model-train:latest ^
  --args=--data_dir,gs://comp4651-pipeline-bucket/pipeline-output/classification,--bucket,comp4651-pipeline-bucket,--model_dir,models/mlp_classifier
```

If you need GPU acceleration, change `machine-type` and add `accelerator-type`/`accelerator-count` in `--worker-pool-spec` (ensure GPU image and quota):

```cmd
--worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=1,accelerator-type=nvidia-tesla-a100,accelerator-count=1,container-image-uri=...
```



## Scheduling & automation

You can schedule Cloud Run jobs in the Cloud Console or use Cloud Scheduler to trigger Cloud Build pipelines that run preprocess → train flows. Example schedules (NYSE Market Open):

- Preprocess: `30 9 * * 1-5` (9:30 AM Mon-Fri)
- Train: `45 9 * * 1-5` (9:45 AM Mon-Fri)


## Troubleshooting & tips

- If a job fails, open Vertex AI → Jobs → select job → Logs to see container stdout/stderr and Python tracebacks.
- Common causes:
  - Missing files in image: rebuild the image after adding files and push a new tag.
  - Missing Python deps: ensure `requirements.txt` updates are included and installed during image build.
  - GCS permission errors: grant the job/service account `storage.objectViewer` (read) and `storage.objectCreator`/`storage.objectAdmin` (write) on the bucket.

Commands to inspect jobs/logs:

```cmd
gcloud ai custom-jobs list --region=us-central1
gcloud ai custom-jobs describe JOB_NAME --region=us-central1
gcloud logging read "resource.type=ml_job AND resource.labels.job_id=JOB_ID" --project=<PROJECT-NAME> --limit=200 --format="value(textPayload)"
```