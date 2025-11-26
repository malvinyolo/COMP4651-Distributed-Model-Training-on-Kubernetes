## Deploying Data-pipeline and Model training onto Vertex AI, Google Cloud
### In this directory, data-pipeline code and model code have been slightly modified to run on GCP

Commands for setting up Data-to-ML pipeline on Google Cloud:
1. Setup project on Google Cloud. 
2. In gcloud CLI, use the command: gcloud config set project <your-project-name>

3. In data-pipeline. 
a) Cloud build image using the command: 
gcloud builds submit --tag=us-central1-docker.pkg.dev/PROJECT/REPO/data-preprocessor:latest .

Example: 
gcloud builds submit --project=sp500-distributed-ml --tag=us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/data-preprocessor:latest .

b) Create Cloud Run Job:
gcloud run jobs create <JOB-NAME> ^
  --image=<REGION>/<PROJECT>/<ARTIFACT_REGISTRY>/<DOCKER_IMAGE> ^
  --region=<REGION> ^
  --args="--mode=<collect/preprocess/all>,--out-gcs=<OUTPUT_GCS_BUCKET>"

Example:
gcloud run jobs create data-all-job ^
  --image=us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/data-preprocessor:latest ^
  --region=us-central1 ^
  --args="--mode=all,--out-gcs=gs://comp4651-pipeline-bucket/pipeline-output"

(Optional, fix the region: gcloud config set run/region us-central1)

c) Run Data Job: 
gcloud run jobs execute data-all-job


4. In training.
a) Cloud build image using the command:
gcloud builds submit --project=sp500-distributed-ml --tag=us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/model-train:latest .

b) Create Cloud Run Job:
gcloud run jobs create model-train-job ^
  --image=us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/model-train:latest ^
  --region=us-central1 ^
  --args=--data_dir=gs://comp4651-pipeline-bucket/pipeline-output/classification ^
  --args=--bucket=comp4651-pipeline-bucket ^
  --args=--model_dir=models/mlp_classifier


gcloud run jobs create model-train-job-2 ^
  --image=us-central1-docker.pkg.dev/sp500-distributed-ml/ml-pipeline/model-train:latest ^
  --region=us-central1 ^
  --args="--data_dir=gs://comp4651-pipeline-bucket/pipeline-output/classification --bucket=comp4651-pipeline-bucket --model_dir=models/mlp_classifier"

c) Run Data Job:
gcloud run jobs execute model-train-job

Optionals:
5. In the Cloud Run console, schedule the Data Preprocessing Jobs and the ML Model Training Jobs to run regularly.

Example settings: 
1. data-all-job
- Schedule: 30 9 * * 1-5 (9:30AM, Mon-Fri)
- Timezone: America/New_York (Market Open Time)
- Region: us-central1

2. model-train-job
- Schedule: 45 9 * * 1-5 (9:45AM, Mon-Fri)
- Timezone: America/New_York 
- Region: us-central1