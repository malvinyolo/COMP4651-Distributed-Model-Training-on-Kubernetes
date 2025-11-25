# COMP4651: Scaling Machine Learning with Cloud Technologies

Stock price prediction model training demonstrating scalability and performance using cloud computing infrastructure.

## Project Overview

This project implements a **regression model** to predict next-day stock returns using historical price data from multiple stocks (AAPL, MSFT, TSLA, JPM, AMZN, XOM, JNJ). The system demonstrates how cloud technologies enable scalable machine learning by leveraging containerization, cloud compute resources, and automated data pipelines.

### Key Features:
- **Data Pipeline**: Automated ETL for stock data collection and preprocessing
- **Model Training**: PyTorch-based LSTM/MLP regression with customizable hyperparameters
- **Containerization**: Docker-based deployment for consistent and portable environments
- **Scalability**: Support for training on individual stocks or combined datasets
- **Cloud Deployment**: Ready for AWS ECR and EC2 deployment

### Architecture:
```
├── data-pipeline/       # Data collection, preprocessing, and dataset scaling
│   ├── src/            # Pipeline modules (data_collector, preprocess, config)
│   ├── data/           # Raw CSV and processed .npz datasets
│   ├── create_large_dataset.py  # Tool to artificially scale datasets
│   └── requirements.txt
│
├── model/              # Training infrastructure
│   ├── src/            # Training logic, models, data loaders
│   ├── outputs/        # Training artifacts, models, and logs
│   ├── Dockerfile      # Container definition
│   ├── TRAINING_GUIDE.md  # Detailed hyperparameter documentation
│   └── requirements.txt
│
└── README.md           # This file
```

---

## Getting Started

### Prerequisites
- **Local (macOS Apple Silicon)**: Docker Desktop, Python 3.10+, pip
- **EC2 (Amazon Linux 2023)**: Docker, Python 3.11, git, AWS CLI

---

## Local Setup (macOS with Apple Silicon)

### Step 1: Clone the Repository
```bash
git clone https://github.com/malvinyolo/COMP4651-Distributed-Model-Training-on-Kubernetes.git
cd COMP4651-Distributed-Model-Training-on-Kubernetes
```

### Step 2: Set Up Data Pipeline
```bash
cd data-pipeline
pip install -r requirements.txt
```

### Step 3: Run Data Pipeline
```bash
cd src
python run_pipeline.py
cd ..
```

This will:
- Download historical stock data for 7 stocks (AAPL, MSFT, TSLA, JPM, AMZN, XOM, JNJ)
- Preprocess and create sequences
- Generate individual stock `.npz` files in `data/processed/classification/`
- Create combined `sp500_regression.npz` dataset

### Step 4: (Optional) Scale Dataset Artificially
```bash
# Create a 50x larger dataset for testing scalability
python create_large_dataset.py --multiplier 50
```

Output: `data/processed/classification/sp500_regression_x50.npz`

### Step 5: Build Docker Image
```bash
cd ../model

# Build for linux/amd64 platform (required on Apple Silicon)
docker buildx build --platform linux/amd64 -t final-trainer --load .
```

### Step 6: Train the Model

**Train on a single stock:**
```bash
docker run --rm \
  --memory="4g" \
  --cpus="2" \
  -v "$(pwd)/../data-pipeline/data/processed:/data:ro" \
  -v "$(pwd)/outputs:/app/outputs:rw" \
  final-trainer \
  python -m src.single.cli \
    --stock AAPL \
    --epochs 50 \
    --batch_size 64 \
    --data_dir /data
```

**Train on the 50x scaled dataset:**
```bash
docker run --rm \
  --memory="4g" \
  --cpus="2" \
  -v "$(pwd)/../data-pipeline/data/processed:/data/classification:ro" \
  -v "$(pwd)/outputs:/app/outputs:rw" \
  final-trainer \
  python -m src.single.cli \
    --stock sp500_regression_x50 \
    --epochs 50 \
    --batch_size 64 \
    --data_dir /data
```

**Train on all stocks sequentially:**
```bash
docker run --rm \
  --memory="4g" \
  --cpus="2" \
  -v "$(pwd)/../data-pipeline/data/processed:/data:ro" \
  -v "$(pwd)/outputs:/app/outputs:rw" \
  final-trainer \
  python train_all_stocks_single.py
```

---

## AWS EC2 Setup (Amazon Linux 2023)

### Step 1: Launch EC2 Instance
- **AMI**: Amazon Linux 2023
- **Instance Type**: t3.medium or larger
- **Storage**: At least 20GB
- **Security Group**: SSH (port 22) access

### Step 2: Connect and Install Dependencies
```bash
# SSH into your EC2 instance
ssh -i your-key.pem ec2-user@your-ec2-ip

# Install git and Python
sudo yum install git -y
sudo dnf install python3.11 -y
sudo yum install python3-pip -y
```

### Step 3: Install and Configure Docker
```bash
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group
whoami
sudo usermod -aG docker ec2-user

# Reboot to apply group changes
sudo reboot
```

After reboot, SSH back in:
```bash
ssh -i your-key.pem ec2-user@your-ec2-ip
```

### Step 4: Clone Repository and Set Up Data
```bash
git clone https://github.com/malvinyolo/COMP4651-Distributed-Model-Training-on-Kubernetes.git
cd COMP4651-Distributed-Model-Training-on-Kubernetes/data-pipeline

pip install -r requirements.txt

cd src
python3 run_pipeline.py
cd ..
```

### Step 5: Build Docker Image
```bash
cd ../model
docker build -t stock-trainer:latest .
```

### Step 6: Configure AWS ECR (if using ECR)

**Get your AWS account number:**
```bash
# On your local machine or in AWS Console, run:
aws configure export-credentials --format env
```

Copy the export commands and paste them in your EC2 terminal:
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

**Verify credentials:**
```bash
aws sts get-caller-identity
# Note the Account ID (e.g., 992382492793)
```

**Tag and push to ECR:**
```bash
# Replace 992382492793 with your Account ID
docker tag stock-trainer:latest 992382492793.dkr.ecr.us-east-1.amazonaws.com/stock-trainer:latest

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
docker login --username AWS --password-stdin 992382492793.dkr.ecr.us-east-1.amazonaws.com

# Push to ECR
docker push 992382492793.dkr.ecr.us-east-1.amazonaws.com/stock-trainer:latest
```

### Step 7: Train on EC2

**Option A: Using local Docker image**
```bash
cd COMP4651-Distributed-Model-Training-on-Kubernetes/model

docker run --rm \
  -v "$(pwd)/../data-pipeline/data/processed:/data:ro" \
  -v "$(pwd)/outputs:/app/outputs:rw" \
  stock-trainer:latest \
  python -m src.single.cli \
    --stock AAPL \
    --epochs 50 \
    --batch_size 64 \
    --data_dir /data
```

**Option B: Using ECR image**
```bash
docker run --rm \
  -v "$(pwd)/../data-pipeline/data/processed:/data:ro" \
  -v "$(pwd)/outputs:/app/outputs:rw" \
  992382492793.dkr.ecr.us-east-1.amazonaws.com/stock-trainer:latest \
  python -m src.single.cli \
    --stock AAPL \
    --epochs 50 \
    --batch_size 64 \
    --data_dir /data
```

**Train on all stocks sequentially:**
```bash
for stock in AAPL MSFT TSLA JPM AMZN XOM JNJ; do
    echo "Training $stock..."
    docker run --rm \
      -v "$(pwd)/../data-pipeline/data/processed:/data:ro" \
      -v "$(pwd)/outputs:/app/outputs:rw" \
      992382492793.dkr.ecr.us-east-1.amazonaws.com/stock-trainer:latest \
      python -m src.single.cli \
        --stock "$stock" \
        --epochs 50 \
        --batch_size 64 \
        --data_dir /data || exit 1
done
echo "All done!"
```

**Run in background with logging:**
```bash
nohup bash -c 'for stock in AAPL MSFT TSLA JPM AMZN XOM JNJ; do
    echo "Training $stock..."
    docker run --rm \
      -v "$(pwd)/../data-pipeline/data/processed:/data:ro" \
      -v "$(pwd)/outputs:/app/outputs:rw" \
      992382492793.dkr.ecr.us-east-1.amazonaws.com/stock-trainer:latest \
      python -m src.single.cli \
        --stock "$stock" \
        --epochs 50 \
        --batch_size 64 \
        --data_dir /data || exit 1
done' > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

---

## Available Datasets

After running the data pipeline, you'll have:

- **Individual stocks**: `AAPL.npz`, `MSFT.npz`, `TSLA.npz`, `JPM.npz`, `AMZN.npz`, `XOM.npz`, `JNJ.npz`
- **Combined dataset**: `sp500_regression.npz` (~6,692 training samples)
- **Scaled datasets**: `sp500_regression_x10.npz`, `sp500_regression_x50.npz`, etc. (if created)

All datasets are located in `data-pipeline/data/processed/classification/`

---

## Training Options

### Basic Usage
```bash
python -m src.single.cli \
  --stock AAPL \
  --epochs 50 \
  --batch_size 64 \
  --data_dir /data
```

### Common Options
- `--stock`: Stock ticker or dataset name (e.g., `AAPL`, `sp500_regression_x50`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 64)
- `--lr`: Learning rate (default: 1e-3)
- `--hidden_dim`: Hidden layer dimension (default: 64)
- `--dropout`: Dropout rate (default: 0.1)
- `--device`: Device to use (`auto`, `cpu`, `cuda`, `mps`)

**For detailed hyperparameter options and advanced training configurations, see [`model/TRAINING_GUIDE.md`](model/TRAINING_GUIDE.md)**

---

## Output Structure

Training outputs are saved to `model/outputs/run_YYYYMMDD_HHMMSS/`:
```
outputs/
└── run_20251116_143025/
    ├── best_model.pt          # Best model checkpoint
    ├── final_model.pt         # Final epoch model
    ├── config.yaml            # Training configuration
    ├── metrics.json           # Training metrics
    └── training_log.txt       # Detailed training log
```

---

## Troubleshooting

### Docker Build Fails on Mac (Apple Silicon)
**Problem**: PyTorch wheel not found for arm64

**Solution**: Use `--platform linux/amd64` flag:
```bash
docker buildx build --platform linux/amd64 -t final-trainer --load .
```

### Permission Denied on EC2 Docker
**Problem**: `permission denied while trying to connect to the Docker daemon`

**Solution**:
```bash
sudo usermod -aG docker ec2-user
sudo reboot
```

### Stock Data Not Found
**Problem**: `ERROR: Stock data not found: /data/classification/STOCK.npz`

**Solution**: Ensure you ran the data pipeline first:
```bash
cd data-pipeline/src
python run_pipeline.py
```

### AWS Credentials Expired
**Problem**: ECR login fails or AWS commands return auth errors

**Solution**: Re-export credentials:
```bash
aws configure export-credentials --format env
# Copy and paste export commands
```

---

## Additional Resources

- **Training Guide**: [`model/TRAINING_GUIDE.md`](model/TRAINING_GUIDE.md) - Detailed hyperparameter documentation
- **Data Pipeline**: [`data-pipeline/`](data-pipeline/) - ETL and preprocessing scripts
- **Model Source**: [`model/src/`](model/src/) - Training implementation

---

## Authors

- Project created by the COMP4651 team(malvinyolo, ChanJiraphat, gugusiow, p3arlk)
- Repository: (https://github.com/malvinyolo/COMP4651-Distributed-Model-Training-on-Kubernetes)

---

## Quick Command Reference

### Local (macOS)
```bash
# Build image
docker buildx build --platform linux/amd64 -t final-trainer --load .

# Train single stock
docker run --rm -v "$(pwd)/../data-pipeline/data/processed:/data:ro" -v "$(pwd)/outputs:/app/outputs:rw" final-trainer python -m src.single.cli --stock AAPL --epochs 50 --batch_size 64 --data_dir /data
```

### EC2
```bash
# Build image
docker build -t stock-trainer:latest .

# Train single stock
docker run --rm -v "$(pwd)/../data-pipeline/data/processed:/data:ro" -v "$(pwd)/outputs:/app/outputs:rw" stock-trainer:latest python -m src.single.cli --stock AAPL --epochs 50 --batch_size 64 --data_dir /data

# Train all stocks
for stock in AAPL MSFT TSLA JPM AMZN XOM JNJ; do docker run --rm -v "$(pwd)/../data-pipeline/data/processed:/data:ro" -v "$(pwd)/outputs:/app/outputs:rw" stock-trainer:latest python -m src.single.cli --stock "$stock" --epochs 50 --batch_size 64 --data_dir /data || exit 1; done
```