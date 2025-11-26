FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training source code
COPY train.py .

# Entry point to run training script, pass arguments via CMD or job config
ENTRYPOINT ["python", "train.py"]