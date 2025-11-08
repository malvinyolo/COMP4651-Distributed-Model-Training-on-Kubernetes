# Use an official PyTorch base image (CPU or GPU version)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /app

# --- Install Python dependencies from both teams' folders ---
# (we copy requirements first so Docker caching works efficiently)
COPY data-pipeline/requirements.txt /tmp/reqs_data.txt
COPY model/requirements.txt /tmp/reqs_model.txt

RUN pip install --no-cache-dir -r /tmp/reqs_data.txt
RUN pip install --no-cache-dir -r /tmp/reqs_model.txt

# --- Copy project source code ---
COPY data-pipeline/ /app/data-pipeline/
COPY model/ /app/model/

# Add project root to Python path
ENV PYTHONPATH=/app

# Set default training entrypoint to our distributed wrapper
ENTRYPOINT ["python", "model/src/distributed_train.py"]