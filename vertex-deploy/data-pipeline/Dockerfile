FROM python:3.10-slim

WORKDIR /app
# system deps if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy pipeline code
COPY src/ src/
# COPY data/ data/
# default envs (override in cloud)
ENV PYTHONUNBUFFERED=1

# entrypoint: run pipeline; accept args via CMD
ENTRYPOINT ["python","src/run_pipeline.py"]