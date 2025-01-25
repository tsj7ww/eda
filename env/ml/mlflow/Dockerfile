FROM python:3.9-slim

WORKDIR /mlflow

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0