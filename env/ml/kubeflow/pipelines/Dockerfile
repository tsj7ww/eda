FROM python:3.9-slim

WORKDIR /pipelines

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.pipelines.txt .
RUN pip install -r requirements.pipelines.txt

COPY pipelines/ .

EXPOSE 8888

CMD ["python", "api_server.py"]