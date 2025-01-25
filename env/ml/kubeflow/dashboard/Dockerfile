FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.dashboard.txt .
RUN pip install -r requirements.dashboard.txt

COPY dashboard/ .

EXPOSE 8080

CMD ["python", "app.py"]