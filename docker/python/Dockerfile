FROM python:3.10-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bash-completion \
    build-essential \
    curl \
    git \
    libpq-dev \
    gawk \
    sed \
    grep \
    ripgrep \
    parallel \
    jq \
    xmlstarlet \
    htop \
    glances \
    ncdu \
    netcat-openbsd \
    socat \
    nmap \
    tmux \
    vim \
    tree \
    pigz \
    pbzip2 \
    csvkit \
    datamash \
    bc \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install Kaggle CLI and create alias
RUN pip install --no-cache-dir kaggle && \
    mkdir -p /root/.kaggle && \
    chmod 755 /root/.kaggle

# Copy requirements file
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Set up bash environment
COPY bashrc /root/.bashrc
ENV SHELL=/bin/bash

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]