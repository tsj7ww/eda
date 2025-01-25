FROM python:3.10-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Bash
    bash \
    bash-completion \
    # System
    build-essential \
    curl \
    git \
    libpq-dev \
    # Tools
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
    # Stan packages
    # wget \
    # cmake \
    # gfortran \
    # libopenblas-dev \
    # liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Set up bash environment
COPY bashrc /root/.bashrc
ENV SHELL=/bin/bash

# Install Kaggle CLI and create alias
RUN pip install --no-cache-dir kaggle && \
    mkdir -p /root/.kaggle && \
    chmod 755 /root/.kaggle

# Install CmdStan
# RUN mkdir -p ${STAN_HOME} \
#     && wget https://github.com/stan-dev/cmdstan/releases/download/v2.32.2/cmdstan-2.32.2.tar.gz \
#     && tar -xzf cmdstan-2.32.2.tar.gz -C ${STAN_HOME} --strip-components=1 \
#     && rm cmdstan-2.32.2.tar.gz \
#     && cd ${STAN_HOME} \
#     && make build

# Set up CmdStan for PyStan
# RUN python -m cmdstanpy.install_cmdstan

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]