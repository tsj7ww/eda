FROM python:3.11-slim

# Set environment variables
# ARG JUPYTER_TOKEN
# ENV JUPYTER_TOKEN=$JUPYTER_TOKEN

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

# Install pipenv and virtualenv
RUN pip install --no-cache-dir pipenv virtualenv
# Create a virtual environment named 'eda'
RUN python -m virtualenv /opt/venv/eda
# Copy Pipfile
COPY Pipfile .
##################
### Cold Start ###
# RUN pipenv lock
### Warm Start ###
COPY Pipfile.lock .
##################
# Generate requirements.txt from Pipfile using pipenv
RUN pipenv requirements > requirements.txt
# Install packages in the virtual environment
RUN . /opt/venv/eda/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m ipykernel install --user --name=eda --display-name="EDA"

# Set up bash environment
COPY bashrc /root/.bashrc
ENV SHELL=/bin/bash
ENV PATH="/opt/venv/eda/bin:$PATH"

# Install Kaggle CLI and create alias
RUN pip install --no-cache-dir kaggle && \
    mkdir -p /root/.kaggle && \
    chmod 755 /root/.kaggle

# # Install CmdStan
# RUN mkdir -p ${STAN_HOME} \
#     && wget https://github.com/stan-dev/cmdstan/releases/download/v2.32.2/cmdstan-2.32.2.tar.gz \
#     && tar -xzf cmdstan-2.32.2.tar.gz -C ${STAN_HOME} --strip-components=1 \
#     && rm cmdstan-2.32.2.tar.gz \
#     && cd ${STAN_HOME} \
#     && make build
# # Set up CmdStan for PyStan
# RUN python -m cmdstanpy.install_cmdstan

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]