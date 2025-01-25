FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    cmake \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    make LLAMA_METAL=1

# Create working directory
WORKDIR /workspace

# Copy necessary files
COPY jupyter_notebook_config.py /root/.jupyter/
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Expose ports for Jupyter and Llama server
EXPOSE 8888 8080

# Start script
CMD ["/start.sh"]