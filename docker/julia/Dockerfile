FROM debian:bullseye-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Julia
ENV JULIA_VERSION=1.9.3
RUN curl -L https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-${JULIA_VERSION}-linux-x86_64.tar.gz -o julia.tar.gz && \
    tar -xzf julia.tar.gz -C /opt && \
    rm julia.tar.gz && \
    ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia

WORKDIR /home

# Install JupyterLab
RUN pip3 install --user jupyterlab

# Copy Project.toml
COPY Project.toml /home/

# Install Julia packages and create kernel
RUN julia -e ' \
    using Pkg; \
    Pkg.activate("."); \
    Pkg.instantiate(); \
    using IJulia; \
    installkernel("EDA", env=Dict("JULIA_NUM_THREADS" => "auto")); \
    '

# Add local bin to PATH
ENV PATH="/home/.local/bin:${PATH}"

WORKDIR /workspace
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
