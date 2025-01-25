FROM rocker/rstudio:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    # Stan packages
    # wget \
    # cmake \
    # gfortran \
    # libopenblas-dev \
    # liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy package installation files
COPY install_packages.R /tmp/
COPY packages.R /tmp/

# Install R packages
RUN Rscript /tmp/install_packages.R

# Create directory for persistent data
RUN mkdir -p /home/rstudio/workspace

# Set working directory
WORKDIR /home/rstudio/workspace

# Install CmdStan
# RUN mkdir -p ${STAN_HOME} \
#     && wget https://github.com/stan-dev/cmdstan/releases/download/v2.32.2/cmdstan-2.32.2.tar.gz \
#     && tar -xzf cmdstan-2.32.2.tar.gz -C ${STAN_HOME} --strip-components=1 \
#     && rm cmdstan-2.32.2.tar.gz \
#     && cd ${STAN_HOME} \
#     && make build

# Set up CmdStan for PyStan
# RUN python -m cmdstanpy.install_cmdstan

# Expose port 8787 for RStudio Server
EXPOSE 8787