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

# Expose port 8787 for RStudio Server
EXPOSE 8787