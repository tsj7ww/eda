#!/bin/bash

# Define the directory structure
DATA_DIR="."
RAW_DIR="$DATA_DIR/raw"
PROCESSED_DIR="$DATA_DIR/processed"

# Kaggle creds
if [ ! -f /root/.kaggle/kaggle.json ]; then
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > /root/.kaggle/kaggle.json
    chmod 600 /root/.kaggle/kaggle.json
fi

# Check if data directory exists
# if [ ! -d "$DATA_DIR" ]; then
#     echo "Creating data directory structure..."
    
#     # Create main data directory
#     mkdir -p "$DATA_DIR"
    
#     # Create raw and processed subdirectories
#     mkdir -p "$RAW_DIR"
#     mkdir -p "$PROCESSED_DIR"
    
#     echo "Directory structure created successfully!"
# else
#     echo "Data directory structure already exists."
# fi