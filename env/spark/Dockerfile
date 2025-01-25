# Dockerfile
FROM apache/spark-py:v3.5.0

USER root

# Create working directory
WORKDIR /workspace

# Copy requirements if needed
COPY requirements.txt .
RUN pip install -r requirements.txt

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]