# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for pyodbc, cfgrib, and GRIB processing
RUN apt-get update && \
    apt-get install -y \
        gcc g++ \
        unixodbc-dev \
        libssl-dev \
        libffi-dev \
        libpq-dev \
        libeccodes-dev \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY deploy/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy deploy folder
COPY deploy/ ./deploy/

# Run inference script
CMD ["python", "deploy/inference.py"]
