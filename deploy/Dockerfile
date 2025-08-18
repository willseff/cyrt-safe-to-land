# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Install system dependencies for pyodbc and cfgrib
RUN apt-get update && \
    apt-get install -y gcc g++ unixodbc-dev libssl-dev libffi-dev libpq-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy deploy folder
COPY deploy/ ./deploy/

# Copy model file
COPY deploy/weather_landing_bundle.pth ./deploy/weather_landing_bundle.pth

# Set environment variables for Azure credentials (override in deployment)
ENV ADLS_CONNECTION_STRING="<your_adls_connection_string>"
ENV ADLS_CONTAINER="<your_container>"
ENV ADLS_BLOB_NAME="realtime_forecast.grib2"
ENV AZURE_SQL_CONN_STR="<your_sql_connection_string>"

# Run inference script
CMD ["python", "deploy/inference.py"]
