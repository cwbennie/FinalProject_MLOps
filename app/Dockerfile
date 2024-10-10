# Use a slim Python image as the base
FROM python:3.12-slim
ENV MLFLOW_RUNID="d4f8d43642bd4396b27f3f8d972c3fd4"
ENV MLFLOW_EXPNAME="final_training"
ENV MLFLOW_URI="https://mlops-680-370980073666.us-west2.run.app"

# Set working directory inside the container
WORKDIR /app

# Upgrade pip and install system dependencies
RUN pip install -U pip && \
    apt-get update && \
    apt-get install -y python3-dev build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/scripts /app/data

# Copy application files
COPY mlflow_data /app/mlflow_data
COPY scripts/ /app/scripts/
COPY epl_app.py /app/

# Expose the port for the FastAPI application
EXPOSE 8080

# Use separate ENTRYPOINT and CMD
ENTRYPOINT [ "uvicorn" ]
CMD [ "epl_app:app", "--host", "0.0.0.0", "--port", "8080"]
