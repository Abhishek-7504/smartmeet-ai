# Stage 1: base Python image
FROM python:3.10-slim

WORKDIR /app

# Copy ingestion code
COPY ingestion/ .

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

# Default to running our ingest tool
ENTRYPOINT ["python", "ingest.py"]
