# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Install pandoc for DOCX conversion (optional but recommended)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create cache and output directories
RUN mkdir -p /app/.cache /app/output

COPY ./src ./src
COPY ./scripts ./scripts

# Default command shows help; override in docker run
ENTRYPOINT ["python", "-m", "onenote_exporter.cli"]
