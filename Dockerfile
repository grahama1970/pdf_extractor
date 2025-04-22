# Use Python 3.11 slim as base image (required by pyproject.toml)
FROM python:3.11-slim

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies for all required packages
RUN apt-get update && apt-get install -y \
    poppler-utils \
    ghostscript \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    # Additional deps for camelot-py
    python3-tk \
    libxml2-dev \
    libxslt1-dev \
    # Additional deps for pdf2image
    poppler-utils \
    # Additional deps for pandoc
    pandoc \
    # For playwright
    wget \
    curl \
    gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Install with pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Install spaCy language model
RUN python -m spacy download en_core_web_sm

# Install Playwright browsers
RUN playwright install chromium

# Create directories for file storage if they don't exist
RUN mkdir -p /app/uploads /app/output /app/corrections /app/extracted_data

# Expose the port FastAPI will run on
EXPOSE 5000

# Run the FastAPI server with Gunicorn
CMD ["gunicorn", "api:app", "-b", "0.0.0.0:5000", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "300"]