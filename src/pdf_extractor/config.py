"""
Configuration settings for the PDF extractor application.

This module defines default configuration values and loads environment variables
for both the FastAPI server and CLI application. It includes settings for
directories, Label Studio integration, and ArangoDB connection.

Usage:
    Import configuration values:
    ```python
    from .config import DEFAULT_OUTPUT_DIR, LABEL_STUDIO_URL
    ```
"""

import os
from pathlib import Path

# Directory paths
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
DEFAULT_CORRECTIONS_DIR = os.getenv("CORRECTIONS_DIR", "corrections")
DEFAULT_UPLOADS_DIR = os.getenv("UPLOADS_DIR", "uploads")

# Label Studio settings
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080/api")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN", "")
LABEL_STUDIO_USERNAME = os.getenv("LABEL_STUDIO_USERNAME", "admin@example.com")
LABEL_STUDIO_PASSWORD = os.getenv("LABEL_STUDIO_PASSWORD", "admin")

# ArangoDB settings
ARANGO_HOST = os.getenv("ARANGO_HOST", "http://localhost:8529")
ARANGO_USER = os.getenv("ARANGO_USER", "root")
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD", "password")
ARANGO_DB = os.getenv("ARANGO_DB", "pdf_extractor")

# Resource settings
MAX_RAM_GB = int(os.getenv("MAX_RAM_GB", "256"))
GPU_REQUIRED = os.getenv("GPU_REQUIRED", "true").lower() in ["true", "1", "yes"]
GPU_MEMORY_GB = int(os.getenv("GPU_MEMORY_GB", "24"))

# Ensure directory paths exist
for directory in [DEFAULT_OUTPUT_DIR, DEFAULT_CORRECTIONS_DIR, DEFAULT_UPLOADS_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)