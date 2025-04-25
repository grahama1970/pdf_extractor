#!/usr/bin/env python3
"""
PDF Extractor ArangoDB Integration Module

This module provides functions for integrating the PDF extractor with ArangoDB,
enabling efficient storage and retrieval of extracted PDF content using various
query methods: basic text search, fulltext search, BM25 search, and preparation
for semantic and hybrid search.
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import ArangoDB connection module
try:
    from pdf_extractor.arangodb_borked.connection import get_db, create_collections
except ImportError as e:
    logger.error(f"Failed to import connection module: {e}")
    logger.error("Please ensure the ArangoDB connection module is available")

# PDF collection and view names - can be overridden with environment variables
PDF_COLLECTION_NAME = os.getenv('PDF_COLLECTION_NAME', 'pdf_documents')
PDF_VIEW_NAME = os.getenv('PDF_VIEW_NAME', 'pdf_search_view')
