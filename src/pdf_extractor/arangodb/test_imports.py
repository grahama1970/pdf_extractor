#!/usr/bin/env python
"""
Test if imports for search modules work correctly.

This script checks that all necessary modules for the search functionality
can be properly imported. It should be run after fixing any import issues
in the search modules.
"""

import sys
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:HH:mm:ss} | {level:<7} | {message}"
)

def test_imports():
    """Test if all search module imports work correctly."""
    success = True
    
    # Try importing configuration
    try:
        from pdf_extractor.arangodb.config import (
            SEARCH_FIELDS,
            ALL_DATA_FIELDS_PREVIEW,
            TEXT_ANALYZER,
            TAG_ANALYZER,
            VIEW_NAME,
            COLLECTION_NAME,
        )
        logger.info("✅ Successfully imported configuration")
    except ImportError as e:
        logger.error(f"❌ Failed to import configuration: {e}")
        success = False
    
    # Try importing BM25 search
    try:
        from pdf_extractor.arangodb.search_api.bm25 import search_bm25
        logger.info("✅ Successfully imported BM25 search")
    except ImportError as e:
        logger.error(f"❌ Failed to import BM25 search: {e}")
        success = False
    
    # Try importing semantic search
    try:
        from pdf_extractor.arangodb.search_api.semantic import search_semantic
        logger.info("✅ Successfully imported semantic search")
    except ImportError as e:
        logger.error(f"❌ Failed to import semantic search: {e}")
        success = False
    
    # Try importing hybrid search
    try:
        from pdf_extractor.arangodb.search_api.hybrid import search_hybrid
        logger.info("✅ Successfully imported hybrid search")
    except ImportError as e:
        logger.error(f"❌ Failed to import hybrid search: {e}")
        success = False
    
    # Try importing basic search
    try:
        from pdf_extractor.arangodb.search_api.search_basic import (
            find_lessons_by_tags_advanced,
            find_lessons_by_text_like,
        )
        logger.info("✅ Successfully imported basic search")
    except ImportError as e:
        logger.error(f"❌ Failed to import basic search: {e}")
        success = False
    
    # Try importing utility functions
    try:
        from pdf_extractor.arangodb.embedding_utils import get_embedding
        from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
        logger.info("✅ Successfully imported utility functions")
    except ImportError as e:
        logger.error(f"❌ Failed to import utility functions: {e}")
        success = False
    
    return success

if __name__ == "__main__":
    logger.info("Testing imports for search modules...")
    
    if test_imports():
        logger.info("✅ All imports work successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Import tests failed. Please fix the issues above.")
        sys.exit(1)
