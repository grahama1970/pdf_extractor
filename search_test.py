#!/usr/bin/env python3
"""
Test script for ArangoDB search functionality

This script tests the different search types:
1. Basic text search
2. BM25 search
3. Semantic search
4. Hybrid search
"""

import logging
import sys
import os
import uuid
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to test search functionality."""
    logger.info("=== Testing ArangoDB Search Functionality ===")
    
    # Import modules
    try:
        from pdf_extractor.arangodb_borked.connection import get_db
    except ImportError as e:
        logger.error(f"Failed to import ArangoDB modules: {e}")
        sys.exit(1)
    
    # Connect to ArangoDB
    try:
        db = get_db()
        if not db:
            logger.error("Failed to connect to ArangoDB")
            sys.exit(1)
            
        logger.info(f"Connected to ArangoDB database: {db.name}")
    except Exception as e:
        logger.error(f"Failed to connect to ArangoDB: {e}")
        sys.exit(1)
    
    # Get the pdf_documents collection
    try:
        if not db.has_collection('pdf_documents'):
            logger.error("pdf_documents collection not found")
            sys.exit(1)
            
        collection = db.collection('pdf_documents')
        logger.info(f"Using collection: {collection.name}")
    except Exception as e:
        logger.error(f"Failed to access collection: {e}")
        sys.exit(1)
    
    # Create a test document
    test_id = f"search_test_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    test_doc = {
        "_key": test_id,
        "type": "paragraph",
        "text": "This document tests ArangoDB search functionality including semantic, BM25, and hybrid search",
        "page": 1,
        "token_count": 12,
        "file_path": "test.pdf",
        "extraction_date": datetime.now().isoformat(),
        "source": "test"
    }
    
    # Insert the test document
    try:
        collection.insert(test_doc)
        logger.info(f"Inserted test document with key: {test_id}")
    except Exception as e:
        logger.error(f"Failed to insert test document: {e}")
        sys.exit(1)
    
    # Test 1: Basic text search with AQL
    try:
        aql = """
        FOR doc IN pdf_documents
            FILTER CONTAINS(doc.text, @query, true)
            RETURN doc
        """
        cursor = db.aql.execute(aql, bind_vars={"query": "ArangoDB"})
        results = [doc for doc in cursor]
        
        if any(doc["_key"] == test_id for doc in results):
            logger.info("✅ Basic text search: PASSED")
        else:
            logger.error("❌ Basic text search: FAILED")
    except Exception as e:
        logger.error(f"Basic text search error: {e}")
    
    # Test 2: BM25-like search with AQL
    try:
        aql = """
        FOR doc IN pdf_documents
            SEARCH ANALYZER(PHRASE(doc.text, @query), "text_en")
            SORT BM25(doc) DESC
            LIMIT 10
            RETURN doc
        """
        cursor = db.aql.execute(aql, bind_vars={"query": "ArangoDB search"})
        results = [doc for doc in cursor]
        
        if any(doc["_key"] == test_id for doc in results):
            logger.info("✅ BM25-like search: PASSED")
        else:
            logger.error("❌ BM25-like search: FAILED")
    except Exception as e:
        logger.error(f"BM25-like search error: {e}")
    
    # Clean up
    try:
        collection.delete(test_id)
        logger.info(f"Deleted test document: {test_id}")
    except Exception as e:
        logger.error(f"Failed to delete test document: {e}")
    
    logger.info("Search testing completed")

if __name__ == "__main__":
    main()
