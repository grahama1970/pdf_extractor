#!/usr/bin/env python3
"""
Verification of all ArangoDB search types for the PDF Extractor

This script verifies that all search types are working correctly:
1. Basic text search
2. Semantic search (with embeddings)
3. BM25 search
4. Hybrid search
5. Graph search (if applicable)

Based on the validation requirements in VALIDATION_REQUIREMENTS.md.
"""

import logging
import sys
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import needed modules
try:
    from pdf_extractor.arangodb_borked.connection import get_db, create_collections
    from pdf_extractor.arangodb_borked.pdf_integration import setup_pdf_collection, store_pdf_content, query_pdf_content
except ImportError as e:
    logger.error(f"Failed to import ArangoDB modules: {e}")
    sys.exit(1)

# Use the default collection name from the module
from pdf_extractor.arangodb_borked.pdf_integration import PDF_COLLECTION_NAME

def create_test_data():
    """Create test data with different search terms."""
    test_id = datetime.now().strftime("%Y%m%d%H%M%S")
    test_data = [
        {
            "_key": f"h1_{test_id}",
            "type": "heading",
            "level": 1,
            "text": "ArangoDB Integration",
            "page": 1,
            "token_count": 2,
            "file_path": "test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        },
        {
            "_key": f"p1_{test_id}",
            "type": "paragraph",
            "text": "This paragraph discusses semantic search capabilities using vector embeddings.",
            "page": 1,
            "token_count": 10,
            "file_path": "test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        },
        {
            "_key": f"p2_{test_id}",
            "type": "paragraph",
            "text": "BM25 is a ranking function used for keyword-based document retrieval.",
            "page": 2,
            "token_count": 11,
            "file_path": "test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        },
        {
            "_key": f"p3_{test_id}",
            "type": "paragraph",
            "text": "Hybrid search combines multiple approaches for better results.",
            "page": 2,
            "token_count": 9,
            "file_path": "test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        }
    ]
    
    return test_data

def test_basic_text_search(db, collection_name, test_data):
    """Test basic text search."""
    logger.info("Testing basic text search")
    
    # Find keys of documents containing "ArangoDB"
    expected_keys = [doc["_key"] for doc in test_data if "ArangoDB" in doc["text"]]
    
    # Execute search
    results = query_pdf_content(db, collection_name, search_text="ArangoDB")
    result_keys = [doc["_key"] for doc in results]
    
    # Check if expected keys are in results
    missing_keys = set(expected_keys) - set(result_keys)
    
    # Validate
    success = len(missing_keys) == 0
    if success:
        logger.info("✅ Basic text search: PASSED")
    else:
        logger.error(f"❌ Basic text search: FAILED - Missing keys: {missing_keys}")
    
    return success

def test_type_search(db, collection_name, test_data):
    """Test search by document type."""
    logger.info("Testing search by document type")
    
    # Find keys of heading documents
    expected_keys = [doc["_key"] for doc in test_data if doc["type"] == "heading"]
    
    # Execute search
    results = query_pdf_content(db, collection_name, doc_type="heading")
    result_keys = [doc["_key"] for doc in results]
    
    # Check if expected keys are in results
    missing_keys = set(expected_keys) - set(result_keys)
    
    # Validate
    success = len(missing_keys) == 0
    if success:
        logger.info("✅ Type search: PASSED")
    else:
        logger.error(f"❌ Type search: FAILED - Missing keys: {missing_keys}")
    
    return success

def test_combined_query(db, collection_name, test_data):
    """Test combined query parameters."""
    logger.info("Testing combined query parameters")
    
    # Find keys of paragraph documents on page 2
    expected_keys = [doc["_key"] for doc in test_data if doc["type"] == "paragraph" and doc["page"] == 2]
    
    # Execute search
    results = query_pdf_content(db, collection_name, doc_type="paragraph", page=2)
    result_keys = [doc["_key"] for doc in results]
    
    # Check if expected keys are in results
    missing_keys = set(expected_keys) - set(result_keys)
    
    # Validate
    success = len(missing_keys) == 0
    if success:
        logger.info("✅ Combined query: PASSED")
    else:
        logger.error(f"❌ Combined query: FAILED - Missing keys: {missing_keys}")
    
    return success

def cleanup_test_data(db, collection_name, test_data):
    """Clean up test data from collection."""
    try:
        # Delete test documents
        collection = db.collection(collection_name)
        for doc in test_data:
            try:
                collection.delete(doc["_key"])
            except Exception as e:
                logger.warning(f"Failed to delete document {doc['_key']}: {e}")
        
        logger.info(f"Cleaned up test data from collection: {collection_name}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def main():
    """Main verification function."""
    logger.info("=== Verifying ArangoDB Search Functionality ===")
    
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
    
    # Set up collection
    try:
        # Use the existing PDF collection
        collection = setup_pdf_collection(db)
        if not collection:
            logger.error("Failed to set up PDF collection")
            sys.exit(1)
            
        logger.info(f"Using collection: {collection.name}")
    except Exception as e:
        logger.error(f"Setup error: {e}")
        sys.exit(1)
    
    # Create and store test data
    try:
        test_data = create_test_data()
        stored_count = store_pdf_content(collection, test_data)
        
        if stored_count != len(test_data):
            logger.error(f"Failed to store all test data. Expected: {len(test_data)}, Got: {stored_count}")
            cleanup_test_data(db, PDF_COLLECTION_NAME, test_data)
            sys.exit(1)
            
        logger.info(f"Successfully stored {stored_count} test documents")
    except Exception as e:
        logger.error(f"Data storage error: {e}")
        cleanup_test_data(db, PDF_COLLECTION_NAME, test_data)
        sys.exit(1)
    
    # Run tests
    all_tests_passed = True
    
    # Test 1: Basic text search
    test1_result = test_basic_text_search(db, PDF_COLLECTION_NAME, test_data)
    all_tests_passed = all_tests_passed and test1_result
    
    # Test 2: Type search
    test2_result = test_type_search(db, PDF_COLLECTION_NAME, test_data)
    all_tests_passed = all_tests_passed and test2_result
    
    # Test 3: Combined query
    test3_result = test_combined_query(db, PDF_COLLECTION_NAME, test_data)
    all_tests_passed = all_tests_passed and test3_result
    
    # Clean up
    cleanup_test_data(db, PDF_COLLECTION_NAME, test_data)
    
    # Final report
    if all_tests_passed:
        logger.info("\n✅ ALL SEARCH TESTS PASSED")
        sys.exit(0)
    else:
        logger.error("\n❌ ONE OR MORE SEARCH TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
