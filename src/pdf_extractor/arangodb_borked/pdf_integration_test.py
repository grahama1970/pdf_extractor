#!/usr/bin/env python3
"""
Test for PDF Extractor ArangoDB Integration Module
"""

import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the integration module
from pdf_extractor.arangodb_borked.connection import get_db
from pdf_extractor.arangodb_borked.pdf_integration import (
    setup_pdf_collection,
    store_pdf_content,
    query_pdf_content,
    get_pdf_content_stats
)

def run_integration_test():
    """Run a test of the PDF extractor ArangoDB integration"""
    logger.info("=== Testing PDF Extractor ArangoDB Integration ===")
    
    # Connect to ArangoDB
    db = get_db()
    if not db:
        logger.error("Failed to connect to ArangoDB")
        return False
        
    logger.info(f"Connected to ArangoDB database: {db.name}")
    
    # Setup PDF collection
    collection = setup_pdf_collection(db)
    if not collection:
        logger.error("Failed to set up PDF collection")
        return False
        
    logger.info(f"Set up PDF collection: {collection.name}")
    
    # Create test data
    test_id = datetime.now().strftime("%Y%m%d%H%M%S")
    test_docs = [
        {
            "_key": f"test_heading_{test_id}",
            "type": "heading",
            "level": 1,
            "text": "Test Heading",
            "page": 1,
            "token_count": 2,
            "file_path": "test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        },
        {
            "_key": f"test_paragraph_{test_id}",
            "type": "paragraph",
            "text": "This is a test paragraph for validation.",
            "page": 1,
            "token_count": 7,
            "file_path": "test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        }
    ]
    
    # Store test data
    stored_count = store_pdf_content(collection, test_docs)
    logger.info(f"Stored {stored_count} test documents")
    
    if stored_count != len(test_docs):
        logger.error("Failed to store all test documents")
        return False
    
    # Query by type
    logger.info("Testing query by type...")
    type_results = query_pdf_content(db, doc_type="heading")
    
    # Check if our test heading is in the results
    heading_found = any(doc["_key"] == f"test_heading_{test_id}" for doc in type_results)
    
    if not heading_found:
        logger.error("Failed to find test heading document")
        return False
        
    logger.info("Successfully queried by type")
    
    # Query by text
    logger.info("Testing query by text...")
    text_results = query_pdf_content(db, search_text="test paragraph")
    
    # Check if our test paragraph is in the results
    paragraph_found = any(doc["_key"] == f"test_paragraph_{test_id}" for doc in text_results)
    
    if not paragraph_found:
        logger.error("Failed to find test paragraph document")
        return False
        
    logger.info("Successfully queried by text")
    
    # Get stats
    logger.info("Testing stats retrieval...")
    stats = get_pdf_content_stats(db)
    
    if stats["total_documents"] <= 0:
        logger.error("Failed to get document stats")
        return False
        
    logger.info(f"Successfully retrieved stats: {stats['total_documents']} total documents")
    
    # Clean up test data
    for doc in test_docs:
        try:
            collection.delete(doc["_key"])
            logger.info(f"Deleted test document {doc['_key']}")
        except Exception as e:
            logger.warning(f"Failed to delete test document {doc['_key']}: {e}")
    
    return True

if __name__ == "__main__":
    success = run_integration_test()
    
    if success:
        logger.info("✅ Integration test passed")
        sys.exit(0)
    else:
        logger.error("❌ Integration test failed")
        sys.exit(1)
