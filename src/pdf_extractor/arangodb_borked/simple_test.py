#!/usr/bin/env python3
"""
Simple Test for PDF Extractor ArangoDB Integration
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

# Import ArangoDB modules
from pdf_extractor.arangodb_borked.connection import get_db, create_collections

def test_arangodb_connection():
    """Test connection to ArangoDB"""
    try:
        db = get_db()
        if not db:
            logger.error("Failed to connect to ArangoDB")
            return False
            
        logger.info(f"Successfully connected to ArangoDB database: {db.name}")
        
        # List collections
        collections = [c["name"] for c in db.collections()]
        logger.info(f"Available collections: {collections}")
        
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def test_create_collection():
    """Test creating a collection"""
    try:
        db = get_db()
        if not db:
            logger.error("Failed to connect to ArangoDB")
            return False
            
        # Generate a unique test collection name
        test_id = datetime.now().strftime("%Y%m%d%H%M%S")
        test_collection_name = f"pdf_test_{test_id}"
        
        # Create the collection
        collections = create_collections(db, [test_collection_name])
        
        if test_collection_name not in collections:
            logger.error(f"Failed to create test collection: {test_collection_name}")
            return False
            
        collection = collections[test_collection_name]
        logger.info(f"Successfully created test collection: {test_collection_name}")
        
        # Create an index using the updated API
        collection.add_index({
            'type': 'hash',
            'fields': ['type'],
            'unique': False
        })
        logger.info("Successfully created hash index")
        
        # Insert a test document
        test_doc = {
            "_key": f"test_{test_id}",
            "type": "test",
            "text": "Test document",
            "file_path": "test.pdf",
            "extraction_date": datetime.now().isoformat()
        }
        
        result = collection.insert(test_doc)
        logger.info(f"Successfully inserted test document with key: {result['_key']}")
        
        # Retrieve the document
        doc = collection.get(result["_key"])
        if not doc:
            logger.error("Failed to retrieve test document")
            return False
            
        logger.info("Successfully retrieved test document")
        
        # Clean up
        db.delete_collection(test_collection_name)
        logger.info(f"Successfully deleted test collection: {test_collection_name}")
        
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("=== Running Simple ArangoDB Integration Test ===")
    
    # Test connection
    connection_result = test_arangodb_connection()
    if not connection_result:
        logger.error("❌ Connection test failed")
        sys.exit(1)
    else:
        logger.info("✅ Connection test passed")
    
    # Test collection creation
    collection_result = test_create_collection()
    if not collection_result:
        logger.error("❌ Collection creation test failed")
        sys.exit(1)
    else:
        logger.info("✅ Collection creation test passed")
    
    logger.info("✅ All tests passed")
    sys.exit(0)
