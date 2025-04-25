#!/usr/bin/env python3
"""
Simple ArangoDB Integration Test
"""

import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import connection module
from pdf_extractor.arangodb_borked.connection import get_db

def test_integration():
    """Test ArangoDB integration"""
    try:
        # Connect to database
        db = get_db()
        if not db:
            logger.error("Failed to connect to ArangoDB")
            return False
            
        logger.info(f"Connected to ArangoDB database: {db.name}")
        
        # Check collections
        collections = db.collections()
        collection_names = [c["name"] for c in collections]
        logger.info(f"Available collections: {collection_names}")
        
        # Access pdf_documents collection
        collection_name = "pdf_documents"
        if collection_name not in collection_names:
            logger.info(f"Creating collection: {collection_name}")
            collection = db.create_collection(collection_name)
        else:
            logger.info(f"Using existing collection: {collection_name}")
            collection = db.collection(collection_name)
        
        # Create test document
        test_id = datetime.now().strftime("%Y%m%d%H%M%S")
        test_doc = {
            "_key": f"test_{test_id}",
            "type": "heading",
            "level": 1,
            "text": "Test Heading",
            "page": 1,
            "file_path": "test.pdf",
            "extraction_date": datetime.now().isoformat()
        }
        
        # Insert document
        result = collection.insert(test_doc)
        logger.info(f"Inserted document: {result['_key']}")
        
        # Query document
        aql = f"""
        FOR doc IN {collection_name}
            FILTER doc._key == @key
            RETURN doc
        """
        cursor = db.aql.execute(aql, bind_vars={"key": test_doc["_key"]})
        results = [doc for doc in cursor]
        
        if not results:
            logger.error("Failed to find document")
            return False
            
        logger.info(f"Retrieved document: {results[0]['_key']}")
        
        # Delete document
        collection.delete(test_doc["_key"])
        logger.info(f"Deleted document: {test_doc['_key']}")
        
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("=== Simple ArangoDB Integration Test ===")
    
    success = test_integration()
    
    if success:
        logger.info("✅ Integration test passed")
        sys.exit(0)
    else:
        logger.error("❌ Integration test failed")
        sys.exit(1)
