#!/usr/bin/env python3
"""
Direct test of ArangoDB functionality
"""

import sys
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import connection module directly
from pdf_extractor.arangodb_borked.connection import get_db

def test_direct_integration():
    """Test ArangoDB functionality directly"""
    try:
        # Connect to database
        db = get_db()
        if not db:
            logger.error("Failed to connect to ArangoDB")
            return False
            
        logger.info(f"Connected to ArangoDB database: {db.name}")
        
        # Test collections
        collections = db.collections()
        collection_names = [c["name"] for c in collections]
        logger.info(f"Collections: {collection_names}")
        
        # Check if pdf_documents exists
        collection_name = "pdf_documents"
        if collection_name not in collection_names:
            logger.error(f"Collection {collection_name} not found")
            return False
            
        # Get the collection
        collection = db.collection(collection_name)
        logger.info(f"Successfully accessed collection: {collection.name}")
        
        # Create a test document
        test_id = datetime.now().strftime("%Y%m%d%H%M%S")
        test_doc = {
            "_key": f"direct_test_{test_id}",
            "type": "test",
            "text": "Direct test document",
            "file_path": "test.pdf",
            "page": 1,
            "extraction_date": datetime.now().isoformat()
        }
        
        # Insert document
        try:
            result = collection.insert(test_doc)
            logger.info(f"Successfully inserted document with key: {result['_key']}")
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            logger.error(traceback.format_exc())
            return False
            
        # Test query
        try:
            aql = f"""
            FOR doc IN {collection_name}
                FILTER doc._key == @key
                RETURN doc
            """
            
            cursor = db.aql.execute(aql, bind_vars={"key": test_doc["_key"]})
            results = [doc for doc in cursor]
            
            if not results:
                logger.error("Failed to find inserted document")
                return False
                
            logger.info(f"Successfully retrieved document: {results[0]['_key']}")
        except Exception as e:
            logger.error(f"Failed to query document: {e}")
            logger.error(traceback.format_exc())
            return False
            
        # Clean up
        try:
            collection.delete(test_doc["_key"])
            logger.info(f"Successfully deleted document: {test_doc['_key']}")
        except Exception as e:
            logger.warning(f"Failed to delete document: {e}")
            
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("=== Direct Test of ArangoDB Functionality ===")
    
    success = test_direct_integration()
    
    if success:
        logger.info("✅ Direct test passed")
        sys.exit(0)
    else:
        logger.error("❌ Direct test failed")
        sys.exit(1)
