#!/usr/bin/env python3
"""
Direct test of ArangoDB search functionality for PDF Extractor

This script directly connects to ArangoDB and tests basic search capabilities.
"""

import logging
import sys
import uuid
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main test function."""
    logger.info("=== Direct Test of ArangoDB Search Functionality ===")
    
    # Import needed modules
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
            
        logger.info(f"Connected to database: {db.name}")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)
    
    # Check available collections
    try:
        collections = db.collections()
        collection_names = [c['name'] for c in collections]
        logger.info(f"Available collections: {collection_names}")
        
        if 'pdf_documents' not in collection_names:
            logger.error("pdf_documents collection not found")
            sys.exit(1)
            
        pdf_collection = db.collection('pdf_documents')
        logger.info(f"Accessed pdf_documents collection")
    except Exception as e:
        logger.error(f"Failed to access collections: {e}")
        sys.exit(1)
    
    # Insert test document
    test_id = f"search_test_{uuid.uuid4().hex[:8]}"
    test_doc = {
        "_key": test_id,
        "type": "paragraph",
        "text": "This is a test document for ArangoDB search",
        "page": 1,
        "token_count": 8,
        "file_path": "test.pdf",
        "extraction_date": datetime.now().isoformat(),
        "source": "test"
    }
    
    try:
        pdf_collection.insert(test_doc)
        logger.info(f"Inserted test document with key: {test_id}")
    except Exception as e:
        logger.error(f"Failed to insert test document: {e}")
        sys.exit(1)
    
    # Test 1: Basic AQL query
    try:
        aql = f"""
        FOR doc IN pdf_documents
            FILTER doc._key == @key
            RETURN doc
        """
        
        cursor = db.aql.execute(aql, bind_vars={"key": test_id})
        results = [doc for doc in cursor]
        
        if len(results) == 1 and results[0]["_key"] == test_id:
            logger.info("✅ Basic AQL query: PASSED")
        else:
            logger.error("❌ Basic AQL query: FAILED")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Basic AQL query error: {e}")
        sys.exit(1)
    
    # Test 2: Text search
    try:
        aql = f"""
        FOR doc IN pdf_documents
            FILTER CONTAINS(doc.text, "ArangoDB", true)
            RETURN doc
        """
        
        cursor = db.aql.execute(aql)
        results = [doc for doc in cursor]
        
        if any(doc["_key"] == test_id for doc in results):
            logger.info("✅ Text search: PASSED")
        else:
            logger.error("❌ Text search: FAILED")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Text search error: {e}")
        sys.exit(1)
    
    # Test 3: Type query
    try:
        aql = f"""
        FOR doc IN pdf_documents
            FILTER doc.type == "paragraph"
            RETURN doc
        """
        
        cursor = db.aql.execute(aql)
        results = [doc for doc in cursor]
        
        if any(doc["_key"] == test_id for doc in results):
            logger.info("✅ Type query: PASSED")
        else:
            logger.error("❌ Type query: FAILED")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Type query error: {e}")
        sys.exit(1)
    
    # Clean up
    try:
        pdf_collection.delete(test_id)
        logger.info(f"Deleted test document: {test_id}")
    except Exception as e:
        logger.error(f"Failed to delete test document: {e}")
    
    logger.info("All search tests completed successfully")

if __name__ == "__main__":
    main()
