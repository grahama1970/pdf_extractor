"""
Test the keyword search functionality.

This script tests the search_keyword function from the pdf_extractor.arangodb.search_api.keyword module
by performing a keyword search and validating the results.
"""

import sys
import os
import uuid
from typing import Dict, List, Any, Optional
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:HH:mm:ss} | {level:<7} | {message}"
)

# Import necessary modules
try:
    from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
    from pdf_extractor.arangodb.search_api.keyword import search_keyword
    from pdf_extractor.arangodb.config import TEXT_ANALYZER, TAG_ANALYZER
except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}")
    logger.error("Please ensure the script is run from the project root or the necessary paths are in PYTHONPATH.")
    sys.exit(1)

def create_search_view(db, view_name, collection_name):
    """Create an ArangoSearch view for testing."""
    try:
        # Define view properties
        view_properties = {
            "links": {
                collection_name: {
                    "fields": {
                        "problem": {"analyzers": [TEXT_ANALYZER]},
                        "solution": {"analyzers": [TEXT_ANALYZER]},
                        "context": {"analyzers": [TEXT_ANALYZER]},
                        "tags": {"analyzers": [TAG_ANALYZER]},
                    },
                    "includeAllFields": False,
                    "storeValues": "id",
                    "trackListPositions": False,
                }
            },
            "primarySort": [{"field": "_key", "direction": "asc"}],
            "storedValues": [
                {"fields": ["problem", "solution", "context", "tags"], "compression": "lz4"},
            ],
            "consolidationPolicy": {
                "type": "tier",
                "threshold": 0.1,
                "segmentsMin": 1,
                "segmentsMax": 10,
            },
        }
        
        # Check if view exists using views list
        views = db.views()
        view_exists = any(v['name'] == view_name for v in views)
        
        if view_exists:
            logger.info(f"View '{view_name}' already exists")
            db.update_view_properties(view_name, view_properties)
            logger.info(f"Updated properties for view '{view_name}'")
        else:
            # View doesn't exist, create it
            db.create_arangosearch_view(view_name, properties=view_properties)
            logger.info(f"Created view '{view_name}'")
            
        return True
    except Exception as e:
        logger.error(f"Error creating ArangoSearch view: {e}")
        return False

def setup_test_collection(db):
    """Set up a test collection and view for testing keyword search."""
    test_id = str(uuid.uuid4())[:6]
    test_collection = f"test_collection_{test_id}"
    test_view = f"test_view_{test_id}"
    
    try:
        # Create test collection
        if not db.has_collection(test_collection):
            db.create_collection(test_collection)
            logger.info(f"Created test collection '{test_collection}'")
        
        # Create test documents
        collection = db.collection(test_collection)
        test_docs = [
            {
                "_key": f"doc1_{test_id}",
                "problem": "Python error when processing JSON data",
                "solution": "Use try/except blocks to handle JSON parsing exceptions",
                "context": "Common in web API integrations",
                "tags": ["python", "json", "error-handling"]
            },
            {
                "_key": f"doc2_{test_id}",
                "problem": "Database connection timeout in production",
                "solution": "Implement connection pooling and retry mechanism",
                "context": "High traffic periods",
                "tags": ["database", "timeout", "production"]
            },
            {
                "_key": f"doc3_{test_id}",
                "problem": "Python script runs out of memory with large datasets",
                "solution": "Use chunking to process large data incrementally",
                "context": "Big data processing",
                "tags": ["python", "memory", "optimization"]
            }
        ]
        
        for doc in test_docs:
            collection.insert(doc)
        logger.info(f"Inserted {len(test_docs)} test documents")
        
        # Create test view
        view_created = create_search_view(db, test_view, test_collection)
        if not view_created:
            logger.error(f"Failed to create test view '{test_view}'")
            return None, None
        
        logger.info(f"Created test view '{test_view}'")
        return test_collection, test_view
    
    except Exception as e:
        logger.error(f"Error setting up test collection and view: {e}")
        return None, None

def cleanup_test_resources(db, collection_name, view_name):
    """Clean up test resources after testing."""
    try:
        # Delete view if it exists
        try:
            views = db.views()
            view_exists = any(v['name'] == view_name for v in views)
            if view_exists:
                db.delete_view(view_name)
                logger.info(f"Deleted test view '{view_name}'")
        except Exception as e:
            logger.warning(f"Failed to delete view '{view_name}': {e}")
        
        # Delete collection if it exists
        if collection_name and db.has_collection(collection_name):
            db.delete_collection(collection_name)
            logger.info(f"Deleted test collection '{collection_name}'")
            
        return True
    except Exception as e:
        logger.error(f"Error cleaning up test resources: {e}")
        return False

def test_keyword_search():
    """
    Test the keyword search functionality.
    
    Expected behavior:
    1. Connect to ArangoDB
    2. Create a test collection and view with a unique ID
    3. Insert test documents
    4. Perform a keyword search
    5. Validate the results
    6. Clean up test resources
    
    Returns:
        bool: True if test passes, False otherwise
    """
    logger.info("Testing keyword search...")
    test_collection = None
    test_view = None
    
    try:
        # Connect to ArangoDB
        logger.info("Connecting to ArangoDB...")
        client = connect_arango()
        if not client:
            logger.error("Failed to connect to ArangoDB")
            return False
        
        # Get database
        logger.info("Getting database...")
        db = ensure_database(client)
        if not db:
            logger.error("Failed to ensure database exists")
            return False
        
        # Setup test collection and view
        logger.info("Setting up test resources...")
        test_collection, test_view = setup_test_collection(db)
        if not test_collection or not test_view:
            logger.error("Failed to set up test resources")
            return False
        
        # Add a small delay to allow indexing
        import time
        logger.info("Waiting 2 seconds for indexing...")
        time.sleep(2)
        
        # Execute keyword search
        logger.info("Performing keyword search...")
        search_term = "python"
        results = search_keyword(
            db=db,
            search_term=search_term,
            similarity_threshold=90.0,  # Lower threshold for testing
            top_n=5,
            view_name=test_view,
            collection_name=test_collection
        )
        
        # Validate results
        logger.info(f"Validating results...")
        
        # Check that the results dictionary has the expected structure
        if not isinstance(results, dict):
            logger.error(f"Expected results to be a dictionary, got {type(results)}")
            return False
            
        if "results" not in results:
            logger.error("Results dictionary missing 'results' key")
            return False
            
        if "total" not in results:
            logger.error("Results dictionary missing 'total' key")
            return False
            
        # Check that we found at least one result
        if results["total"] < 1:
            logger.error(f"Expected at least 1 result, got {results['total']}")
            return False
            
        # Check that results contain documents with the expected fields
        for i, item in enumerate(results["results"]):
            if "doc" not in item:
                logger.error(f"Result {i} missing 'doc' field")
                return False
                
            if "keyword_score" not in item:
                logger.error(f"Result {i} missing 'keyword_score' field")
                return False
                
            doc = item["doc"]
            if "_key" not in doc:
                logger.error(f"Result {i} doc missing '_key' field")
                return False
        
        logger.success(f"Found {results['total']} results for keyword '{search_term}'")
        
        # Print a sample result
        if results["results"]:
            sample = results["results"][0]
            logger.info(f"Sample result - Key: {sample['doc'].get('_key')}, Score: {sample['keyword_score']:.2f}")
            logger.info(f"  Problem: {sample['doc'].get('problem', '')}")
            
        return True
    
    except Exception as e:
        logger.exception(f"Error during keyword search test: {e}")
        return False
    
    finally:
        # Clean up test resources
        if test_collection or test_view:
            logger.info("Cleaning up test resources...")
            cleanup_test_resources(db, test_collection, test_view)

if __name__ == "__main__":
    logger.info("Starting keyword search test...")
    
    if test_keyword_search():
        logger.success("✅ Keyword search works!")
        sys.exit(0)
    else:
        logger.error("❌ Keyword search test FAILED")
        sys.exit(1)
