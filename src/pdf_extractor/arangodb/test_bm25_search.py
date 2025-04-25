#!/usr/bin/env python
"""
Test the BM25 search functionality.

This script tests the search_bm25 function from the pdf_extractor.arangodb.search_api.bm25 module
by performing a simple search and validating the results against expected outputs.
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
    from pdf_extractor.arangodb.arango_setup import (
        connect_arango, 
        ensure_database,
    )
    from pdf_extractor.arangodb.search_api.bm25 import search_bm25
    # Import config variables
    from pdf_extractor.arangodb.config import (
        COLLECTION_NAME,
        VIEW_NAME,
        SEARCH_FIELDS,
        TEXT_ANALYZER,
        TAG_ANALYZER,
    )
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
                        "tags": {"analyzers": [TAG_ANALYZER]},
                    },
                    "includeAllFields": False,
                    "storeValues": "id",
                    "trackListPositions": False,
                }
            },
            "primarySort": [{"field": "_key", "direction": "asc"}],
            "storedValues": [
                {"fields": ["problem", "solution", "tags"], "compression": "lz4"},
            ],
            "consolidationPolicy": {
                "type": "tier",
                "threshold": 0.1,
                "segmentsMin": 1,
                "segmentsMax": 10,
            },
        }
        
        # Check if view exists using try/except since has_view might not be available
        try:
            # Try to get view properties - if it doesn't exist, an exception will be raised
            db.view(view_name)
            logger.info(f"View '{view_name}' already exists")
            db.update_view_properties(view_name, view_properties)
            logger.info(f"Updated properties for view '{view_name}'")
        except Exception:
            # View doesn't exist, create it
            db.create_arangosearch_view(view_name, properties=view_properties)
            logger.info(f"Created view '{view_name}'")
            
        return True
    except Exception as e:
        logger.error(f"Error creating ArangoSearch view: {e}")
        return False

def setup_test_collection(db):
    """Set up a test collection and view for testing BM25 search."""
    test_id = str(uuid.uuid4())[:6]
    test_coll_name = f"test_collection_{test_id}"
    test_view_name = f"test_view_{test_id}"
    
    try:
        # Create test collection
        if not db.has_collection(test_coll_name):
            db.create_collection(test_coll_name)
            logger.info(f"Created test collection '{test_coll_name}'")
        
        # Create test documents
        collection = db.collection(test_coll_name)
        test_docs = [
            {
                "_key": f"doc1_{test_id}",
                "problem": "Python error when processing JSON data",
                "solution": "Use try/except blocks to handle JSON parsing exceptions",
                "tags": ["python", "json", "error-handling"]
            },
            {
                "_key": f"doc2_{test_id}",
                "problem": "Database connection timeout in production",
                "solution": "Implement connection pooling and retry mechanism",
                "tags": ["database", "timeout", "production"]
            },
            {
                "_key": f"doc3_{test_id}",
                "problem": "Python script runs out of memory with large datasets",
                "solution": "Use chunking to process large data incrementally",
                "tags": ["python", "memory", "optimization"]
            }
        ]
        
        for doc in test_docs:
            collection.insert(doc)
        logger.info(f"Inserted {len(test_docs)} test documents")
        
        # Create test view using our custom function
        view_created = create_search_view(db, test_view_name, test_coll_name)
        if not view_created:
            logger.error(f"Failed to create test view '{test_view_name}'")
            return None, None
        
        logger.info(f"Created test view '{test_view_name}'")
        return test_coll_name, test_view_name
    
    except Exception as e:
        logger.error(f"Error setting up test collection and view: {e}")
        return None, None

def cleanup_test_resources(db, collection_name, view_name):
    """Clean up test resources after testing."""
    try:
        # Delete view if it exists
        try:
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

def test_bm25_search():
    """
    Test the BM25 search functionality.
    
    Expected behavior:
    1. Connect to ArangoDB
    2. Create a test collection and view
    3. Perform a BM25 search using a simple query
    4. Validate the structure and contents of the results
    5. Clean up test resources
    """
    logger.info("Testing BM25 search...")
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
        
        # Sample search parameters
        search_text = "python"  # Simple search term that should match technical documents
        tags = None  # No tag filtering for basic test
        top_n = 5  # Return top 5 results
        
        try:
            # Perform BM25 search
            logger.info(f"Performing BM25 search with query: '{search_text}'")
            results = search_bm25(
                db=db,
                search_text=search_text,
                bm25_threshold=0.1,
                top_n=top_n,
                offset=0,
                tags=tags,
                view_name=test_view
            )
            
            # Validate results structure
            logger.info("Validating results structure...")
            
            # Check that results contain expected fields
            expected_fields = ["results", "total", "offset", "limit"]
            missing_fields = [field for field in expected_fields if field not in results]
            if missing_fields:
                logger.error(f"Results missing expected fields: {missing_fields}")
                return False
            
            # Check that results count matches limit (or is less if fewer results found)
            actual_count = len(results.get("results", []))
            expected_count = min(top_n, results.get("total", 0))
            
            # We should have at least 2 results (docs 1 and 3 match "python")
            if actual_count < 2:
                logger.error(f"Expected at least 2 results, got {actual_count}")
                return False
                
            # Check that each result has a doc and bm25_score
            for i, result in enumerate(results.get("results", [])):
                if "doc" not in result:
                    logger.error(f"Result {i} missing 'doc' field")
                    return False
                if "bm25_score" not in result:
                    logger.error(f"Result {i} missing 'bm25_score' field")
                    return False
                
                # Check that doc has a _key
                if "_key" not in result["doc"]:
                    logger.error(f"Result {i} doc missing '_key' field")
                    return False
                    
            # If we've made it this far, all validations passed
            logger.success(f"Successfully retrieved {actual_count} results (total={results.get('total', 0)})")
            
            # Print a sample result for manual verification
            if results.get("results"):
                sample = results["results"][0]
                logger.info(f"Sample result - Key: {sample['doc'].get('_key')}, Score: {sample['bm25_score']}")
                
            return True
        
        except Exception as e:
            logger.exception(f"Error during BM25 search test: {e}")
            return False
    
    except Exception as e:
        logger.exception(f"Error in BM25 search test: {e}")
        return False
    
    finally:
        # Clean up test resources
        if test_collection or test_view:
            logger.info("Cleaning up test resources...")
            cleanup_test_resources(db, test_collection, test_view)

if __name__ == "__main__":
    logger.info("Starting BM25 search test...")
    
    if test_bm25_search():
        logger.success("✅ BM25 search test PASSED")
        sys.exit(0)
    else:
        logger.error("❌ BM25 search test FAILED")
        sys.exit(1)
