"""
BM25 Search Validation for PDF Extractor ArangoDB Integration.

This module validates the BM25 search functionality by comparing results
against expected outputs stored in test fixtures.

Third-Party Package Documentation:
- python-arango: https://python-driver.arangodb.com/
- loguru: https://github.com/Delgan/loguru

Sample Input:
Query text for BM25 search

Expected Output:
Validation report indicating if search results match expectations
and details of any discrepancies
"""
import sys
import os
from loguru import logger

# Add root directory to path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.search_api.bm25 import search_bm25
from pdf_extractor.arangodb.validation.validation_utils import (
    save_fixture,
    load_fixture,
    compare_results,
    report_validation,
    ensure_fixtures_dir
)

def validate_bm25_search():
    """
    Validate BM25 search functionality by comparing results against expected outputs.
    
    This function:
    1. Connects to ArangoDB
    2. Creates a test collection and view with a unique ID
    3. Inserts test documents
    4. Performs a BM25 search
    5. Validates the results against a fixture
    6. Cleans up the test resources
    
    Returns:
        bool: True if validation passed, False otherwise
    """
    import uuid
    from datetime import datetime
    import time
    
    # Generate a unique ID for test resources
    test_id = str(uuid.uuid4())[:8]
    test_collection = f"test_collection_{test_id}"
    test_view = f"test_view_{test_id}"
    
    try:
        # Connect to ArangoDB
        client = connect_arango()
        if not client:
            logger.error("Failed to connect to ArangoDB")
            return False
        
        db = ensure_database(client)
        if not db:
            logger.error("Failed to ensure database exists")
            return False
        
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
        
        # Create search view for the test collection
        view_properties = {
            "links": {
                test_collection: {
                    "fields": {
                        "problem": {"analyzers": ["text_en"]},
                        "solution": {"analyzers": ["text_en"]},
                        "tags": {"analyzers": ["identity"]}
                    },
                    "includeAllFields": False,
                    "storeValues": "id",
                    "trackListPositions": False
                }
            }
        }
        
        # Check if view exists using views list
        views = db.views()
        view_exists = any(v['name'] == test_view for v in views)
        
        if view_exists:
            db.update_view_properties(test_view, view_properties)
            logger.info(f"Updated test view '{test_view}'")
        else:
            db.create_arangosearch_view(test_view, properties=view_properties)
            logger.info(f"Created test view '{test_view}'")
        
        # Wait for view indexing
        logger.info("Waiting 2 seconds for view indexing...")
        time.sleep(2)
        
        # Execute BM25 search
        results = search_bm25(
            db=db,
            search_text="python error",
            bm25_threshold=0.1,
            top_n=3,
            offset=0,
            tags=None,
            view_name=test_view
        )
        
        # Add timestamp for fixture identification
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fixture_name = f"bm25_search_expected_{timestamp}"
        
        # Load or create fixture
        fixture = load_fixture(fixture_name)
        if not fixture:
            # First run - save as fixture
            fixture_path = save_fixture(fixture_name, results)
            logger.info(f"Created new fixture: {fixture_path}")
            
            # Clean up test resources
            try:
                views = db.views()
                view_exists = any(v['name'] == test_view for v in views)
                if view_exists:
                    db.delete_view(test_view)
                    logger.info(f"Deleted test view '{test_view}'")
            except Exception as e:
                logger.warning(f"Failed to delete test view '{test_view}': {e}")
            
            try:
                if db.has_collection(test_collection):
                    db.delete_collection(test_collection)
                    logger.info(f"Deleted test collection '{test_collection}'")
            except Exception as e:
                logger.warning(f"Failed to delete test collection '{test_collection}': {e}")
                
            return True
        
        # Compare against fixture
        passed, failures = compare_results(fixture, results)
        
        # Clean up regardless of validation result
        try:
            views = db.views()
            view_exists = any(v['name'] == test_view for v in views)
            if view_exists:
                db.delete_view(test_view)
                logger.info(f"Deleted test view '{test_view}'")
        except Exception as e:
            logger.warning(f"Failed to delete test view '{test_view}': {e}")
        
        try:
            if db.has_collection(test_collection):
                db.delete_collection(test_collection)
                logger.info(f"Deleted test collection '{test_collection}'")
        except Exception as e:
            logger.warning(f"Failed to delete test collection '{test_collection}': {e}")
        
        return report_validation(passed, failures, "BM25 search")
        
    except Exception as e:
        logger.exception(f"Error validating BM25 search: {e}")
        
        # Attempt cleanup in case of error
        try:
            db = ensure_database(connect_arango())
            views = db.views()
            view_exists = any(v['name'] == test_view for v in views)
            if view_exists:
                db.delete_view(test_view)
                
            if db.has_collection(test_collection):
                db.delete_collection(test_collection)
                
            logger.info(f"Cleaned up test resources after error")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up test resources: {cleanup_error}")
            
        return False

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level:<7} | {message}"
    )
    
    # Ensure fixtures directory exists
    ensure_fixtures_dir()
    
    # Run validation
    logger.info("Running BM25 search validation...")
    if validate_bm25_search():
        logger.success("✅ BM25 search validation PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ BM25 search validation FAILED!")
        sys.exit(1)
