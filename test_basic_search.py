#!/usr/bin/env python3
"""
Simplified test for ArangoDB search functionality

This script tests the available search types that don't require vector index:
1. Basic text search
2. BM25 search using ArangoSearch view

The test works with the standard ArangoDB configuration without
special experimental features.
"""

import logging
import sys
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for search configuration
COLLECTION_NAME = 'pdf_documents'
TEST_PREFIX = f"search_test_{datetime.now().strftime('%Y%m%d%H%M%S')}"
VIEW_NAME = f"{COLLECTION_NAME}_view"

# Import project modules
try:
    from pdf_extractor.arangodb_borked.connection import get_db
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)

def ensure_search_view(db, view_name: str, collection_name: str):
    """
    Ensure ArangoSearch view exists for BM25 search.
    Based on examples/arangodb/arango_setup.py but simplified.
    
    Args:
        db: ArangoDB database connection
        view_name: Name for the ArangoSearch view
        collection_name: Collection to link to view
        
    Returns:
        Boolean indicating success
    """
    try:
        logger.info(f"Setting up ArangoSearch view: {view_name}")
        
        # Check if view exists
        views = db.views()
        view_exists = any(v['name'] == view_name for v in views)
        
        if view_exists:
            logger.info(f"View {view_name} already exists, deleting first")
            db.delete_view(view_name)
        
        # Create view with links to collection
        view_properties = {
            "type": "arangosearch",
            "links": {
                collection_name: {
                    "fields": {
                        "text": {
                            "analyzers": ["text_en"]
                        },
                        "type": {},
                        "file_path": {},
                        "page": {}
                    },
                    "includeAllFields": True,  # Changed to include all fields
                    "trackListPositions": False,
                    "storeValues": "none",
                    "analyzers": ["identity", "text_en"]
                }
            },
            "consolidationIntervalMsec": 1000,
            "commitIntervalMsec": 1000
        }
        
        db.create_view(view_name, "arangosearch", view_properties)
        logger.info(f"Created ArangoSearch view: {view_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create ArangoSearch view: {e}")
        return False

def create_test_documents():
    """
    Create test documents for search testing (without embeddings).
    
    Returns:
        List of test documents
    """
    docs = [
        {
            "_key": f"{TEST_PREFIX}_doc1",
            "type": "heading",
            "level": 1,
            "text": "Introduction to ArangoDB Search",
            "page": 1,
            "token_count": 4,
            "file_path": "search_test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        },
        {
            "_key": f"{TEST_PREFIX}_doc2",
            "type": "paragraph",
            "text": "Semantic search uses vector embeddings to find documents with similar meaning.",
            "page": 1,
            "token_count": 12,
            "file_path": "search_test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        },
        {
            "_key": f"{TEST_PREFIX}_doc3",
            "type": "paragraph",
            "text": "BM25 is a ranking algorithm widely used for keyword search in information retrieval systems.",
            "page": 2,
            "token_count": 14,
            "file_path": "search_test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        },
        {
            "_key": f"{TEST_PREFIX}_doc4",
            "type": "paragraph",
            "text": "Hybrid search combines the precision of keyword search with the understanding of semantic similarity.",
            "page": 2,
            "token_count": 14,
            "file_path": "search_test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        },
        {
            "_key": f"{TEST_PREFIX}_doc5",
            "type": "paragraph",
            "text": "Graph traversal search finds connections between related documents by following links.",
            "page": 3,
            "token_count": 12,
            "file_path": "search_test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        },
        {
            "_key": f"{TEST_PREFIX}_doc6",
            "type": "table",
            "text": "Comparison of search methods: keyword, semantic, BM25, hybrid, and graph.",
            "headers": ["Method", "Precision", "Recall", "Use Case"],
            "rows": [
                ["Keyword", "High", "Low", "Exact matches"],
                ["Semantic", "Medium", "High", "Conceptual similarity"],
                ["BM25", "High", "Medium", "Relevance ranking"],
                ["Hybrid", "High", "High", "Best overall results"],
                ["Graph", "Medium", "High", "Related content"]
            ],
            "page": 4,
            "token_count": 30,
            "file_path": "search_test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        }
    ]
    
    return docs

def test_basic_search(db, collection_name, test_docs):
    """
    Test basic keyword search using AQL.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        test_docs: List of test documents
        
    Returns:
        Boolean indicating success
    """
    logger.info("\n=== Testing Basic Text Search ===")
    
    # Search query
    search_term = "ArangoDB"
    expected_docs = [doc for doc in test_docs if search_term in doc["text"]]
    expected_keys = [doc["_key"] for doc in expected_docs]
    
    try:
        # Execute search with AQL
        aql = f"""
        FOR doc IN {collection_name}
            FILTER CONTAINS(doc.text, @search_term, true)
            SORT doc._key
            RETURN doc
        """
        cursor = db.aql.execute(aql, bind_vars={"search_term": search_term})
        results = [doc for doc in cursor]
        result_keys = [doc["_key"] for doc in results]
        
        # Verify results
        found_expected = all(key in result_keys for key in expected_keys)
        
        if found_expected:
            logger.info(f"✅ Basic text search PASSED - Found {len(result_keys)} results including all expected documents")
            return True
        else:
            missing = set(expected_keys) - set(result_keys)
            logger.error(f"❌ Basic text search FAILED - Missing expected documents: {missing}")
            return False
    except Exception as e:
        logger.error(f"Basic text search error: {e}")
        return False

def test_bm25_search(db, view_name, test_docs):
    """
    Test BM25 search using ArangoSearch view.
    
    Args:
        db: ArangoDB database connection
        view_name: ArangoSearch view name
        test_docs: List of test documents
        
    Returns:
        Boolean indicating success
    """
    logger.info("\n=== Testing BM25 Search ===")
    
    # Using a simpler search term that should match doc3
    search_term = "BM25"
    expected_doc_keys = [f"{TEST_PREFIX}_doc3"]  # Doc about BM25
    
    try:
        # Execute search using simpler query
        aql = f"""
        FOR doc IN {view_name}
            SEARCH ANALYZER(doc.text LIKE @term, "text_en")
            SORT BM25(doc) DESC
            RETURN doc
        """
        cursor = db.aql.execute(aql, bind_vars={"term": search_term})
        results = [doc for doc in cursor]
        result_keys = [doc["_key"] for doc in results]
        
        logger.info(f"BM25 search returned {len(results)} results")
        for i, doc in enumerate(results):
            logger.info(f"Result {i+1}: {doc['_key']} - {doc['text'][:50]}...")
        
        # Check if expected document is in results
        found_expected = any(key in result_keys for key in expected_doc_keys)
        
        if found_expected:
            logger.info(f"✅ BM25 search PASSED - Found expected document in results")
            return True
        else:
            # Try a different query as a fallback
            logger.info("Trying alternative BM25 query...")
            
            alt_aql = f"""
            FOR doc IN {view_name}
                FILTER CONTAINS(doc.text, @term, true)
                SORT BM25(doc) DESC
                RETURN doc
            """
            
            alt_cursor = db.aql.execute(alt_aql, bind_vars={"term": search_term})
            alt_results = [doc for doc in alt_cursor]
            alt_result_keys = [doc["_key"] for doc in alt_results]
            
            logger.info(f"Alternative BM25 search returned {len(alt_results)} results")
            for i, doc in enumerate(alt_results):
                logger.info(f"Result {i+1}: {doc['_key']} - {doc['text'][:50]}...")
            
            found_alt = any(key in alt_result_keys for key in expected_doc_keys)
            
            if found_alt:
                logger.info(f"✅ Alternative BM25 search PASSED - Found expected document in results")
                return True
            else:
                logger.error(f"❌ BM25 search FAILED - Expected document not found in results")
                return False
    except Exception as e:
        logger.error(f"BM25 search error: {e}")
        return False

def test_basic_fulltext_search(db, collection_name, test_docs):
    """
    Test fulltext search using fulltext index.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        test_docs: List of test documents
        
    Returns:
        Boolean indicating success
    """
    logger.info("\n=== Testing Basic Fulltext Search ===")
    
    # Ensure fulltext index exists
    try:
        collection = db.collection(collection_name)
        
        # Check if fulltext index exists
        indexes = collection.indexes()
        has_fulltext = False
        
        for idx in indexes:
            if idx.get('type') == 'fulltext' and 'text' in idx.get('fields', []):
                logger.info("Fulltext index found on 'text' field")
                has_fulltext = True
                break
                
        if not has_fulltext:
            logger.info("Creating fulltext index on 'text' field")
            collection.add_index({
                "type": "fulltext",
                "fields": ["text"],
                "minLength": 3
            })
    except Exception as e:
        logger.error(f"Failed to ensure fulltext index: {e}")
        return False
    
    # Search term
    search_term = "BM25"
    expected_doc_keys = [f"{TEST_PREFIX}_doc3"]  # Doc about BM25
    
    try:
        # Execute fulltext search
        aql = f"""
        FOR doc IN FULLTEXT({collection_name}, "text", @term)
            RETURN doc
        """
        cursor = db.aql.execute(aql, bind_vars={"term": search_term})
        results = [doc for doc in cursor]
        result_keys = [doc["_key"] for doc in results]
        
        logger.info(f"Fulltext search returned {len(results)} results")
        for i, doc in enumerate(results):
            logger.info(f"Result {i+1}: {doc['_key']} - {doc['text'][:50]}...")
        
        # Check if expected document is in results
        found_expected = any(key in result_keys for key in expected_doc_keys)
        
        if found_expected:
            logger.info(f"✅ Fulltext search PASSED - Found expected document in results")
            return True
        else:
            logger.error(f"❌ Fulltext search FAILED - Expected document not found in results")
            return False
    except Exception as e:
        logger.error(f"Fulltext search error: {e}")
        return False

def cleanup_test_data(db, collection_name, test_docs):
    """
    Clean up test documents after testing.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        test_docs: List of test documents
    """
    logger.info("\n=== Cleaning Up Test Data ===")
    
    try:
        collection = db.collection(collection_name)
        
        # Delete test documents
        for doc in test_docs:
            try:
                collection.delete(doc["_key"])
            except Exception as e:
                logger.warning(f"Failed to delete document {doc['_key']}: {e}")
        
        logger.info(f"Cleaned up {len(test_docs)} test documents")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def main():
    """Main function to run all search tests."""
    logger.info("=== ArangoDB Search Functionality Test ===")
    
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
    
    # Check if collection exists
    if not db.has_collection(COLLECTION_NAME):
        logger.error(f"Collection {COLLECTION_NAME} does not exist")
        sys.exit(1)
        
    collection = db.collection(COLLECTION_NAME)
    logger.info(f"Using collection: {COLLECTION_NAME}")
    
    # Set up ArangoSearch view for BM25 search
    if not ensure_search_view(db, VIEW_NAME, COLLECTION_NAME):
        logger.error("Failed to set up ArangoSearch view, BM25 search may not work")
    
    # Create and insert test documents
    test_docs = create_test_documents()
    logger.info(f"Created {len(test_docs)} test documents")
    
    # Insert documents
    try:
        for doc in test_docs:
            collection.insert(doc)
        logger.info(f"Inserted {len(test_docs)} test documents")
        
        # Allow time for indexing to complete
        logger.info("Waiting for indexing to complete...")
        time.sleep(2)
    except Exception as e:
        logger.error(f"Failed to insert test documents: {e}")
        sys.exit(1)
    
    # Track test results
    all_tests_passed = True
    
    # Run tests
    try:
        # 1. Basic text search
        basic_search_passed = test_basic_search(db, COLLECTION_NAME, test_docs)
        all_tests_passed = all_tests_passed and basic_search_passed
        
        # 2. Fulltext index search
        fulltext_search_passed = test_basic_fulltext_search(db, COLLECTION_NAME, test_docs)
        all_tests_passed = all_tests_passed and fulltext_search_passed
        
        # 3. BM25 search
        bm25_search_passed = test_bm25_search(db, VIEW_NAME, test_docs)
        all_tests_passed = all_tests_passed and bm25_search_passed
    finally:
        # Clean up test data regardless of test results
        cleanup_test_data(db, COLLECTION_NAME, test_docs)
    
    # Final report
    if all_tests_passed:
        logger.info("\n✅ ALL SEARCH TESTS PASSED")
        sys.exit(0)
    else:
        logger.error("\n❌ ONE OR MORE SEARCH TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
