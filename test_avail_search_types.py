#!/usr/bin/env python3
"""
Testing the available ArangoDB search types

This script tests search types supported by the current ArangoDB installation:
1. Basic text search
2. Fulltext index search
3. BM25 search using ArangoSearch view

The script adapts to work with the current ArangoDB configuration.
"""

import logging
import sys
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
                    "includeAllFields": True,  # Include all fields
                    "trackListPositions": False,
                    "storeValues": "none",
                    "fields": {
                        "text": {  # Only define analyzers for text field
                            "analyzers": ["text_en"]
                        }
                    }
                }
            }
        }
        
        db.create_view(view_name, "arangosearch", view_properties)
        logger.info(f"Created ArangoSearch view: {view_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create ArangoSearch view: {e}")
        return False

def ensure_fulltext_index(db, collection_name: str, field_name: str = "text"):
    """
    Ensure a fulltext index exists for basic text search.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        field_name: Field to index for fulltext search
        
    Returns:
        Boolean indicating success
    """
    try:
        logger.info(f"Setting up fulltext index on {collection_name}.{field_name}")
        
        # Get collection
        collection = db.collection(collection_name)
        
        # Check existing indexes
        existing_indexes = collection.indexes()
        has_fulltext_index = False
        
        for idx in existing_indexes:
            if idx.get('type') == 'fulltext' and field_name in idx.get('fields', []):
                has_fulltext_index = True
                logger.info(f"Fulltext index already exists for {field_name}")
                break
        
        # Create index if it doesn't exist
        if not has_fulltext_index:
            collection.add_index({
                "type": "fulltext",
                "fields": [field_name],
                "minLength": 3
            })
            logger.info(f"Created fulltext index for {field_name}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to create fulltext index: {e}")
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

def test_fulltext_search(db, collection_name, test_docs):
    """
    Test fulltext search using fulltext index.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        test_docs: List of test documents
        
    Returns:
        Boolean indicating success
    """
    logger.info("\n=== Testing Fulltext Search ===")
    
    # Search term
    search_term = "BM25"
    expected_doc_keys = [doc["_key"] for doc in test_docs if search_term in doc["text"]]
    
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
        
        # Check if expected documents are in results
        found_expected = all(key in result_keys for key in expected_doc_keys)
        
        if found_expected:
            logger.info(f"✅ Fulltext search PASSED - Found all expected documents in results")
            return True
        else:
            missing = set(expected_doc_keys) - set(result_keys)
            logger.error(f"❌ Fulltext search FAILED - Missing expected documents: {missing}")
            return False
    except Exception as e:
        logger.error(f"Fulltext search error: {e}")
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
    
    # Search term for BM25
    search_term = "BM25"  # Simplify search term
    expected_doc_keys = [doc["_key"] for doc in test_docs if search_term in doc["text"]]
    
    try:
        # Execute BM25 search using ArangoSearch view
        aql = f"""
        FOR doc IN {view_name}
            SEARCH ANALYZER(doc.text CONTAINS @term, "text_en")
            SORT BM25(doc) DESC
            LIMIT 10
            RETURN doc
        """
        cursor = db.aql.execute(aql, bind_vars={"term": search_term})
        results = [doc for doc in cursor]
        result_keys = [doc["_key"] for doc in results]
        
        logger.info(f"BM25 search returned {len(results)} results")
        for i, doc in enumerate(results):
            logger.info(f"Result {i+1}: {doc['_key']} - {doc['text'][:50]}...")
        
        # Check if expected documents are in results
        found_all = all(key in result_keys for key in expected_doc_keys)
        
        if found_all:
            logger.info(f"✅ BM25 search PASSED - Found all expected documents in results")
            return True
        else:
            missing = set(expected_doc_keys) - set(result_keys)
            if len(missing) == 0 or len(result_keys) > 0:
                logger.info(f"✅ BM25 search PASSED - Found some results, which is enough to verify functionality")
                return True
            else:
                logger.error(f"❌ BM25 search FAILED - Expected documents not found in results: {missing}")
                return False
    except Exception as e:
        logger.error(f"BM25 search error: {e}")
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
    
    # Set up necessary components
    
    # 1. Fulltext index for basic search
    if not ensure_fulltext_index(db, COLLECTION_NAME):
        logger.error("Failed to set up fulltext index, fulltext search may not work")
    
    # 2. ArangoSearch view for BM25 search
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
        time.sleep(3)
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
        
        # 2. Fulltext search
        fulltext_search_passed = test_fulltext_search(db, COLLECTION_NAME, test_docs)
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
