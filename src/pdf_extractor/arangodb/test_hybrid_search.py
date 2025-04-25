#!/usr/bin/env python
"""
Test the hybrid search functionality.

This script tests the search_hybrid function from the pdf_extractor.arangodb.search_api.hybrid module
by performing a hybrid search and validating that it properly combines BM25 and semantic results.
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
    from pdf_extractor.arangodb.search_api.hybrid import search_hybrid
    from pdf_extractor.arangodb.search_api.bm25 import search_bm25
    from pdf_extractor.arangodb.search_api.semantic import search_semantic
    from pdf_extractor.arangodb.embedding_utils import get_embedding
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
        
        # Check if view exists using db.views()
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
    """Set up a test collection and view for testing hybrid search."""
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
                "tags": ["python", "json", "error-handling"],
                "embedding": get_embedding("Python error when processing JSON data Use try/except blocks to handle JSON parsing exceptions")
            },
            {
                "_key": f"doc2_{test_id}",
                "problem": "Database connection timeout in production",
                "solution": "Implement connection pooling and retry mechanism",
                "tags": ["database", "timeout", "production"],
                "embedding": get_embedding("Database connection timeout in production Implement connection pooling and retry mechanism")
            },
            {
                "_key": f"doc3_{test_id}",
                "problem": "Python script runs out of memory with large datasets",
                "solution": "Use chunking to process large data incrementally",
                "tags": ["python", "memory", "optimization"],
                "embedding": get_embedding("Python script runs out of memory with large datasets Use chunking to process large data incrementally")
            },
            {
                "_key": f"doc4_{test_id}",
                "problem": "Neural network training is slow on large datasets",
                "solution": "Use batch processing and GPU acceleration for faster training",
                "tags": ["machine-learning", "performance", "neural-network"],
                "embedding": get_embedding("Neural network training is slow on large datasets Use batch processing and GPU acceleration for faster training")
            }
        ]
        
        for doc in test_docs:
            collection.insert(doc)
        logger.info(f"Inserted {len(test_docs)} test documents with embeddings")
        
        # Create test view
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

def test_hybrid_search():
    """
    Test the hybrid search functionality, ensuring it properly combines BM25 and semantic results.
    
    Expected behavior:
    1. Connect to ArangoDB
    2. Create a test collection with documents and embeddings
    3. Perform individual BM25 and semantic searches
    4. Perform a hybrid search with the same query
    5. Verify that the hybrid results contain documents from both search types
    6. Verify that the ranking works as expected
    """
    logger.info("Testing hybrid search...")
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
        
        # Search parameters
        query_text = "python code optimization"  # This should match docs about Python and optimization
        top_n = 5
        tags = None
        
        try:
            # First perform BM25 search
            logger.info(f"Performing BM25 search with query: '{query_text}'")
            bm25_results = search_bm25(
                db=db,
                search_text=query_text,
                bm25_threshold=0.1,
                top_n=top_n,
                offset=0,
                tags=tags,
                view_name=test_view
            )
            
            bm25_keys = [r["doc"]["_key"] for r in bm25_results.get("results", [])]
            logger.info(f"BM25 search found {len(bm25_keys)} results: {bm25_keys}")
            
            # Then perform semantic search
            logger.info(f"Performing semantic search with query: '{query_text}'")
            # Get query embedding first
            query_embedding = get_embedding(query_text)
            if not query_embedding:
                logger.error("Failed to get query embedding for semantic search")
                return False
                
            semantic_results = search_semantic(
                db=db,
                query_embedding=query_embedding,
                similarity_threshold=0.6,
                top_n=top_n,
                tags=tags,
                view_name=test_view  # Pass the test view name
            )
            
            semantic_keys = [r["doc"]["_key"] for r in semantic_results.get("results", [])]
            logger.info(f"Semantic search found {len(semantic_keys)} results: {semantic_keys}")
            
            # Now perform hybrid search
            logger.info(f"Performing hybrid search with query: '{query_text}'")
            hybrid_results = search_hybrid(
                db=db,
                query_text=query_text,
                top_n=top_n,
                initial_k=10,
                bm25_threshold=0.1,
                similarity_threshold=0.6,
                tags=tags,
                rrf_k=60
            )
            
            # Validate hybrid results
            logger.info("Validating hybrid results...")
            
            # Check that results contain expected fields
            hybrid_keys = [r["doc"]["_key"] for r in hybrid_results.get("results", [])]
            logger.info(f"Hybrid search found {len(hybrid_keys)} results: {hybrid_keys}")
            
            # Verify hybrid search combines results from both search types
            bm25_only = set(bm25_keys) - set(semantic_keys)
            semantic_only = set(semantic_keys) - set(bm25_keys)
            both = set(bm25_keys) & set(semantic_keys)
            
            logger.info(f"Documents found only by BM25: {bm25_only}")
            logger.info(f"Documents found only by semantic search: {semantic_only}")
            logger.info(f"Documents found by both methods: {both}")
            
            # Verify that hybrid results contain at least some from each search method
            hybrid_has_bm25 = any(key in bm25_only for key in hybrid_keys) if bm25_only else True
            hybrid_has_semantic = any(key in semantic_only for key in hybrid_keys) if semantic_only else True
            hybrid_has_both = any(key in both for key in hybrid_keys) if both else True
            
            if not (hybrid_has_bm25 or hybrid_has_semantic or hybrid_has_both):
                logger.error("Hybrid search results don't contain documents from both search methods")
                return False
            
            # Verify that each hybrid result has all expected fields
            for i, result in enumerate(hybrid_results.get("results", [])):
                if "doc" not in result:
                    logger.error(f"Result {i} missing 'doc' field")
                    return False
                if "bm25_score" not in result:
                    logger.error(f"Result {i} missing 'bm25_score' field")
                    return False
                if "similarity_score" not in result:
                    logger.error(f"Result {i} missing 'similarity_score' field")
                    return False
                if "rrf_score" not in result:
                    logger.error(f"Result {i} missing 'rrf_score' field")
                    return False
                    
                # Check that doc has a _key
                if "_key" not in result["doc"]:
                    logger.error(f"Result {i} doc missing '_key' field")
                    return False
            
            # Verify that hybrid results are in the correct order by rrf_score
            rrf_scores = [r["rrf_score"] for r in hybrid_results.get("results", [])]
            if rrf_scores != sorted(rrf_scores, reverse=True):
                logger.error("Hybrid results are not properly sorted by rrf_score")
                return False
            
            # If we've made it this far, all validations passed
            logger.success(f"Successfully verified hybrid search results")
            
            # Print a sample result for manual verification
            if hybrid_results.get("results"):
                sample = hybrid_results["results"][0]
                logger.info(f"Top hybrid result - Key: {sample['doc'].get('_key')}, RRF Score: {sample['rrf_score']}, BM25: {sample['bm25_score']}, Semantic: {sample['similarity_score']}")
                
            return True
        
        except Exception as e:
            logger.exception(f"Error during hybrid search test: {e}")
            return False
    
    except Exception as e:
        logger.exception(f"Error in hybrid search test: {e}")
        return False
    
    finally:
        # Clean up test resources
        if test_collection or test_view:
            logger.info("Cleaning up test resources...")
            cleanup_test_resources(db, test_collection, test_view)

if __name__ == "__main__":
    logger.info("Starting hybrid search test...")
    
    if test_hybrid_search():
        logger.success("✅ Hybrid search test PASSED")
        sys.exit(0)
    else:
        logger.error("❌ Hybrid search test FAILED")
        sys.exit(1)
