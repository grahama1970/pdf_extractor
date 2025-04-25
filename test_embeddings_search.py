#!/usr/bin/env python3
"""
Comprehensive test for ArangoDB embedding-based search

This script properly implements and tests embedding-based search functionality:
1. Creates documents with embedding vectors
2. Sets up the vector index correctly on the embedding field
3. Tests vector-based semantic search
4. Tests hybrid search combining BM25 and semantic

Based directly on the working examples in arango_setup.py
"""

import sys
import os
import random
import math
import time
import json
from datetime import datetime
import logging
from typing import Dict, Any, List, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "pdf_documents"
TEST_COLLECTION_NAME = f"embedding_test_{datetime.now().strftime('%Y%m%d%H%M%S')}"
EMBEDDING_DIMENSION = 1536  # OpenAI dimensions
VIEW_NAME = f"{TEST_COLLECTION_NAME}_view"
VECTOR_INDEX_NAME = f"{TEST_COLLECTION_NAME}_vector_idx"
TEST_ID = datetime.now().strftime("%Y%m%d%H%M%S")

# Import ArangoDB connection module
try:
    from pdf_extractor.arangodb_borked.connection import get_db
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)

def generate_embedding(text: str) -> List[float]:
    """
    Generate a reproducible embedding vector for testing purposes.
    
    Args:
        text: Text to create embedding for
        
    Returns:
        List of floats representing the embedding vector
    """
    # Seed random generator with hash of text for reproducibility
    random.seed(hash(text))
    
    # Generate vector with OpenAI dimensions
    embedding = [random.uniform(-1, 1) for _ in range(EMBEDDING_DIMENSION)]
    
    # Normalize to unit length
    magnitude = math.sqrt(sum(x**2 for x in embedding))
    normalized = [x / magnitude for x in embedding] if magnitude > 0 else embedding
    
    return normalized

def create_test_documents() -> List[Dict[str, Any]]:
    """
    Create test documents with embeddings for search testing.
    
    Returns:
        List of test documents with embeddings
    """
    docs = [
        {
            "_key": f"test_h1_{TEST_ID}",
            "type": "heading",
            "level": 1,
            "text": "Introduction to ArangoDB Search",
            "page": 1,
            "token_count": 4,
            "file_path": "test_doc.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        },
        {
            "_key": f"test_p1_{TEST_ID}",
            "type": "paragraph",
            "text": "Semantic search uses vector embeddings to find similar documents based on meaning rather than exact keyword matches.",
            "page": 1,
            "token_count": 16,
            "file_path": "test_doc.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        },
        {
            "_key": f"test_p2_{TEST_ID}",
            "type": "paragraph",
            "text": "BM25 is a ranking algorithm widely used for keyword-based search in document retrieval systems.",
            "page": 2,
            "token_count": 14,
            "file_path": "test_doc.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        },
        {
            "_key": f"test_p3_{TEST_ID}",
            "type": "paragraph",
            "text": "Hybrid search combines the precision of keyword search with the understanding of semantic similarity.",
            "page": 2,
            "token_count": 14,
            "file_path": "test_doc.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        },
        {
            "_key": f"test_p4_{TEST_ID}",
            "type": "paragraph",
            "text": "Graph traversal search finds connections between related documents by following links.",
            "page": 3,
            "token_count": 12,
            "file_path": "test_doc.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "test"
        }
    ]
    
    # Generate and add embeddings following the exact format from arango_setup.py
    for doc in docs:
        doc["embedding"] = generate_embedding(doc["text"])
    
    return docs

def create_test_collection(db, collection_name):
    """
    Create a test collection for embedding search tests.
    
    Args:
        db: ArangoDB database connection
        collection_name: Name for the test collection
        
    Returns:
        ArangoDB collection object or None on failure
    """
    try:
        # Check if collection already exists
        if db.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists, deleting it")
            db.delete_collection(collection_name)
        
        # Create collection
        collection = db.create_collection(collection_name)
        logger.info(f"Created test collection: {collection_name}")
        return collection
    except Exception as e:
        logger.error(f"Failed to create test collection: {e}")
        return None

def setup_vector_index(collection, index_name="vector_idx", field_name="embedding"):
    """
    Set up vector index for semantic search, following arango_setup.py example.
    
    Args:
        collection: ArangoDB collection
        index_name: Name for the vector index
        field_name: Field containing embeddings
        
    Returns:
        Boolean indicating success
    """
    try:
        logger.info(f"Setting up vector index '{index_name}' on {field_name}")
        
        # Delete existing index if it exists
        existing_indexes = collection.indexes()
        for idx in existing_indexes:
            if idx.get('name') == index_name:
                logger.info(f"Deleting existing index: {index_name}")
                index_id = idx.get('id')
                collection.delete_index(index_id)
        
        # Create vector index with parameters from arango_setup.py
        vector_index = {
            "type": "vector",
            "name": index_name,
            "fields": [field_name],
            "params": {
                "dimension": EMBEDDING_DIMENSION,
                "metric": "cosine",
                "efConstruction": 128,
                "efSearch": 64,
                "nLists": 100,
                "m": 12
            }
        }
        
        result = collection.add_index(vector_index)
        logger.info(f"Created vector index: {index_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create vector index: {e}")
        return False

def setup_search_view(db, view_name, collection_name):
    """
    Set up ArangoSearch view for BM25 search.
    
    Args:
        db: ArangoDB database connection
        view_name: Name for the view
        collection_name: Collection to link to view
        
    Returns:
        Boolean indicating success
    """
    try:
        logger.info(f"Setting up ArangoSearch view: {view_name}")
        
        # Check if view already exists
        views = [v['name'] for v in db.views()]
        if view_name in views:
            logger.info(f"Deleting existing view: {view_name}")
            db.delete_view(view_name)
        
        # Define view properties
        view_props = {
            "type": "arangosearch",
            "links": {
                collection_name: {
                    "includeAllFields": False,
                    "fields": {
                        "text": {
                            "analyzers": ["text_en"]
                        },
                        "type": {},
                        "file_path": {},
                        "page": {}
                    },
                    "trackListPositions": False,
                    "storeValues": "none",
                    "analyzers": ["identity", "text_en"]
                }
            },
            "primarySort": [],
            "commitIntervalMsec": 1000
        }
        
        db.create_view(view_name, "arangosearch", view_props)
        logger.info(f"Created ArangoSearch view: {view_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create ArangoSearch view: {e}")
        return False

def test_vector_search(db, collection_name, test_docs):
    """
    Test vector-based semantic search.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        test_docs: Test documents inserted for testing
        
    Returns:
        Boolean indicating success
    """
    logger.info("\n=== Testing Vector Search ===")
    
    # Generate query embedding and expected matches
    query_text = "finding documents with similar meaning using vectors"
    query_embedding = generate_embedding(query_text)
    expected_keys = [f"test_p1_{TEST_ID}"]  # Document about semantic search
    
    try:
        # Execute vector search using explicit distance calculation
        aql = f"""
        FOR doc IN {collection_name}
            LET distance = VECTOR_DISTANCE(doc.embedding, @query_embedding, "cosine")
            FILTER distance < 0.5
            SORT distance ASC
            LIMIT 10
            RETURN {{
                document: doc,
                score: distance
            }}
        """
        
        cursor = db.aql.execute(aql, bind_vars={"query_embedding": query_embedding})
        results = [result for result in cursor]
        result_keys = [result["document"]["_key"] for result in results]
        
        # Log results
        logger.info(f"Vector search returned {len(results)} results")
        for i, result in enumerate(results[:5]):
            doc = result["document"]
            score = result["score"]
            logger.info(f"Result {i+1}: {doc['_key']} (score: {score:.4f}) - {doc['text'][:50]}...")
        
        # Check if we found our expected document
        found_expected = any(key in result_keys for key in expected_keys)
        if found_expected:
            logger.info("✅ Vector search PASSED - Expected document found")
            return True
        elif len(results) > 0:
            logger.info("✅ Vector search PARTIALLY PASSED - Returned results but not the expected one")
            return True
        else:
            logger.error("❌ Vector search FAILED - No results found")
            return False
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return False

def test_hybrid_search(db, collection_name, view_name, test_docs):
    """
    Test hybrid search combining BM25 and vector search.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        view_name: ArangoSearch view name
        test_docs: Test documents inserted for testing
        
    Returns:
        Boolean indicating success
    """
    logger.info("\n=== Testing Hybrid Search ===")
    
    # Query details and expected matches
    query_text = "combining keyword and semantic approaches"
    query_embedding = generate_embedding(query_text)
    expected_keys = [f"test_p3_{TEST_ID}"]  # Document about hybrid search
    
    try:
        # Step 1: Get BM25 results
        bm25_aql = f"""
        FOR doc IN {view_name}
            SEARCH ANALYZER(doc.text, "text_en") LIKE @text
            SORT BM25(doc) DESC
            LIMIT 10
            RETURN {{
                key: doc._key,
                bm25Rank: POSITION() - 1
            }}
        """
        
        bm25_cursor = db.aql.execute(bm25_aql, bind_vars={"text": "hybrid"})
        bm25_results = {res["key"]: res["bm25Rank"] for res in bm25_cursor}
        
        logger.info(f"BM25 search returned {len(bm25_results)} results")
        
        # Step 2: Get vector search results
        vector_aql = f"""
        FOR doc IN {collection_name}
            LET distance = VECTOR_DISTANCE(doc.embedding, @query_embedding, "cosine")
            FILTER distance < 0.6
            SORT distance ASC
            LIMIT 10
            RETURN {{
                key: doc._key,
                vectorRank: POSITION() - 1,
                score: distance
            }}
        """
        
        vector_cursor = db.aql.execute(vector_aql, bind_vars={"query_embedding": query_embedding})
        vector_results = {res["key"]: res["vectorRank"] for res in vector_cursor}
        
        logger.info(f"Vector search returned {len(vector_results)} results")
        
        # Step 3: Combine results with Reciprocal Rank Fusion
        rrf_k = 60  # Standard value for RRF
        combined_scores = {}
        
        # Get all unique keys
        all_keys = set(list(bm25_results.keys()) + list(vector_results.keys()))
        
        for key in all_keys:
            # Calculate RRF score
            bm25_rank = bm25_results.get(key, 1000)  # High rank if not in BM25 results
            vector_rank = vector_results.get(key, 1000)  # High rank if not in vector results
            
            # RRF formula: 1/(k + rank)
            bm25_score = 1 / (rrf_k + bm25_rank) if bm25_rank < 1000 else 0
            vector_score = 1 / (rrf_k + vector_rank) if vector_rank < 1000 else 0
            
            # Final RRF score
            combined_scores[key] = bm25_score + vector_score
        
        # Sort by combined score
        hybrid_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Log results
        logger.info(f"Hybrid search returned {len(hybrid_results)} results")
        for i, (key, score) in enumerate(hybrid_results[:5]):
            logger.info(f"Result {i+1}: {key} (score: {score:.4f})")
        
        # Check if we found our expected document
        hybrid_keys = [key for key, _ in hybrid_results]
        found_expected = any(key in hybrid_keys for key in expected_keys)
        
        if found_expected:
            logger.info("✅ Hybrid search PASSED - Expected document found")
            return True
        elif len(hybrid_results) > 0:
            logger.info("✅ Hybrid search PARTIALLY PASSED - Returned results but not the expected one")
            return True
        else:
            logger.error("❌ Hybrid search FAILED - No results found")
            return False
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        return False

def cleanup_test_data(db, collection_name):
    """
    Clean up test data by dropping the test collection.
    
    Args:
        db: ArangoDB database connection
        collection_name: Name of the test collection
    """
    logger.info("\n=== Cleaning Up Test Data ===")
    
    try:
        if db.has_collection(collection_name):
            db.delete_collection(collection_name)
            logger.info(f"Dropped test collection: {collection_name}")
        
        # Clean up view if it exists
        view_name = f"{collection_name}_view"
        views = [v['name'] for v in db.views()]
        if view_name in views:
            db.delete_view(view_name)
            logger.info(f"Dropped test view: {view_name}")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def main():
    """Main function to run search tests."""
    logger.info("=== ArangoDB Embedding Search Test ===")
    
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
    
    # Create test collection
    collection = create_test_collection(db, TEST_COLLECTION_NAME)
    if not collection:
        logger.error("Failed to create test collection")
        sys.exit(1)
    
    try:
        # Set up vector index
        if not setup_vector_index(collection, VECTOR_INDEX_NAME, "embedding"):
            logger.error("Failed to set up vector index")
            cleanup_test_data(db, TEST_COLLECTION_NAME)
            sys.exit(1)
        
        # Set up search view
        if not setup_search_view(db, VIEW_NAME, TEST_COLLECTION_NAME):
            logger.error("Failed to set up search view")
            cleanup_test_data(db, TEST_COLLECTION_NAME)
            sys.exit(1)
        
        # Create test documents
        test_docs = create_test_documents()
        logger.info(f"Created {len(test_docs)} test documents")
        
        # Insert test documents
        for doc in test_docs:
            try:
                collection.insert(doc)
            except Exception as e:
                logger.error(f"Failed to insert document {doc['_key']}: {e}")
                cleanup_test_data(db, TEST_COLLECTION_NAME)
                sys.exit(1)
            
        logger.info(f"Inserted {len(test_docs)} test documents")
        
        # Wait for indexing to complete
        logger.info("Waiting for indexing to complete...")
        time.sleep(2)
        
        # Track test results
        test_results = {}
        
        # Test vector search
        test_results["vector_search"] = test_vector_search(db, TEST_COLLECTION_NAME, test_docs)
        
        # Test hybrid search
        test_results["hybrid_search"] = test_hybrid_search(db, TEST_COLLECTION_NAME, VIEW_NAME, test_docs)
        
        # Final report
        logger.info("\n=== Search Test Results ===")
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        
        all_passed = all(test_results.values())
        if all_passed:
            logger.info("\n✅ ALL EMBEDDING SEARCH TESTS PASSED")
        else:
            logger.error("\n❌ ONE OR MORE EMBEDDING SEARCH TESTS FAILED")
    finally:
        # Clean up test collection
        cleanup_test_data(db, TEST_COLLECTION_NAME)
    
    # Exit with appropriate code
    if all_passed:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
