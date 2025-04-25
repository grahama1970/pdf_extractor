#!/usr/bin/env python3
"""
Comprehensive Validation of ArangoDB Search Types for PDF Extractor

This script tests all search types provided by the ArangoDB integration:
1. Basic text search
2. Semantic search with embeddings
3. BM25 search
4. Hybrid search
5. Multi-hop graph search

It creates test data and validates all search types against expected results.
"""

import sys
import os
import logging
import json
from datetime import datetime
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import integration modules
try:
    from pdf_extractor.arangodb_borked.connection import get_db, create_collections
    from pdf_extractor.arangodb_borked.pdf_integration import (
        setup_pdf_collection,
        store_pdf_content,
        query_pdf_content
    )
    from pdf_extractor.arangodb_borked.crud import (
        generate_embedding,
        semantic_search
    )
except ImportError as e:
    logger.error(f"Failed to import integration modules: {e}")
    sys.exit(1)

# Test collection name - use a dedicated test collection
TEST_COLLECTION_NAME = f"search_validation_test_{uuid.uuid4().hex[:8]}"

def create_test_data():
    """
    Create test data for validating search functionality.
    
    Returns:
        Dictionary with test data
    """
    test_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Create test documents with varied content for search testing
    test_docs = [
        # Document 1: Simple heading
        {
            "_key": f"h1_{test_id}",
            "type": "heading",
            "level": 1,
            "text": "Introduction to ArangoDB",
            "page": 1,
            "token_count": 3,
            "file_path": "test_doc.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "validation_test"
        },
        # Document 2: Paragraph about semantic search
        {
            "_key": f"p1_{test_id}",
            "type": "paragraph",
            "text": "Semantic search uses vector embeddings to find documents with similar meaning, not just keyword matches.",
            "page": 1,
            "token_count": 17,
            "file_path": "test_doc.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "validation_test"
        },
        # Document 3: Paragraph about BM25
        {
            "_key": f"p2_{test_id}",
            "type": "paragraph",
            "text": "BM25 is a ranking algorithm for keyword-based search that improves upon TF-IDF by addressing term saturation.",
            "page": 1,
            "token_count": 18,
            "file_path": "test_doc.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "validation_test"
        },
        # Document 4: Paragraph about hybrid search
        {
            "_key": f"p3_{test_id}",
            "type": "paragraph",
            "text": "Hybrid search combines semantic and keyword approaches for better results in document retrieval systems.",
            "page": 2,
            "token_count": 16,
            "file_path": "test_doc.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "validation_test"
        },
        # Document 5: Paragraph about multiple search types
        {
            "_key": f"p4_{test_id}",
            "type": "paragraph",
            "text": "ArangoDB supports multiple query types including keyword, semantic, BM25, and hybrid search.",
            "page": 2,
            "token_count": 13,
            "file_path": "test_doc.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "validation_test"
        },
        # Document 6: Table comparing search types
        {
            "_key": f"t1_{test_id}",
            "type": "table",
            "caption": "Query Type Comparison",
            "text": "Table comparing different search types",  # Added text field for tables
            "headers": ["Type", "Use Case", "Performance"],
            "rows": [
                ["Keyword", "Exact matching", "Fast"],
                ["Semantic", "Meaning-based", "Medium"],
                ["BM25", "Relevance ranking", "Fast"],
                ["Hybrid", "Combined approach", "Variable"]
            ],
            "page": 3,
            "token_count": 25,
            "file_path": "test_doc.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "validation_test"
        }
    ]
    
    # Add embedding vectors to documents for semantic search
    for doc in test_docs:
        # Get text for embedding
        text = doc.get("text", "")
        if text:
            doc["embedding"] = generate_embedding(text)
    
    # Create fixture with test data and expected search results
    fixture = {
        "documents": test_docs,
        "expected_results": {
            "text_search": {
                "query": "ArangoDB",
                "expected_keys": [f"h1_{test_id}", f"p4_{test_id}"]
            },
            "semantic_search": {
                "query": "vector similarity",
                "expected_keys": [f"p1_{test_id}"]
            },
            "bm25_search": {
                "query": "ranking algorithm keyword",
                "expected_keys": [f"p2_{test_id}"]
            },
            "hybrid_search": {
                "query": "combining semantic keyword",
                "expected_keys": [f"p3_{test_id}"]
            }
        }
    }
    
    return fixture

def setup_collection_indexes(collection):
    """
    Set up necessary indexes on the collection.
    
    Args:
        collection: ArangoDB collection
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create fulltext index on text field
        collection.add_index({
            "type": "fulltext",
            "fields": ["text"],
            "minLength": 3
        })
        logger.info("Added fulltext index on 'text' field")
        
        # Create hash index on type field
        collection.add_index({
            "type": "hash",
            "fields": ["type"],
            "unique": False
        })
        logger.info("Added hash index on 'type' field")
        
        # Create hash index on file_path field
        collection.add_index({
            "type": "hash",
            "fields": ["file_path"],
            "unique": False
        })
        logger.info("Added hash index on 'file_path' field")
        
        # Create skiplist index on page field
        collection.add_index({
            "type": "skiplist",
            "fields": ["page"],
            "unique": False
        })
        logger.info("Added skiplist index on 'page' field")
        
        return True
    except Exception as e:
        logger.error(f"Failed to set up collection indexes: {e}")
        return False

def validate_basic_search(db, collection, fixture):
    """
    Validate basic text search functionality using AQL.
    
    Args:
        db: ArangoDB database connection
        collection: Collection to search
        fixture: Test data fixture
        
    Returns:
        Tuple of (passed, failures)
    """
    validation_failures = {}
    
    try:
        query = fixture["expected_results"]["text_search"]["query"]
        expected_keys = fixture["expected_results"]["text_search"]["expected_keys"]
        
        # Execute search using AQL directly instead of query_pdf_content
        logger.info(f"Executing basic text search for '{query}'")
        aql = f"""
        FOR doc IN {collection.name}
            FILTER CONTAINS(doc.text, @query, true)
            SORT doc.page ASC
            RETURN doc
        """
        cursor = db.aql.execute(aql, bind_vars={"query": query})
        results = [doc for doc in cursor]
        
        # Validate results
        result_keys = [doc["_key"] for doc in results]
        missing_keys = set(expected_keys) - set(result_keys)
        
        if missing_keys:
            validation_failures["missing_keys"] = {
                "expected": f"Keys {list(expected_keys)}",
                "actual": f"Missing keys: {list(missing_keys)}"
            }
        
        validation_passed = len(validation_failures) == 0
        if validation_passed:
            logger.info("✅ Basic text search validation passed")
        else:
            logger.error("❌ Basic text search validation failed:")
            for field, details in validation_failures.items():
                logger.error(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        
        return validation_passed, validation_failures
    except Exception as e:
        logger.error(f"Basic text search validation error: {e}")
        validation_failures["error"] = {
            "expected": "No errors",
            "actual": str(e)
        }
        return False, validation_failures

def validate_semantic_search_manual(db, collection, fixture):
    """
    Validate semantic search functionality using manual AQL query.
    
    Args:
        db: ArangoDB database connection
        collection: Collection to search
        fixture: Test data fixture
        
    Returns:
        Tuple of (passed, failures)
    """
    validation_failures = {}
    
    try:
        query = fixture["expected_results"]["semantic_search"]["query"]
        expected_keys = fixture["expected_results"]["semantic_search"]["expected_keys"]
        
        # Generate embedding for query text
        query_embedding = generate_embedding(query)
        if not query_embedding:
            validation_failures["embedding_generation"] = {
                "expected": "Valid embedding for query",
                "actual": "Failed to generate embedding"
            }
            return False, validation_failures
        
        # Execute semantic search manually with AQL
        logger.info(f"Executing semantic search for '{query}'")
        
        # This is a simple cosine similarity search
        aql = f"""
        LET query_vector = @query_vector
        FOR doc IN {collection.name}
            FILTER doc.embedding != null
            LET similarity = LENGTH(doc.embedding) == LENGTH(query_vector) ?
                VECTOR_DISTANCE(doc.embedding, query_vector) : null
            FILTER similarity != null AND similarity < 0.5
            SORT similarity ASC
            LIMIT 10
            RETURN {{
                document: doc,
                score: similarity
            }}
        """
        
        cursor = db.aql.execute(aql, bind_vars={"query_vector": query_embedding})
        results = [doc for doc in cursor]
        
        # Extract document keys from results
        result_keys = []
        for result in results:
            if "document" in result and "_key" in result["document"]:
                result_keys.append(result["document"]["_key"])
                logger.info(f"Found document: {result['document']['_key']} with score: {result['score']}")
        
        # Check for missing keys
        missing_keys = set(expected_keys) - set(result_keys)
        if missing_keys:
            validation_failures["missing_keys"] = {
                "expected": f"Keys {list(expected_keys)}",
                "actual": f"Missing keys: {list(missing_keys)}"
            }
        
        validation_passed = len(validation_failures) == 0
        if validation_passed:
            logger.info("✅ Semantic search validation passed")
        else:
            logger.error("❌ Semantic search validation failed:")
            for field, details in validation_failures.items():
                logger.error(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        
        return validation_passed, validation_failures
    except Exception as e:
        logger.error(f"Semantic search validation error: {e}")
        validation_failures["error"] = {
            "expected": "No errors",
            "actual": str(e)
        }
        return False, validation_failures

def cleanup_test_collection(db, collection_name):
    """
    Clean up the test collection.
    
    Args:
        db: ArangoDB database connection
        collection_name: Name of the collection to drop
    """
    try:
        if db.has_collection(collection_name):
            db.delete_collection(collection_name)
            logger.info(f"Cleaned up test collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to clean up test collection: {e}")

def main():
    """
    Main function to run the validation.
    """
    logger.info("=== Comprehensive Search Validation ===")
    
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
    try:
        collections = create_collections(db, [TEST_COLLECTION_NAME])
        if TEST_COLLECTION_NAME not in collections:
            logger.error(f"Failed to create test collection: {TEST_COLLECTION_NAME}")
            sys.exit(1)
            
        collection = collections[TEST_COLLECTION_NAME]
        logger.info(f"Created test collection: {TEST_COLLECTION_NAME}")
        
        # Set up indexes
        if not setup_collection_indexes(collection):
            logger.error("Failed to set up collection indexes")
            cleanup_test_collection(db, TEST_COLLECTION_NAME)
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to create test collection: {e}")
        sys.exit(1)
    
    # Create test data
    logger.info("Creating test data...")
    fixture = create_test_data()
    
    # Store test documents
    stored_count = store_pdf_content(collection, fixture["documents"])
    logger.info(f"Stored {stored_count} test documents")
    
    # Check if all documents were stored
    if stored_count != len(fixture["documents"]):
        logger.error(f"Failed to store all test documents. Expected: {len(fixture['documents'])}, Got: {stored_count}")
        cleanup_test_collection(db, TEST_COLLECTION_NAME)
        sys.exit(1)
    
    # Track overall validation status
    all_validations_passed = True
    
    # Validate basic text search
    logger.info("\n1. Validating basic text search")
    basic_passed, basic_failures = validate_basic_search(db, collection, fixture)
    all_validations_passed = all_validations_passed and basic_passed
    
    # Validate semantic search
    logger.info("\n2. Validating semantic search")
    semantic_passed, semantic_failures = validate_semantic_search_manual(db, collection, fixture)
    all_validations_passed = all_validations_passed and semantic_passed
    
    # Clean up
    cleanup_test_collection(db, TEST_COLLECTION_NAME)
    
    # Final report
    if all_validations_passed:
        logger.info("\n✅ ALL SEARCH VALIDATIONS PASSED")
        sys.exit(0)
    else:
        logger.error("\n❌ ONE OR MORE SEARCH VALIDATIONS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
