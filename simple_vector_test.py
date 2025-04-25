#!/usr/bin/env python3
"""
Minimal test for vector search in ArangoDB
"""

import sys
import os
import random
import math
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TEST_COLLECTION_NAME = f"simple_vector_test_{int(time.time())}"
EMBEDDING_DIMENSION = 1536  # OpenAI dimension

# Import ArangoDB connection module
try:
    from pdf_extractor.arangodb_borked.connection import get_db
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)

def generate_embedding(text):
    """Generate a test embedding"""
    random.seed(hash(text))
    embedding = [random.uniform(-1, 1) for _ in range(EMBEDDING_DIMENSION)]
    
    # Normalize to unit length
    magnitude = math.sqrt(sum(x**2 for x in embedding))
    normalized = [x / magnitude for x in embedding] if magnitude > 0 else embedding
    
    return normalized

def main():
    """Main function to test minimal vector setup"""
    logger.info("=== Minimal ArangoDB Vector Test ===")
    
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
        if db.has_collection(TEST_COLLECTION_NAME):
            db.delete_collection(TEST_COLLECTION_NAME)
            
        collection = db.create_collection(TEST_COLLECTION_NAME)
        logger.info(f"Created test collection: {TEST_COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Failed to create test collection: {e}")
        sys.exit(1)
    
    try:
        # Insert test documents with embeddings
        logger.info("Inserting test documents with embeddings...")
        
        docs = []
        for i in range(10):
            text = f"Test document {i} for vector search"
            doc = {
                "_key": f"doc_{i}",
                "text": text,
                "embedding": generate_embedding(text)
            }
            collection.insert(doc)
            docs.append(doc)
            
        logger.info(f"Inserted {len(docs)} documents with embeddings")
        
        # Create vector index with minimal parameters
        logger.info("Creating vector index...")
        
        vector_index = {
            "type": "vector",
            "fields": ["embedding"],
            "params": {
                "dimension": EMBEDDING_DIMENSION,
                "metric": "cosine",
                "nLists": 1  # Minimal value for small dataset
            }
        }
        
        index_result = collection.add_index(vector_index)
        logger.info(f"Created vector index: {index_result}")
        
        # Test vector search
        logger.info("Testing vector search...")
        
        query_embedding = generate_embedding("Test document 0 for vector search")
        
        aql = f"""
        FOR doc IN {TEST_COLLECTION_NAME}
            LET distance = VECTOR_DISTANCE(doc.embedding, @query_embedding, "cosine")
            FILTER distance < 0.5
            SORT distance ASC
            LIMIT 5
            RETURN {{
                document: doc._key,
                score: distance
            }}
        """
        
        cursor = db.aql.execute(aql, bind_vars={"query_embedding": query_embedding})
        results = [result for result in cursor]
        
        if results:
            logger.info(f"Vector search successful - found {len(results)} results")
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: {result['document']} (score: {result['score']})")
            logger.info("✅ Vector search is working correctly")
        else:
            logger.warning("Vector search returned no results")
    except Exception as e:
        logger.error(f"Error during test: {e}")
    finally:
        # Clean up
        if db.has_collection(TEST_COLLECTION_NAME):
            db.delete_collection(TEST_COLLECTION_NAME)
            logger.info(f"Cleaned up test collection: {TEST_COLLECTION_NAME}")

if __name__ == "__main__":
    main()
