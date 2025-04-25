#!/usr/bin/env python3
"""
Setup script for ArangoDB search views and indexes

This script creates the necessary ArangoDB views and indexes needed
for all search types (BM25, Semantic, Hybrid) to work correctly.
Based on the working examples provided in examples/arangodb/search_api.
"""

import logging
import sys
import os
from arango import ArangoClient
from arango.exceptions import ServerConnectionError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from our project
try:
    from pdf_extractor.arangodb_borked.connection import get_db
except ImportError as e:
    logger.error(f"Failed to import ArangoDB modules: {e}")
    sys.exit(1)

# Constants
COLLECTION_NAME = 'pdf_documents'
VIEW_NAME = f"{COLLECTION_NAME}_view"

def ensure_search_view(db):
    """
    Ensure the ArangoSearch view exists with proper configuration.
    
    Args:
        db: ArangoDB database connection
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        logger.info(f"Setting up ArangoSearch view: {VIEW_NAME}")
        
        # List existing views
        existing_views = db.views()
        existing_view_names = [v['name'] for v in existing_views]
        
        if VIEW_NAME in existing_view_names:
            logger.info(f"View {VIEW_NAME} already exists, updating properties")
            # Delete existing view to recreate it (simplest approach)
            db.delete_view(VIEW_NAME)
            logger.info(f"Deleted existing view: {VIEW_NAME}")
        
        # Create new ArangoSearch view
        logger.info(f"Creating ArangoSearch view: {VIEW_NAME}")
        properties = {
            "links": {
                COLLECTION_NAME: {
                    "analyzers": [
                        "identity",
                        "text_en"
                    ],
                    "fields": {
                        "text": {
                            "analyzers": ["text_en"]
                        },
                        "type": {},
                        "file_path": {},
                        "page": {}
                    },
                    "includeAllFields": False,
                    "storeValues": "none",
                    "trackListPositions": False
                }
            }
        }
        
        # Create view
        db.create_view(VIEW_NAME, "arangosearch", properties)
        logger.info(f"Successfully created view: {VIEW_NAME}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to setup ArangoSearch view: {e}")
        return False

def ensure_vector_index(db, collection_name, field_name="embedding"):
    """
    Ensure a vector index exists for semantic search.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        field_name: Field containing vector embeddings
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        logger.info(f"Setting up vector index on {collection_name}.{field_name}")
        
        # Get collection
        collection = db.collection(collection_name)
        
        # Check existing indexes
        existing_indexes = collection.indexes()
        has_vector_index = False
        
        for idx in existing_indexes:
            if idx.get('type') == 'inverted' and field_name in idx.get('fields', []):
                has_vector_index = True
                logger.info(f"Vector index already exists for {field_name}")
                break
        
        # Create index if it doesn't exist
        if not has_vector_index:
            try:
                collection.add_index({
                    "type": "inverted",
                    "fields": [field_name],
                    "analyzerDefinitions": [{
                        "name": "vector",
                        "type": "vector",
                        "properties": {
                            "dimensions": 1536,  # OpenAI embedding dimensions
                            "algorithm": "hnsw",
                            "distance": "euclidean"
                        }
                    }]
                })
                logger.info(f"Created vector index for {field_name}")
            except Exception as e:
                logger.warning(f"Failed to create vector index with custom analyzer: {e}")
                logger.info("Falling back to simpler inverted index")
                # Fallback to simpler inverted index
                collection.add_index({
                    "type": "inverted",
                    "fields": [field_name]
                })
                logger.info(f"Created basic inverted index for {field_name}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to create vector index: {e}")
        return False

def ensure_fulltext_index(db, collection_name, field_name="text"):
    """
    Ensure a fulltext index exists for basic text search.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        field_name: Field to index for fulltext search
        
    Returns:
        Boolean indicating success or failure
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

def setup_collection_indexes(db, collection_name):
    """
    Set up all necessary indexes for the collection.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        logger.info(f"Setting up indexes for collection: {collection_name}")
        
        # Get collection
        collection = db.collection(collection_name)
        
        # Check existing indexes
        existing_indexes = collection.indexes()
        has_type_index = False
        has_file_path_index = False
        has_page_index = False
        
        for idx in existing_indexes:
            idx_type = idx.get('type')
            idx_fields = idx.get('fields', [])
            
            if idx_type == 'hash' and 'type' in idx_fields:
                has_type_index = True
                logger.info("Hash index on 'type' field already exists")
                
            if idx_type == 'hash' and 'file_path' in idx_fields:
                has_file_path_index = True
                logger.info("Hash index on 'file_path' field already exists")
                
            if idx_type == 'skiplist' and 'page' in idx_fields:
                has_page_index = True
                logger.info("Skiplist index on 'page' field already exists")
        
        # Create indexes if needed
        if not has_type_index:
            collection.add_index({
                "type": "hash",
                "fields": ["type"],
                "unique": False
            })
            logger.info("Created hash index on 'type' field")
            
        if not has_file_path_index:
            collection.add_index({
                "type": "hash",
                "fields": ["file_path"],
                "unique": False
            })
            logger.info("Created hash index on 'file_path' field")
            
        if not has_page_index:
            collection.add_index({
                "type": "skiplist",
                "fields": ["page"],
                "unique": False
            })
            logger.info("Created skiplist index on 'page' field")
            
        return True
    except Exception as e:
        logger.error(f"Failed to create basic indexes: {e}")
        return False

def main():
    """Main function to set up views and indexes."""
    logger.info("=== Setting up ArangoDB views and indexes ===")
    
    # Connect to ArangoDB
    try:
        db = get_db()
        if not db:
            logger.error("Failed to connect to ArangoDB")
            sys.exit(1)
            
        logger.info(f"Connected to ArangoDB database: {db.name}")
    except ServerConnectionError as e:
        logger.error(f"Failed to connect to ArangoDB: {e}")
        sys.exit(1)
    
    # Check if collection exists
    if not db.has_collection(COLLECTION_NAME):
        logger.error(f"Collection {COLLECTION_NAME} does not exist")
        sys.exit(1)
    
    # Set up basic indexes
    if not setup_collection_indexes(db, COLLECTION_NAME):
        logger.error("Failed to set up basic indexes")
        sys.exit(1)
    
    # Set up fulltext index
    if not ensure_fulltext_index(db, COLLECTION_NAME):
        logger.error("Failed to set up fulltext index")
        sys.exit(1)
    
    # Set up vector index
    if not ensure_vector_index(db, COLLECTION_NAME):
        logger.error("Failed to set up vector index")
        sys.exit(1)
    
    # Set up search view
    if not ensure_search_view(db):
        logger.error("Failed to set up search view")
        sys.exit(1)
    
    logger.info("All ArangoDB views and indexes set up successfully")

if __name__ == "__main__":
    main()
