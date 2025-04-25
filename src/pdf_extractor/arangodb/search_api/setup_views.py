#!/usr/bin/env python3
# src/pdf_extractor/arangodb/search_api/setup_views.py

import sys
from typing import Dict, Any, List, Optional
from loguru import logger

from arango.database import StandardDatabase

from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME as DOC_COLLECTION_NAME,
    VIEW_NAME,
    TEXT_ANALYZER,
    TAG_ANALYZER,
    EMBEDDING_DIMENSIONS
)
from pdf_extractor.arangodb.message_history_config import (
    MESSAGE_COLLECTION_NAME
)

def setup_search_views(db: StandardDatabase) -> bool:
    """
    Set up search views for both documents and messages.
    """
    try:
        # Define view properties
        view_properties = {
            "links": {
                # Document collection
                DOC_COLLECTION_NAME: {
                    "fields": {
                        "content": {
                            "analyzers": [TEXT_ANALYZER]
                        },
                        "tags": {
                            "analyzers": [TAG_ANALYZER]
                        }
                    },
                    "includeAllFields": False,
                    "storeValues": "none",
                    "trackListPositions": False
                },
                # Message history collection
                MESSAGE_COLLECTION_NAME: {
                    "fields": {
                        "content": {
                            "analyzers": [TEXT_ANALYZER]
                        },
                        "metadata.tags": {
                            "analyzers": [TAG_ANALYZER]
                        }
                    },
                    "includeAllFields": False,
                    "storeValues": "none",
                    "trackListPositions": False
                }
            }
        }
        
        # Create or update the view
        if db.has_view(VIEW_NAME):
            db.update_view(VIEW_NAME, view_properties)
            logger.info(f"Updated search view: {VIEW_NAME}")
        else:
            db.create_view(VIEW_NAME, "arangosearch", view_properties)
            logger.info(f"Created search view: {VIEW_NAME}")
        
        # Set up vector indices for both collections
        ensure_vector_index(db, DOC_COLLECTION_NAME)
        ensure_vector_index(db, MESSAGE_COLLECTION_NAME)
        
        return True
    except Exception as e:
        logger.error(f"Failed to set up search views: {e}")
        return False


def ensure_vector_index(
    db: StandardDatabase,
    collection_name: str,
    field_name: str = "embedding",
    dimensions: int = EMBEDDING_DIMENSIONS
) -> bool:
    """
    Ensure a vector index exists for a collection.
    """
    try:
        # Get the collection
        collection = db.collection(collection_name)
        
        # Check if vector index already exists
        vector_index = None
        for index in collection.indexes():
            if index.get("type") == "vector" and field_name in index.get("fields", []):
                vector_index = index
                break
        
        if vector_index:
            logger.info(f"Vector index already exists on {collection_name}.{field_name}")
            return True
        
        # Create the vector index
        index_props = {
            "type": "vector",
            "fields": [field_name],
            "name": f"idx_{collection_name}_{field_name}",
            "params": {
                "dimension": dimensions,
                "metric": "cosine",
                "nLists": 2
            }
        }
        
        collection.add_index(index_props)
        logger.info(f"Created vector index on {collection_name}.{field_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to ensure vector index: {e}")
        return False


if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Connect to ArangoDB
    client = connect_arango()
    if not client:
        print("❌ Failed to connect to ArangoDB")
        sys.exit(1)
    
    db = ensure_database(client)
    if not db:
        print("❌ Failed to ensure database exists")
        sys.exit(1)
    
    # Set up search views
    if setup_search_views(db):
        print("✅ Search views setup completed successfully")
    else:
        print("❌ Failed to set up search views")
        sys.exit(1)
