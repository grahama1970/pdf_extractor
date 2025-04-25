#!/usr/bin/env python3
"""
Setup script for ArangoDB collections and views needed for validation.
This script sets up the necessary ArangoDB collections and views for testing
the search functions.
"""

import sys
import os
from typing import Dict, Any, List, Optional
from loguru import logger

from arango.client import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import (
    ArangoClientError,
    ArangoServerError,
    DatabaseCreateError,
    CollectionCreateError,
    GraphCreateError,
)

# ArangoDB Connection Settings (from config file)
ARANGO_HOST = os.environ.get("ARANGO_HOST", "http://localhost:8529")
ARANGO_USER = os.environ.get("ARANGO_USER", "root")
ARANGO_PASSWORD = os.environ.get("ARANGO_PASSWORD", "openSesame")  # Password from our test
ARANGO_DB_NAME = os.environ.get("ARANGO_DB_NAME", "pdf_extractor")

# Collection & View Names (from config file)
COLLECTION_NAME = "documents"
EDGE_COLLECTION_NAME = "relationships"
VIEW_NAME = "document_view"
GRAPH_NAME = "knowledge_graph"

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

def connect_to_arangodb() -> ArangoClient:
    """Connect to ArangoDB server."""
    logger.info(f"Connecting to ArangoDB at {ARANGO_HOST}...")
    try:
        client = ArangoClient(hosts=ARANGO_HOST)
        # Verify connection
        sys_db = client.db("_system", username=ARANGO_USER, password=ARANGO_PASSWORD)
        _ = sys_db.collections()
        logger.info("Successfully connected to ArangoDB.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to ArangoDB: {e}")
        sys.exit(1)

def ensure_database(client: ArangoClient) -> StandardDatabase:
    """Ensure the database exists."""
    try:
        sys_db = client.db("_system", username=ARANGO_USER, password=ARANGO_PASSWORD)
        
        if not sys_db.has_database(ARANGO_DB_NAME):
            logger.info(f"Creating database '{ARANGO_DB_NAME}'...")
            sys_db.create_database(ARANGO_DB_NAME)
            logger.info(f"Database '{ARANGO_DB_NAME}' created.")
        else:
            logger.info(f"Database '{ARANGO_DB_NAME}' already exists.")
        
        return client.db(ARANGO_DB_NAME, username=ARANGO_USER, password=ARANGO_PASSWORD)
    except Exception as e:
        logger.error(f"Failed to ensure database: {e}")
        sys.exit(1)

def ensure_collection(db: StandardDatabase, collection_name: str) -> None:
    """Ensure a collection exists."""
    try:
        if not db.has_collection(collection_name):
            logger.info(f"Creating collection '{collection_name}'...")
            db.create_collection(collection_name)
            logger.info(f"Collection '{collection_name}' created.")
        else:
            logger.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logger.error(f"Failed to ensure collection '{collection_name}': {e}")
        sys.exit(1)

def ensure_edge_collection(db: StandardDatabase, collection_name: str) -> None:
    """Ensure an edge collection exists."""
    try:
        if not db.has_collection(collection_name):
            logger.info(f"Creating edge collection '{collection_name}'...")
            db.create_collection(collection_name, edge=True)
            logger.info(f"Edge collection '{collection_name}' created.")
        else:
            logger.info(f"Edge collection '{collection_name}' already exists.")
    except Exception as e:
        logger.error(f"Failed to ensure edge collection '{collection_name}': {e}")
        sys.exit(1)

def ensure_view(db: StandardDatabase, view_name: str) -> None:
    """Ensure a search view exists."""
    try:
        view_properties = {
            "type": "arangosearch",
            "links": {
                COLLECTION_NAME: {
                    "analyzers": ["text_en"],
                    "includeAllFields": True,
                    "storeValues": "none",
                    "trackListPositions": False
                }
            }
        }
        
        # Check if view exists by trying to get it
        existing_views = db.views()
        view_exists = any(v['name'] == view_name for v in existing_views)
        
        if not view_exists:
            logger.info(f"Creating search view '{view_name}'...")
            db.create_view(view_name, view_properties)
            logger.info(f"Search view '{view_name}' created.")
        else:
            logger.info(f"Updating search view '{view_name}'...")
            db.update_view(view_name, view_properties)
            logger.info(f"Search view '{view_name}' updated.")
    except Exception as e:
        logger.error(f"Failed to ensure view '{view_name}': {e}")
        sys.exit(1)

def ensure_graph(db: StandardDatabase, graph_name: str, edge_collection: str, vertex_collection: str) -> None:
    """Ensure a graph exists."""
    try:
        if not db.has_graph(graph_name):
            logger.info(f"Creating graph '{graph_name}'...")
            edge_definition = {
                "edge_collection": edge_collection,
                "from_vertex_collections": [vertex_collection],
                "to_vertex_collections": [vertex_collection]
            }
            db.create_graph(graph_name, [edge_definition])
            logger.info(f"Graph '{graph_name}' created.")
        else:
            logger.info(f"Graph '{graph_name}' already exists.")
    except Exception as e:
        logger.error(f"Failed to ensure graph '{graph_name}': {e}")
        sys.exit(1)

def create_test_documents(db: StandardDatabase, collection_name: str) -> None:
    """Create some test documents for validation."""
    try:
        collection = db.collection(collection_name)
        
        # Check if we already have documents
        if collection.count() > 0:
            logger.info(f"Collection '{collection_name}' already has {collection.count()} documents.")
            return
        
        # Create test documents
        test_docs = [
            {
                "_key": "test_doc_1",
                "problem": "Python error when processing JSON data",
                "solution": "Use try/except blocks to handle JSON parsing exceptions",
                "context": "Error handling in data processing",
                "tags": ["python", "json", "error-handling"]
            },
            {
                "_key": "test_doc_2",
                "problem": "Python script runs out of memory with large datasets",
                "solution": "Use chunking to process large data incrementally",
                "context": "Performance optimization",
                "tags": ["python", "memory", "optimization"]
            },
            {
                "_key": "test_doc_3",
                "problem": "Need to search documents efficiently",
                "solution": "Use ArangoDB's search capabilities with proper indexing",
                "context": "Database search optimization",
                "tags": ["database", "search", "optimization"]
            }
        ]
        
        # Insert the documents
        for doc in test_docs:
            collection.insert(doc)
            logger.info(f"Inserted document '{doc['_key']}'")
        
        logger.info(f"Created {len(test_docs)} test documents in '{collection_name}'")
    except Exception as e:
        logger.error(f"Failed to create test documents: {e}")
        sys.exit(1)

def create_test_relationships(db: StandardDatabase, edge_collection: str, vertex_collection: str) -> None:
    """Create some test relationships for validation."""
    try:
        edge_coll = db.collection(edge_collection)
        vertex_coll = db.collection(vertex_collection)
        
        # Check if we already have edges
        if edge_coll.count() > 0:
            logger.info(f"Edge collection '{edge_collection}' already has {edge_coll.count()} relationships.")
            return
        
        # Verify vertex documents exist
        doc_keys = ["test_doc_1", "test_doc_2", "test_doc_3"]
        existing_keys = []
        
        for key in doc_keys:
            if vertex_coll.has(key):
                existing_keys.append(key)
        
        if len(existing_keys) < 2:
            logger.warning(f"Not enough documents to create relationships. Found only {len(existing_keys)} documents.")
            return
        
        # Create relationships
        relationships = [
            {
                "_from": f"{vertex_collection}/test_doc_1",
                "_to": f"{vertex_collection}/test_doc_2",
                "type": "related_to",
                "weight": 0.8
            },
            {
                "_from": f"{vertex_collection}/test_doc_2",
                "_to": f"{vertex_collection}/test_doc_3",
                "type": "similar_to",
                "weight": 0.6
            }
        ]
        
        # Insert the relationships
        for edge in relationships:
            edge_coll.insert(edge)
            logger.info(f"Created relationship from {edge['_from']} to {edge['_to']}")
        
        logger.info(f"Created {len(relationships)} test relationships in '{edge_collection}'")
    except Exception as e:
        logger.error(f"Failed to create test relationships: {e}")
        sys.exit(1)

def main():
    """Set up ArangoDB for testing the search functions."""
    # Connect to ArangoDB
    client = connect_to_arangodb()
    
    # Ensure database exists
    db = ensure_database(client)
    
    # Ensure collections exist
    ensure_collection(db, COLLECTION_NAME)
    ensure_edge_collection(db, EDGE_COLLECTION_NAME)
    
    # Create test data
    create_test_documents(db, COLLECTION_NAME)
    
    # Create test relationships
    create_test_relationships(db, EDGE_COLLECTION_NAME, COLLECTION_NAME)
    
    # Ensure graph exists
    ensure_graph(db, GRAPH_NAME, EDGE_COLLECTION_NAME, COLLECTION_NAME)
    
    # Ensure view exists
    ensure_view(db, VIEW_NAME)
    
    # Additional collection for message search testing
    message_collection = "messages"
    ensure_collection(db, message_collection)
    
    # Success message
    logger.info("✅ ArangoDB setup completed successfully.")
    logger.info("You can now run the validation script to test the search functions.")

if __name__ == "__main__":
    main()
