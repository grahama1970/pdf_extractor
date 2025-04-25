# src/pdf_extractor/arangodb/arango_setup.py
import sys
import json
from typing import Dict, Any, Optional, List
import os
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
from pdf_extractor.arangodb.config import (
    ARANGO_HOST,
    ARANGO_USER,
    ARANGO_PASSWORD,
    ARANGO_DB_NAME,
    COLLECTION_NAME,
    EDGE_COLLECTION_NAME,
    GRAPH_NAME,
)

def connect_arango() -> ArangoClient:
    """Establishes a connection to the ArangoDB server."""
    logger.info(f"Connecting to ArangoDB at {ARANGO_HOST}...")
    try:
        client = ArangoClient(hosts=ARANGO_HOST)
        # Verify connection by trying to access _system db
        sys_db = client.db("_system", username=ARANGO_USER, password=ARANGO_PASSWORD)
        _ = sys_db.collections()  # Simple operation to check connectivity
        logger.info("Successfully connected to ArangoDB instance.")
        return client
    except (ArangoClientError, ArangoServerError) as e:
        logger.error(f"Failed to connect to ArangoDB: {e}")
        sys.exit(1)

def ensure_database(client: ArangoClient) -> StandardDatabase:
    """Ensures the specified database exists."""
    try:
        sys_db = client.db("_system", username=ARANGO_USER, password=ARANGO_PASSWORD)
        current_databases = sys_db.databases()
        
        if ARANGO_DB_NAME not in current_databases:
            logger.info(f"Database '{ARANGO_DB_NAME}' not found. Creating...")
            sys_db.create_database(ARANGO_DB_NAME)
            logger.info(f"Database '{ARANGO_DB_NAME}' created successfully.")
        else:
            logger.info(f"Database '{ARANGO_DB_NAME}' already exists.")
        
        # Return the handle to the specific database
        return client.db(ARANGO_DB_NAME, username=ARANGO_USER, password=ARANGO_PASSWORD)
    except (DatabaseCreateError, ArangoServerError, ArangoClientError) as e:
        logger.error(f"Failed to ensure database: {e}")
        sys.exit(1)

def ensure_collection(db: StandardDatabase, collection_name: str) -> None:
    """Ensures the specified DOCUMENT collection exists in ArangoDB."""
    try:
        if not db.has_collection(collection_name):
            logger.info(f"Collection '{collection_name}' not found. Creating...")
            db.create_collection(collection_name)
            logger.info(f"Collection '{collection_name}' created successfully.")
        else:
            logger.info(f"Collection '{collection_name}' already exists.")
    except (CollectionCreateError, ArangoServerError) as e:
        logger.error(f"Failed to ensure collection '{collection_name}': {e}")
        sys.exit(1)

def ensure_edge_collection(db: StandardDatabase) -> None:
    """Ensures the specified EDGE collection exists."""
    try:
        if not db.has_collection(EDGE_COLLECTION_NAME):
            logger.info(f"Edge collection '{EDGE_COLLECTION_NAME}' not found. Creating...")
            db.create_collection(EDGE_COLLECTION_NAME, edge=True)
            logger.info(f"Edge collection '{EDGE_COLLECTION_NAME}' created.")
        else:
            logger.info(f"Edge collection '{EDGE_COLLECTION_NAME}' already exists.")
    except (CollectionCreateError, ArangoServerError) as e:
        logger.error(f"Failed to ensure edge collection: {e}")
        sys.exit(1)

def ensure_graph(db: StandardDatabase) -> None:
    """Ensures the graph defining relationships exists."""
    try:
        # Check if vertex and edge collections exist first
        if not db.has_collection(COLLECTION_NAME) or not db.has_collection(EDGE_COLLECTION_NAME):
            logger.error(f"Cannot ensure graph: Required collections not found.")
            sys.exit(1)
            
        if not db.has_graph(GRAPH_NAME):
            logger.info(f"Graph '{GRAPH_NAME}' not found. Creating...")
            # Define the edge relationship within the graph
            edge_definition = {
                "edge_collection": EDGE_COLLECTION_NAME,
                "from_vertex_collections": [COLLECTION_NAME],
                "to_vertex_collections": [COLLECTION_NAME],
            }
            db.create_graph(GRAPH_NAME, edge_definitions=[edge_definition])
            logger.info(f"Graph '{GRAPH_NAME}' created.")
        else:
            logger.info(f"Graph '{GRAPH_NAME}' already exists.")
    except (GraphCreateError, ArangoServerError) as e:
        logger.error(f"Failed to ensure graph: {e}")
        sys.exit(1)

def validate_setup(db: StandardDatabase, fixture_path: str) -> bool:
    """Validate database setup against fixture."""
    try:
        with open(fixture_path, "r") as f:
            expected = json.load(f)
        
        # Check if collections exist
        collections = [c["name"] for c in db.collections() if not c["name"].startswith("_")]
        for expected_collection in expected["collections"]:
            if expected_collection not in collections:
                logger.error(f"Missing collection: {expected_collection}")
                return False
        
        # Check if graph exists
        if not db.has_graph(expected["graph_name"]):
            logger.error(f"Missing graph: {expected['graph_name']}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Run setup process
    client = connect_arango()
    db = ensure_database(client)
    ensure_collection(db, COLLECTION_NAME)
    ensure_edge_collection(db)
    ensure_graph(db)
    
    # Validate setup
    validation_passed = validate_setup(db, "src/test_fixtures/setup_expected.json")
    
    if validation_passed:
        print("✅ Graph setup validation passed")
        sys.exit(0)
    else:
        print("❌ Graph setup validation failed")
        sys.exit(1)
