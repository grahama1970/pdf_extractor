# src/pdf_extractor/arangodb/arango_setup.py
import sys
import json
from typing import Dict, Any, Optional, List, Tuple
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

# Define embedding field constant
EMBEDDING_FIELD = "embedding"

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

def ensure_edge_collection(db: StandardDatabase, collection_name: str = EDGE_COLLECTION_NAME) -> None:
    """Ensures the specified EDGE collection exists."""
    try:
        if not db.has_collection(collection_name):
            logger.info(f"Edge collection '{collection_name}' not found. Creating...")
            db.create_collection(collection_name, edge=True)
            logger.info(f"Edge collection '{collection_name}' created.")
        else:
            logger.info(f"Edge collection '{collection_name}' already exists.")
    except (CollectionCreateError, ArangoServerError) as e:
        logger.error(f"Failed to ensure edge collection: {e}")
        sys.exit(1)

def ensure_graph(db: StandardDatabase, graph_name: str = GRAPH_NAME, 
                 edge_collection: str = EDGE_COLLECTION_NAME, 
                 vertex_collection: str = COLLECTION_NAME) -> None:
    """Ensures the graph defining relationships exists."""
    try:
        # Check if vertex and edge collections exist first
        if not db.has_collection(vertex_collection) or not db.has_collection(edge_collection):
            logger.error(f"Cannot ensure graph: Required collections not found.")
            sys.exit(1)
            
        if not db.has_graph(graph_name):
            logger.info(f"Graph '{graph_name}' not found. Creating...")
            # Define the edge relationship within the graph
            edge_definition = {
                "edge_collection": edge_collection,
                "from_vertex_collections": [vertex_collection],
                "to_vertex_collections": [vertex_collection],
            }
            db.create_graph(graph_name, edge_definitions=[edge_definition])
            logger.info(f"Graph '{graph_name}' created.")
        else:
            logger.info(f"Graph '{graph_name}' already exists.")
    except (GraphCreateError, ArangoServerError) as e:
        logger.error(f"Failed to ensure graph: {e}")
        sys.exit(1)

def validate_setup(db: StandardDatabase, fixture_path: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate database setup against fixture.
    
    Args:
        db: ArangoDB database connection
        fixture_path: Path to the fixture file containing expected results
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    # Track all validation failures
    validation_failures = {}
    
    try:
        # Load fixture data
        with open(fixture_path, "r") as f:
            expected_data = json.load(f)
        
        # Check if collections exist
        collections = [c["name"] for c in db.collections() if not c["name"].startswith("_")]
        
        for expected_collection in expected_data.get("collections", []):
            if expected_collection not in collections:
                validation_failures[f"missing_collection_{expected_collection}"] = {
                    "expected": f"Collection {expected_collection} exists",
                    "actual": f"Collection {expected_collection} not found"
                }
        
        # Check if graph exists
        graph_name = expected_data.get("graph_name", GRAPH_NAME)
        if not db.has_graph(graph_name):
            validation_failures[f"missing_graph_{graph_name}"] = {
                "expected": f"Graph {graph_name} exists",
                "actual": f"Graph {graph_name} not found"
            }
        
        # Check edge collection
        edge_collection = expected_data.get("edge_collection", EDGE_COLLECTION_NAME)
        if not db.has_collection(edge_collection):
            validation_failures[f"missing_edge_collection_{edge_collection}"] = {
                "expected": f"Edge collection {edge_collection} exists",
                "actual": f"Edge collection {edge_collection} not found"
            }
        else:
            # Check if it's an edge collection
            collection_info = db.collection(edge_collection)
            if not collection_info.properties().get("edge", False):
                validation_failures[f"collection_type_{edge_collection}"] = {
                    "expected": "Edge collection",
                    "actual": "Document collection"
                }
        
        return len(validation_failures) == 0, validation_failures
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False, {"validation_error": {"expected": "Successful validation", "actual": str(e)}}

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Path to test fixture
    fixture_path = "src/test_fixtures/setup_expected.json"
    
    try:
        # Create a test fixture if it doesn't exist
        try:
            with open(fixture_path, "r") as f:
                fixture_exists = True
        except FileNotFoundError:
            # Create a minimal fixture file
            with open(fixture_path, "w") as f:
                json.dump({
                    "collections": [COLLECTION_NAME, EDGE_COLLECTION_NAME],
                    "graph_name": GRAPH_NAME,
                    "edge_collection": EDGE_COLLECTION_NAME
                }, f)
        
        # Run setup process
        client = connect_arango()
        db = ensure_database(client)
        ensure_collection(db, COLLECTION_NAME)
        ensure_edge_collection(db)
        ensure_graph(db)
        
        # Validate setup
        validation_passed, validation_failures = validate_setup(db, fixture_path)
        
        # Report validation status
        if validation_passed:
            print("✅ VALIDATION COMPLETE - Graph setup validation passed")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED - Graph setup validation failed")
            print(f"FAILURE DETAILS:")
            for field, details in validation_failures.items():
                print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
            print(f"Total errors: {len(validation_failures)} fields mismatched")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
