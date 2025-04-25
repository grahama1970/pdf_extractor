# src/pdf_extractor/arangodb/message_history_setup.py
"""
Setup module for message history collections in ArangoDB.

This module provides functions to create and configure the
message history collection and related graph structure.
"""
import sys
from typing import Dict, Any, List, Optional, Union
from loguru import logger
from arango.database import StandardDatabase
from arango.exceptions import (
    ArangoClientError,
    ArangoServerError,
    CollectionCreateError,
    GraphCreateError,
    IndexCreateError
)

from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.message_history_config import (
    MESSAGE_COLLECTION_NAME,
    MESSAGE_EDGE_COLLECTION_NAME,
    MESSAGE_GRAPH_NAME,
    MESSAGE_INDEXES
)

def ensure_message_collection(db: StandardDatabase) -> bool:
    """
    Ensure the message history collection exists.
    
    Args:
        db: ArangoDB database handle
        
    Returns:
        bool: True if collection exists or was created, False otherwise
    """
    try:
        # Check if collection already exists
        if not db.has_collection(MESSAGE_COLLECTION_NAME):
            logger.info(f"Creating message history collection: {MESSAGE_COLLECTION_NAME}")
            db.create_collection(MESSAGE_COLLECTION_NAME)
            logger.info(f"Message history collection created successfully")
        else:
            logger.info(f"Message history collection already exists: {MESSAGE_COLLECTION_NAME}")
        
        # Create indexes for efficient querying
        collection = db.collection(MESSAGE_COLLECTION_NAME)
        existing_indexes = collection.indexes()
        existing_index_fields = []
        
        # Extract fields from existing indexes
        for idx in existing_indexes:
            if idx["type"] != "primary" and "fields" in idx:
                existing_index_fields.append(tuple(idx["fields"]))
        
        # Create missing indexes
        for index_config in MESSAGE_INDEXES:
            fields = tuple(index_config["fields"])
            if fields not in existing_index_fields:
                logger.info(f"Creating index on fields: {fields}")
                collection.add_persistent_index(
                    fields=index_config["fields"],
                    unique=index_config.get("unique", False)
                )
                logger.info(f"Index created successfully")
        
        return True
    
    except (CollectionCreateError, IndexCreateError, ArangoServerError, ArangoClientError) as e:
        logger.error(f"Failed to ensure message collection: {e}")
        return False

def ensure_message_edge_collection(db: StandardDatabase) -> bool:
    """
    Ensure the message relationship edge collection exists.
    
    Args:
        db: ArangoDB database handle
        
    Returns:
        bool: True if collection exists or was created, False otherwise
    """
    try:
        # Check if edge collection already exists
        if not db.has_collection(MESSAGE_EDGE_COLLECTION_NAME):
            logger.info(f"Creating message edge collection: {MESSAGE_EDGE_COLLECTION_NAME}")
            db.create_collection(MESSAGE_EDGE_COLLECTION_NAME, edge=True)
            logger.info(f"Message edge collection created successfully")
        else:
            logger.info(f"Message edge collection already exists: {MESSAGE_EDGE_COLLECTION_NAME}")
        
        return True
    
    except (CollectionCreateError, ArangoServerError, ArangoClientError) as e:
        logger.error(f"Failed to ensure message edge collection: {e}")
        return False

def ensure_message_graph(db: StandardDatabase) -> bool:
    """
    Ensure the message graph exists.
    
    Args:
        db: ArangoDB database handle
        
    Returns:
        bool: True if graph exists or was created, False otherwise
    """
    try:
        # Check if graph already exists
        if not db.has_graph(MESSAGE_GRAPH_NAME):
            logger.info(f"Creating message graph: {MESSAGE_GRAPH_NAME}")
            
            # Define edge definition for message relationships
            edge_definition = {
                "edge_collection": MESSAGE_EDGE_COLLECTION_NAME,
                "from_vertex_collections": [MESSAGE_COLLECTION_NAME],
                "to_vertex_collections": [MESSAGE_COLLECTION_NAME]
            }
            
            # Create graph with edge definition
            db.create_graph(MESSAGE_GRAPH_NAME, edge_definitions=[edge_definition])
            logger.info(f"Message graph created successfully")
        else:
            logger.info(f"Message graph already exists: {MESSAGE_GRAPH_NAME}")
        
        return True
    
    except (GraphCreateError, ArangoServerError, ArangoClientError) as e:
        logger.error(f"Failed to ensure message graph: {e}")
        return False

def initialize_message_history(db: StandardDatabase) -> bool:
    """
    Initialize all message history components.
    
    Args:
        db: ArangoDB database handle
        
    Returns:
        bool: True if all components were initialized successfully
    """
    message_collection_ok = ensure_message_collection(db)
    edge_collection_ok = ensure_message_edge_collection(db)
    graph_ok = ensure_message_graph(db)
    
    return message_collection_ok and edge_collection_ok and graph_ok

def validate_message_setup(db: StandardDatabase) -> bool:
    """
    Validate that message history setup was successful.
    
    Args:
        db: ArangoDB database handle
        
    Returns:
        bool: True if all components exist and are configured correctly
    """
    try:
        # Check collections and graph exist
        collections_exist = (
            db.has_collection(MESSAGE_COLLECTION_NAME) and
            db.has_collection(MESSAGE_EDGE_COLLECTION_NAME) and
            db.has_graph(MESSAGE_GRAPH_NAME)
        )
        
        if not collections_exist:
            return False
        
        # Verify indexes exist
        collection = db.collection(MESSAGE_COLLECTION_NAME)
        indexes = collection.indexes()
        
        required_index_fields = [tuple(idx["fields"]) for idx in MESSAGE_INDEXES]
        existing_index_fields = []
        
        for idx in indexes:
            if idx["type"] != "primary" and "fields" in idx:
                existing_index_fields.append(tuple(idx["fields"]))
        
        for required_fields in required_index_fields:
            if required_fields not in existing_index_fields:
                logger.error(f"Missing index on fields: {required_fields}")
                return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error validating message history setup: {e}")
        return False

if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Connect to ArangoDB and ensure database exists
    client = connect_arango()
    if not client:
        print("❌ Failed to connect to ArangoDB")
        sys.exit(1)
    
    db = ensure_database(client)
    if not db:
        print("❌ Failed to ensure database exists")
        sys.exit(1)
    
    # Initialize message history components
    if initialize_message_history(db):
        logger.info("Message history initialization completed")
        
        # Validate setup
        if validate_message_setup(db):
            print("✅ Message history setup validation passed")
            sys.exit(0)
        else:
            print("❌ Message history setup validation failed")
            sys.exit(1)
    else:
        print("❌ Message history initialization failed")
        sys.exit(1)
