# src/pdf_extractor/arangodb/message_document_links.py
"""
Module for creating and managing relationships between messages and documents.

This module provides functions to link messages to documents in the main collection,
enabling full conversation context with document references.
"""
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger

from arango.database import StandardDatabase
from arango.exceptions import ArangoServerError, DocumentInsertError

from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.message_history_setup import initialize_message_history
from pdf_extractor.arangodb.message_history_config import (
    MESSAGE_COLLECTION_NAME,
    MESSAGE_EDGE_COLLECTION_NAME,
    RELATIONSHIP_TYPE_REFERS_TO
)
from pdf_extractor.arangodb.config import COLLECTION_NAME

def link_message_to_document(
    db: StandardDatabase,
    message_key: str,
    document_key: str,
    rationale: str = "Message refers to this document"
) -> Optional[Dict[str, Any]]:
    """
    Create a relationship between a message and a document.
    
    Args:
        db: ArangoDB database handle
        message_key: Key of the message
        document_key: Key of the document
        rationale: Reason for the relationship
        
    Returns:
        Optional[Dict[str, Any]]: Edge document if successful, None otherwise
    """
    try:
        # Create the edge
        edge_data = {
            "_from": f"{MESSAGE_COLLECTION_NAME}/{message_key}",
            "_to": f"{COLLECTION_NAME}/{document_key}",
            "type": RELATIONSHIP_TYPE_REFERS_TO,
            "rationale": rationale,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Insert the edge
        edge_collection = db.collection(MESSAGE_EDGE_COLLECTION_NAME)
        result = edge_collection.insert(edge_data, return_new=True)
        
        logger.info(f"Linked message {message_key} to document {document_key}")
        return result["new"]
    except (DocumentInsertError, ArangoServerError) as e:
        logger.error(f"Failed to link message to document: {e}")
        return None

def get_documents_for_message(
    db: StandardDatabase,
    message_key: str
) -> List[Dict[str, Any]]:
    """
    Get all documents linked to a message.
    
    Args:
        db: ArangoDB database handle
        message_key: Key of the message
        
    Returns:
        List[Dict[str, Any]]: List of documents linked to the message
    """
    try:
        # Query documents linked to this message
        aql = f"""
        FOR v, e IN 1..1 OUTBOUND @start_vertex @edge_collection
        FILTER e.type == @rel_type
        RETURN v
        """
        
        bind_vars = {
            "start_vertex": f"{MESSAGE_COLLECTION_NAME}/{message_key}",
            "edge_collection": MESSAGE_EDGE_COLLECTION_NAME,
            "rel_type": RELATIONSHIP_TYPE_REFERS_TO
        }
        
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        documents = list(cursor)
        
        logger.info(f"Found {len(documents)} documents for message {message_key}")
        return documents
    except Exception as e:
        logger.error(f"Failed to get documents for message: {e}")
        return []

def get_messages_for_document(
    db: StandardDatabase,
    document_key: str
) -> List[Dict[str, Any]]:
    """
    Get all messages that reference a document.
    
    Args:
        db: ArangoDB database handle
        document_key: Key of the document
        
    Returns:
        List[Dict[str, Any]]: List of messages that reference the document
    """
    try:
        # Query messages that reference this document
        aql = f"""
        FOR v, e IN 1..1 INBOUND @start_vertex @edge_collection
        FILTER e.type == @rel_type
        RETURN v
        """
        
        bind_vars = {
            "start_vertex": f"{COLLECTION_NAME}/{document_key}",
            "edge_collection": MESSAGE_EDGE_COLLECTION_NAME,
            "rel_type": RELATIONSHIP_TYPE_REFERS_TO
        }
        
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        messages = list(cursor)
        
        logger.info(f"Found {len(messages)} messages for document {document_key}")
        return messages
    except Exception as e:
        logger.error(f"Failed to get messages for document: {e}")
        return []

def validate_document_links(results: Dict[str, Any], expected_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate document links results.
    
    Args:
        results: The results to validate
        expected_path: Path to the expected results file
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (is_valid, validation_failures)
    """
    validation_failures = {}
    
    # Check if we have document links
    if not results.get("document_links"):
        validation_failures["document_links"] = {
            "expected": "present",
            "actual": "missing"
        }
    
    # Check if we have the expected number of links
    link_count = len(results.get("document_links", []))
    if link_count < 1:
        validation_failures["link_count"] = {
            "expected": "at least 1",
            "actual": link_count
        }
    
    # Check if we have messages
    if not results.get("messages"):
        validation_failures["messages"] = {
            "expected": "present",
            "actual": "missing"
        }
    
    # Check if we have the expected number of messages
    message_count = len(results.get("messages", []))
    if message_count < 1:
        validation_failures["message_count"] = {
            "expected": "at least 1",
            "actual": message_count
        }
    
    return len(validation_failures) == 0, validation_failures

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
    
    # Initialize message history collections if needed
    initialize_message_history(db)
    
    # Create test message and document
    message_collection = db.collection(MESSAGE_COLLECTION_NAME)
    document_collection = db.collection(COLLECTION_NAME)
    
    message_key = f"test_msg_{uuid.uuid4().hex[:8]}"
    document_key = f"test_doc_{uuid.uuid4().hex[:8]}"
    
    # Insert test message
    message = {
        "_key": message_key,
        "conversation_id": str(uuid.uuid4()),
        "message_type": "USER",
        "content": "This is a test message referencing a document",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    message_collection.insert(message)
    
    # Insert test document
    document = {
        "_key": document_key,
        "content": "This is a test document",
        "tags": ["test"]
    }
    document_collection.insert(document)
    
    # Create the link
    link = link_message_to_document(db, message_key, document_key)
    if not link:
        print("❌ Failed to link message to document")
        message_collection.delete(message_key)
        document_collection.delete(document_key)
        sys.exit(1)
    
    # Get documents for message
    documents = get_documents_for_message(db, message_key)
    
    # Get messages for document
    messages = get_messages_for_document(db, document_key)
    
    # Validate results
    results = {
        "document_links": [link],
        "documents": documents,
        "messages": messages
    }
    
    validation_passed, validation_failures = validate_document_links(
        results, "src/test_fixtures/document_links_expected.json"
    )
    
    # Clean up test data
    message_collection.delete(message_key)
    document_collection.delete(document_key)
    
    if validation_passed:
        print("✅ Document links validation passed")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED - Document links don't match expected values")
        print("FAILURE DETAILS:")
        for field, details in validation_failures.items():
            print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        print(f"Total errors: {len(validation_failures)} fields mismatched")
        sys.exit(1)
