# src/pdf_extractor/arangodb/conversation_context.py
"""
Module for analyzing conversation context through graph traversal.

This module provides functions to traverse conversation graphs and analyze
the context of conversations, including referenced documents.
"""
import sys
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger

from arango.database import StandardDatabase

from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.message_history_setup import initialize_message_history
from pdf_extractor.arangodb.message_history_config import (
    MESSAGE_COLLECTION_NAME,
    MESSAGE_EDGE_COLLECTION_NAME,
    MESSAGE_GRAPH_NAME,
    RELATIONSHIP_TYPE_NEXT,
    RELATIONSHIP_TYPE_REFERS_TO,
    RELATIONSHIP_TYPE_CLARIFIES,
    RELATIONSHIP_TYPE_ANSWERS
)
from pdf_extractor.arangodb.config import COLLECTION_NAME
from pdf_extractor.arangodb.message_history_api import add_message
from pdf_extractor.arangodb.message_document_links import link_message_to_document

def get_conversation_context(
    db: StandardDatabase,
    message_key: str,
    max_depth: int = 3,
    include_documents: bool = True
) -> Dict[str, Any]:
    """
    Get the context surrounding a message, including previous messages and documents.
    
    Args:
        db: ArangoDB database handle
        message_key: Key of the message to analyze
        max_depth: Maximum traversal depth for previous messages
        include_documents: Whether to include linked documents
        
    Returns:
        Dict[str, Any]: Conversation context
    """
    try:
        # Get previous messages
        previous_messages = get_previous_messages(db, message_key, max_depth)
        
        # Get next messages
        next_messages = get_next_messages(db, message_key, max_depth)
        
        # Get documents if requested
        documents = []
        if include_documents:
            documents = get_related_documents(db, [message_key] + 
                                             [m["_key"] for m in previous_messages] + 
                                             [m["_key"] for m in next_messages])
        
        return {
            "message_key": message_key,
            "previous_messages": previous_messages,
            "next_messages": next_messages,
            "related_documents": documents
        }
    except Exception as e:
        logger.error(f"Failed to get conversation context: {e}")
        return {
            "message_key": message_key,
            "previous_messages": [],
            "next_messages": [],
            "related_documents": []
        }

def get_previous_messages(
    db: StandardDatabase,
    message_key: str,
    max_depth: int = 3
) -> List[Dict[str, Any]]:
    """
    Get previous messages in the conversation.
    
    Args:
        db: ArangoDB database handle
        message_key: Key of the starting message
        max_depth: Maximum traversal depth
        
    Returns:
        List[Dict[str, Any]]: Previous messages
    """
    try:
        # Query to get previous messages
        aql = f"""
        FOR v, e, p IN 1..@max_depth INBOUND @start_vertex @edge_collection
        FILTER e.type == @rel_type
        SORT p.vertices[0].timestamp DESC
        RETURN v
        """
        
        bind_vars = {
            "start_vertex": f"{MESSAGE_COLLECTION_NAME}/{message_key}",
            "edge_collection": MESSAGE_EDGE_COLLECTION_NAME,
            "rel_type": RELATIONSHIP_TYPE_NEXT,
            "max_depth": max_depth
        }
        
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        messages = list(cursor)
        
        logger.info(f"Found {len(messages)} previous messages for {message_key}")
        return messages
    except Exception as e:
        logger.error(f"Failed to get previous messages: {e}")
        return []

def get_next_messages(
    db: StandardDatabase,
    message_key: str,
    max_depth: int = 3
) -> List[Dict[str, Any]]:
    """
    Get next messages in the conversation.
    
    Args:
        db: ArangoDB database handle
        message_key: Key of the starting message
        max_depth: Maximum traversal depth
        
    Returns:
        List[Dict[str, Any]]: Next messages
    """
    try:
        # Query to get next messages
        aql = f"""
        FOR v, e, p IN 1..@max_depth OUTBOUND @start_vertex @edge_collection
        FILTER e.type == @rel_type
        SORT p.vertices[0].timestamp ASC
        RETURN v
        """
        
        bind_vars = {
            "start_vertex": f"{MESSAGE_COLLECTION_NAME}/{message_key}",
            "edge_collection": MESSAGE_EDGE_COLLECTION_NAME,
            "rel_type": RELATIONSHIP_TYPE_NEXT,
            "max_depth": max_depth
        }
        
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        messages = list(cursor)
        
        logger.info(f"Found {len(messages)} next messages for {message_key}")
        return messages
    except Exception as e:
        logger.error(f"Failed to get next messages: {e}")
        return []

def get_related_documents(
    db: StandardDatabase,
    message_keys: List[str]
) -> List[Dict[str, Any]]:
    """
    Get documents related to a list of messages.
    
    Args:
        db: ArangoDB database handle
        message_keys: List of message keys
        
    Returns:
        List[Dict[str, Any]]: Related documents
    """
    if not message_keys:
        return []
    
    try:
        # Create a list of message vertex IDs
        message_vertices = [f"{MESSAGE_COLLECTION_NAME}/{key}" for key in message_keys]
        
        # Query to get documents related to any of the messages
        aql = f"""
        FOR start_vertex IN @start_vertices
        FOR v, e IN 1..1 OUTBOUND start_vertex @edge_collection
        FILTER e.type == @rel_type
        RETURN DISTINCT v
        """
        
        bind_vars = {
            "start_vertices": message_vertices,
            "edge_collection": MESSAGE_EDGE_COLLECTION_NAME,
            "rel_type": RELATIONSHIP_TYPE_REFERS_TO
        }
        
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        documents = list(cursor)
        
        logger.info(f"Found {len(documents)} related documents for {len(message_keys)} messages")
        return documents
    except Exception as e:
        logger.error(f"Failed to get related documents: {e}")
        return []

def analyze_conversation_references(
    db: StandardDatabase,
    conversation_id: str
) -> Dict[str, Any]:
    """
    Analyze document references in a conversation.
    
    Args:
        db: ArangoDB database handle
        conversation_id: ID of the conversation
        
    Returns:
        Dict[str, Any]: Analysis of document references
    """
    try:
        # Get all messages in the conversation
        aql_messages = f"""
        FOR msg IN {MESSAGE_COLLECTION_NAME}
        FILTER msg.conversation_id == @conversation_id
        SORT msg.timestamp ASC
        RETURN msg
        """
        
        cursor_messages = db.aql.execute(aql_messages, bind_vars={"conversation_id": conversation_id})
        messages = list(cursor_messages)
        
        # Get document references for each message
        document_refs = {}
        message_refs = {}
        
        for message in messages:
            message_key = message["_key"]
            
            # Get documents referenced by this message
            aql_docs = f"""
            FOR v, e IN 1..1 OUTBOUND @start_vertex @edge_collection
            FILTER e.type == @rel_type
            RETURN {{"document": v, "edge": e}}
            """
            
            bind_vars = {
                "start_vertex": f"{MESSAGE_COLLECTION_NAME}/{message_key}",
                "edge_collection": MESSAGE_EDGE_COLLECTION_NAME,
                "rel_type": RELATIONSHIP_TYPE_REFERS_TO
            }
            
            cursor_docs = db.aql.execute(aql_docs, bind_vars=bind_vars)
            docs = list(cursor_docs)
            
            if docs:
                message_refs[message_key] = docs
                
                for doc_data in docs:
                    doc_key = doc_data["document"]["_key"]
                    if doc_key not in document_refs:
                        document_refs[doc_key] = []
                    document_refs[doc_key].append({
                        "message": message,
                        "edge": doc_data["edge"]
                    })
        
        return {
            "conversation_id": conversation_id,
            "message_count": len(messages),
            "documents_referenced": len(document_refs),
            "messages_with_references": len(message_refs),
            "document_references": document_refs,
            "message_references": message_refs
        }
    except Exception as e:
        logger.error(f"Failed to analyze conversation references: {e}")
        return {
            "conversation_id": conversation_id,
            "message_count": 0,
            "documents_referenced": 0,
            "messages_with_references": 0,
            "document_references": {},
            "message_references": {}
        }

def validate_conversation_context(
    context: Dict[str, Any], 
    expected_file: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate conversation context analysis results.
    
    Args:
        context: Conversation context to validate
        expected_file: Path to expected results file
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (is_valid, validation_failures)
    """
    validation_failures = {}
    
    # Check message key
    if not context.get("message_key"):
        validation_failures["message_key"] = {
            "expected": "present",
            "actual": "missing"
        }
    
    # Check previous messages
    if "previous_messages" not in context:
        validation_failures["previous_messages"] = {
            "expected": "present",
            "actual": "missing"
        }
    
    # Check next messages
    if "next_messages" not in context:
        validation_failures["next_messages"] = {
            "expected": "present",
            "actual": "missing"
        }
    
    # Check related documents
    if "related_documents" not in context:
        validation_failures["related_documents"] = {
            "expected": "present",
            "actual": "missing"
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
    
    # Create a test conversation with linked documents
    conversation_id = str(uuid.uuid4())
    document_collection = db.collection(COLLECTION_NAME)
    
    # Create test document
    document_key = f"test_doc_{uuid.uuid4().hex[:8]}"
    document = {
        "_key": document_key,
        "content": "This is a test document for conversation context",
        "tags": ["test", "context"]
    }
    document_collection.insert(document)
    
    # Create a series of messages in the conversation
    current_time = datetime.now(timezone.utc)
    
    # First message
    first_message = add_message(
        db=db,
        conversation_id=conversation_id,
        message_type="USER",
        content="I have a question about document processing",
        timestamp=(current_time - timedelta(minutes=5)).isoformat()
    )
    
    # Second message (agent response)
    second_message = add_message(
        db=db,
        conversation_id=conversation_id,
        message_type="AGENT",
        content="I can help with document processing. Let me check our documentation.",
        timestamp=(current_time - timedelta(minutes=4)).isoformat(),
        previous_message_key=first_message["_key"]
    )
    
    # Link second message to document
    link_message_to_document(
        db=db,
        message_key=second_message["_key"],
        document_key=document_key,
        rationale="Reference to relevant documentation"
    )
    
    # Third message (user follow-up)
    third_message = add_message(
        db=db,
        conversation_id=conversation_id,
        message_type="USER",
        content="Thanks! How do I implement the extraction pipeline?",
        timestamp=(current_time - timedelta(minutes=3)).isoformat(),
        previous_message_key=second_message["_key"]
    )
    
    # Get context for the third message
    context = get_conversation_context(db, third_message["_key"])
    
    # Validate context
    validation_passed, validation_failures = validate_conversation_context(
        context, "src/test_fixtures/context_expected.json"
    )
    
    # Clean up test data
    db.collection(MESSAGE_COLLECTION_NAME).delete(first_message["_key"])
    db.collection(MESSAGE_COLLECTION_NAME).delete(second_message["_key"])
    db.collection(MESSAGE_COLLECTION_NAME).delete(third_message["_key"])
    document_collection.delete(document_key)
    
    if validation_passed:
        print("✅ Conversation context validation passed")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED - Conversation context doesn't match expected values")
        print("FAILURE DETAILS:")
        for field, details in validation_failures.items():
            print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        print(f"Total errors: {len(validation_failures)} fields mismatched")
        sys.exit(1)
