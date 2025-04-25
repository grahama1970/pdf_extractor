# src/pdf_extractor/arangodb/crud.py
"""
ArangoDB CRUD Operations Module

This module provides standardized functions for Create, Read, Update, Delete (CRUD)
operations on ArangoDB collections, with specific support for message history
and document collections in the PDF extractor project.

Key Features:
- Generalized CRUD operations that work with any collection
- Specialized functions for message_history operations
- Support for relationship management between collections
- Consistent error handling and logging
"""

import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger

from arango.database import StandardDatabase
from arango.exceptions import (
    ArangoClientError,
    ArangoServerError,
    DocumentInsertError,
    DocumentGetError,
    DocumentUpdateError,
    DocumentDeleteError
)

from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.message_history_config import (
    MESSAGE_COLLECTION_NAME,
    MESSAGE_EDGE_COLLECTION_NAME,
    MESSAGE_GRAPH_NAME,
    RELATIONSHIP_TYPE_NEXT,
    RELATIONSHIP_TYPE_REFERS_TO
)
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME as DOC_COLLECTION_NAME,
    EDGE_COLLECTION_NAME as DOC_EDGE_COLLECTION_NAME,
    GRAPH_NAME as DOC_GRAPH_NAME
)


# -------------------- Generic CRUD Operations --------------------


def create_document(
    db: StandardDatabase,
    collection_name: str,
    document: Dict[str, Any],
    document_key: Optional[str] = None,
    return_new: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Insert a document into a collection.
    
    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        document: Document data to insert
        document_key: Optional key for the document (auto-generated if not provided)
        return_new: Whether to return the new document
        
    Returns:
        Optional[Dict[str, Any]]: The inserted document or metadata if successful, None otherwise
    """
    try:
        # Generate a key if not provided
        if document_key:
            document["_key"] = document_key
        elif "_key" not in document:
            document["_key"] = str(uuid.uuid4())
        
        # Add timestamp if not present
        if "timestamp" not in document:
            document["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Get the collection and insert document
        collection = db.collection(collection_name)
        result = collection.insert(document, return_new=return_new)
        
        logger.info(f"Created document in {collection_name}: {result.get('_key', result)}")
        return result["new"] if return_new and "new" in result else result
    
    except DocumentInsertError as e:
        logger.error(f"Failed to create document in {collection_name}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error creating document in {collection_name}: {e}")
        return None

def get_document(
    db: StandardDatabase, 
    collection_name: str, 
    document_key: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a document by key.
    
    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        document_key: Key of the document to retrieve
        
    Returns:
        Optional[Dict[str, Any]]: The document if found, None otherwise
    """
    try:
        collection = db.collection(collection_name)
        document = collection.get(document_key)
        
        if document:
            logger.debug(f"Retrieved document from {collection_name}: {document_key}")
        else:
            logger.warning(f"Document not found in {collection_name}: {document_key}")
        
        return document
    
    except DocumentGetError as e:
        logger.error(f"Failed to get document from {collection_name}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error getting document from {collection_name}: {e}")
        return None


def update_document(
    db: StandardDatabase,
    collection_name: str,
    document_key: str,
    updates: Dict[str, Any],
    return_new: bool = True,
    check_rev: bool = False,
    rev: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Update a document with new values.
    
    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        document_key: Key of the document to update
        updates: Dictionary of fields to update
        return_new: Whether to return the updated document
        check_rev: Whether to check document revision
        rev: Document revision (required if check_rev is True)
        
    Returns:
        Optional[Dict[str, Any]]: The updated document if successful, None otherwise
    """
    try:
        # Add updated timestamp
        if "timestamp" not in updates:
            updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        # Get the collection and update document
        collection = db.collection(collection_name)
        
        # Add revision if needed
        params = {}
        if check_rev and rev:
            params["rev"] = rev
        
        result = collection.update(
            document=document_key,
            check_rev=check_rev,
            merge=True,
            data=updates,
            return_new=return_new,
            **params
        )
        
        logger.info(f"Updated document in {collection_name}: {document_key}")
        return result["new"] if return_new and "new" in result else result
    
    except DocumentUpdateError as e:
        logger.error(f"Failed to update document in {collection_name}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error updating document in {collection_name}: {e}")
        return None


def delete_document(
    db: StandardDatabase,
    collection_name: str,
    document_key: str,
    ignore_missing: bool = True,
    return_old: bool = False,
    check_rev: bool = False,
    rev: Optional[str] = None
) -> bool:
    """
    Delete a document from a collection.
    
    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        document_key: Key of the document to delete
        ignore_missing: Whether to ignore if document doesn't exist
        return_old: Whether to return the old document
        check_rev: Whether to check document revision
        rev: Document revision (required if check_rev is True)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the collection and delete document
        collection = db.collection(collection_name)
        
        # Add revision if needed
        params = {}
        if check_rev and rev:
            params["rev"] = rev
        
        result = collection.delete(
            document=document_key,
            ignore_missing=ignore_missing,
            return_old=return_old,
            check_rev=check_rev,
            **params
        )
        
        if result is False and ignore_missing:
            logger.info(f"Document not found for deletion in {collection_name}: {document_key}")
            return True
        
        logger.info(f"Deleted document from {collection_name}: {document_key}")
        return True
    
    except DocumentDeleteError as e:
        logger.error(f"Failed to delete document from {collection_name}: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error deleting document from {collection_name}: {e}")
        return False

def query_documents(
    db: StandardDatabase,
    collection_name: str,
    filter_clause: str = "",
    sort_clause: str = "",
    limit: int = 100,
    offset: int = 0,
    bind_vars: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Query documents from a collection.
    
    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        filter_clause: AQL filter clause (e.g., "FILTER doc.field == @value")
        sort_clause: AQL sort clause (e.g., "SORT doc.field DESC")
        limit: Maximum number of documents to return
        offset: Number of documents to skip
        bind_vars: Bind variables for the query
        
    Returns:
        List[Dict[str, Any]]: List of documents matching the query
    """
    try:
        # Build AQL query
        aql = f"""
        FOR doc IN {collection_name}
        {filter_clause}
        {sort_clause}
        LIMIT {offset}, {limit}
        RETURN doc
        """
        
        # Set default bind variables
        if bind_vars is None:
            bind_vars = {}
        
        # Execute query
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)
        
        logger.info(f"Query returned {len(results)} documents from {collection_name}")
        return results
    
    except Exception as e:
        logger.exception(f"Error querying documents from {collection_name}: {e}")
        return []


# -------------------- Message History Specific Operations --------------------


def create_message(
    db: StandardDatabase,
    conversation_id: str,
    message_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
    previous_message_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Create a message in the message history collection.
    
    Args:
        db: ArangoDB database handle
        conversation_id: ID of the conversation
        message_type: Type of message (USER, AGENT, SYSTEM)
        content: Message content
        metadata: Optional metadata
        timestamp: Optional timestamp (ISO format)
        previous_message_key: Optional key of the previous message to link to
        
    Returns:
        Optional[Dict[str, Any]]: The created message if successful, None otherwise
    """
    # Prepare message
    message_key = str(uuid.uuid4())
    message = {
        "_key": message_key,
        "conversation_id": conversation_id,
        "message_type": message_type,
        "content": content,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {}
    }
    
    # Create the message
    result = create_document(db, MESSAGE_COLLECTION_NAME, message)
    
    # Create relationship if previous message is provided
    if result and previous_message_key:
        # Create edge between messages
        edge = {
            "_from": f"{MESSAGE_COLLECTION_NAME}/{previous_message_key}",
            "_to": f"{MESSAGE_COLLECTION_NAME}/{message_key}",
            "type": RELATIONSHIP_TYPE_NEXT,
            "timestamp": message["timestamp"]
        }
        create_document(db, MESSAGE_EDGE_COLLECTION_NAME, edge)
    
    return result


def get_message(
    db: StandardDatabase,
    message_key: str
) -> Optional[Dict[str, Any]]:
    """
    Get a message by key.
    
    Args:
        db: ArangoDB database handle
        message_key: Key of the message
        
    Returns:
        Optional[Dict[str, Any]]: The message if found, None otherwise
    """
    return get_document(db, MESSAGE_COLLECTION_NAME, message_key)


def update_message(
    db: StandardDatabase,
    message_key: str,
    updates: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Update a message.
    
    Args:
        db: ArangoDB database handle
        message_key: Key of the message
        updates: Fields to update
        
    Returns:
        Optional[Dict[str, Any]]: The updated message if successful, None otherwise
    """
    return update_document(db, MESSAGE_COLLECTION_NAME, message_key, updates)


def delete_message(
    db: StandardDatabase,
    message_key: str,
    delete_relationships: bool = True
) -> bool:
    """
    Delete a message.
    
    Args:
        db: ArangoDB database handle
        message_key: Key of the message
        delete_relationships: Whether to delete related edges
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Delete relationships if requested
    if delete_relationships:
        try:
            # Delete outgoing edges
            aql_out = f"""
            FOR edge IN {MESSAGE_EDGE_COLLECTION_NAME}
            FILTER edge._from == @from
            RETURN edge._key
            """
            cursor_out = db.aql.execute(
                aql_out, 
                bind_vars={"from": f"{MESSAGE_COLLECTION_NAME}/{message_key}"}
            )
            for edge_key in cursor_out:
                delete_document(db, MESSAGE_EDGE_COLLECTION_NAME, edge_key)
            
            # Delete incoming edges
            aql_in = f"""
            FOR edge IN {MESSAGE_EDGE_COLLECTION_NAME}
            FILTER edge._to == @to
            RETURN edge._key
            """
            cursor_in = db.aql.execute(
                aql_in, 
                bind_vars={"to": f"{MESSAGE_COLLECTION_NAME}/{message_key}"}
            )
            for edge_key in cursor_in:
                delete_document(db, MESSAGE_EDGE_COLLECTION_NAME, edge_key)
        
        except Exception as e:
            logger.error(f"Error deleting message relationships: {e}")
            return False
    
    # Delete the message
    return delete_document(db, MESSAGE_COLLECTION_NAME, message_key)

def get_conversation_messages(
    db: StandardDatabase,
    conversation_id: str,
    limit: int = 100,
    offset: int = 0,
    sort_order: str = "asc"
) -> List[Dict[str, Any]]:
    """
    Get all messages for a conversation.
    
    Args:
        db: ArangoDB database handle
        conversation_id: ID of the conversation
        limit: Maximum number of messages to return
        offset: Number of messages to skip
        sort_order: Sort order ("asc" or "desc")
        
    Returns:
        List[Dict[str, Any]]: List of messages
    """
    # Validate sort order
    sort_direction = "ASC" if sort_order.lower() == "asc" else "DESC"
    
    # Build filter and sort clauses
    filter_clause = "FILTER doc.conversation_id == @conversation_id"
    sort_clause = f"SORT doc.timestamp {sort_direction}"
    
    # Query messages
    return query_documents(
        db,
        MESSAGE_COLLECTION_NAME,
        filter_clause,
        sort_clause,
        limit,
        offset,
        {"conversation_id": conversation_id}
    )


def delete_conversation(
    db: StandardDatabase,
    conversation_id: str
) -> bool:
    """
    Delete all messages for a conversation.
    
    Args:
        db: ArangoDB database handle
        conversation_id: ID of the conversation
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get all message keys for the conversation
        aql_keys = f"""
        FOR doc IN {MESSAGE_COLLECTION_NAME}
        FILTER doc.conversation_id == @conversation_id
        RETURN doc._key
        """
        cursor_keys = db.aql.execute(aql_keys, bind_vars={"conversation_id": conversation_id})
        message_keys = list(cursor_keys)
        
        if not message_keys:
            logger.info(f"No messages found for conversation: {conversation_id}")
            return True
        
        # Delete all messages (including relationships)
        for key in message_keys:
            delete_message(db, key)
        
        logger.info(f"Deleted {len(message_keys)} messages for conversation: {conversation_id}")
        return True
    
    except Exception as e:
        logger.exception(f"Failed to delete conversation: {e}")
        return False


# -------------------- Document-Message Relationship Operations --------------------


def link_message_to_document(
    db: StandardDatabase,
    message_key: str,
    document_key: str,
    relationship_type: str = RELATIONSHIP_TYPE_REFERS_TO,
    rationale: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Create a relationship between a message and a document.
    
    Args:
        db: ArangoDB database handle
        message_key: Key of the message
        document_key: Key of the document
        relationship_type: Type of relationship
        rationale: Optional explanation for the relationship
        
    Returns:
        Optional[Dict[str, Any]]: The created edge if successful, None otherwise
    """
    edge = {
        "_from": f"{MESSAGE_COLLECTION_NAME}/{message_key}",
        "_to": f"{DOC_COLLECTION_NAME}/{document_key}",
        "type": relationship_type,
        "rationale": rationale or "Message references document",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    return create_document(db, MESSAGE_EDGE_COLLECTION_NAME, edge)


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
        List[Dict[str, Any]]: List of linked documents
    """
    try:
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
        List[Dict[str, Any]]: List of messages referencing the document
    """
    try:
        aql = f"""
        FOR v, e IN 1..1 INBOUND @start_vertex @edge_collection
        FILTER e.type == @rel_type
        RETURN v
        """
        
        bind_vars = {
            "start_vertex": f"{DOC_COLLECTION_NAME}/{document_key}",
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


# -------------------- Validation --------------------


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
    
    # Test CRUD operations
    print("\nTesting CRUD operations...")
    
    # Create a test message
    test_message = create_message(
        db=db,
        conversation_id=str(uuid.uuid4()),
        message_type="SYSTEM",
        content="This is a test message for CRUD validation",
        metadata={"test": True}
    )
    
    if not test_message:
        print("❌ Failed to create test message")
        sys.exit(1)
    
    test_key = test_message["_key"]
    print(f"✅ Created test message: {test_key}")
    
    # Get the message
    retrieved = get_message(db, test_key)
    if not retrieved:
        print("❌ Failed to retrieve test message")
        delete_message(db, test_key)
        sys.exit(1)
    
    print(f"✅ Retrieved test message: {test_key}")
    
    # Update the message
    updated = update_message(db, test_key, {"content": "Updated test message"})
    if not updated:
        print("❌ Failed to update test message")
        delete_message(db, test_key)
        sys.exit(1)
    
    print(f"✅ Updated test message: {test_key}")
    
    # Delete the message
    deleted = delete_message(db, test_key)
    if not deleted:
        print("❌ Failed to delete test message")
        sys.exit(1)
    
    print(f"✅ Deleted test message: {test_key}")
    print("\n✅ CRUD operations validation passed")
