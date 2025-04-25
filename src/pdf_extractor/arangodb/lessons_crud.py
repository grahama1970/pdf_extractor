"""
Lessons CRUD Operations for PDF Extractor ArangoDB Integration.

This module provides functions for creating, reading, updating, and deleting
lessons in the ArangoDB database.

Third-Party Package Documentation:
- python-arango: https://python-driver.arangodb.com/
- loguru: https://github.com/Delgan/loguru

Sample Input:
Lesson data dictionary with fields for problem, solution, tags, etc.

Expected Output:
Results of CRUD operations including confirmation of success and returned data
"""
import sys
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid
from loguru import logger

# Add root directory to path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from arango.database import StandardDatabase
from arango.collection import StandardCollection
from arango.exceptions import (
    DocumentCreateError,
    DocumentGetError,
    DocumentUpdateError,
    DocumentDeleteError,
    ArangoServerError,
    AQLQueryExecuteError
)

# Import configuration
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME,
)
from pdf_extractor.arangodb.embedding_utils import get_embedding

def add_lesson(
    db: StandardDatabase,
    lesson_data: Dict[str, Any],
    collection_name: str = COLLECTION_NAME,
    embed_text: bool = True,
) -> Dict[str, Any]:
    """
    Add a new lesson to the database.
    
    Args:
        db: ArangoDB database connection
        lesson_data: Dictionary containing lesson data
        collection_name: Name of the collection to insert into
        embed_text: Whether to generate and store an embedding for the lesson
        
    Returns:
        Dictionary containing the inserted document with _id, _key, etc.
        
    Raises:
        DocumentCreateError: If the document could not be created
        ValueError: If required fields are missing
    """
    # Validate required fields
    required_fields = ["problem", "solution"]
    for field in required_fields:
        if field not in lesson_data or not lesson_data[field]:
            raise ValueError(f"Required field '{field}' is missing or empty in lesson data")
    
    # Ensure collection exists
    if not db.has_collection(collection_name):
        logger.error(f"Collection '{collection_name}' does not exist")
        raise ValueError(f"Collection '{collection_name}' does not exist")
    
    collection = db.collection(collection_name)
    
    # Add timestamp if not provided
    if "timestamp" not in lesson_data:
        lesson_data["timestamp"] = datetime.now().isoformat()
    
    # Add unique ID if not provided
    if "_key" not in lesson_data:
        lesson_data["_key"] = f"lesson_{str(uuid.uuid4())[:8]}"
    
    # Generate embedding if requested
    if embed_text:
        # Combine relevant text fields for embedding
        text_for_embedding = " ".join(
            str(lesson_data.get(field, ""))
            for field in ["problem", "solution", "context", "example"]
            if field in lesson_data
        )
        
        if text_for_embedding:
            embedding = get_embedding(text_for_embedding)
            if embedding:
                lesson_data["embedding"] = embedding
                logger.debug(f"Generated embedding for lesson with key '{lesson_data['_key']}'")
            else:
                logger.warning(f"Failed to generate embedding for lesson with key '{lesson_data['_key']}'")
    
    # Insert document
    try:
        result = collection.insert(lesson_data, return_new=True)
        logger.info(f"Added lesson with key '{result['_key']}'")
        return result.get("new", result)  # Return the full document if available
    except DocumentCreateError as e:
        logger.error(f"Failed to create lesson: {e}")
        raise

def get_lesson(
    db: StandardDatabase,
    lesson_key: str,
    collection_name: str = COLLECTION_NAME,
) -> Optional[Dict[str, Any]]:
    """
    Get a lesson by its key.
    
    Args:
        db: ArangoDB database connection
        lesson_key: Key of the lesson to retrieve
        collection_name: Name of the collection to read from
        
    Returns:
        Dictionary containing the lesson document or None if not found
        
    Raises:
        DocumentGetError: If there was an error retrieving the document
    """
    # Ensure collection exists
    if not db.has_collection(collection_name):
        logger.error(f"Collection '{collection_name}' does not exist")
        raise ValueError(f"Collection '{collection_name}' does not exist")
    
    collection = db.collection(collection_name)
    
    # Get document
    try:
        result = collection.get(lesson_key)
        if result:
            logger.debug(f"Retrieved lesson with key '{lesson_key}'")
        else:
            logger.info(f"Lesson with key '{lesson_key}' not found")
        return result
    except DocumentGetError as e:
        logger.error(f"Error retrieving lesson with key '{lesson_key}': {e}")
        raise

def update_lesson(
    db: StandardDatabase,
    lesson_key: str,
    update_data: Dict[str, Any],
    collection_name: str = COLLECTION_NAME,
    update_embedding: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Update an existing lesson.
    
    Args:
        db: ArangoDB database connection
        lesson_key: Key of the lesson to update
        update_data: Dictionary containing fields to update
        collection_name: Name of the collection containing the lesson
        update_embedding: Whether to update the embedding if text fields change
        
    Returns:
        Dictionary containing the updated document or None if not found
        
    Raises:
        DocumentUpdateError: If there was an error updating the document
        ValueError: If the lesson key doesn't exist
    """
    # Ensure collection exists
    if not db.has_collection(collection_name):
        logger.error(f"Collection '{collection_name}' does not exist")
        raise ValueError(f"Collection '{collection_name}' does not exist")
    
    collection = db.collection(collection_name)
    
    # Check if document exists
    existing_doc = collection.get(lesson_key)
    if not existing_doc:
        logger.error(f"Lesson with key '{lesson_key}' not found")
        return None
    
    # Update embedding if text fields changed and update_embedding is True
    if update_embedding and any(
        field in update_data for field in ["problem", "solution", "context", "example"]
    ):
        # Prepare the full document with updates
        updated_doc = {**existing_doc, **update_data}
        
        # Combine relevant text fields for embedding
        text_for_embedding = " ".join(
            str(updated_doc.get(field, ""))
            for field in ["problem", "solution", "context", "example"]
            if field in updated_doc
        )
        
        if text_for_embedding:
            embedding = get_embedding(text_for_embedding)
            if embedding:
                update_data["embedding"] = embedding
                logger.debug(f"Updated embedding for lesson with key '{lesson_key}'")
            else:
                logger.warning(f"Failed to update embedding for lesson with key '{lesson_key}'")
    
    # Update document
    try:
        result = collection.update(lesson_key, update_data, return_new=True)
        logger.info(f"Updated lesson with key '{lesson_key}'")
        return result.get("new", result)  # Return the full updated document if available
    except DocumentUpdateError as e:
        logger.error(f"Error updating lesson with key '{lesson_key}': {e}")
        raise

def delete_lesson(
    db: StandardDatabase,
    lesson_key: str,
    collection_name: str = COLLECTION_NAME,
) -> bool:
    """
    Delete a lesson from the database.
    
    Args:
        db: ArangoDB database connection
        lesson_key: Key of the lesson to delete
        collection_name: Name of the collection containing the lesson
        
    Returns:
        True if the lesson was deleted, False if it didn't exist
        
    Raises:
        DocumentDeleteError: If there was an error deleting the document
    """
    # Ensure collection exists
    if not db.has_collection(collection_name):
        logger.error(f"Collection '{collection_name}' does not exist")
        raise ValueError(f"Collection '{collection_name}' does not exist")
    
    collection = db.collection(collection_name)
    
    # Check if document exists
    if not collection.get(lesson_key):
        logger.info(f"Lesson with key '{lesson_key}' not found")
        return False
    
    # Delete document
    try:
        collection.delete(lesson_key)
        logger.info(f"Deleted lesson with key '{lesson_key}'")
        return True
    except DocumentDeleteError as e:
        logger.error(f"Error deleting lesson with key '{lesson_key}': {e}")
        raise

def list_lessons(
    db: StandardDatabase,
    limit: int = 100,
    offset: int = 0,
    collection_name: str = COLLECTION_NAME,
) -> Dict[str, Any]:
    """
    List lessons with pagination.
    
    Args:
        db: ArangoDB database connection
        limit: Maximum number of lessons to return
        offset: Starting position for pagination
        collection_name: Name of the collection containing lessons
        
    Returns:
        Dictionary containing results and pagination info
        
    Raises:
        AQLQueryExecuteError: If there was an error executing the AQL query
    """
    # Ensure collection exists
    if not db.has_collection(collection_name):
        logger.error(f"Collection '{collection_name}' does not exist")
        raise ValueError(f"Collection '{collection_name}' does not exist")
    
    # Query lessons with pagination
    try:
        aql = f"""
            LET lessons = (
                FOR doc IN {collection_name}
                    SORT doc.timestamp DESC
                    LIMIT @offset, @limit
                    RETURN doc
            )
            
            LET total_count = LENGTH({collection_name})
            
            RETURN {{
                "results": lessons,
                "total": total_count,
                "offset": @offset,
                "limit": @limit
            }}
        """
        
        bind_vars = {
            "offset": offset,
            "limit": limit
        }
        
        cursor = db.aql.execute(aql, bind_vars=bind_vars, count=True)
        result = next(cursor, None)
        
        if result:
            logger.info(f"Listed {len(result.get('results', []))} lessons (total: {result.get('total', 0)})")
            return result
        else:
            logger.info(f"No lessons found or error in query")
            return {"results": [], "total": 0, "offset": offset, "limit": limit}
    except AQLQueryExecuteError as e:
        logger.error(f"Error listing lessons: {e}")
        raise

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level:<7} | {message}"
    )
    
    # Example usage
    from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
    
    client = connect_arango()
    db = ensure_database(client)
    
    # Example: Add a lesson
    try:
        example_lesson = {
            "problem": "Finding optimal PDF extraction parameters",
            "solution": "Use grid search to find optimal parameters for different PDF types",
            "tags": ["pdf", "extraction", "optimization"],
            "context": "When extracting text from PDFs, different parameters work better for different document types",
            "example": "For scientific papers, use tighter character spacing; for forms, use larger block detection"
        }
        
        added_lesson = add_lesson(db, example_lesson)
        logger.info(f"Added lesson with key {added_lesson.get('_key')}")
        
        # Example: Update the lesson
        updated_lesson = update_lesson(
            db,
            added_lesson.get("_key"),
            {"tags": ["pdf", "extraction", "optimization", "grid-search"]}
        )
        logger.info(f"Updated lesson: {updated_lesson.get('tags')}")
        
        # Example: Get the lesson
        retrieved_lesson = get_lesson(db, added_lesson.get("_key"))
        logger.info(f"Retrieved lesson: {retrieved_lesson.get('problem')}")
        
        # Example: List lessons
        lessons = list_lessons(db, limit=10)
        logger.info(f"Listed {len(lessons.get('results', []))} of {lessons.get('total', 0)} lessons")
        
        # Example: Delete the lesson
        deleted = delete_lesson(db, added_lesson.get("_key"))
        logger.info(f"Deleted lesson: {deleted}")
        
        logger.info("All CRUD operations completed successfully")
    except Exception as e:
        logger.error(f"Error during CRUD operations example: {e}")
        raise
