#!/usr/bin/env python
"""
CLI Interface for Managing Lessons Learned in ArangoDB.

This module provides a command-line interface for managing lessons learned 
documents in ArangoDB. It supports creating, reading, updating, and deleting
lesson documents through a simple CLI interface.

Dependencies:
- typer: For building the CLI
- loguru: For logging
- pydantic: For data validation
- arango: For ArangoDB interaction

Usage:
    Add a new lesson:
    ```
    uv run src/pdf_extractor/arangodb/lessons_cli.py add \
        --problem "Problem description" \
        --solution "Solution approach" \
        --tags "pdf,extraction" \
        --project "pdf_extractor"
    ```

    Get a lesson by key:
    ```
    uv run src/pdf_extractor/arangodb/lessons_cli.py get lesson_key
    ```

    List all lessons or filter by tags:
    ```
    uv run src/pdf_extractor/arangodb/lessons_cli.py list
    uv run src/pdf_extractor/arangodb/lessons_cli.py list --tags "pdf,table"
    ```

    Update a lesson:
    ```
    uv run src/pdf_extractor/arangodb/lessons_cli.py update lesson_key \
        --solution "Updated solution"
    ```

    Delete a lesson:
    ```
    uv run src/pdf_extractor/arangodb/lessons_cli.py delete lesson_key
    ```
"""

import os
import sys
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import typer
from loguru import logger
from pydantic import BaseModel, Field, validator
from arango import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import ArangoServerError, DocumentGetError

# --- Add parent directory to path for module imports ---
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

# --- Import project modules ---
try:
    from pdf_extractor.arangodb.lessons import (
        add_lesson,
        get_lesson,
        update_lesson,
        delete_lesson,
    )
    from pdf_extractor.arangodb.config import (
        COLLECTION_NAME,
        ARANGO_HOST,
        ARANGO_USER,
        ARANGO_PASSWORD,
        ARANGO_DB_NAME,
    )
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# --- Schema Validation Models ---
class LessonItem(BaseModel):
    """Schema for a single lesson item within the lessons array."""
    category: str = Field(..., description="Category of the lesson")
    title: str = Field(..., description="Brief title of the lesson")
    description: str = Field(..., description="Short description of the lesson")
    details: str = Field(..., description="Detailed explanation")
    benefit: str = Field(..., description="Benefits gained from this lesson")

class LessonDocument(BaseModel):
    """Schema for a complete lesson document."""
    _key: Optional[str] = Field(None, description="Unique identifier for the lesson")
    project: str = Field(..., description="Project name")
    module: str = Field(..., description="Module or component name")
    created_date: Optional[str] = Field(None, description="Creation date (YYYY-MM-DD)")
    author: Optional[str] = Field("Claude", description="Author of the lesson")
    tags: List[str] = Field(..., description="Tags for categorization")
    problem: str = Field(..., description="Problem statement")
    solution: str = Field(..., description="Solution approach")
    lessons: Optional[List[LessonItem]] = Field([], description="List of specific lessons learned")

    @validator("created_date", pre=True, always=True)
    def set_created_date(cls, v):
        """Set the current date if not provided."""
        if v is None:
            return datetime.now().strftime("%Y-%m-%d")
        return v

    @validator("_key", pre=True, always=True)
    def set_key(cls, v, values):
        """Generate a key if not provided."""
        if v is None and "project" in values and "module" in values:
            return f"{values['project']}_{values['module']}_{uuid.uuid4().hex[:8]}"
        return v

# --- CLI App Setup ---
app = typer.Typer(
    name="lessons-cli",
    help="CLI for managing Lessons Learned documents in ArangoDB.",
)

# --- Helper Functions ---
def get_db_connection() -> StandardDatabase:
    """Establish a connection to ArangoDB and return the database object."""
    try:
        # Connect to ArangoDB
        client = ArangoClient(hosts=ARANGO_HOST)
        
        # Connect to the database
        db = client.db(
            name=ARANGO_DB_NAME,
            username=ARANGO_USER,
            password=ARANGO_PASSWORD,
        )
        
        # Ensure the collection exists
        if not db.has_collection(COLLECTION_NAME):
            db.create_collection(COLLECTION_NAME)
            logger.info(f"Created collection: {COLLECTION_NAME}")
        
        return db
    except ArangoServerError as e:
        logger.error(f"Failed to connect to ArangoDB: {e}")
        typer.echo(f"Error: Failed to connect to ArangoDB: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception(f"Unexpected error connecting to database: {e}")
        typer.echo(f"Error: {str(e)}")
        raise typer.Exit(code=1)

def format_output(data: Dict[str, Any], pretty: bool = True) -> None:
    """Format and print data to console."""
    if pretty:
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(json.dumps(data))

# --- CLI Commands ---
@app.command("add")
def cli_add_lesson(
    problem: str = typer.Option(..., help="Problem description"),
    solution: str = typer.Option(..., help="Solution approach"),
    project: str = typer.Option(..., help="Project name"),
    module: str = typer.Option(..., help="Module or component name"),
    tags: str = typer.Option("", help="Comma-separated tags"),
    author: str = typer.Option("Claude", help="Author name"),
    key: Optional[str] = typer.Option(None, help="Custom key (optional)"),
    pretty: bool = typer.Option(True, help="Pretty print the output"),
) -> None:
    """
    Add a new lesson learned document to ArangoDB.
    
    The lesson will be stored with required fields and metadata.
    """
    # Process tags
    tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
    
    # Prepare the lesson document
    lesson_data = {
        "problem": problem,
        "solution": solution,
        "project": project,
        "module": module,
        "tags": tag_list,
        "author": author,
        "lessons": [],
        "created_date": datetime.now().strftime("%Y-%m-%d"),
    }
    
    # Add custom key if provided
    if key:
        lesson_data["_key"] = key
        
    # Validate with Pydantic model
    try:
        validated_data = LessonDocument(**lesson_data).dict(exclude_none=True)
    except Exception as e:
        logger.error(f"Validation error: {e}")
        typer.echo(f"Error: Invalid lesson data: {e}")
        raise typer.Exit(code=1)
    
    # Add to database
    db = get_db_connection()
    result = add_lesson(db, validated_data)
    
    if result:
        logger.info(f"Added lesson with key: {result.get('_key')}")
        typer.echo(f"Successfully added lesson with key: {result.get('_key')}")
        if pretty:
            format_output(result, pretty)
    else:
        logger.error("Failed to add lesson")
        typer.echo("Error: Failed to add lesson")
        raise typer.Exit(code=1)

@app.command("get")
def cli_get_lesson(
    key: str = typer.Argument(..., help="Lesson key to retrieve"),
    pretty: bool = typer.Option(True, help="Pretty print the output"),
) -> None:
    """
    Retrieve a lesson learned document by its key.
    """
    db = get_db_connection()
    lesson = get_lesson(db, key)
    
    if lesson:
        format_output(lesson, pretty)
    else:
        logger.error(f"Lesson not found: {key}")
        typer.echo(f"Error: Lesson not found with key: {key}")
        raise typer.Exit(code=1)

@app.command("list")
def cli_list_lessons(
    tags: Optional[str] = typer.Option(None, help="Filter by comma-separated tags"),
    project: Optional[str] = typer.Option(None, help="Filter by project name"),
    limit: int = typer.Option(20, help="Maximum number of lessons to retrieve"),
    pretty: bool = typer.Option(True, help="Pretty print the output"),
) -> None:
    """
    List lessons learned documents with optional filtering.
    """
    db = get_db_connection()
    
    # Prepare AQL query
    query_filters = []
    bind_vars = {"collection": COLLECTION_NAME, "limit": limit}
    
    # Add tag filter if provided
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        query_filters.append("LENGTH(INTERSECTION(doc.tags, @tags)) > 0")
        bind_vars["tags"] = tag_list
    
    # Add project filter if provided
    if project:
        query_filters.append("doc.project == @project")
        bind_vars["project"] = project
    
    # Build the query
    aql = f"""
    FOR doc IN {COLLECTION_NAME}
    """
    
    if query_filters:
        aql += "FILTER " + " AND ".join(query_filters)
    
    aql += """
    SORT doc.created_date DESC
    LIMIT @limit
    RETURN doc
    """
    
    try:
        # Execute the query
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        lessons = [doc for doc in cursor]
        
        if not lessons:
            if tags or project:
                typer.echo("No lessons found matching the filter criteria.")
            else:
                typer.echo("No lessons found in the database.")
            return
        
        # Print as JSON array
        format_output(lessons, pretty)
        
    except Exception as e:
        logger.exception(f"Error listing lessons: {e}")
        typer.echo(f"Error: Failed to list lessons: {e}")
        raise typer.Exit(code=1)

@app.command("update")
def cli_update_lesson(
    key: str = typer.Argument(..., help="Lesson key to update"),
    problem: Optional[str] = typer.Option(None, help="Updated problem description"),
    solution: Optional[str] = typer.Option(None, help="Updated solution approach"),
    tags: Optional[str] = typer.Option(None, help="Updated comma-separated tags"),
    add_lesson_item: bool = typer.Option(False, help="Add a new lesson item"),
    pretty: bool = typer.Option(True, help="Pretty print the output"),
) -> None:
    """
    Update an existing lesson learned document.
    """
    db = get_db_connection()
    
    # Get the existing lesson first
    existing = get_lesson(db, key)
    if not existing:
        logger.error(f"Lesson not found: {key}")
        typer.echo(f"Error: Lesson not found with key: {key}")
        raise typer.Exit(code=1)
    
    # Prepare the update data
    update_data = {}
    
    if problem:
        update_data["problem"] = problem
    
    if solution:
        update_data["solution"] = solution
    
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        update_data["tags"] = tag_list
    
    # Add a new lesson item if requested
    if add_lesson_item:
        category = typer.prompt("Category")
        title = typer.prompt("Title")
        description = typer.prompt("Description")
        details = typer.prompt("Details")
        benefit = typer.prompt("Benefit")
        
        new_item = {
            "category": category,
            "title": title,
            "description": description,
            "details": details,
            "benefit": benefit
        }
        
        # Validate the lesson item
        try:
            LessonItem(**new_item)
        except Exception as e:
            logger.error(f"Invalid lesson item: {e}")
            typer.echo(f"Error: Invalid lesson item: {e}")
            raise typer.Exit(code=1)
        
        # Add to existing lessons array or create a new one
        lessons = existing.get("lessons", []) 
        lessons.append(new_item)
        update_data["lessons"] = lessons
    
    # If no updates were provided and not adding a lesson item
    if not update_data and not add_lesson_item:
        typer.echo("No updates provided. Use --help to see available options.")
        return
    
    # Update the lesson
    result = update_lesson(db, key, update_data)
    
    if result:
        logger.info(f"Updated lesson: {key}")
        typer.echo(f"Successfully updated lesson: {key}")
        
        # Get the updated document to show changes
        updated = get_lesson(db, key)
        if updated and pretty:
            format_output(updated, pretty)
    else:
        logger.error(f"Failed to update lesson: {key}")
        typer.echo(f"Error: Failed to update lesson: {key}")
        raise typer.Exit(code=1)

@app.command("delete")
def cli_delete_lesson(
    key: str = typer.Argument(..., help="Lesson key to delete"),
    confirm: bool = typer.Option(True, help="Confirm before deletion"),
) -> None:
    """
    Delete a lesson learned document.
    """
    db = get_db_connection()
    
    # Check if lesson exists
    existing = get_lesson(db, key)
    if not existing:
        logger.error(f"Lesson not found: {key}")
        typer.echo(f"Error: Lesson not found with key: {key}")
        raise typer.Exit(code=1)
    
    # Confirm deletion
    if confirm:
        confirmed = typer.confirm(f"Are you sure you want to delete lesson '{key}'?")
        if not confirmed:
            typer.echo("Deletion cancelled.")
            return
    
    # Delete the lesson
    result = delete_lesson(db, key)
    
    if result:
        logger.info(f"Deleted lesson: {key}")
        typer.echo(f"Successfully deleted lesson: {key}")
    else:
        logger.error(f"Failed to delete lesson: {key}")
        typer.echo(f"Error: Failed to delete lesson: {key}")
        raise typer.Exit(code=1)

# --- Validation and Standalone Execution ---
def validate_execution() -> bool:
    """Validate the CLI execution by testing a simple list operation."""
    try:
        # Connect to the database
        db = get_db_connection()
        
        # Test a simple query
        aql = f"FOR doc IN {COLLECTION_NAME} LIMIT 1 RETURN doc"
        try:
            cursor = db.aql.execute(aql)
            result = list(cursor)
            logger.info(f"Validation successful: Retrieved {len(result)} documents")
            
            # Create a test fixture with expected results
            test_fixture_dir = os.path.abspath(os.path.join(_root, "test_fixtures"))
            os.makedirs(test_fixture_dir, exist_ok=True)
            
            test_fixture = {
                "validation_successful": True,
                "collection": COLLECTION_NAME,
                "host": ARANGO_HOST,
                "database": ARANGO_DB_NAME,
                "records_found": len(result)
            }
            
            fixture_path = os.path.join(test_fixture_dir, "lessons_cli_validation.json")
            with open(fixture_path, "w") as f:
                json.dump(test_fixture, f, indent=2)
                
            logger.info(f"Created validation fixture: {fixture_path}")
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    except Exception as e:
        logger.exception(f"Validation error: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr, 
        level="INFO", 
        format="{time:HH:mm:ss} | {level:<7} | {message}"
    )
    
    # Display connection info
    logger.info(f"Using ArangoDB at: {ARANGO_HOST}, Database: {ARANGO_DB_NAME}")
    
    # Validate environment
    if validate_execution():
        logger.info("✅ Environment validation passed")
        app()
    else:
        logger.error("❌ Environment validation failed")
        typer.echo("Error: Failed to connect to ArangoDB or validate environment")
        sys.exit(1)
