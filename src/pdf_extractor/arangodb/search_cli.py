#!/usr/bin/env python3
# src/pdf_extractor/arangodb/search_cli.py

import sys
import json
from typing import Dict, Any, List, Optional
import typer
from loguru import logger

from arango.database import StandardDatabase

from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME as DOC_COLLECTION_NAME,
    VIEW_NAME
)
from pdf_extractor.arangodb.message_history_config import (
    MESSAGE_COLLECTION_NAME,
    MESSAGE_TYPE_USER,
    MESSAGE_TYPE_AGENT,
    MESSAGE_TYPE_SYSTEM
)
from pdf_extractor.arangodb.search_api.message_search import search_messages, unified_search

def register_search_commands(app: typer.Typer) -> None:
    """Register commands for search functionality in CLI app."""
    
    search_cli = typer.Typer(name="search", help="Search functionality for documents and messages")
    
    @search_cli.command("unified")
    def unified_search_cmd(
        query: str = typer.Argument(..., help="Search query text"),
        search_type: str = typer.Option(
            "hybrid", "--type", "-t", help="Search type: hybrid, bm25, or semantic"
        ),
        collections: Optional[str] = typer.Option(
            None, "--collections", "-c", help="Collections to search in (comma-separated)"
        ),
        exclude: Optional[str] = typer.Option(
            None, "--exclude", "-e", help="Collections to exclude (comma-separated)"
        ),
        limit: int = typer.Option(
            10, "--limit", "-l", help="Maximum number of results to return"
        ),
        json_output: bool = typer.Option(
            False, "--json", "-j", help="Output results as JSON"
        ),
    ):
        """
        Perform a unified search across documents and messages.
        """
        client = connect_arango()
        db = ensure_database(client)
        
        # If collections were specified as a comma-separated string, split them
        search_collections = None
        if collections:
            search_collections = [c.strip() for c in collections.split(",")]
        
        # If exclude collections were specified as a comma-separated string, split them
        exclude_collections = None
        if exclude:
            exclude_collections = [c.strip() for c in exclude.split(",")]
        
        results = unified_search(
            db=db,
            query=query,
            search_type=search_type,
            collections=search_collections,
            exclude_collections=exclude_collections,
            top_n=limit
        )
        
        # Display results
        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return
        
        # Text output
        if not results.get("results"):
            typer.echo(f"No results found for query: {query}")
            return
        
        typer.echo(f"Found {len(results.get('results', []))} results across {len(results.get('collections_searched', []))} collections:")
        typer.echo(f"Collections searched: {', '.join(results.get('collections_searched', []))}")
        typer.echo("")
        
        for i, result in enumerate(results.get("results", []), 1):
            collection = result.get("collection", "unknown")
            score = result.get("score", 0)
            doc_id = result.get("_id", "unknown")
            content = result.get("content", "")
            
            typer.echo(f"{i}. [{collection}] Score: {score:.4f}")
            typer.echo(f"   ID: {doc_id}")
            # Truncate long content for display
            if len(content) > 100:
                content = content[:100] + "..."
            typer.echo(f"   Content: {content}")
            typer.echo("")
    
    @search_cli.command("messages")
    def search_messages_cmd(
        query: str = typer.Argument(..., help="Search query text"),
        search_type: str = typer.Option(
            "hybrid", "--type", "-t", help="Search type: hybrid, bm25, or semantic"
        ),
        conversation_id: Optional[str] = typer.Option(
            None, "--conversation", "-c", help="Filter by conversation ID"
        ),
        message_type: Optional[str] = typer.Option(
            None, "--message-type", "-m", 
            help="Filter by message type (USER, AGENT, SYSTEM)"
        ),
        limit: int = typer.Option(
            10, "--limit", "-l", help="Maximum number of results to return"
        ),
        json_output: bool = typer.Option(
            False, "--json", "-j", help="Output results as JSON"
        ),
    ):
        """
        Search for messages in message history.
        """
        client = connect_arango()
        db = ensure_database(client)
        
        results = search_messages(
            db=db,
            query=query,
            search_type=search_type,
            top_n=limit,
            conversation_id=conversation_id,
            message_type=message_type
        )
        
        # Display results
        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return
        
        # Text output
        if not results.get("results"):
            typer.echo(f"No message results found for query: {query}")
            return
        
        typer.echo(f"Found {len(results.get('results', []))} messages:")
        typer.echo("")
        
        for i, result in enumerate(results.get("results", []), 1):
            score = result.get("score", 0)
            msg_id = result.get("_id", "unknown")
            content = result.get("content", "")
            msg_type = result.get("message_type", "unknown")
            conversation = result.get("conversation_id", "unknown")
            
            typer.echo(f"{i}. [{msg_type}] Score: {score:.4f}")
            typer.echo(f"   ID: {msg_id}")
            typer.echo(f"   Conversation: {conversation}")
            # Truncate long content for display
            if len(content) > 100:
                content = content[:100] + "..."
            typer.echo(f"   Content: {content}")
            typer.echo("")
    
    # Add the search CLI group to the app
    app.add_typer(search_cli)

if __name__ == "__main__":
    # Create a standalone CLI app for testing
    app = typer.Typer()
    register_search_commands(app)
    app()
