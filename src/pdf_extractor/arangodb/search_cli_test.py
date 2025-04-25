#!/usr/bin/env python3
"""
Test script for search_cli functionality without database connection
"""
import sys
import json
from typing import Dict, Any, List, Optional
import typer
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

app = typer.Typer()
search_cli = typer.Typer(name="search", help="Search functionality for testing")
app.add_typer(search_cli)

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
    Test search for messages in message history.
    """
    # Print parameters to verify they're correctly processed
    params = {
        "query": query,
        "search_type": search_type,
        "conversation_id": conversation_id,
        "message_type": message_type,
        "limit": limit,
        "json_output": json_output
    }
    
    if json_output:
        print(json.dumps(params, indent=2))
    else:
        print("\n=== Message Search Parameters ===")
        for key, value in params.items():
            print(f"{key}: {value}")

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
    Test unified search across documents and messages.
    """
    # Print parameters to verify they're correctly processed
    params = {
        "query": query,
        "search_type": search_type,
        "collections": collections.split(",") if collections else None,
        "exclude_collections": exclude.split(",") if exclude else None,
        "limit": limit,
        "json_output": json_output
    }
    
    if json_output:
        print(json.dumps(params, indent=2))
    else:
        print("\n=== Unified Search Parameters ===")
        for key, value in params.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    app()
