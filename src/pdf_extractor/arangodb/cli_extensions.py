# src/pdf_extractor/arangodb/cli_extensions.py
import sys
import json
import uuid
from typing import Dict, Any, List, Optional
import typer
from loguru import logger
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME, RELATIONSHIP_TYPE_SIMILAR, RELATIONSHIP_TYPE_SHARED_TOPIC,
    RELATIONSHIP_TYPE_REFERENCES, RELATIONSHIP_TYPE_PREREQUISITE, RELATIONSHIP_TYPE_CAUSAL,
    RATIONALE_MIN_LENGTH, CONFIDENCE_SCORE_RANGE
)
from pdf_extractor.arangodb.relationship_api import (
    add_relationship, delete_relationship, get_relationships
)
from pdf_extractor.arangodb.agent_decision import (
    evaluate_relationship_need, 
    identify_relationship_candidates, 
    create_strategic_relationship
)
from pdf_extractor.arangodb.advanced_query_solution import solve_query
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database

def register_agent_commands(app: typer.Typer) -> None:
    """Register commands for relationship management in CLI app."""
    
    graph_cli = typer.Typer(name="graph", help="Manage document relationships")
    
    @graph_cli.command("add")
    def add_relationship_cmd(
        from_key: str = typer.Argument(..., help="Source document key"),
        to_key: str = typer.Argument(..., help="Target document key"),
        relationship_type: str = typer.Option(
            RELATIONSHIP_TYPE_SIMILAR, "--type", "-t", 
            help="Relationship type"
        ),
        rationale: str = typer.Option(
            "", "--rationale", "-r", 
            help=f"Relationship rationale (min {RATIONALE_MIN_LENGTH} chars)"
        ),
        confidence: int = typer.Option(
            3, "--confidence", "-c",
            help=f"Confidence score ({CONFIDENCE_SCORE_RANGE[0]}-{CONFIDENCE_SCORE_RANGE[1]}, lower is better)",
            min=CONFIDENCE_SCORE_RANGE[0], max=CONFIDENCE_SCORE_RANGE[1]
        ),
        interactive: bool = typer.Option(
            False, "--interactive", "-i",
            help="Use interactive prompts for rationale and confidence"
        )
    ):
        """Add a relationship between two documents."""
        client = connect_arango()
        db = ensure_database(client)
        
        # Check if documents exist
        vertex_collection = db.collection(COLLECTION_NAME)
        if not vertex_collection.has(from_key) or not vertex_collection.has(to_key):
            typer.echo(f"Error: One or both document keys not found")
            sys.exit(1)
        
        # Interactive mode
        if interactive:
            typer.echo(f"Creating relationship: {from_key} -> {to_key}")
            typer.echo(f"Relationship type: {relationship_type}")
            
            # Get document details
            from_doc = vertex_collection.get(from_key)
            to_doc = vertex_collection.get(to_key)
            typer.echo(f"Source document: {from_doc.get('problem', from_key)}")
            typer.echo(f"Target document: {to_doc.get('problem', to_key)}")
            
            # Interactive rationale input
            rationale = typer.prompt(f"Enter rationale (min {RATIONALE_MIN_LENGTH} chars)")
            while len(rationale) < RATIONALE_MIN_LENGTH:
                typer.echo(f"Rationale too short ({len(rationale)} < {RATIONALE_MIN_LENGTH})")
                rationale = typer.prompt(f"Enter rationale")
            
            # Interactive confidence input
            confidence_str = typer.prompt(
                f"Enter confidence score ({CONFIDENCE_SCORE_RANGE[0]}-{CONFIDENCE_SCORE_RANGE[1]}, lower is better)",
                default="3"
            )
            while not confidence_str.isdigit() or int(confidence_str) < CONFIDENCE_SCORE_RANGE[0] or int(confidence_str) > CONFIDENCE_SCORE_RANGE[1]:
                typer.echo(f"Invalid score, must be {CONFIDENCE_SCORE_RANGE[0]}-{CONFIDENCE_SCORE_RANGE[1]}")
                confidence_str = typer.prompt(f"Enter confidence score")
            confidence = int(confidence_str)
        
        # Not interactive, but check rationale length
        elif len(rationale) < RATIONALE_MIN_LENGTH:
            typer.echo(f"Error: Rationale too short ({len(rationale)} < {RATIONALE_MIN_LENGTH})")
            sys.exit(1)
        
        # Add the relationship
        result = add_relationship(db, from_key, to_key, rationale, relationship_type, confidence)
        if result:
            typer.echo(f"Relationship created successfully")
        else:
            typer.echo(f"Failed to create relationship")
            sys.exit(1)
    
    @graph_cli.command("list")
    def list_relationships_cmd(
        doc_key: str = typer.Argument(..., help="Document key"),
        direction: str = typer.Option(
            "ANY", "--direction", "-d",
            help="Direction (INBOUND, OUTBOUND, ANY)"
        ),
        output_format: str = typer.Option(
            "table", "--format", "-f",
            help="Output format (table, json)"
        )
    ):
        """List relationships for a document."""
        client = connect_arango()
        db = ensure_database(client)
        
        # Check if document exists
        vertex_collection = db.collection(COLLECTION_NAME)
        if not vertex_collection.has(doc_key):
            typer.echo(f"Error: Document key not found")
            sys.exit(1)
        
        # Get relationships
        relationships = get_relationships(db, doc_key, direction)
        
        # Display results
        if output_format == "json":
            typer.echo(json.dumps(relationships, indent=2))
        else:
            typer.echo(f"Relationships for document {doc_key} ({len(relationships)} found):")
            if not relationships:
                typer.echo("  No relationships found")
                return
            
            for i, rel in enumerate(relationships, 1):
                rel_type = rel.get("type", "UNKNOWN")
                rel_from = rel.get("_from", "").split("/")[-1]
                rel_to = rel.get("_to", "").split("/")[-1]
                rel_score = rel.get("confidence_score", "N/A")
                
                typer.echo(f"  {i}. {rel_from} --[{rel_type} ({rel_score})]-> {rel_to}")
                typer.echo(f"     Rationale: {rel.get('rationale', '')[:50]}...")
    
    @graph_cli.command("delete")
    def delete_relationship_cmd(
        edge_key: str = typer.Argument(..., help="Edge key to delete"),
        force: bool = typer.Option(
            False, "--force", "-f",
            help="Delete without confirmation"
        )
    ):
        """Delete a relationship."""
        client = connect_arango()
        db = ensure_database(client)
        
        if not force:
            confirm = typer.confirm(f"Are you sure you want to delete relationship {edge_key}?")
            if not confirm:
                typer.echo("Deletion cancelled")
                return
        
        result = delete_relationship(db, edge_key)
        if result:
            typer.echo(f"Relationship {edge_key} deleted successfully")
        else:
            typer.echo(f"Failed to delete relationship {edge_key}")
            sys.exit(1)
    
    @graph_cli.command("evaluate-need")
    def evaluate_relationship_need_cmd(
        query: str = typer.Argument(..., help="Query text")
    ):
        """Evaluate if relationships are needed for a query."""
        client = connect_arango()
        db = ensure_database(client)
        
        result = evaluate_relationship_need(db, query)
        
        typer.echo(f"Relationship need score: {result['need_score']}/10")
        typer.echo(f"Explanation: {result['explanation']}")
        typer.echo(f"Search results count: {result['search_results'].get('count', 0)}")
    
    @graph_cli.command("suggest")
    def suggest_relationships_cmd(
        query: str = typer.Argument(..., help="Query text"),
        limit: int = typer.Option(
            5, "--limit", "-n",
            help="Maximum number of suggestions"
        )
    ):
        """Suggest potential relationships based on search results."""
        client = connect_arango()
        db = ensure_database(client)
        
        # Get search results
        result = evaluate_relationship_need(db, query)
        search_results = result["search_results"]
        
        # Get suggestions
        suggestions = identify_relationship_candidates(db, query, search_results)
        
        # Display results
        typer.echo(f"Found {len(suggestions)} potential relationships:")
        for i, suggestion in enumerate(suggestions[:limit], 1):
            from_key = suggestion.get("from_key", "")
            to_key = suggestion.get("to_key", "")
            rel_type = suggestion.get("suggested_type", "UNKNOWN")
            score = suggestion.get("score", 0.0)
            
            typer.echo(f"  {i}. {from_key} --[{rel_type} ({score:.2f})]-> {to_key}")
            typer.echo(f"     Explanation: {suggestion.get('explanation', '')}")
    
    @graph_cli.command("query")
    def query_cmd(
        query: str = typer.Argument(..., help="Query text"),
        use_relationships: bool = typer.Option(
            True, "--use-relationships/--no-relationships",
            help="Whether to use relationships in query"
        )
    ):
        """Run a query with relationship awareness."""
        client = connect_arango()
        db = ensure_database(client)
        
        results = solve_query(db, query, use_relationships=use_relationships)
        
        typer.echo(f"Query results (count: {results.get('count', 0)}, attempt: {results.get('attempt', 0)}):")
        for i, result in enumerate(results.get("results", []), 1):
            doc = result.get("doc", {})
            doc_key = doc.get("_key", "")
            doc_content = doc.get("content", "")[:50]
            
            if "relationship" in result:
                rel = result.get("relationship", {})
                rel_type = rel.get("type", "")
                source = rel.get("source_key", "")
                typer.echo(f"  {i}. {doc_key} (via {rel_type} from {source}): {doc_content}...")
            else:
                typer.echo(f"  {i}. {doc_key}: {doc_content}...")
    
    # Add the graph commands to the main CLI app
    app.add_typer(graph_cli)

def create_test_app():
    """Create a test application."""
    app = typer.Typer()
    register_agent_commands(app)
    return app

# Test function for CLI extensions
def test_cli_extensions():
    """Test if CLI extensions import correctly."""
    app = typer.Typer()
    register_agent_commands(app)
    return True

if __name__ == "__main__":
    app = create_test_app()
    app()
        - Provide the conversation ID as the main argument
        - Use --limit to control the number of messages returned
        - Use --offset to paginate through large conversations
        - Use --sort to choose chronological (asc) or reverse (desc) order
        """
        # Connect to the database
        client = connect_arango()
        db = ensure_database(client)
        
        # Get messages
        results = get_conversation_messages(db, conversation_id, limit, offset, sort)
        
        # Output the result
        if json_output:
            console.print(json.dumps(results))
        else:
            # Create a table for the messages
            table = Table(title=f"Conversation: {conversation_id}")
            table.add_column("#", style="dim")
            table.add_column("Key", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Timestamp", style="yellow")
            table.add_column("Content", style="green")
            
            for i, msg in enumerate(results, offset + 1):
                # Format content for better display
                content = msg.get("content", "")
                content_preview = content if len(content) < 50 else content[:47] + "..."
                
                table.add_row(
                    str(i),
                    msg.get("_key", "N/A"),
                    msg.get("message_type", "N/A"),
                    msg.get("timestamp", "N/A"),
                    content_preview
                )
            
            console.print(table)
            console.print(f"[dim]Showing {len(results)} of {len(results)} messages[/dim]")
    
    @message_history.command("delete-conversation")
    def delete_conversation_cmd(
        conversation_id: str = typer.Argument(
            ...,
            help="ID of the conversation to delete"
        ),
        confirm: bool = typer.Option(
            False,
            "--yes", "-y",
            help="Skip confirmation prompt"
        )
    ):
        """
        [Action] Delete an entire conversation with all its messages.
        
        WHEN TO USE: When you need to remove an entire conversation and all its
        messages from the database.
        
        WHY TO USE: For bulk removal of conversations that are no longer needed,
        contain sensitive information, or need to be purged for compliance reasons.
        
        HOW TO USE:
        - Provide the conversation ID as the main argument
        - Use --yes to skip the confirmation prompt
        """
        # Connect to the database
        client = connect_arango()
        db = ensure_database(client)
        
        # Confirm deletion
        if not confirm:
            # Get message count first
            count_query = f"""
            FOR doc IN claude_message_history
            FILTER doc.conversation_id == @conversation_id
            COLLECT WITH COUNT INTO message_count
            RETURN message_count
            """
            cursor = db.aql.execute(count_query, bind_vars={"conversation_id": conversation_id})
            message_count = next(cursor, 0)
            
            console.print(f"[yellow]Warning: This will delete the conversation '{conversation_id}' with {message_count} messages.[/yellow]")
            confirm = typer.confirm("Are you sure you want to continue?")
            if not confirm:
                console.print("[yellow]Operation cancelled.[/yellow]")
                return
        
        # Delete the conversation
        result = delete_conversation(db, conversation_id)
        
        # Output the result
        if result:
            console.print(f"[bold green]✓ Conversation deleted successfully: {conversation_id}[/bold green]")
        else:
            console.print(f"[bold red]✗ Failed to delete conversation: {conversation_id}[/bold red]")
            sys.exit(1)
    
    # Add commands for document-message relationships
    
    @message_history.command("link-document")
    def link_message_to_document_cmd(
        message_key: str = typer.Option(
            ...,
            "--message", "-m",
            help="Key of the message"
        ),
        document_key: str = typer.Option(
            ...,
            "--document", "-d",
            help="Key of the document"
        ),
        rationale: str = typer.Option(
            "Message references document",
            "--rationale", "-r",
            help="Explanation for the relationship"
        ),
        json_output: bool = typer.Option(
            False,
            "--json", "-j",
            help="Output result as JSON"
        )
    ):
        """
        [Action] Create a relationship between a message and a document.
        
        WHEN TO USE: When a message refers to or is related to a specific document,
        and you want to track this relationship in the graph database.
        
        WHY TO USE: To build a knowledge graph that connects conversations and documents,
        enabling advanced queries that can traverse these connections.
        
        HOW TO USE:
        - Provide the message key and document key
        - Add a rationale explaining the relationship
        - The relationship is directional (message → document)
        """
        # Connect to the database
        client = connect_arango()
        db = ensure_database(client)
        
        # Create the relationship
        result = link_message_to_document(db, message_key, document_key, RELATIONSHIP_TYPE_REFERS_TO, rationale)
        
        # Output the result
        if json_output:
            console.print(json.dumps(result))
        else:
            if result:
                console.print(f"[bold green]✓ Message-document relationship created[/bold green]")
                console.print(f"From message: {message_key}")
                console.print(f"To document: {document_key}")
            else:
                console.print("[bold red]✗ Failed to create relationship[/bold red]")
                sys.exit(1)
    
    @message_history.command("document-references")
    def get_documents_for_message_cmd(
        message_key: str = typer.Argument(
            ...,
            help="Key of the message"
        ),
        json_output: bool = typer.Option(
            False,
            "--json", "-j",
            help="Output result as JSON"
        )
    ):
        """
        [Query] Find all documents referenced by a message.
        
        WHEN TO USE: When you need to identify which documents are connected to a specific
        message in the conversation history.
        
        WHY TO USE: To discover document relationships, analyze which documents were
        referenced in a conversation, or verify that relationships were correctly created.
        
        HOW TO USE:
        - Provide the message key as the main argument
        - The command will return all documents linked to that message
        """
        # Connect to the database
        client = connect_arango()
        db = ensure_database(client)
        
        # Get documents
        results = get_documents_for_message(db, message_key)
        
        # Output the result
        if json_output:
            console.print(json.dumps(results))
        else:
            if results:
                # Create a table for the documents
                table = Table(title=f"Documents Referenced by Message: {message_key}")
                table.add_column("#", style="dim")
                table.add_column("Key", style="cyan")
                table.add_column("Content", style="green")
                table.add_column("Tags", style="yellow")
                
                for i, doc in enumerate(results, 1):
                    # Format content for better display
                    content = doc.get("content", "")
                    content_preview = content if len(content) < 50 else content[:47] + "..."
                    
                    # Format tags
                    tags = doc.get("tags", [])
                    tags_str = ", ".join(tags) if tags else "N/A"
                    
                    table.add_row(
                        str(i),
                        doc.get("_key", "N/A"),
                        content_preview,
                        tags_str
                    )
                
                console.print(table)
                console.print(f"Found {len(results)} documents referenced by this message")
            else:
                console.print(f"No documents found for message: {message_key}")
    
    @message_history.command("message-references")
    def get_messages_for_document_cmd(
        document_key: str = typer.Argument(
            ...,
            help="Key of the document"
        ),
        json_output: bool = typer.Option(
            False,
            "--json", "-j",
            help="Output result as JSON"
        )
    ):
        """
        [Query] Find all messages that reference a document.
        
        WHEN TO USE: When you need to identify which messages are connected to a specific
        document in the database.
        
        WHY TO USE: To discover which conversations mention a document, analyze document
        usage patterns, or verify that relationships were correctly created.
        
        HOW TO USE:
        - Provide the document key as the main argument
        - The command will return all messages linked to that document
        """
        # Connect to the database
        client = connect_arango()
        db = ensure_database(client)
        
        # Get messages
        results = get_messages_for_document(db, document_key)
        
        # Output the result
        if json_output:
            console.print(json.dumps(results))
        else:
            if results:
                # Create a table for the messages
                table = Table(title=f"Messages Referencing Document: {document_key}")
                table.add_column("#", style="dim")
                table.add_column("Key", style="cyan")
                table.add_column("Type", style="magenta")
                table.add_column("Conversation ID", style="blue")
                table.add_column("Content", style="green")
                
                for i, msg in enumerate(results, 1):
                    # Format content for better display
                    content = msg.get("content", "")
                    content_preview = content if len(content) < 50 else content[:47] + "..."
                    
                    table.add_row(
                        str(i),
                        msg.get("_key", "N/A"),
                        msg.get("message_type", "N/A"),
                        msg.get("conversation_id", "N/A"),
                        content_preview
                    )
                
                console.print(table)
                console.print(f"Found {len(results)} messages referencing this document")
            else:
                console.print(f"No messages found for document: {document_key}")


# Register the agent commands when this module is imported
if __name__ == "__main__":
    app = typer.Typer()
    register_agent_commands(app)
    app()
