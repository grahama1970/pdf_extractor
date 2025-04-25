#!/usr/bin/env python3
# src/pdf_extractor/arangodb/cli_integration.py

import typer
from pdf_extractor.arangodb.search_cli import register_search_commands

def register_cli_commands() -> typer.Typer:
    """
    Register all CLI commands for the PDF extractor.
    """
    app = typer.Typer(help="PDF Extractor CLI")
    
    # Register search commands
    register_search_commands(app)
    
    return app

if __name__ == "__main__":
    app = register_cli_commands()
    app()
