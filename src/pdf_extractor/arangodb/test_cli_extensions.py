# src/pdf_extractor/arangodb/test_cli_extensions.py
import sys
import typer
from pdf_extractor.arangodb.cli_extensions import register_agent_commands

def test_cli_extensions():
    """Test CLI extensions import."""
    app = typer.Typer()
    register_agent_commands(app)
    print("✅ CLI extensions validation passed")
    return True

if __name__ == "__main__":
    if test_cli_extensions():
        print("✅ CLI extensions test passed")
    else:
        print("❌ CLI extensions test failed")
        sys.exit(1)
