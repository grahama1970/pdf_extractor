#!/usr/bin/env python3
# test_cli_extensions.py - Test script for CLI search extensions

import sys
import subprocess
from pathlib import Path

def test_search_cli():
    Test the search CLI commands.
    print(Testing search CLI commands...)
    
    # Define the test query
    test_query = test query
    
    # Get the path to the search_cli.py script
    script_path = Path(src/pdf_extractor/arangodb/search_cli.py)
    if not script_path.exists():
        print(f❌ Error: Script not found at {script_path})
        return False
    
    # Test running the script with --help
    print(nTesting help command...)
    try:
        result = subprocess.run([sys.executable, str(script_path), --help], 
                               capture_output=True, text=True, check=True)
        print(f✅ Help command executed successfully)
        print(fOutput preview:n{result.stdout[:200]}...)
    except subprocess.CalledProcessError as e:
        print(f❌ Error executing help command: {e})
        print(fOutput: {e.stdout})
        print(fError: {e.stderr})
        return False
    
    # Test the 'messages' command with --help
    print(nTesting messages command help...)
    try:
        result = subprocess.run([sys.executable, str(script_path), messages, --help], 
                               capture_output=True, text=True, check=True)
        print(f✅ Messages help command executed successfully)
        print(fOutput preview:n{result.stdout[:200]}...)
    except subprocess.CalledProcessError as e:
        print(f❌ Error executing messages help command: {e})
        print(fOutput: {e.stdout})
        print(fError: {e.stderr})
        return False
    
    # Test the 'unified' command with --help
    print(nTesting unified command help...)
    try:
        result = subprocess.run([sys.executable, str(script_path), unified, --help], 
                               capture_output=True, text=True, check=True)
        print(f✅ Unified help command executed successfully)
        print(fOutput preview:n{result.stdout[:200]}...)
    except subprocess.CalledProcessError as e:
        print(f❌ Error executing unified help command: {e})
        print(fOutput: {e.stdout})
        print(fError: {e.stderr})
        return False
    
    print(n✅ All CLI help tests passed!)
    
    # Note: We don't actually run the search commands here as they would
    # require a database connection and test data
    print(nNote: Actual search command execution tests require a database connection.)
    print(To manually test the commands with actual data, run:)
    print(f python {script_path} unified your query)
    print(f python {script_path} messages your query)
    
    return True

if __name__ == __main__:
    success = test_search_cli()
    sys.exit(0 if success else 1)
