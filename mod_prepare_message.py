#!/usr/bin/env python3
# mod_prepare_message.py
"""
This script fixes the syntax error in message_history_api.py and adds embedding functionality
"""

import sys
import re

def fix_prepare_message(file_path):
    """Fix the prepare_message function in the specified file"""
    with open(file_path, "r") as f:
        content = f.read()
    
    # Find the prepare_message function and fix the syntax error
    pattern = r"(\s+message = \{\s+.*?"metadata": metadata or \{\})(\s+)("embedding"|\})"
    replacement = r"\1,\2\3"
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # If we didnt find the pattern, maybe its already fixed or has a different issue
    if new_content == content:
        # Try another approach - explicitly check for the embedding field
        if "embedding: embedding" not in content and """embedding": embedding" not in content:
            # Embedding field is missing, add it
            pattern = r"(\s+"metadata": metadata or \{\})(\s+\})"
            replacement = r"\1,
        "embedding": embedding  # Add embedding for semantic search\2"
            new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(file_path, "w") as f:
        f.write(new_content)
    
    print(f"Updated {file_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mod_prepare_message.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    success = fix_prepare_message(file_path)
    sys.exit(0 if success else 1)

