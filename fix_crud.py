#!/usr/bin/env python3
import sys
import re

def main():
    file_path = 'src/pdf_extractor/arangodb/crud.py'
    
    # Read file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Fix update_document function
    update_pattern = r'result = collection\.update\([^)]*\)'
    update_replacement = '''result = collection.update(
            document=document,  # Pass a single document with _key
            check_rev=check_rev,
            merge=True,
            return_new=return_new
        )'''
    content = content.replace(
        'body=updates',
        'data=updates'
    )
    
    content = re.sub(update_pattern, update_replacement, content)
    
    # Fix replace_document function
    replace_pattern = r'result = collection\.replace\([^)]*\)'
    replace_replacement = '''result = collection.replace(
            document=document,
            check_rev=check_rev,
            return_new=return_new
        )'''
    content = content.replace(
        'body=document',
        'data=document'
    )
    
    content = re.sub(replace_pattern, replace_replacement, content)
    
    # Write updated content
    with open(file_path, 'w') as file:
        file.write(content)
    
    print(Successfully fixed crud.py)

if __name__ == __main__:
    main()
