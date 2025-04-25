#!/usr/bin/env python3
# update_message_api.py

import re
import sys

def update_message_api(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the missing comma between metadata and embedding in the message dictionary
    content = content.replace('metadata: metadata or {}', 'metadata: metadata or {},')
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f'Updated {file_path}')
    return True

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python update_message_api.py <file_path>')
        sys.exit(1)
    
    file_path = sys.argv[1]
    if update_message_api(file_path):
        print('✅ Successfully updated the file')
        sys.exit(0)
    else:
        print('❌ Failed to update the file')
        sys.exit(1)
