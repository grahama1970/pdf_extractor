#!/usr/bin/env python3

# Read the original file
with open('src/table_extractor.py', 'r') as f:
    content = f.read()

# Create a backup
with open('src/table_extractor.py.final', 'w') as f:
    f.write(content)

# Write a new file with our updates
with open('src/table_extractor.py.new', 'w') as f:
    f.write('''