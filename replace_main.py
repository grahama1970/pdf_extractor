#!/usr/bin/env python3
# Script to replace the main section in pdf_integration.py

import re

# Get the content up to the if __name__ == __main__: line
with open('src/pdf_extractor/arangodb/pdf_integration.py', 'r') as f:
    content = f.read()

# Find the if __name__ section
pattern = rif __name__ == __main__:
match = re.search(pattern, content)

if match:
    # Keep everything before the if __name__ section
    main_section_start = match.start()
    content_before_main = content[:main_section_start]
    
    # Read the corrected main section
    with open('src/pdf_extractor/arangodb/pdf_integration_main.py', 'r') as f:
        main_section = f.read()
    
    # Combine and write back
    with open('src/pdf_extractor/arangodb/pdf_integration.py', 'w') as f:
        f.write(content_before_main + main_section)
    
    print('Successfully replaced the main section in pdf_integration.py')
else:
    print('Could not find the main section in the file')
