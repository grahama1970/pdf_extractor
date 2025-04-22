#!/usr/bin/env python3
'''
Script to update table_extractor.py with improved table merger functionality.
'''
import os
import re
from pathlib import Path

# Path to table_extractor.py
table_extractor_path = Path('src/table_extractor.py')

# Make a backup
backup_path = table_extractor_path.with_suffix('.py.bak2')
os.system(f'cp {table_extractor_path} {backup_path}')
print(f'Created backup: {backup_path}')

# Read the file
with open(table_extractor_path, 'r') as f:
    content = f.read()

# Update 1: Add improved_table_merger import
import_pattern = r'try:\s+import camelot\.io as camelot.*?CAMELOT_AVAILABLE = False'
import_replacement = '''try:
    import camelot.io as camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    logger.warning(camelot-py not found. Table extraction will not be available.)
    camelot = None
    CAMELOT_AVAILABLE = False

# Import our improved table merger
try:
    from improved_table_merger import process_and_merge_tables
    IMPROVED_MERGER_AVAILABLE = True
except ImportError:
    logger.warning(improved_table_merger not found. Using basic table processing.)
    IMPROVED_MERGER_AVAILABLE = False'''

content = re.sub(import_pattern, import_replacement, content, flags=re.DOTALL)

# Update 2: Add multi-page merging to extract_tables
return_pattern = r'(\s+results\.append\(table_data\)\s+)(\s+return results)'
return_replacement = r'''\1
        # Apply multi-page table merging if available
        if len(results) > 1 and IMPROVED_MERGER_AVAILABLE:
            logger.info(Applying multi-page table merging)
            original_count = len(results)
            
            # Use our improved table merger
            results = process_and_merge_tables(tables)
            
            merged_count = len(results)
            if merged_count < original_count:
                logger.info(fMerged {original_count - merged_count} multi-page tables)
\2'''

content = re.sub(return_pattern, return_replacement, content, flags=re.DOTALL)

# Update 3: Update docstring to mention table merging
docstring_pattern = r'(This module provides functions for extracting tables from PDFs using Camelot-py,\s+with fallback mechanisms and validation\. It supports both \'lattice\' and \'stream\'\s+extraction methods and includes automatic fallback for low-confidence tables\.)'
docstring_replacement = r'\1\nIt also handles multi-page table detection and merging.'

content = re.sub(docstring_pattern, docstring_replacement, content, flags=re.DOTALL)

# Write the updated content
with open(table_extractor_path, 'w') as f:
    f.write(content)

print(f'Updated {table_extractor_path} with improved table merger functionality.')
