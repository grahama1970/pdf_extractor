#!/usr/bin/env python3
'''
Apply the final fix to the improved_table_merger.py and table_extractor.py files
'''
import re

# Step 1: Fix the process_and_merge_tables function in improved_table_merger.py
with open('src/improved_table_merger.py', 'r') as f:
    content = f.read()

# Find the process_and_merge_tables function and update it to work with both types
old_function = '''def process_and_merge_tables(camelot_tables: List[Any], merge_strategy: str = conservative) -> List[Dict[str, Any]]:
    