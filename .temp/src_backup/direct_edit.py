#!/usr/bin/env python3
'''
Direct edit script for table_extractor.py
'''
import sys

def insert_after(lines, target, new_content):
    Insert new content after target line
    for i, line in enumerate(lines):
        if target in line:
            return lines[:i+1] + new_content + lines[i+1:]
    return lines

def add_param_to_function(lines, func_name, param_decl):
    Add parameter to function declaration
    # Find the function
    func_start = -1
    param_end = -1
    for i, line in enumerate(lines):
        if func_name in line and 'def ' in line:
            func_start = i
            break
    
    if func_start == -1:
        return lines
    
    # Find the end of parameters
    for i in range(func_start, len(lines)):
        if ')' in lines[i] and '->' in lines[i]:
            param_end = i
            break
    
    if param_end == -1:
        return lines
    
    # Split the line with closing parenthesis
    line_parts = lines[param_end].split(')')
    # Insert the new parameter before the closing parenthesis
    new_line = line_parts[0] + param_decl + ')' + line_parts[1]
    lines[param_end] = new_line
    
    return lines

# Read the file
with open('src/table_extractor.py', 'r') as f:
    lines = f.readlines()

# Make a backup
with open('src/table_extractor.py.bak5', 'w') as f:
    f.writelines(lines)

# 1. Add the imports for improved_table_merger
import_lines = [
    '# Import improved table merger\n',
    'try:\n',
    '    from improved_table_merger import process_and_merge_tables\n',
    '    IMPROVED_MERGER_AVAILABLE = True\n',
    'except ImportError:\n',
    '    logger.warning(improved_table_merger not found. Using basic table processing.)\n',
    '    IMPROVED_MERGER_AVAILABLE = False\n',
    '\n'
]
lines = insert_after(lines, '# Import camelot', import_lines)

# 2. Add the merge_strategy parameter to extract_tables function
param_decl = ',\n    merge_strategy: str = conservative'
lines = add_param_to_function(lines, 'def extract_tables', param_decl)

# 3. Add documentation for the merge_strategy parameter
doc_lines = ['        merge_strategy: Strategy for table merging (aggressive, conservative, none)\n']
lines = insert_after(lines, 'flavor: Table extraction method', doc_lines)

# 4. Update the multi-page merging code
for i, line in enumerate(lines):
    if 'Apply multi-page table merging' in line:
        merge_start = i
        for j in range(i, len(lines)):
            if 'return results' in lines[j]:
                merge_end = j
                break
        else:
            merge_end = -1
        
        if merge_end != -1:
            # Replace the merging block
            merge_lines = [
                '        # Apply multi-page table merging if available\n',
                '        if len(results) > 1 and IMPROVED_MERGER_AVAILABLE:\n',
                f'            logger.info(fApplying multi-page table merging with strategy: {{merge_strategy}})\n',
                '            original_count = len(results)\n',
                '            \n',
                '            # Use our improved table merger with specified strategy\n',
                '            results = process_and_merge_tables(tables, merge_strategy=merge_strategy)\n',
                '            \n',
                '            merged_count = len(results)\n',
                '            if merged_count < original_count:\n',
                '                logger.info(fMerged {original_count - merged_count} multi-page tables)\n',
                '            \n'
            ]
            lines = lines[:merge_start] + merge_lines + lines[merge_end:]
        break

# Write the updated file
with open('src/table_extractor.py', 'w') as f:
    f.writelines(lines)

print(Successfully updated table_extractor.py with improved table merging capabilities.)
