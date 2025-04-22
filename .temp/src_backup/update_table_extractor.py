#!/usr/bin/env python3
"""
Manual update script for table_extractor.py
"""

import os

# Backup the file
os.system('cp src/table_extractor.py src/table_extractor.py.bak4')

# Read the original file
with open('src/table_extractor.py', 'r') as f:
    lines = f.readlines()

# Update function signature to add merge_strategy parameter
new_lines = []
function_def_found = False

for i, line in enumerate(lines):
    if 'def extract_tables(' in line and not function_def_found:
        function_def_found = True
        new_lines.append('def extract_tables(\n')
        new_lines.append('    pdf_path: Union[str, Path], \n')
        new_lines.append('    pages: Union[str, List[int]] = "all",\n')
        new_lines.append('    flavor: str = CAMELOT_DEFAULT_FLAVOR,\n')
        new_lines.append('    merge_strategy: str = "conservative",\n')
        new_lines.append(') -> List[Dict[str, Any]]:\n')
        
        # Skip the original function definition lines
        while i < len(lines) and lines[i].strip() != '":' and '-> List[Dict[str, Any]]:' not in lines[i]:
            i += 1
            
        # Skip the return type line if we haven't already
        if i < len(lines) and '-> List[Dict[str, Any]]:' in lines[i]:
            i += 1
            
        continue
    
    # Update the function docstring
    if function_def_found and '    Args:' in line:
        new_lines.append(line)
        new_lines.append('        pdf_path: Path to PDF file\n')
        new_lines.append('        pages: Page numbers to process ("all" or list of numbers)\n')
        new_lines.append('        flavor: Table extraction method ("lattice" or "stream")\n')
        new_lines.append('        merge_strategy: Strategy for table merging ("aggressive", "conservative", "none")\n')
        
        # Skip the original argument description lines
        while i < len(lines) and '    Returns:' not in lines[i]:
            i += 1
            
        continue
    
    # Update the multi-page table merging code
    if 'Applying multi-page table merging' in line:
        new_lines.append('        # Apply multi-page table merging if available\n')
        new_lines.append('        if len(results) > 1 and IMPROVED_MERGER_AVAILABLE:\n')
        new_lines.append(f'            logger.info(f"Applying multi-page table merging with strategy: {{merge_strategy}}")\n')
        new_lines.append('            original_count = len(results)\n')
        new_lines.append('            \n')
        new_lines.append('            # Use our improved table merger with specified strategy\n')
        new_lines.append('            results = process_and_merge_tables(tables, merge_strategy=merge_strategy)\n')
        new_lines.append('            \n')
        new_lines.append('            merged_count = len(results)\n')
        new_lines.append('            if merged_count < original_count:\n')
        new_lines.append('                logger.info(f"Merged {original_count - merged_count} multi-page tables")\n')
        
        # Skip the original merging code lines
        while i < len(lines) and 'return results' not in lines[i]:
            i += 1
            
        continue
    
    # Keep other lines unchanged
    new_lines.append(line)

# Write the updated file
with open('src/table_extractor.py', 'w') as f:
    f.writelines(new_lines)

print('Table extractor updated successfully')
