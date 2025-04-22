#!/usr/bin/env python3
'''
Script to fix table_extractor.py
'''

# Read the file
with open('src/table_extractor.py', 'r') as f:
    lines = f.readlines()

# Update docstring
for i, line in enumerate(lines):
    if 'extraction methods and includes automatic fallback for low-confidence tables.' in line:
        lines[i] = line.rstrip() + ' It also handles multi-page table detection and merging.\n'
        break

# Add improved_table_merger import
import_added = False
for i, line in enumerate(lines):
    if 'CAMELOT_AVAILABLE = False' in line:
        import_lines = [
            '\n# Import our improved table merger\n',
            'try:\n',
            '    from improved_table_merger import process_and_merge_tables\n',
            '    IMPROVED_MERGER_AVAILABLE = True\n',
            'except ImportError:\n',
            '    logger.warning("improved_table_merger not found. Using basic table processing.")\n',
            '    IMPROVED_MERGER_AVAILABLE = False\n'
        ]
        lines[i+1:i+1] = import_lines
        import_added = True
        break

# Add multi-page table merging code
for i, line in enumerate(lines):
    if 'results.append(table_data)' in line:
        # Find the return statement
        for j in range(i+1, len(lines)):
            if 'return results' in lines[j]:
                # Add merging code before return
                merge_lines = [
                    '        # Apply multi-page table merging if available\n',
                    '        if len(results) > 1 and IMPROVED_MERGER_AVAILABLE:\n',
                    '            logger.info("Applying multi-page table merging")\n',
                    '            original_count = len(results)\n',
                    '            \n',
                    '            # Use our improved table merger\n',
                    '            results = process_and_merge_tables(tables)\n',
                    '            \n',
                    '            merged_count = len(results)\n',
                    '            if merged_count < original_count:\n',
                    '                logger.info(f"Merged {original_count - merged_count} multi-page tables")\n'
                ]
                lines[j:j] = merge_lines
                break
        break

# Write the updated file
with open('src/table_extractor.py', 'w') as f:
    f.writelines(lines)

print('Updated table_extractor.py successfully')
