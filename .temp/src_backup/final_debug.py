#!/usr/bin/env python3
import sys
import traceback
from pathlib import Path

sys.path.append('/home/graham/workspace/experiments/pdf_extractor/src')
from table_extractor import _process_table
import camelot.io as camelot
from improved_table_merger import process_and_merge_tables

# Configuration
pdf_path = Path('/home/graham/workspace/experiments/pdf_extractor/src/input/BHT_CV32A65X.pdf')

print('DEBUGGING PDF EXTRACTION AND TABLE MERGING')
print('=======================================')

# Step 1: Extract tables with camelot
try:
    print('\n1. Extracting tables with camelot')
    tables = camelot.read_pdf(
        str(pdf_path),
        pages='1-2',
        flavor='lattice',
        line_scale=15,
        process_background=True
    )
    
    print(f'   - Extracted {len(tables)} tables')
    for i, table in enumerate(tables):
        print(f'   - Table {i+1}: page={table.page}, shape={len(table.data)}x{len(table.data[0]) if table.data else 0}')
        
    # Step 2: Process tables into dictionaries
    print('\n2. Processing tables to dictionaries')
    processed_tables = []
    for table in tables:
        processed = _process_table(table)
        processed_tables.append(processed)
    
    print(f'   - Processed {len(processed_tables)} table dictionaries')
    
    # Step 3: Test merging
    print('\n3. Testing table merging with different strategies')
    for strategy in ['conservative', 'aggressive', 'none']:
        print(f'\nStrategy: {strategy}')
        try:
            # Define threshold based on strategy
            threshold = 0.8 if strategy == 'conservative' else 0.6 if strategy == 'aggressive' else 1.0
            
            # Import directly to avoid any confusion
            from improved_table_merger import merge_multi_page_tables
            merged = merge_multi_page_tables(processed_tables, threshold)
            print(f'   - Result: {len(merged)} tables after merging')
            
            # Show details of merged tables
            for i, table in enumerate(merged):
                print(f'   - Merged table {i+1}:')
                print(f'     Page: {table.get(page, unknown)}')
                print(f'     Rows: {table.get(rows, 0)}')
                print(f'     Is multi-page: {table.get(is_multi_page, False)}')
                if page_range in table:
                    print(f'     Page range: {table.get(page_range, )}')
                    
        except Exception as e:
            print(f'   - Error testing strategy {strategy}: {e}')
            traceback.print_exc()
    
    # Step 4: Identify fix for table_extractor.py
    print('\n4. Fix for table_extractor.py:')
    print('   1. Process tables before merging:')
    print('      processed_tables = [_process_table(table) for table in tables]')
    print('      results = process_and_merge_tables(processed_tables, merge_strategy=merge_strategy)')
    print('   2. OR: Update process_and_merge_tables to handle camelot tables directly')
    
except Exception as e:
    print(f'Error: {e}')
    traceback.print_exc()
