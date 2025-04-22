#!/usr/bin/env python3
import sys
import traceback
from pathlib import Path

sys.path.append('/home/graham/workspace/experiments/pdf_extractor/src')
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
        processed = {
            page: int(table.page),
            data: table.data,
            accuracy: table.parsing_report.get(accuracy, 0),
            rows: len(table.data),
            cols: len(table.data[0]) if table.data else 0,
        }
        try:
            processed[bbox] = tuple(table._bbox)
        except:
            pass
        processed_tables.append(processed)
    
    print(f'   - Processed {len(processed_tables)} table dictionaries')
    
    # Step 3: Test merging
    print('\n3. Testing table merging')
    
    # First try with our improved_table_merger
    from improved_table_merger import merge_multi_page_tables
    merged = merge_multi_page_tables(processed_tables, 0.8)
    print(f'   - Merged tables result: {len(merged)} tables')
    
    # Try passing Camelot tables directly
    try:
        print('\n4. Testing with tables argument directly')
        result = process_and_merge_tables(tables)
        print(f'   - Direct merging result: {len(result)} tables')
    except Exception as e:
        print(f'   - Error when passing tables directly: {e}')
        print('   - Traceback:')
        traceback.print_exc()
        print('\n   Analyzing error:')
        print('   - Expected dictionary with keys: page, data, etc.')
        print('   - Received camelot table object')
        print('   - This indicates process_and_merge_tables needs processed dicts, not raw camelot tables')
        
    # Update fix for table_extractor.py
    print('\n5. Suggested fix:')
    print('   - In table_extractor.py, change:')
    print('     results = process_and_merge_tables(tables, merge_strategy=merge_strategy)')
    print('   - To:')
    print('     processed_tables = [_process_table(table) for table in tables]')
    print('     results = process_and_merge_tables(processed_tables, merge_strategy=merge_strategy)')
        
except Exception as e:
    print(f'Error: {e}')
    traceback.print_exc()
