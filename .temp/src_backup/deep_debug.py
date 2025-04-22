#!/usr/bin/env python3
'''
Deep debugging of PDF extraction and table merging
'''
import sys
import traceback
import logging
from pathlib import Path

# Configure more detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sys.path.append('/home/graham/workspace/experiments/pdf_extractor/src')
from table_extractor import extract_tables
import camelot.io as camelot
from improved_table_merger import process_and_merge_tables

def debug_individual_components():
    Debug each component separately to identify where the issue is
    pdf_path = Path('/home/graham/workspace/experiments/pdf_extractor/src/input/BHT_CV32A65X.pdf')
    
    # Step 1: Extract with camelot directly
    print('\n1. DIRECT CAMELOT EXTRACTION')
    print('============================')
    try:
        tables = camelot.read_pdf(
            str(pdf_path),
            pages='1-2',
            flavor='lattice',
            line_scale=15,
            process_background=True
        )
        print(f'Camelot extracted {len(tables)} tables')
        for i, table in enumerate(tables):
            print(f'  Table {i+1}: page={table.page}, accuracy={table.parsing_report.get(accuracy, 0)}')
            print(f'  Data shape: {len(table.data)}x{len(table.data[0]) if table.data else 0}')
        
        # Step 2: Process camelot tables to dictionaries
        print('\n2. PROCESSING TABLES TO DICTIONARIES')
        print('===================================')
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
            except (AttributeError, TypeError):
                print(f'  No bounding box for table on page {table.page}')
            processed_tables.append(processed)
        
        print(f'Processed {len(processed_tables)} tables into dictionaries')
        
        # Step 3: Test table merging directly
        print('\n3. TESTING TABLE MERGING DIRECTLY')
        print('================================')
        for strategy in ['conservative', 'aggressive', 'none']:
            print(f'  Strategy: {strategy}')
            try:
                merged_tables = process_and_merge_tables(tables) 
                print(f'  Merged tables directly: {len(merged_tables)} result tables')
            except Exception as e:
                print(f'  Error with direct merging: {e}')
                traceback.print_exc()
            
            try:
                from improved_table_merger import merge_multi_page_tables
                threshold = 0.8 if strategy == 'conservative' else 0.6 if strategy == 'aggressive' else 1.0
                merged_processed = merge_multi_page_tables(processed_tables, threshold) 
                print(f'  Merged processed dictionaries: {len(merged_processed)} result tables')
            except Exception as e:
                print(f'  Error with processed merging: {e}')
                traceback.print_exc()
    
    except Exception as e:
        print(f'Error in camelot extraction: {e}')
        traceback.print_exc()

if __name__ == __main__:
    debug_individual_components()
