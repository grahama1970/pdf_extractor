#!/usr/bin/env python3
'''
Debug the table extraction process
'''
import sys
import traceback
from pathlib import Path

sys.path.append('/home/graham/workspace/experiments/pdf_extractor/src')
from table_extractor import extract_tables

# Set up test file
pdf_path = Path('/home/graham/workspace/experiments/pdf_extractor/src/input/BHT_CV32A65X.pdf')

try:
    print('Attempting table extraction...')
    tables = extract_tables(pdf_path, pages='1-2', flavor='lattice')
    print(f'Successfully extracted {len(tables)} tables')
except Exception as e:
    print(f'Error during extraction: {e}')
    traceback.print_exc()
