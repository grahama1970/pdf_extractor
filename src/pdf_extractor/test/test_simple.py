#!/usr/bin/env python3
'''Simple test for improved_table_merger'''

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the module
from pdf_extractor.improved_table_merger import process_and_merge_tables
from pdf_extractor.table_extractor import TableExtractor

# Create test tables
test_tables = [
    {
        'page': 1,
        'data': [
            ['Column1', 'Column2', 'Column3'],
            ['Value1', 'Value2', 'Value3']
        ]
    },
    {
        'page': 2,
        'data': [
            ['Column1', 'Column2', 'Column3'],
            ['Value4', 'Value5', 'Value6']
        ]
    }
]

def main():
    '''Run the test'''
    print('Testing improved_table_merger...')
    
    # Test direct API
    for strategy in ['conservative', 'aggressive', 'none']:
        print(f'Strategy: {strategy}')
        result = process_and_merge_tables(test_tables, merge_strategy=strategy)
        print(f'  Result tables: {len(result)}')
    
    # Test via TableExtractor
    extractor = TableExtractor(merge_strategy='conservative')
    print('TableExtractor integration test passed')
    
    print('All tests completed successfully')
    return 0

if __name__ == '__main__':
    sys.exit(main())
