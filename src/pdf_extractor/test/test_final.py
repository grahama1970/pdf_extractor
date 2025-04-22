#!/usr/bin/env python3
'''
Simple test script for the improved table merger
'''

import sys
from pathlib import Path

# Add the parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the table merger
from pdf_extractor.improved_table_merger import process_and_merge_tables

def main():
    '''Test the improved table merger with synthetic data'''
    # Create test tables
    tables = [
        {
            'page': 1,
            'data': [
                ['Header1', 'Header2'],
                ['Value1', 'Value2']
            ]
        },
        {
            'page': 2,
            'data': [
                ['Header1', 'Header2'],
                ['Value3', 'Value4']
            ]
        }
    ]
    
    print('Testing improved table merger...')
    print('Original tables:', len(tables))
    
    # Test with conservative strategy
    result1 = process_and_merge_tables(tables, merge_strategy='conservative')
    print('Conservative strategy result:', len(result1))
    
    # Test with aggressive strategy
    result2 = process_and_merge_tables(tables, merge_strategy='aggressive')
    print('Aggressive strategy result:', len(result2))
    
    # Test with no merging
    result3 = process_and_merge_tables(tables, merge_strategy='none')
    print('None strategy result:', len(result3))
    
    print('All tests completed successfully!')
    return 0

if __name__ == '__main__':
    sys.exit(main())
