#!/usr/bin/env python3
'''
Test for the improved table merger with synthetic test data.
'''

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the improved table merger
from pdf_extractor.improved_table_merger import process_and_merge_tables

def create_test_tables():
    '''Create a set of synthetic test tables.'''
    return [
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
                ['Value4', 'Value5', 'Value6'],
                ['Value7', 'Value8', 'Value9']
            ]
        },
        {
            'page': 3,
            'data': [
                ['Different', 'Headers', 'Here'],
                ['Data1', 'Data2', 'Data3']
            ]
        }
    ]

def main():
    '''Run tests with synthetic data for each merge strategy.'''
    print('Testing improved table merger with synthetic data...')
    
    # Create test tables
    tables = create_test_tables()
    print("  {0}: {1} tables -> {2} tables".format(strategy, result["original_count"], result["merged_count"]))
    
    # Test with different strategies
    results = {}
    for strategy in ['conservative', 'aggressive', 'none']:
        print(f'\nTesting with {strategy} strategy:')
        
        # Use improved table merger
        merged_tables = process_and_merge_tables(tables, merge_strategy=strategy)
        print("  {0}: {1} tables -> {2} tables".format(strategy, result["original_count"], result["merged_count"]))
        
        # Store results
        results[strategy] = {
            'strategy': strategy,
            'original_count': len(tables),
            'merged_count': len(merged_tables)
        }
        
        # Save results to output directory
        output_dir = Path(__file__).parent.parent.parent.parent / 'output'
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f'synthetic_merger_{strategy}.json'
        print(f'Saving results to {output_file}')
        
        with open(output_file, 'w') as f:
            json.dump(results[strategy], f, indent=2)
    
    # Final summary
    print('\nMerge Strategy Results:')
    for strategy, result in results.items():
        print("  {0}: {1} tables -> {2} tables".format(strategy, result["original_count"], result["merged_count"]))
    
    print('\nAll tests completed successfully!')
    return 0

if __name__ == '__main__':
    sys.exit(main())
