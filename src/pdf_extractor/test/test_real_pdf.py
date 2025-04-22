#!/usr/bin/env python3
'''
Test improved table merger with real PDF examples from the input directory.
'''

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import necessary modules
from pdf_extractor.table_extractor import TableExtractor

def test_pdf(pdf_path, merge_strategy='conservative'):
    '''Test table extraction and merging with a real PDF.'''
    print(f'Testing PDF: {pdf_path}')
    print(f'Merge strategy: {merge_strategy}')
    
    # Create extractor with specified strategy
    extractor = TableExtractor(merge_strategy=merge_strategy)
    
    # Extract tables
    result = extractor.extract_tables(pdf_path)
    
    # Print results
    table_count = len(result['tables'])
    multi_page_count = len(result.get('multi_page_tables', []))
    print(f'Extracted {table_count} tables')
    print(f'Detected {multi_page_count} multi-page tables')
    
    return result

def main():
    '''Run tests with real PDFs.'''
    # Get a list of sample PDFs
    input_dir = Path(__file__).parent.parent.parent / 'input'
    sample_pdfs = list(input_dir.glob('*.pdf'))
    
    if not sample_pdfs:
        print('No sample PDFs found in the input directory')
        return 1
    
    # Use the first PDF for testing
    sample_pdf = str(sample_pdfs[0])
    print(f'Using sample PDF: {sample_pdf}')
    
    # Test with different strategies
    for strategy in ['conservative', 'aggressive', 'none']:
        print(f'\nTesting with {strategy} strategy:')
        result = test_pdf(sample_pdf, merge_strategy=strategy)
        
        # Save results to output directory
        output_dir = Path(__file__).parent.parent.parent.parent / 'output'
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f'result_{strategy}.json'
        print(f'Saving results to {output_file}')
        
        # We can't directly serialize the result, so we'll just save the counts
        summary = {
            'pdf': sample_pdf,
            'strategy': strategy,
            'table_count': len(result['tables']),
            'multi_page_count': len(result.get('multi_page_tables', [])),
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    print('\nAll tests completed successfully!')
    return 0

if __name__ == '__main__':
    sys.exit(main())
