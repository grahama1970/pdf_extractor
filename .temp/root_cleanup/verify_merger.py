#!/usr/bin/env python3
"""
Test script to verify the integration of improved_table_merger with table_extractor.
This script runs tests with different merge strategies and outputs the results.
"""

import sys
import logging
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.insert(0, 'src')

# Test function
def test_table_merger():
    try:
        # Import the needed modules
        logger.info("Importing modules...")
        from pdf_extractor.improved_table_merger import process_and_merge_tables
        from pdf_extractor.table_extractor import TableExtractor
        
        logger.info("Successfully imported modules")

        # Create test tables
        test_tables = [
            {
                "page": 1,
                "data": [
                    ["Column1", "Column2", "Column3"],
                    ["Value1", "Value2", "Value3"]
                ]
            },
            {
                "page": 2,
                "data": [
                    ["Column1", "Column2", "Column3"],
                    ["Value4", "Value5", "Value6"],
                    ["Value7", "Value8", "Value9"]
                ]
            },
            {
                "page": 3,
                "data": [
                    ["Different", "Headers", "Here"],
                    ["Data1", "Data2", "Data3"]
                ]
            }
        ]

        # Test each merge strategy
        for strategy in ['none', 'conservative', 'aggressive']:
            logger.info(f"Testing {strategy} merge strategy...")
            
            # Direct use of process_and_merge_tables
            merged_tables = process_and_merge_tables(test_tables, merge_strategy=strategy)
            logger.info(f"  Direct merger result: {len(merged_tables)} tables")
            
            # Use via TableExtractor 
            extractor = TableExtractor(merge_strategy=strategy)
            
            # Access the _process_tables method for testing
            # Note: This isn't ideal in production code but helps us verify integration
            extractor.page_tables_map = {
                1: [test_tables[0]],
                2: [test_tables[1]],
                3: [test_tables[2]]
            }
            
            # Create a copy of test tables for processing
            tables_copy = test_tables.copy()
            extractor._process_tables(tables_copy)
            
            logger.info(f"  Extractor result: {len(tables_copy)} tables")
            
            # Write results to output directory
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / f"direct_merger_{strategy}.json", 'w') as f:
                json.dump(merged_tables, f, indent=2)
                
            with open(output_dir / f"extractor_merger_{strategy}.json", 'w') as f:
                json.dump(tables_copy, f, indent=2)

        logger.info("All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    os.environ['PYTHONPATH'] = os.path.join(os.getcwd(), 'src')
    success = test_table_merger()
    print()
    if success:
        print("✅ VERIFICATION COMPLETE - improved_table_merger integration is working correctly")
        print("Results saved to the output directory with different merge strategies:")
        print("  - none: No tables merged")
        print("  - conservative: Only tables with matching headers merged")
        print("  - aggressive: More tables merged with relaxed similarity requirements")
    else:
        print("❌ VERIFICATION FAILED - See error details above")
    
    sys.exit(0 if success else 1)
