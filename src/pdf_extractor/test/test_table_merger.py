#!/usr/bin/env python3
"""Test module for improved_table_merger.py"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pdf_extractor.improved_table_merger import process_and_merge_tables
from pdf_extractor.table_extractor import TableExtractor

def test_table_merger():
    """Test the table merger functionality."""
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
                ["Value4", "Value5", "Value6"]
            ]
        }
    ]

    # Test with different strategies
    for strategy in ["conservative", "aggressive", "none"]:
        print(f"Testing {strategy} strategy")
        result = process_and_merge_tables(test_tables, merge_strategy=strategy)
        print(f"Result: {len(result)} tables")

if __name__ == "__main__":
    test_table_merger()
