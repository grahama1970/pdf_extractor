#!/usr/bin/env python3
"""
Test script for the improved table merger
"""
import sys
from improved_table_merger import merge_multi_page_tables, has_matching_headers

def test_merge_strategies():
    print("TESTING DIFFERENT MERGE STRATEGIES")
    print("==================================\n")
    
    # Define test tables that should be merged
    test_tables = [
        {
            "page": 1,
            "data": [
                ["Signal", "IO", "Description", "Connection", "Type"],
                ["signal1", "in", "test1", "conn1", "type1"]
            ],
            "bbox": (50, 700, 550, 750),
            "rows": 2,
            "cols": 5,
            "accuracy": 100.0
        },
        {
            "page": 2,
            "data": [
                ["Signal", "IO", "Description", "Connection", "Type"],
                ["signal2", "out", "test2", "conn2", "type2"],
                ["signal3", "in", "test3", "conn3", "type3"]
            ],
            "bbox": (50, 700, 550, 750),
            "rows": 3,
            "cols": 5,
            "accuracy": 100.0
        },
        {
            "page": 3,
            "data": [
                ["Different", "Header", "Structure"],
                ["data1", "data2", "data3"]
            ],
            "bbox": (50, 700, 550, 750),
            "rows": 2,
            "cols": 3,
            "accuracy": 95.0
        }
    ]
    
    # Add tricky test case with slightly different headers
    test_tables_tricky = [
        {
            "page": 1,
            "data": [
                ["Signal Name", "I/O", "Description", "Connection", "Type"],
                ["signal1", "in", "test1", "conn1", "type1"]
            ],
            "bbox": (50, 700, 550, 750),
            "rows": 2,
            "cols": 5,
            "accuracy": 100.0
        },
        {
            "page": 2,
            "data": [
                ["Signal", "IO", "Desc.", "Connection", "Type"],
                ["signal2", "out", "test2", "conn2", "type2"],
                ["signal3", "in", "test3", "conn3", "type3"]
            ],
            "bbox": (50, 700, 550, 750),
            "rows": 3,
            "cols": 5,
            "accuracy": 100.0
        }
    ]
    
    # Test with different merge strategies
    strategies = {
        "conservative": 0.8,
        "aggressive": 0.6,
        "none": 1.1  # Threshold so high nothing will merge
    }
    
    for strategy_name, threshold in strategies.items():
        print(f"\nTesting with merge strategy: {strategy_name} (threshold: {threshold})")
        
        # Standard test case
        merged_standard = merge_multi_page_tables(test_tables, similarity_threshold=threshold)
        print(f"- Standard case: {len(test_tables)} tables -> {len(merged_standard)} tables")
        
        # Tricky test case with slightly different headers
        merged_tricky = merge_multi_page_tables(test_tables_tricky, similarity_threshold=threshold)
        print(f"- Tricky headers: {len(test_tables_tricky)} tables -> {len(merged_tricky)} tables")
        
        # Test header matching function directly
        headers1 = ["Signal Name", "I/O", "Description", "Connection", "Type"]
        headers2 = ["Signal", "IO", "Desc.", "Connection", "Type"]
        match_result = has_matching_headers(headers1, headers2, similarity_threshold=0.6)
        print(f"- Header matching test (0.6 threshold): {match_result}")
        
        if strategy_name == "none":
            # Should not merge any tables
            assert len(merged_standard) == len(test_tables), "Tables should not be merged in 'none' strategy"
            assert len(merged_tricky) == len(test_tables_tricky), "Tables should not be merged in 'none' strategy"
            print("✓ No tables merged with 'none' strategy")
        elif strategy_name == "conservative":
            # Should merge standard tables but possibly not tricky ones
            assert len(merged_standard) == 2, "Conservative strategy should merge standard tables"
            print("✓ Standard tables correctly merged with 'conservative' strategy")
        elif strategy_name == "aggressive":
            # Should merge both standard and tricky tables
            assert len(merged_standard) == 2, "Aggressive strategy should merge standard tables"
            assert len(merged_tricky) == 1, "Aggressive strategy should merge tricky tables with different headers"
            print("✓ Both standard and tricky tables correctly merged with 'aggressive' strategy")

if __name__ == "__main__":
    test_merge_strategies()
    print("\nAll tests passed successfully!")
