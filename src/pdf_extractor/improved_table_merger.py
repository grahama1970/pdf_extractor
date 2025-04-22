#!/usr/bin/env python3
"""
Advanced table merging for multi-page PDF tables.

This module provides sophisticated multi-page table detection and merging
functionality, designed to work with tables extracted from PDF documents.
It uses multiple similarity metrics to determine if tables should be merged,
including header content matching, bounding box spatial analysis, and structural
comparison.

Example usage:
    >>> from improved_table_merger import process_and_merge_tables
    >>> # Assuming tables is a list of camelot table objects
    >>> processed_tables = process_and_merge_tables(tables)
    >>> print(f"After merging, {len(processed_tables)} tables remain")
"""

import logging
import sys
import copy
import warnings
from typing import List, Dict, Any, Tuple, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def has_matching_headers(headers1: List[str], headers2: List[str], similarity_threshold: float = 0.7) -> bool:
    """
    Determine if two sets of headers are likely the same table headers.
    Handles variations in formatting, whitespace, and partial matches.
    
    Args:
        headers1: List of header strings from first table
        headers2: List of header strings from second table
        similarity_threshold: Minimum ratio of matching headers
        
    Returns:
        Boolean indicating if headers likely match
    """
    # Handle case where header count doesn't match
    if len(headers1) != len(headers2):
        return False
        
    # Normalize headers to improve matching
    normalized1 = [h.strip().lower() for h in headers1]
    normalized2 = [h.strip().lower() for h in headers2]
    
    # Count exact matches
    exact_matches = sum(1 for h1, h2 in zip(normalized1, normalized2) if h1 == h2)
    
    # Count partial matches (for truncated headers)
    partial_matches = 0
    for h1, h2 in zip(normalized1, normalized2):
        if h1 and h2 and (h1 in h2 or h2 in h1) and h1 != h2:
            partial_matches += 1
            
    # Calculate similarity score
    total_headers = len(headers1)
    if total_headers == 0:
        return False
        
    similarity = (exact_matches + 0.5 * partial_matches) / total_headers
    
    return similarity >= similarity_threshold

def calculate_table_similarity(table1: Dict[str, Any], table2: Dict[str, Any]) -> float:
    """
    Calculate similarity between two tables using multiple metrics.
    
    Args:
        table1: First table dictionary with 'data' and optionally 'bbox' keys
        table2: Second table dictionary with 'data' and optionally 'bbox' keys
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Extract table data
    data1 = table1.get("data", [])
    data2 = table2.get("data", [])
    
    # If either table is empty, they cannot be similar
    if not data1 or not data2:
        return 0.0
    
    # Collect similarity scores from different metrics
    scores = []
    
    # 1. Column count - tables must have same number of columns
    if len(data1[0]) != len(data2[0]):
        return 0.0
    
    # 2. Header similarity
    if len(data1) > 0 and len(data2) > 0:
        headers1 = [h.strip() for h in data1[0]]
        headers2 = [h.strip() for h in data2[0]]
        
        # Use enhanced header matching
        header_match = has_matching_headers(headers1, headers2)
        header_similarity = 0.9 if header_match else 0.3
        scores.append(header_similarity)
    
    # 3. Bounding box similarity (if available)
    if "bbox" in table1 and "bbox" in table2:
        bbox1 = table1["bbox"]
        bbox2 = table2["bbox"]
        
        # Check horizontal alignment (left and right edges)
        left_diff = abs(bbox1[0] - bbox2[0])
        right_diff = abs(bbox1[2] - bbox2[2])
        width = max(bbox1[2] - bbox1[0], 100)  # Avoid division by zero
        
        x_alignment = 1.0 - min(left_diff / width, right_diff / width, 1.0)
        bbox_similarity = max(0.0, x_alignment)  # Ensure positive
        scores.append(bbox_similarity)
    
    # Calculate final similarity score with simple averaging
    if not scores:
        return 0.5 if len(data1[0]) == len(data2[0]) else 0.0  # Default to moderate similarity if columns match
        
    return sum(scores) / len(scores)

def should_merge_tables(table1: Dict[str, Any], table2: Dict[str, Any], threshold: float = 0.7) -> bool:
    """
    Determine if two tables should be merged based on similarity and page sequence.
    
    Args:
        table1: First table dictionary
        table2: Second table dictionary
        threshold: Minimum similarity threshold for merging (default: 0.7)
        
    Returns:
        True if tables should be merged, False otherwise
    """
    # Tables must be on consecutive pages
    if table2.get("page", 0) != table1.get("page", 0) + 1:
        return False
    
    # Calculate similarity
    similarity = calculate_table_similarity(table1, table2)
    logger.debug(f"Table similarity: {similarity:.2f}, threshold: {threshold}")
    
    # Return decision
    return similarity >= threshold

def merge_table_data_safely(table1: Dict[str, Any], table2: Dict[str, Any]) -> Optional[List[List[str]]]:
    """
    Merge tables with additional validation to ensure data integrity.
    
    Args:
        table1: First table dictionary with 'data' key
        table2: Second table dictionary with 'data' key
        
    Returns:
        Merged table data or None if safe merge isn't possible
    """
    data1 = table1.get("data", [])
    data2 = table2.get("data", [])
    
    # Empty table handling
    if not data1:
        return data2
    if not data2:
        return data1
        
    # Column count validation
    if data1 and data2 and len(data1[0]) != len(data2[0]):
        # Don't merge tables with different column counts
        logger.warning(f"Column count mismatch: {len(data1[0])} vs {len(data2[0])}")
        return None
    
    # Header handling
    headers1 = data1[0] if data1 else []
    headers2 = data2[0] if data2 else []
    has_matching_header = has_matching_headers(headers1, headers2)
    
    # Create merged data conservatively
    merged_data = data1.copy()
    start_idx = 1 if has_matching_header else 0
    
    # Add data from second table
    for i in range(start_idx, len(data2)):
        merged_data.append(data2[i])
    
    return merged_data

def merge_multi_page_tables(tables: List[Dict[str, Any]], similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Identify and merge tables that span multiple pages.
    
    Args:
        tables: List of table dictionaries, each with 'page' and 'data' keys
        similarity_threshold: Minimum similarity for merging (default: 0.7)
        
    Returns:
        List of tables with multi-page tables merged
    """
    if not tables:
        return []
    
    # Sort tables by page number
    sorted_tables = sorted(tables, key=lambda t: t.get("page", 0))
    
    # Initialize result with first table
    merged_tables = [copy.deepcopy(sorted_tables[0])]
    
    # Process remaining tables
    for current_table in sorted_tables[1:]:
        # Check if current table should be merged with previous
        if merged_tables and should_merge_tables(merged_tables[-1], current_table, similarity_threshold):
            # Get the last table
            last_table = merged_tables[-1]
            
            try:
                # Merge data with safety checks
                merged_data = merge_table_data_safely(last_table, current_table)
                
                # Skip if merge failed
                if merged_data is None:
                    logger.warning(f"Failed to safely merge table from page {current_table.get('page')} - adding as separate table")
                    merged_tables.append(copy.deepcopy(current_table))
                    continue
                
                # Update metadata
                last_table["data"] = merged_data
                last_table["rows"] = len(merged_data)
                last_table["is_multi_page"] = True
                
                # Create or update page range string
                start_page = str(last_table.get("page", 0))
                current_page = str(current_table.get("page", 0))
                
                # Update or create page_range
                if "page_range" in last_table:
                    # Extract the end page from the existing range
                    range_parts = last_table["page_range"].split("-")
                    if len(range_parts) > 1:
                        start_page = range_parts[0]  # Keep original start page
                    last_table["page_range"] = f"{start_page}-{current_page}"
                else:
                    last_table["page_range"] = f"{start_page}-{current_page}"
                
                # Log the merge
                logger.info(f"Merged table from page {current_page} into table starting on page {start_page}")
            
            except Exception as e:
                logger.warning(f"Error merging tables: {e} - adding as separate table")
                merged_tables.append(copy.deepcopy(current_table))
                continue
        else:
            # Add as new table
            merged_tables.append(copy.deepcopy(current_table))
    
    return merged_tables

def process_and_merge_tables(tables: List[Any], merge_strategy: str = "conservative") -> List[Dict[str, Any]]:
    """
    Process tables and merge multi-page tables with configurable strategy.
    Works with both camelot table objects and dictionaries.
    
    Args:
        tables: List of camelot table objects or processed table dictionaries
        merge_strategy: Strategy for merging ("aggressive", "conservative", "none")
        
    Returns:
        List of processed table dictionaries
    """
    # Check if we received camelot table objects or dictionaries
    processed_tables = []
    
    # If first item is a dictionary with needed keys, assume all are dictionaries
    if tables and isinstance(tables[0], dict) and "data" in tables[0]:
        processed_tables = tables
    else:
        # Process camelot table objects
        for table in tables:
            try:
                processed = {
                    "page": int(table.page),
                    "data": table.data,
                    "accuracy": table.parsing_report.get("accuracy", 0),
                    "rows": len(table.data),
                    "cols": len(table.data[0]) if table.data else 0,
                }
                
                # Add bounding box if available
                try:
                    processed["bbox"] = tuple(table._bbox)
                except (AttributeError, TypeError):
                    logger.debug(f"Bounding box not available for table on page {table.page}")
                
                processed_tables.append(processed)
            except AttributeError as e:
                logger.warning(f"Error processing table: {e}")
                # Skip invalid tables instead of failing
    
    # Skip merging if requested or if we have no tables
    if merge_strategy == "none" or not processed_tables:
        logger.info("Table merging disabled")
        return processed_tables
        
    # Set threshold based on strategy
    similarity_threshold = 0.8 if merge_strategy == "conservative" else 0.6
    
    # Apply multi-page table merging
    merged_tables = merge_multi_page_tables(processed_tables, similarity_threshold)
    logger.info(f"Table merging ({merge_strategy}): {len(processed_tables)} tables -> {len(merged_tables)} merged tables")
    
    return merged_tables

if __name__ == "__main__":
    print("IMPROVED TABLE MERGER VALIDATION")
    print("===============================")
    
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
    
    # Define expected results for validation
    EXPECTED_RESULTS = {
        "merged_count": 2,  # Should merge tables 1 and 2, keep table 3 separate
        "first_table": {
            "rows": 4,  # Header + row from table 1 + 2 rows from table 2
            "is_multi_page": True,
            "page_range": "1-2"
        }
    }
    
    # Test with different merge strategies
    for strategy in ["conservative", "aggressive", "none"]:
        print(f"\nTesting with merge strategy: {strategy}")
        
        # Standard test case
        merged_standard = process_and_merge_tables(test_tables, merge_strategy=strategy)
        print(f"- Standard case: {len(test_tables)} tables -> {len(merged_standard)} tables")
        
        # Tricky test case with slightly different headers
        merged_tricky = merge_multi_page_tables(test_tables_tricky, similarity_threshold=0.8 if strategy == "conservative" else 0.6 if strategy == "aggressive" else 1.0)
        print(f"- Tricky headers: {len(test_tables_tricky)} tables -> {len(merged_tricky)} tables")
        
        if strategy == "none":
            # Should not merge any tables
            assert len(merged_standard) == len(test_tables), "Tables should not be merged in 'none' strategy"
            assert len(merged_tricky) == len(test_tables_tricky), "Tables should not be merged in 'none' strategy"
        elif strategy == "conservative":
            # Should merge standard tables but possibly not tricky ones
            assert len(merged_standard) == 2, "Conservative strategy should merge standard tables"
        elif strategy == "aggressive":
            # Should merge both standard and tricky tables
            assert len(merged_standard) == 2, "Aggressive strategy should merge standard tables"
            assert len(merged_tricky) == 1, "Aggressive strategy should merge tricky tables"
    
    # Perform table merging for validation
    merged_tables = merge_multi_page_tables(test_tables)
    
    # Validate results
    validation_failures = {}
    
    # Check merged count
    if len(merged_tables) != EXPECTED_RESULTS["merged_count"]:
        validation_failures["merged_count"] = {
            "expected": EXPECTED_RESULTS["merged_count"],
            "actual": len(merged_tables)
        }
    
    # Check first table properties if it exists
    if merged_tables:
        first_table = merged_tables[0]
        expected_first = EXPECTED_RESULTS["first_table"]
        
        # Check row count
        if first_table.get("rows", 0) != expected_first["rows"]:
            validation_failures["first_table_rows"] = {
                "expected": expected_first["rows"],
                "actual": first_table.get("rows", 0)
            }
        
        # Check multi-page flag
        if first_table.get("is_multi_page", False) != expected_first["is_multi_page"]:
            validation_failures["first_table_is_multi_page"] = {
                "expected": expected_first["is_multi_page"],
                "actual": first_table.get("is_multi_page", False)
            }
        
        # Check page range
        if first_table.get("page_range", "") != expected_first["page_range"]:
            validation_failures["first_table_page_range"] = {
                "expected": expected_first["page_range"],
                "actual": first_table.get("page_range", "")
            }
    else:
        validation_failures["merged_tables_empty"] = {
            "expected": "At least one merged table",
            "actual": "No tables after merging"
        }
    
    # Report validation status
    if not validation_failures:
        print("\n✅ VALIDATION COMPLETE - All table merging results match expected values")
        print(f"- Merged {len(test_tables)} tables into {len(merged_tables)} tables")
        print(f"- First table has {merged_tables[0]['rows']} rows, spanning pages {merged_tables[0]['page_range']}")
        print(f"- Tables with different structures maintained separately")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED - Table merging results don't match expected values")
        print("FAILURE DETAILS:")
        for field, details in validation_failures.items():
            print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        print(f"Total errors: {len(validation_failures)} fields mismatched")
        sys.exit(1)
