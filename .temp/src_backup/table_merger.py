#!/usr/bin/env python3

import logging
import sys
import copy
import warnings
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_table_similarity(table1, table2):
    # Calculate similarity between two tables
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
        
        # Count matching headers
        matching_headers = sum(1 for h1, h2 in zip(headers1, headers2) if h1 == h2)
        header_similarity = matching_headers / len(headers1) if headers1 else 0.0
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
        return 0.0
        
    return sum(scores) / len(scores)

def should_merge_tables(table1, table2, threshold=0.7):
    # Determine if two tables should be merged
    # Tables must be on consecutive pages
    if table2.get("page", 0) != table1.get("page", 0) + 1:
        return False
    
    # Calculate similarity
    similarity = calculate_table_similarity(table1, table2)
    
    # Return decision
    return similarity >= threshold

def merge_table_data(table1, table2):
    # Merge data from two tables
    data1 = table1.get("data", [])
    data2 = table2.get("data", [])
    
    # Handle empty tables
    if not data1:
        return data2
    if not data2:
        return data1
    
    # Check for matching headers
    headers1 = data1[0] if data1 else []
    headers2 = data2[0] if data2 else []
    
    # Determine if headers are similar
    matching_headers = sum(1 for h1, h2 in zip(headers1, headers2) if h1.strip() == h2.strip())
    has_matching_header = False
    if headers1 and len(headers1) > 0:
        if matching_headers / len(headers1) >= 0.5:
            has_matching_header = True
    
    # Merge the data
    merged_data = data1.copy()
    
    # Skip second table header if similar to first table header
    start_idx = 1 if has_matching_header else 0
    for i in range(start_idx, len(data2)):
        merged_data.append(data2[i])
    
    return merged_data

def merge_multi_page_tables(tables, similarity_threshold=0.7):
    # Identify and merge tables that span multiple pages
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
            
            # Merge data
            merged_data = merge_table_data(last_table, current_table)
            
            # Update metadata
            last_table["data"] = merged_data
            last_table["rows"] = len(merged_data)
            last_table["is_multi_page"] = True
            
            # Create page range string
            start_page = str(last_table.get("page", 0))
            end_page = str(current_table.get("page", 0))
            last_table["page_range"] = start_page + "-" + end_page
            
            # Log the merge
            logger.info("Merged table from page " + end_page + " into table on page " + start_page)
        else:
            # Add as new table
            merged_tables.append(copy.deepcopy(current_table))
    
    return merged_tables

def process_and_merge_tables(camelot_tables):
    # Process camelot tables and merge multi-page tables
    processed_tables = []
    
    for table in camelot_tables:
        processed = {
            "page": int(table.page),
            "data": table.data,
            "accuracy": table.parsing_report.get("accuracy", 0),
            "bbox": tuple(table._bbox),  # Protected access needed for bbox
            "rows": len(table.data),
            "cols": len(table.data[0]) if table.data else 0,
        }
        processed_tables.append(processed)
    
    # Apply multi-page table merging
    return merge_multi_page_tables(processed_tables)
