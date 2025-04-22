#!/usr/bin/env python3

import sys
from pathlib import Path

# Import table_extractor
sys.path.append("/home/graham/workspace/experiments/pdf_extractor/src")
from table_extractor import extract_tables

# Test strategies
for strategy in ["conservative", "aggressive", "none"]:
    print(f"\nStrategy: {strategy}")
    tables = extract_tables("/home/graham/workspace/experiments/pdf_extractor/src/input/BHT_CV32A65X.pdf", pages="1-3", flavor="lattice", merge_strategy=strategy)
    print(f"Extracted {len(tables)} tables")
    for i, table in enumerate(tables):
        print(f"  Table {i+1}: page={table.get("page", "unknown")}, rows={table.get("rows", 0)}")
        if table.get("is_multi_page", False):
            print(f"  Multi-page: Yes ({table.get("page_range", "N/A")})")

