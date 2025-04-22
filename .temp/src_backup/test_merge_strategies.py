#!/usr/bin/env python3
import sys
from pathlib import Path

from table_extractor import extract_tables

# Test with different strategies
print("TESTING MERGE STRATEGIES")

for strategy in ["conservative", "aggressive", "none"]:
    print(f"
Strategy: {strategy}")
    path = "/home/graham/workspace/experiments/pdf_extractor/src/input/BHT_CV32A65X.pdf"
    tables = extract_tables(path, pages="1-3", flavor="lattice", merge_strategy=strategy)
    print(f"Extracted {len(tables)} tables")
