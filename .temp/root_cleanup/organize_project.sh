#!/bin/bash
# Script to organize the project files and clean up temporary files

# Create needed directories
mkdir -p src/pdf_extractor/test
mkdir -p docs

# Move implementation files to correct locations if they're not already there
if [ -f src/improved_table_merger.py ] && [ ! -f src/pdf_extractor/improved_table_merger.py ]; then
  cp src/improved_table_merger.py src/pdf_extractor/improved_table_merger.py
  echo Moved improved_table_merger.py to the correct module location
fi

# Move test files to test directory
if [ -f src/pdf_extractor/test/test_table_merger.py ]; then
  echo Test files already in correct location
else
  cp test_merger_strategies.py src/pdf_extractor/test/test_table_merger.py 2>/dev/null || true
  cp verify_merger.py src/pdf_extractor/test/verify_integration.py 2>/dev/null || true
  echo Moved test files to the correct test location
fi

# Create documentation summary
cat > docs/table_merger.md << 'EOT'
# Table Merger Module

## Overview
The  module provides sophisticated multi-page table detection and merging 
functionality for PDF tables. It uses multiple similarity metrics to determine if tables 
should be merged, including header content matching and structural comparison.

## Usage


## Merge Strategies
- **conservative**: Default strategy, only merges tables with highly similar headers
- **aggressive**: Merges tables with more relaxed similarity requirements
- **none**: Disables table merging functionality

## Implementation Details
The module is designed to be a 5% customization on top of existing PDF extraction
capabilities, following the project's 95/5 rule. It adds essential multi-page table
handling without reimplementing functionality that exists in established packages.
EOT

echo Created documentation in docs/table_merger.md

# Clean up temporary scripts 
for file in fix_table_extractor.py update_table_extractor.py edit_table_extractor.py final_debug.py final_fix.py; do
  if [ -f  ]; then
    mkdir -p .temp
    mv  .temp/
    echo Moved temporary script to .temp directory
  fi
done

# Copy the project status to a more discoverable location
cp .claude/project_status.md docs/project_status.md

echo ✅ Project organization complete!
echo - Implementation files in src/pdf_extractor/
echo - Test files in src/pdf_extractor/test/
echo - Documentation in docs/
echo - Temporary files moved to .temp/
