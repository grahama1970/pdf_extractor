#!/bin/bash
# Script to clean up and organize the PDF Extractor project

# Define directories
mkdir -p src/pdf_extractor/experiments
mkdir -p src/pdf_extractor/test
mkdir -p .temp

# Make sure __init__.py files exist
touch src/pdf_extractor/__init__.py
touch src/pdf_extractor/test/__init__.py
touch src/pdf_extractor/experiments/__init__.py

# Move core files to pdf_extractor module
for file in api.py cli.py markdown_extractor.py marker_processor.py pdf_to_json_converter.py qwen_processor.py table_extraction.py utils.py; do
  if [ -f src/ ]; then
    cp src/ src/pdf_extractor/
    echo Moved to src/pdf_extractor/
  fi
done

# Move debug/experimental files to experiments directory
for file in debug_camelot.py debug_extraction.py deep_debug.py direct_edit.py edit_table_extractor.py final_debug.py final_fix.py simple_debug.py test_camelot.py test_camelot2.py test_camelot3.py; do
  if [ -f src/ ]; then
    cp src/ src/pdf_extractor/experiments/
    echo Moved to src/pdf_extractor/experiments/
  fi
done

# Move temporary files to .temp directory
for file in fix_table_extractor.py update_table_extractor.py temp_edit.txt; do
  if [ -f  ]; then
    cp  .temp/
    echo Moved to .temp directory
  fi
done

echo Project organization complete!
