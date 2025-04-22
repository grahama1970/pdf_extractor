#!/bin/bash
# Simple script to move files to .temp directory

# Create temp directory if it doesn't exist
mkdir -p .temp/src_backup

# Move files from src to temp
for file in src/*.py src/*.bak src/*.new src/*.txt; do
  if [ -f  ]; then
    filename=
    echo Moving to .temp/src_backup/
    mv  .temp/src_backup/
  fi
done

echo Files moved to .temp/src_backup
