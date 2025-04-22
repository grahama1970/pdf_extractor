#!/bin/bash
# Script to safely clean up redundant files in the src directory

# Create backup directory
mkdir -p .temp/src_backup
echo Created backup directory: .temp/src_backup

# Backup and remove files from src directory
for file in src/*.py src/*.bak src/*.new src/*.clean src/*.txt; do
  if [ -f  ]; then
    filename=
    echo Backing up ...
    cp  .temp/src_backup/
    echo Removing ...
    rm 
  fi
done

echo Cleanup complete! All files have been backed up to .temp/src_backup before removal.
echo If you need to restore any files, you can find them in the backup directory.
