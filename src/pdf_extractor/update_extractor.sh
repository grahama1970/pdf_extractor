#!/bin/bash
# Update table_extractor.py to use improved_table_merger.py

# Define paths
EXTRACTOR_PATH="src/pdf_extractor/table_extractor.py"
BACKUP_PATH="src/pdf_extractor/table_extractor.py.bak"

# Create backup
cp "" ""
echo "Created backup at "

# Update file using sed commands
# 1. Add import for improved_table_merger
sed -i '/^logger = logging.getLogger(__name__)/ a \
# Import our improved table merger\
try:\
    from .improved_table_merger import process_and_merge_tables\
    IMPROVED_MERGER_AVAILABLE = True\
except ImportError:\
    logger.warning("improved_table_merger not found. Using basic table processing.")\
    IMPROVED_MERGER_AVAILABLE = False\
' ""

# 2. Update docstring to mention table merging
sed -i 's/handles special cases like multi-page tables and merged cells./handles special cases like multi-page tables and merged cells. It also includes configurable table merging with different strategies./' ""

# 3. Update __init__ method to add merge_strategy parameter
sed -i 's/def __init__(self, \n                 multi_page_detection: bool = True,\n                 similarity_threshold: float = 0.7):/def __init__(self, \n                 multi_page_detection: bool = True,\n                 similarity_threshold: float = 0.7,\n                 merge_strategy: str = "conservative"):/' ""

# 4. Add merge_strategy attribute to __init__ body
sed -i '/self.multi_page_tables = \[\]  # List of detected multi-page tables/ a \        self.merge_strategy = merge_strategy  # Strategy for merging tables: "conservative", "aggressive", or "none"' ""

# 5. Add table merging code to _process_tables method
sed -i '/# Detect multi-page tables/{
i \        # Apply multi-page table merging if available\
        if IMPROVED_MERGER_AVAILABLE and len(tables) > 1:\
            logger.info(f"Applying multi-page table merging with {self.merge_strategy} strategy")\
            original_count = len(tables)\
            \
            # Use our improved table merger\
            merged_tables = process_and_merge_tables(tables, merge_strategy=self.merge_strategy)\
            \
            # Update our tables list with the merged result\
            tables.clear()\
            tables.extend(merged_tables)\
            \
            merged_count = len(tables)\
            if merged_count < original_count:\
                logger.info(f"Merged {original_count - merged_count} multi-page tables")\
}' ""

echo "✅ Successfully updated table_extractor.py with improved_table_merger integration"
echo
echo "To use the improved table merger, initialize TableExtractor with a merge strategy:"
echo "  - extractor = TableExtractor(merge_strategy='conservative')  # Default strategy"
echo "  - extractor = TableExtractor(merge_strategy='aggressive')    # More aggressive merging"
echo "  - extractor = TableExtractor(merge_strategy='none')          # Disable merging"
