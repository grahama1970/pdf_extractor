from typing import List, Dict, Any, Optional, Tuple, Set
import re
import logging

logger = logging.getLogger(__name__)

# Import our improved table merger
try:
    from .improved_table_merger import process_and_merge_tables
    IMPROVED_MERGER_AVAILABLE = True
except ImportError:
    logger.warning("improved_table_merger not found. Using basic table processing.")
    IMPROVED_MERGER_AVAILABLE = False

class TableExtractor:
    """Enhanced table extraction from PDF documents.
    
    This class provides functionality to extract tables from PDFs and 
    handles special cases like multi-page tables and merged cells. It also includes
    configurable table merging with different strategies.
    """
    
    def __init__(self, 
                 multi_page_detection: bool = True,
                 similarity_threshold: float = 0.7,
                 merge_strategy: str = "conservative"):
        """Initialize the table extractor.
        
        Args:
            multi_page_detection: Flag to enable multi-page table detection
            similarity_threshold: Threshold for table similarity detection
            merge_strategy: Strategy for merging tables ("conservative", "aggressive", or "none")
        """
        self.multi_page_detection = multi_page_detection
        self.similarity_threshold = similarity_threshold
        self.page_tables_map = {}  # Maps page numbers to tables on each page
        self.multi_page_tables = []  # List of detected multi-page tables
        self.merge_strategy = merge_strategy  # Strategy for merging tables
        
    def extract_tables(self, pdf_path: str, all_pages: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """Extract tables from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            all_pages: Flag to extract from all pages (needed for multi-page detection)
            
        Returns:
            Dictionary with tables and multi-page tables
        """
        try:
            # Import marker here to avoid making it a required dependency
            import marker
        except ImportError:
            raise ImportError("The marker package is required for TableExtractor. Install with: uv install marker")
            
        # Extract tables with Marker
        extraction_result = marker.extract(pdf_path, all_pages=all_pages)
        
        # Process tables from extraction result
        tables = extraction_result.get('tables', [])
        
        # Process tables with multi-page detection if enabled
        if self.multi_page_detection and all_pages:
            self._process_tables(tables)
            
            # Add multi-page tables to the result
            return {
                'tables': tables,
                'multi_page_tables': self.multi_page_tables
            }
        
        return {'tables': tables}
    
    def _process_tables(self, tables: List[Dict[str, Any]]) -> None:
        """Process tables and detect multi-page tables.
        
        Args:
            tables: List of tables from extraction
            
        Returns:
            None
        """
        # Group tables by page
        self.page_tables_map = {}
        for table in tables:
            page = table.get('page', 0)
            if page not in self.page_tables_map:
                self.page_tables_map[page] = []
            self.page_tables_map[page].append(table)
        
        # Apply multi-page table merging if available
        if IMPROVED_MERGER_AVAILABLE and len(tables) > 1:
            logger.info(f"Applying multi-page table merging with {self.merge_strategy} strategy")
            original_count = len(tables)
            
            # Use our improved table merger
            merged_tables = process_and_merge_tables(tables, merge_strategy=self.merge_strategy)
            
            # Update our tables list with the merged result
            tables.clear()
            tables.extend(merged_tables)
            
            merged_count = len(tables)
            if merged_count < original_count:
                logger.info(f"Merged {original_count - merged_count} multi-page tables")
        
        # Detect multi-page tables
        self._detect_multi_page_tables()
        
        # Update table metadata with multi-page information
        self._update_table_metadata(tables)
    
    def _detect_multi_page_tables(self) -> None:
        """Detect tables that span multiple pages.
        
        Returns:
            None
        """
        # Get sorted list of pages
        pages = sorted(self.page_tables_map.keys())
        
        # Process all pages
        for i in range(len(pages) - 1):
            current_page = pages[i]
            next_page = pages[i + 1]
            
            # Skip non-consecutive pages
            if next_page != current_page + 1:
                continue
                
            current_tables = self.page_tables_map[current_page]
            next_tables = self.page_tables_map[next_page]
            
            # Check for multi-page tables
            self._find_multi_page_candidates(
                current_tables, next_tables, current_page, next_page
            )
    
    def _find_multi_page_candidates(self, 
                                   tables1: List[Dict[str, Any]], 
                                   tables2: List[Dict[str, Any]],
                                   page1: int,
                                   page2: int) -> None:
        """Find candidate tables for multi-page continuation.
        
        Args:
            tables1: Tables from the first page
            tables2: Tables from the second page
            page1: First page number
            page2: Second page number
            
        Returns:
            None
        """
        # Look for tables that might be continued
        for idx1, table1 in enumerate(tables1):
            # Check if table is at the bottom of the page
            if self._is_table_at_bottom(table1):
                for idx2, table2 in enumerate(tables2):
                    # Check if table is at the top of the next page
                    if self._is_table_at_top(table2):
                        # Calculate similarity between tables
                        similarity = self._calculate_table_similarity(table1, table2)
                        
                        if similarity >= self.similarity_threshold:
                            logger.info(f"Multi-page table detected between pages {page1} and {page2}")
                            
                            # Create multi-page table
                            multi_page_table = {
                                'type': 'multi_page_table',
                                'pages': [page1, page2],
                                'parts': [table1, table2],
                                'similarity': similarity
                            }
                            
                            # Mark tables as parts of a multi-page table
                            table1['is_multi_page'] = True
                            table1['multi_page_id'] = len(self.multi_page_tables)
                            table1['continues_on_page'] = page2
                            
                            table2['is_multi_page'] = True
                            table2['multi_page_id'] = len(self.multi_page_tables)
                            table2['continued_from_page'] = page1
                            
                            self.multi_page_tables.append(multi_page_table)
    
    def _is_table_at_bottom(self, table: Dict[str, Any]) -> bool:
        """Check if a table is likely at the bottom of a page.
        
        Args:
            table: Table dictionary with position information
            
        Returns:
            Boolean indicating if table is at bottom
        """
        # This is a placeholder implementation
        # In a real implementation, we would check the y-coordinate
        # relative to the page height
        return True
    
    def _is_table_at_top(self, table: Dict[str, Any]) -> bool:
        """Check if a table is likely at the top of a page.
        
        Args:
            table: Table dictionary with position information
            
        Returns:
            Boolean indicating if table is at top
        """
        # This is a placeholder implementation
        # In a real implementation, we would check the y-coordinate
        # relative to the page height
        return True
    
    def _calculate_table_similarity(self, table1: Dict[str, Any], table2: Dict[str, Any]) -> float:
        """Calculate similarity between two tables.
        
        Args:
            table1: First table dictionary
            table2: Second table dictionary
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # This is a simplified implementation
        # In a real implementation, we would compare headers, columns, and content
        
        # Get table headers (assumed to be first row)
        headers1 = table1.get('data', [[]])[0] if table1.get('data') else []
        headers2 = table2.get('data', [[]])[0] if table2.get('data') else []
        
        # If headers don't match in count, tables are less likely to be similar
        if len(headers1) != len(headers2):
            return 0.2
        
        # Count matching headers
        matching_count = sum(1 for h1, h2 in zip(headers1, headers2) if h1.strip() == h2.strip())
        
        # Calculate similarity based on matching headers
        if len(headers1) > 0:
            return matching_count / len(headers1)
        
        return 0.0
    
    def _update_table_metadata(self, tables: List[Dict[str, Any]]) -> None:
        """Update table metadata with multi-page information.
        
        Args:
            tables: List of tables to update
            
        Returns:
            None
        """
        # This is a placeholder implementation
        # In a real implementation, we might update table metadata
        # based on the detected multi-page tables
        pass

if __name__ == "__main__":
    # Example usage
    import sys
    logging.basicConfig(level=logging.INFO)
    
    # Check args
    if len(sys.argv) < 2:
        print("Usage: python table_extractor.py <pdf_path> [merge_strategy]\n")
        print("Available merge strategies:\n")
        print("  - conservative: Default strategy, merges tables with high similarity")
        print("  - aggressive: Merges tables with lower similarity requirements")
        print("  - none: Disables table merging\n")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    merge_strategy = sys.argv[2] if len(sys.argv) > 2 else "conservative"
    
    extractor = TableExtractor(merge_strategy=merge_strategy)
    result = extractor.extract_tables(pdf_path)
    
    # Print results
    print(f"Extracted {len(result['tables'])} tables from {pdf_path}")
    if 'multi_page_tables' in result:
        print(f"Detected {len(result['multi_page_tables'])} multi-page tables")
    
    sys.exit(0)
