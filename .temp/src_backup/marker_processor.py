"""
Process PDF content using Marker with specialized handling for multi-page tables.

This module provides functionality for processing PDF documents using the Marker 
library, with enhanced capabilities for detecting and processing tables that 
span multiple pages. It integrates with the pdf_extractor system to support
the complete extraction pipeline.

Third-party package documentation:
- marker: https://github.com/VikParuchuri/marker
- re: https://docs.python.org/3/library/re.html

Example usage:
    >>> from marker_processor import MarkerProcessor
    >>> processor = MarkerProcessor()
    >>> result = processor.process_pdf("example.pdf")
    >>> print(f"Detected {len(result.get('tables', []))} tables")
    >>> multi_page_tables = result.get('multi_page_tables', [])
    >>> print(f"Detected {len(multi_page_tables)} multi-page tables")
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TableMetadata:
    """Metadata for tables extracted with Marker.
    
    Attributes:
        table_id: Unique identifier for the table
        page: Page number where the table appears
        header_row: Flag indicating if the first row is a header
        is_multi_page: Flag indicating if the table spans multiple pages
        continued_from_page: Previous page number if this is a continuation
        continues_on_page: Next page number if this table continues
        structure_similarity: Similarity score for multi-page sections
        column_count: Number of columns in the table
        column_widths: Width of each column
    """
    table_id: str = None
    page: int = None
    header_row: bool = True
    is_multi_page: bool = False
    continued_from_page: Optional[int] = None
    continues_on_page: Optional[int] = None
    structure_similarity: float = 0.0
    column_count: int = 0
    column_widths: List[int] = field(default_factory=list)


class MarkerProcessor:
    """Processor for Marker-based extraction of PDF content.
    
    This class handles the extraction of structured content from PDF documents
    using the Marker library, with special handling for multi-page tables.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """Initialize the Marker processor.
        
        Args:
            similarity_threshold: Threshold for table similarity detection
        """
        self.similarity_threshold = similarity_threshold
        self.tables_by_page = {}  # Maps page numbers to tables on that page
        self.multi_page_tables = []  # List of detected multi-page tables
        
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF file using Marker.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted content
        """
        try:
            # Import marker here to avoid making it a required dependency
            import marker
            logger.info(f"Processing PDF with Marker: {pdf_path}")
        except ImportError:
            error_msg = "The marker package is required for MarkerProcessor. Install with: uv install marker"
            logger.error(error_msg)
            raise ImportError(error_msg)
            
        # Extract content from PDF with Marker
        try:
            extraction_result = marker.extract(pdf_path, all_pages=True)  # Request for all pages (needed for multi-page detection)
            logger.info(f"Marker extraction completed, processing tables")
            
            # Process tables with multi-page detection
            self._process_tables(extraction_result)
            
            # Log results
            table_count = len(extraction_result.get('tables', []))
            multi_page_count = len(self.multi_page_tables)
            logger.info(f"Found {table_count} tables, including {multi_page_count} multi-page tables")
            
            return extraction_result
        except Exception as e:
            logger.error(f"Error during Marker processing: {e}")
            raise RuntimeError(f"Marker extraction failed: {e}")
    
    def _process_tables(self, extraction_result: Dict[str, Any]) -> None:
        """Process tables from extraction result and detect multi-page tables.
        
        Args:
            extraction_result: Extraction result from Marker
            
        Returns:
            None
        """
        # Extract tables from all pages
        if 'tables' in extraction_result:
            tables = extraction_result['tables']
            
            # Group tables by page
            self.tables_by_page = {}
            for table in tables:
                page = table.get('page', 0)
                if page not in self.tables_by_page:
                    self.tables_by_page[page] = []
                
                # Add metadata for easier multi-page processing
                if 'metadata' not in table:
                    table['metadata'] = TableMetadata(
                        table_id=f"table_{page}_{len(self.tables_by_page[page])}",
                        page=page,
                        column_count=len(table.get('header', [])),
                        column_widths=self._calculate_column_widths(table)
                    )
                
                self.tables_by_page[page].append(table)
            
            # Detect multi-page tables
            self._detect_multi_page_tables()
            
            # Update extraction result with multi-page table info
            self._update_extraction_result(extraction_result)
    
    def _calculate_column_widths(self, table: Dict[str, Any]) -> List[int]:
        """Calculate relative column widths from a table.
        
        Args:
            table: Table object
            
        Returns:
            List of relative column widths
        """
        # This is a simplified approach - in a real implementation,
        # you might use the actual widths from the PDF
        header = table.get('header', [])
        return [len(col) for col in header] if header else []
    
    def _detect_multi_page_tables(self) -> None:
        """Detect tables that span multiple pages.
        
        Returns:
            None
        """
        # Get sorted list of pages
        pages = sorted(self.tables_by_page.keys())
        if len(pages) <= 1:
            logger.info("Only one page with tables detected, skipping multi-page detection")
            return
        
        # Check consecutive pages for table continuation
        for i in range(len(pages) - 1):
            current_page = pages[i]
            next_page = pages[i + 1]
            
            # Skip if pages are not consecutive
            if next_page != current_page + 1:
                logger.debug(f"Pages {current_page} and {next_page} are not consecutive, skipping")
                continue
                
            current_tables = self.tables_by_page[current_page]
            next_tables = self.tables_by_page[next_page]
            
            if not current_tables or not next_tables:
                continue
                
            # Check tables at the bottom of current page
            for curr_idx, curr_table in enumerate(current_tables):
                # Consider tables near the end of the page as candidates
                if curr_idx == len(current_tables) - 1:  # Last table on the page
                    # Look for matching tables at the top of the next page
                    for next_idx, next_table in enumerate(next_tables):
                        if next_idx == 0:  # First table on the next page
                            # Compare tables for similarity
                            similarity = self._calculate_table_similarity(curr_table, next_table)
                            logger.debug(f"Table similarity between page {current_page} and {next_page}: {similarity:.2f}")
                            
                            if similarity >= self.similarity_threshold:
                                logger.info(f"Detected multi-page table: pages {current_page}-{next_page} (similarity: {similarity:.2f})")
                                
                                # Mark tables as multi-page
                                curr_meta = curr_table.get('metadata', TableMetadata())
                                next_meta = next_table.get('metadata', TableMetadata())
                                
                                curr_meta.is_multi_page = True
                                curr_meta.continues_on_page = next_page
                                curr_meta.structure_similarity = similarity
                                
                                next_meta.is_multi_page = True
                                next_meta.continued_from_page = current_page
                                next_meta.structure_similarity = similarity
                                
                                curr_table['metadata'] = curr_meta
                                next_table['metadata'] = next_meta
                                
                                # Track multi-page table
                                multi_page_table = {
                                    'type': 'multi_page_table',
                                    'parts': [curr_table, next_table],
                                    'pages': [current_page, next_page],
                                    'similarity': similarity,
                                    'page_range': f"{current_page}-{next_page}"
                                }
                                self.multi_page_tables.append(multi_page_table)
    
    def _calculate_table_similarity(self, table1: Dict[str, Any], table2: Dict[str, Any]) -> float:
        """Calculate similarity score between two tables.
        
        Args:
            table1: First table
            table2: Second table
            
        Returns:
            Similarity score between 0 and 1
        """
        score_components = []
        
        # 1. Compare column count
        header1 = table1.get('header', [])
        header2 = table2.get('header', [])
        
        if header1 and header2:
            col_count_score = 1.0 if len(header1) == len(header2) else 0.5
            score_components.append(col_count_score)
        
        # 2. Compare header content
        if header1 and header2 and len(header1) == len(header2):
            header_match_count = sum(1 for h1, h2 in zip(header1, header2) 
                                    if self._text_similarity(h1, h2) > 0.8)
            header_score = header_match_count / len(header1)
            score_components.append(header_score)
        
        # 3. Compare column width patterns
        meta1 = table1.get('metadata', TableMetadata())
        meta2 = table2.get('metadata', TableMetadata())
        
        if meta1.column_widths and meta2.column_widths and len(meta1.column_widths) == len(meta2.column_widths):
            # Normalize widths
            total1 = sum(meta1.column_widths)
            total2 = sum(meta2.column_widths)
            
            if total1 > 0 and total2 > 0:
                norm1 = [w/total1 for w in meta1.column_widths]
                norm2 = [w/total2 for w in meta2.column_widths]
                
                # Calculate average difference
                width_diffs = [abs(w1 - w2) for w1, w2 in zip(norm1, norm2)]
                avg_diff = sum(width_diffs) / len(width_diffs)
                width_score = 1.0 - min(avg_diff, 1.0)  # Convert to similarity
                
                score_components.append(width_score)
        
        # Return average score
        return sum(score_components) / len(score_components) if score_components else 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple word-based similarity
        words1 = set(re.sub(r'[^\w\s]', '', text1.lower()).split())
        words2 = set(re.sub(r'[^\w\s]', '', text2.lower()).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _update_extraction_result(self, extraction_result: Dict[str, Any]) -> None:
        """Update extraction result with multi-page table information.
        
        Args:
            extraction_result: Extraction result to update
            
        Returns:
            None
        """
        # Add multi-page table information to the result
        if self.multi_page_tables:
            extraction_result['multi_page_tables'] = self.multi_page_tables
            
            # Update individual table metadata
            for page, tables in self.tables_by_page.items():
                for table in tables:
                    meta = table.get('metadata', None)
                    if meta and meta.is_multi_page:
                        # Ensure metadata is properly serialized
                        table['is_multi_page'] = True
                        
                        if meta.continued_from_page is not None:
                            table['continued_from_page'] = meta.continued_from_page
                            
                        if meta.continues_on_page is not None:
                            table['continues_on_page'] = meta.continues_on_page
                            
                        # Add page range for convenience
                        if meta.continued_from_page is not None:
                            table['page_range'] = f"{meta.continued_from_page}-{meta.page}"
                        elif meta.continues_on_page is not None:
                            table['page_range'] = f"{meta.page}-{meta.continues_on_page}"
    
    def get_combined_table_content(self, multi_page_table: Dict[str, Any]) -> List[List[str]]:
        """Get combined content from all parts of a multi-page table.
        
        Args:
            multi_page_table: Multi-page table object
            
        Returns:
            2D array of table content
        """
        combined_content = []
        
        # Combine content from all table parts
        for i, table_part in enumerate(multi_page_table.get('parts', [])):
            # Get rows from this part
            rows = []
            
            # Add header if this is the first part
            if i == 0:  # First part
                header = table_part.get('header', [])
                if header:
                    rows.append(header)
            
            # Add data rows
            data_rows = table_part.get('data', [])
            rows.extend(data_rows)
            
            combined_content.extend(rows)
        
        return combined_content


if __name__ == "__main__":
    """
    Validation code for marker_processor.py
    
    This validation ensures that:
    1. The MarkerProcessor class initializes with correct settings
    2. TableMetadata class supports multi-page table properties
    3. Table similarity calculation works correctly
    """
    import sys
    import json
    from pathlib import Path
    
    # CRITICAL: Define exact expected results as required by VALIDATION_REQUIREMENTS.md
    EXPECTED_RESULTS = {
        "multi_page_detection": True,
        "metadata_fields_count": 9,
        "similarity_threshold": 0.7,
        "table_similarity": {
            "identical_tables": 1.0,
            "similar_tables": 0.8,
            "different_tables": 0.3
        }
    }
    
    # Track validation failures
    validation_failures = {}
    
    print("MARKER PROCESSOR MODULE VERIFICATION")
    print("=====================================")
    
    # Test 1: Initialization with default settings
    processor = MarkerProcessor()
    print("\n1. Testing initialization:")
    print("-------------------------")
    
    # Verify similarity threshold
    if processor.similarity_threshold != EXPECTED_RESULTS["similarity_threshold"]:
        validation_failures["similarity_threshold"] = {
            "expected": EXPECTED_RESULTS["similarity_threshold"],
            "actual": processor.similarity_threshold
        }
        print(f"  ❌ Incorrect similarity threshold: {processor.similarity_threshold}")
    else:
        print(f"  ✓ Correct similarity threshold: {processor.similarity_threshold}")
    
    # Test 2: TableMetadata class features
    print("\n2. Testing TableMetadata class:")
    print("-----------------------------")
    
    metadata = TableMetadata(
        table_id="table_1_0",
        page=1,
        is_multi_page=True,
        continued_from_page=None,
        continues_on_page=2
    )
    
    # Count fields to validate completeness
    metadata_fields = [f for f in dir(metadata) if not f.startswith('_') and not callable(getattr(metadata, f))]
    
    if len(metadata_fields) != EXPECTED_RESULTS["metadata_fields_count"]:
        validation_failures["metadata_fields_count"] = {
            "expected": EXPECTED_RESULTS["metadata_fields_count"],
            "actual": len(metadata_fields)
        }
        print(f"  ❌ Incorrect metadata field count: {len(metadata_fields)}")
    else:
        print(f"  ✓ Correct metadata field count: {len(metadata_fields)}")
    
    # Verify multi-page support in metadata
    multi_page_fields = ['is_multi_page', 'continued_from_page', 'continues_on_page']
    missing_fields = [f for f in multi_page_fields if not hasattr(metadata, f)]
    
    if missing_fields:
        validation_failures["multi_page_metadata"] = {
            "expected": multi_page_fields,
            "actual": [f for f in multi_page_fields if hasattr(metadata, f)]
        }
        print(f"  ❌ Missing required multi-page metadata fields: {missing_fields}")
    else:
        print(f"  ✓ All required multi-page metadata fields present")
    
    # Test 3: Table similarity calculation
    print("\n3. Testing table similarity calculation:")
    print("-------------------------------------")
    
    # Create test tables with different levels of similarity
    identical_tables = [
        {
            'header': ['Column1', 'Column2', 'Column3'],
            'data': [
                ['data1', 'data2', 'data3'],
                ['data4', 'data5', 'data6']
            ],
            'metadata': TableMetadata(
                column_count=3,
                column_widths=[10, 10, 10]
            )
        },
        {
            'header': ['Column1', 'Column2', 'Column3'],
            'data': [
                ['data7', 'data8', 'data9'],
                ['data10', 'data11', 'data12']
            ],
            'metadata': TableMetadata(
                column_count=3,
                column_widths=[10, 10, 10]
            )
        }
    ]
    
    similar_tables = [
        {
            'header': ['Column1', 'Column2', 'Column3'],
            'data': [
                ['data1', 'data2', 'data3']
            ],
            'metadata': TableMetadata(
                column_count=3,
                column_widths=[10, 10, 10]
            )
        },
        {
            'header': ['Column1', 'Column2', 'Different'],
            'data': [
                ['data4', 'data5', 'data6']
            ],
            'metadata': TableMetadata(
                column_count=3,
                column_widths=[12, 8, 10]
            )
        }
    ]
    
    different_tables = [
        {
            'header': ['Column1', 'Column2', 'Column3'],
            'data': [
                ['data1', 'data2', 'data3']
            ],
            'metadata': TableMetadata(
                column_count=3,
                column_widths=[10, 10, 10]
            )
        },
        {
            'header': ['Different1', 'Different2'],
            'data': [
                ['data4', 'data5']
            ],
            'metadata': TableMetadata(
                column_count=2,
                column_widths=[15, 15]
            )
        }
    ]
    
    # Test similarity calculations
    identical_similarity = processor._calculate_table_similarity(identical_tables[0], identical_tables[1])
    similar_similarity = processor._calculate_table_similarity(similar_tables[0], similar_tables[1])
    different_similarity = processor._calculate_table_similarity(different_tables[0], different_tables[1])
    
    print(f"  - Identical tables similarity: {identical_similarity:.2f}")
    print(f"  - Similar tables similarity: {similar_similarity:.2f}")
    print(f"  - Different tables similarity: {different_similarity:.2f}")
    
    # Verify results match expectations
    similarity_tests = {
        "identical_tables": (identical_similarity > 0.9),
        "similar_tables": (similar_similarity >= 0.7 and similar_similarity <= 0.9),
        "different_tables": (different_similarity < 0.7)
    }
    
    for test_name, passed in similarity_tests.items():
        if not passed:
            validation_failures[f"similarity_{test_name}"] = {
                "expected": f"{EXPECTED_RESULTS['table_similarity'][test_name]} (approximate)",
                "actual": identical_similarity if test_name == "identical_tables" else 
                         similar_similarity if test_name == "similar_tables" else
                         different_similarity
            }
            print(f"  ❌ {test_name} similarity test failed")
        else:
            print(f"  ✓ {test_name} similarity test passed")
    
    # Final validation result
    if not validation_failures:
        print("\n✅ VALIDATION COMPLETE - All marker processor features match expected values")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED - Results don't match expected values")
        for field, details in validation_failures.items():
            print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        print(f"Total errors: {len(validation_failures)} fields mismatched")
        sys.exit(1)
