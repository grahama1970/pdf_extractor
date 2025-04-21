"""
Process PDF markers and annotations.

This module handles extraction and processing of PDF markers, annotations,
and table markers, converting them to structured data. It supports both
markdown and JSON marker formats for easier integration with documentation.

Third-party package documentation:
- pypdf: https://pypdf.readthedocs.io/
- pdfplumber: https://github.com/jsvine/pdfplumber

Example usage:
    >>> from mcp_doc_retriever.context7.pdf_extractor.marker_processor import process_marker
    >>> pdf_path = "example.pdf"
    >>> repo_link = "https://github.com/myorg/myrepo"
    >>> nodes = process_marker(pdf_path, repo_link)
    >>> for node in nodes:
    ...     print(f"Found {node['type']} with {len(node.get('children', []))} children")
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import cast, Optional, Sequence, Union, Dict, Any, TypedDict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import types from our package using absolute imports
from mcp_doc_retriever.context7.pdf_extractor.type_definitions import (
    TableData,
    DocumentNode,
    JsonDict,
    ExtractedData,
    TableList,
    DocumentList
)

# Import table_extractor
from mcp_doc_retriever.context7.pdf_extractor.table_extractor import extract_tables

# Define TableMetadata here to avoid circular imports
class TableMetadata(TypedDict, total=False):
    """Metadata for tables extracted from PDFs."""
    page: int
    rows: int
    cols: int
    accuracy: float
    bbox: Union[Tuple[float, float, float, float], List[float]]
    source: str

def process_tables_from_markers(
    marker_json: Optional[JsonDict],
    pdf_path: str,
    repo_link: str
) -> ExtractedData:
    """
    Process tables based on JSON markers.
    
    Args:
        marker_json: JSON data containing table markers
        pdf_path: Path to PDF file
        repo_link: Repository URL for metadata
    
    Returns:
        ExtractedData containing tables or error message
    """
    try:
        if not marker_json:
            return {"status": "error", "content": "No marker data provided", "metadata": {}}
            
        if "tables" not in marker_json:
            return {"status": "error", "content": "No table markers found", "metadata": {}}
            
        raw_tables = extract_tables(pdf_path)
        if not raw_tables:
            return {"status": "error", "content": "No tables extracted", "metadata": {}}
            
        # Convert raw tables to proper TableData
        tables: TableList = [cast(TableData, table) for table in raw_tables]
        
        # Assign unique IDs
        tables = _assign_unique_table_ids(tables)
        
        return {
            "status": "success",
            "content": tables,
            "metadata": {
                "table_count": len(tables),
                "source": "marker",
                "repo": repo_link
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to process tables from markers: {e}")
        return {
            "status": "error",
            "content": str(e),
            "metadata": {}
        }

def _assign_unique_table_ids(tables: TableList) -> TableList:
    """Add unique IDs to tables based on position."""
    result = []
    for i, table in enumerate(tables, 1):
        # Create safe copy and ensure required fields
        table_copy = {
            "page": table.get("page", 0),
            "data": table.get("data", []),
            "accuracy": table.get("accuracy", 0.0),
            "bbox": table.get("bbox", (0.0, 0.0, 0.0, 0.0)),
            "rows": table.get("rows", 0),
            "cols": table.get("cols", 0),
            "id": f"table_{i}"
        }
        result.append(cast(TableData, table_copy))
    return result

def parse_markdown(
    markdown_content: str,
    pdf_path: str, 
    repo_link: str
) -> DocumentList:
    """
    Parse markdown content for table markers and descriptions.
    
    Args:
        markdown_content: Markdown text to parse
        pdf_path: Path to PDF being processed
        repo_link: Repository URL for metadata
    
    Returns:
        List of document nodes containing tables and text
    """
    nodes: DocumentList = []
    
    try:
        # Extract tables based on markdown markers
        result = process_tables_from_markers(
            {"tables": []},  # TODO: Parse table markers from markdown
            pdf_path,
            repo_link
        )
        
        if result["status"] == "success" and isinstance(result["content"], list):
            for table in result["content"]:
                metadata: TableMetadata = {
                    "page": table.get("page", 0),
                    "rows": table.get("rows", 0),
                    "cols": table.get("cols", 0),
                    "accuracy": table.get("accuracy", 0.0),
                    "bbox": table.get("bbox", [0.0, 0.0, 0.0, 0.0])
                }
                
                nodes.append({
                    "type": "table",
                    "content": json.dumps(table.get("data", [])),
                    "metadata": metadata,
                    "children": []
                })
                
    except Exception as e:
        logger.error(f"Failed to parse markdown: {e}")
        
    return nodes

def process_marker(
    pdf_path: str,
    repo_link: str,
    use_markdown: bool = False
) -> DocumentList:
    """
    Process PDF markers using either markdown or JSON format.
    
    Args:
        pdf_path: Path to PDF file
        repo_link: Repository URL for metadata
        use_markdown: Whether to use markdown parsing
        
    Returns:
        List of document nodes containing extracted content
    """
    try:
        if use_markdown:
            # Read and parse markdown file
            md_path = Path(pdf_path).with_suffix(".md")
            if not md_path.exists():
                logger.warning(f"No markdown file found at {md_path}")
                return []
                
            with open(md_path) as f:
                markdown = f.read()
                
            return parse_markdown(markdown, pdf_path, repo_link)
            
        else:
            # Read and parse JSON markers
            json_path = Path(pdf_path).with_suffix(".json")
            if not json_path.exists():
                logger.warning(f"No JSON markers found at {json_path}")
                return []
                
            with open(json_path) as f:
                marker_json = json.load(f)
                
            result = process_tables_from_markers(
                marker_json,
                pdf_path,
                repo_link
            )
            
            if result["status"] == "success" and isinstance(result["content"], list):
                nodes: DocumentList = []
                for table in result["content"]:
                    metadata: TableMetadata = {
                        "page": table.get("page", 0),
                        "rows": table.get("rows", 0),
                        "cols": table.get("cols", 0),
                        "accuracy": table.get("accuracy", 0.0),
                        "bbox": table.get("bbox", [0.0, 0.0, 0.0, 0.0])
                    }
                    nodes.append({
                        "type": "table",
                        "content": json.dumps(table.get("data", [])),
                        "metadata": metadata,
                        "children": []
                    })
                return nodes
                
            return []
            
    except Exception as e:
        logger.error(f"Failed to process markers: {e}")
        return []

if __name__ == "__main__":
    import warnings
    import sys
    
    # Filter out specific camelot warning about table areas that doesn't indicate failure
    warnings.filterwarnings("ignore", message="No tables found in table area.*")
    
    print("MARKER PROCESSOR MODULE VERIFICATION")
    print("===================================")
    
    # CRITICAL: Define exact expected results for validation
    # These must match exactly or the test fails
    EXPECTED_RESULTS = {
        "table_processing": {
            "status": "success",
            "table_count": 2,
            "tables": [
                {
                    "page": 1,
                    "min_rows": 1,
                    "min_cols": 2
                },
                {
                    "page": 2,
                    "min_rows": 2,
                    "min_cols": 2
                }
            ]
        },
        "document_nodes": {
            "node_count": 2,
            "node_types": ["table", "table"]
        }
    }
    
    # Track validation status
    validation_passed = True
    actual_results = {
        "table_processing": {
            "status": "",
            "table_count": 0,
            "tables": []
        },
        "document_nodes": {
            "node_count": 0,
            "node_types": []
        }
    }
    
    # Look for test files in the input directory
    input_dir = Path(__file__).parent / "input"
    test_pdf = input_dir / "BHT_CV32A65X.pdf"
    json_file = test_pdf.with_suffix(".json")
    
    # Create a JSON marker file if it doesn't exist
    if not json_file.exists():
        # Create a test marker file
        test_marker = {
            "tables": [
                {"page": 1, "bbox": [50, 50, 500, 200], "description": "Test table 1"},
                {"page": 2, "bbox": [100, 100, 400, 300], "description": "Test table 2"}
            ]
        }
        with open(json_file, "w") as f:
            json.dump(test_marker, f)
        print(f"Created test JSON marker file: {json_file}")
    
    if test_pdf.exists():
        print(f"Testing with file: {test_pdf}")
        print(f"File exists: {test_pdf.exists()}")
        print(f"File size: {test_pdf.stat().st_size} bytes")
        print(f"JSON marker file: {json_file.exists()}")
        
        try:
            # TEST 1: Table processing from markers
            print("\n1. Testing table processing from markers:")
            print("---------------------------------------")
            
            # Load marker data from the JSON file
            with open(json_file) as f:
                test_marker = json.load(f)
                
            result = process_tables_from_markers(test_marker, str(test_pdf), "https://github.com/test/repo")
            
            # Record results for validation
            actual_results["table_processing"]["status"] = result["status"]
            
            if result["status"] == "success":
                print(f"  ✓ Processed {len(result['content'])} tables")
                actual_results["table_processing"]["table_count"] = len(result["content"])
                
                for i, table in enumerate(result['content'], 1):
                    print(f"  - Table {i}:")
                    print(f"    - Page: {table.get('page', 'unknown')}")
                    print(f"    - Size: {table.get('rows', 0)}x{table.get('cols', 0)}")
                    print(f"    - ID: {table.get('id', 'unknown')}")
                    
                    # Record table info for validation
                    actual_results["table_processing"]["tables"].append({
                        "page": table.get("page", 0),
                        "rows": table.get("rows", 0),
                        "cols": table.get("cols", 0)
                    })
            else:
                print(f"  ✗ Failed: {result.get('content', 'Unknown error')}")
                validation_passed = False
            
            # VALIDATION - Table Processing Results
            print("\n• Validating table processing results:")
            print("------------------------------------")
            
            # Check overall status
            expected_status = EXPECTED_RESULTS["table_processing"]["status"]
            actual_status = actual_results["table_processing"]["status"]
            if actual_status != expected_status:
                print(f"  ✗ FAIL: Status mismatch! Expected '{expected_status}', got '{actual_status}'")
                validation_passed = False
            else:
                print(f"  ✓ PASS: Status matches expected ('{expected_status}')")
            
            # Check table count
            expected_count = EXPECTED_RESULTS["table_processing"]["table_count"]
            actual_count = actual_results["table_processing"]["table_count"]
            if actual_count != expected_count:
                print(f"  ✗ FAIL: Table count mismatch! Expected {expected_count}, got {actual_count}")
                validation_passed = False
            else:
                print(f"  ✓ PASS: Table count matches expected ({expected_count})")
            
            # Validate each table
            expected_tables = EXPECTED_RESULTS["table_processing"]["tables"]
            for i, expected_table in enumerate(expected_tables):
                if i >= len(actual_results["table_processing"]["tables"]):
                    print(f"  ✗ FAIL: Missing expected table at index {i}")
                    validation_passed = False
                    continue
                
                actual_table = actual_results["table_processing"]["tables"][i]
                print(f"\n  • Validating Table {i+1}:")
                
                # Check page
                actual_page = actual_table.get("page", None)
                expected_page = expected_table.get("page")
                if actual_page != expected_page:
                    print(f"    ✗ FAIL: Page mismatch! Expected {expected_page}, got {actual_page}")
                    validation_passed = False
                else:
                    print(f"    ✓ PASS: Page matches expected ({expected_page})")
                
                # Check minimum rows
                actual_rows = actual_table.get("rows", 0)
                min_rows = expected_table.get("min_rows", 0)
                if actual_rows < min_rows:
                    print(f"    ✗ FAIL: Row count too low! Expected at least {min_rows}, got {actual_rows}")
                    validation_passed = False
                else:
                    print(f"    ✓ PASS: Row count meets minimum ({actual_rows} >= {min_rows})")
                
                # Check minimum columns
                actual_cols = actual_table.get("cols", 0)
                min_cols = expected_table.get("min_cols", 0)
                if actual_cols < min_cols:
                    print(f"    ✗ FAIL: Column count too low! Expected at least {min_cols}, got {actual_cols}")
                    validation_passed = False
                else:
                    print(f"    ✓ PASS: Column count meets minimum ({actual_cols} >= {min_cols})")
            
            # TEST 2: Document node generation
            print("\n2. Testing document node generation:")
            print("-----------------------------------")
            nodes = process_marker(str(test_pdf), "https://github.com/test/repo", use_markdown=False)
            
            # Record node info for validation
            actual_results["document_nodes"]["node_count"] = len(nodes)
            actual_results["document_nodes"]["node_types"] = [node.get("type", "unknown") for node in nodes]
            
            if not nodes:
                print(f"  ✗ FAIL: No document nodes generated")
                validation_passed = False
            else:
                print(f"  ✓ Generated {len(nodes)} document nodes")
                for i, node in enumerate(nodes, 1):
                    print(f"  - Node {i}:")
                    print(f"    - Type: {node.get('type', 'unknown')}")
                    metadata = node.get('metadata', {})
                    print(f"    - Page: {metadata.get('page', 'unknown')}")
                    print(f"    - Size: {metadata.get('rows', 0)}x{metadata.get('cols', 0)}")
                    print(f"    - Children: {len(node.get('children', []))}")
            
            # VALIDATION - Document Nodes
            print("\n• Validating document nodes:")
            print("---------------------------")
            
            # Check node count
            expected_count = EXPECTED_RESULTS["document_nodes"]["node_count"]
            actual_count = actual_results["document_nodes"]["node_count"]
            if actual_count != expected_count:
                print(f"  ✗ FAIL: Node count mismatch! Expected {expected_count}, got {actual_count}")
                validation_passed = False
            else:
                print(f"  ✓ PASS: Node count matches expected ({expected_count})")
            
            # Check node types
            expected_types = EXPECTED_RESULTS["document_nodes"]["node_types"]
            for i, expected_type in enumerate(expected_types):
                if i >= len(actual_results["document_nodes"]["node_types"]):
                    print(f"  ✗ FAIL: Missing expected node at index {i}")
                    validation_passed = False
                    continue
                
                actual_type = actual_results["document_nodes"]["node_types"][i]
                if actual_type != expected_type:
                    print(f"  ✗ FAIL: Node {i+1} type mismatch! Expected '{expected_type}', got '{actual_type}'")
                    validation_passed = False
                else:
                    print(f"  ✓ PASS: Node {i+1} type matches expected ('{expected_type}')")
            
            # FINAL VALIDATION - All tests
            if validation_passed:
                print("\n✅ VALIDATION COMPLETE - All results match expected values")
                sys.exit(0)
            else:
                print("\n❌ VALIDATION FAILED - Results don't match expected values")
                print(f"Expected: {json.dumps(EXPECTED_RESULTS, indent=2)}")
                print(f"Got: {json.dumps(actual_results, indent=2)}")
                sys.exit(1)
            
        except Exception as e:
            print(f"\n✗ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"\n✗ Test PDF file not found: {test_pdf}")
        sys.exit(1)
