"""
Parsers for various document formats in PDF extraction workflow.

This module provides parsers for different document formats encountered during PDF
extraction, including structured text parsing, table parsing, and metadata extraction.
It supports customization through filter functions and offers consistent error handling.

Third-party package documentation:
- pypdf: https://pypi.org/project/pypdf/
- pdfplumber: https://github.com/jsvine/pdfplumber
- tabulate: https://pypi.org/project/tabulate/

Example usage:
    >>> from mcp_doc_retriever.context7.pdf_extractor.parsers import parse_structured_text
    >>> text = "# Heading\\n\\nParagraph with **bold** text"
    >>> result = parse_structured_text(text)
    >>> print(f"Found {len(result)} blocks including {result[0]['type']}")
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Union, TypedDict
from loguru import logger

# Handle imports for both standalone and module usage
if __name__ == "__main__":
    # When run as a script, configure system path
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir
    while not (project_root / 'pyproject.toml').exists() and project_root != project_root.parent:
        project_root = project_root.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root / "src"))
    
    # Import after path setup for standalone usage
    try:
        from mcp_doc_retriever.context7.pdf_extractor.types import DocumentNode
    except ImportError:
        logger.warning("Could not import required modules for standalone execution")
        
        # Define minimal version for standalone testing
        class DocumentNode(TypedDict, total=False):
            """Document node with content and metadata."""
            type: str
            content: str
            metadata: Dict[str, Any]
            children: List['DocumentNode']
else:
    # When imported as a module, use relative imports
    try:
        from .types import DocumentNode
    except ImportError:
        # Fallback to absolute imports
        from mcp_doc_retriever.context7.pdf_extractor.types import DocumentNode

def parse_structured_text(
    text: str, 
    filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None
) -> List[Dict[str, Any]]:
    """
    Parse structured text into blocks with type information.
    
    Args:
        text: The text to parse
        filter_func: Optional function to filter blocks based on content
        
    Returns:
        List of dictionaries representing text blocks with type and content
    """
    if not text:
        return []
        
    blocks = []
    lines = text.split("\n")
    current_block = {"type": "paragraph", "content": "", "metadata": {}}
    
    for line in lines:
        # Check for headings
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            # Save current block if non-empty
            if current_block["content"]:
                blocks.append(current_block)
                
            # Create heading block
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            current_block = {
                "type": "heading",
                "content": heading_text,
                "metadata": {"level": level}
            }
            blocks.append(current_block)
            current_block = {"type": "paragraph", "content": "", "metadata": {}}
            continue
            
        # Check for code blocks
        if line.startswith("```"):
            # Save current block if non-empty
            if current_block["content"]:
                blocks.append(current_block)
                
            # Extract language if specified
            language = line[3:].strip()
            # Create code block
            current_block = {
                "type": "code",
                "content": "",
                "metadata": {"language": language}
            }
            blocks.append(current_block)
            current_block = {"type": "paragraph", "content": "", "metadata": {}}
            continue
            
        # Handle empty lines - end current paragraph
        if not line.strip() and current_block["content"]:
            blocks.append(current_block)
            current_block = {"type": "paragraph", "content": "", "metadata": {}}
            continue
            
        # Add line to current block
        if current_block["content"]:
            current_block["content"] += "\n"
        current_block["content"] += line
        
    # Add final block if non-empty
    if current_block["content"]:
        blocks.append(current_block)
        
    # Apply filter if provided
    if filter_func:
        blocks = [block for block in blocks if filter_func(block)]
        
    return blocks

def parse_table_data(
    table_data: List[List[str]]
) -> Dict[str, Any]:
    """
    Parse table data into structured format with headers and rows.
    
    Args:
        table_data: List of lists representing table rows and cells
        
    Returns:
        Dictionary with headers and rows
    """
    if not table_data or len(table_data) < 2:
        return {"headers": [], "rows": []}
        
    headers = [cell.strip() for cell in table_data[0]]
    rows = []
    
    # Process data rows
    for row in table_data[1:]:
        # Ensure row has same number of cells as headers
        if len(row) < len(headers):
            row += [""] * (len(headers) - len(row))
        elif len(row) > len(headers):
            row = row[:len(headers)]
            
        # Strip whitespace from cells
        row = [cell.strip() for cell in row]
        rows.append(row)
        
    return {
        "headers": headers,
        "rows": rows
    }

def extract_metadata(text: str) -> Dict[str, str]:
    """
    Extract metadata from text using common patterns.
    
    Args:
        text: Text to extract metadata from
        
    Returns:
        Dictionary of metadata key-value pairs
    """
    metadata = {}
    
    # Look for common metadata patterns
    # Pattern: Key: Value
    kv_pattern = re.compile(r"^([A-Za-z0-9_\- ]+):\s*(.+)$", re.MULTILINE)
    for match in kv_pattern.finditer(text):
        key = match.group(1).strip().lower().replace(" ", "_")
        value = match.group(2).strip()
        metadata[key] = value
        
    # Look for title patterns
    title_match = re.search(r"^(?:Title|#)\s*[:.\s]*\s*(.+)$", text, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()
        
    # Look for author patterns
    author_match = re.search(r"^(?:Author|By)\s*[:.\s]*\s*(.+)$", text, re.MULTILINE)
    if author_match:
        metadata["author"] = author_match.group(1).strip()
        
    # Look for date patterns
    date_match = re.search(r"^(?:Date|Published)\s*[:.\s]*\s*(.+)$", text, re.MULTILINE)
    if date_match:
        metadata["date"] = date_match.group(1).strip()
        
    return metadata

def convert_to_document_nodes(
    blocks: List[Dict[str, Any]]
) -> List[DocumentNode]:
    """
    Convert parsed blocks to DocumentNode format.
    
    Args:
        blocks: List of parsed blocks
        
    Returns:
        List of DocumentNode objects
    """
    nodes: List[DocumentNode] = []
    
    for block in blocks:
        node: DocumentNode = {
            "type": block["type"],
            "content": block["content"],
            "metadata": block["metadata"],
            "children": []
        }
        nodes.append(node)
        
    return nodes

if __name__ == "__main__":
    import logging
    import warnings
    from loguru import logger
    
    # Set up logging for better debugging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    print("PARSERS MODULE VERIFICATION")
    print("==========================")
    
    # CRITICAL: Define exact expected results for validation
    # These must match exactly or the test fails
    EXPECTED_RESULTS = {
        "structured_text": {
            "block_count": 5,
            "block_types": ["heading", "paragraph", "heading", "code", "paragraph"],
            "heading_levels": [1, 2]
        },
        "table_data": {
            "header_count": 3,
            "row_count": 2
        },
        "metadata": {
            "keys": ["title", "author", "date", "version"],
            "title_value": "Test Document"
        }
    }
    
    # Track validation status
    all_tests_passed = True
    
    # Test 1: Parse structured text
    print("\n1. Testing structured text parsing:")
    print("--------------------------------")
    
    test_text = """# Test Document
    
This is a test paragraph with some content.

## Code Example

```python
def hello_world():
    print("Hello, world!")
```

Final paragraph with additional text.
"""
    
    blocks = parse_structured_text(test_text)
    
    # Display results
    if blocks:
        print(f"Parsed {len(blocks)} blocks:")
        for i, block in enumerate(blocks):
            block_type = block["type"]
            content_preview = block["content"][:30] + "..." if len(block["content"]) > 30 else block["content"]
            print(f"  - Block {i+1}: {block_type} - {content_preview}")
            if block_type == "heading" and "level" in block["metadata"]:
                print(f"    Level: {block['metadata']['level']}")
    else:
        print("No blocks parsed")
        all_tests_passed = False
    
    # VALIDATION - Structured Text Parsing
    print("\n" Validating structured text parsing:")
    print("----------------------------------")
    
    # Check block count
    expected_count = EXPECTED_RESULTS["structured_text"]["block_count"]
    if len(blocks) != expected_count:
        print(f"   FAIL: Block count mismatch! Expected {expected_count}, got {len(blocks)}")
        all_tests_passed = False
    else:
        print(f"   PASS: Block count matches expected ({expected_count})")
    
    # Check block types
    expected_types = EXPECTED_RESULTS["structured_text"]["block_types"]
    actual_types = [block["type"] for block in blocks]
    
    if len(actual_types) != len(expected_types):
        print(f"   FAIL: Block type count mismatch! Expected {len(expected_types)}, got {len(actual_types)}")
        all_tests_passed = False
    else:
        types_match = all(a == e for a, e in zip(actual_types, expected_types))
        if not types_match:
            print(f"   FAIL: Block types don't match expected sequence")
            print(f"    Expected: {expected_types}")
            print(f"    Got: {actual_types}")
            all_tests_passed = False
        else:
            print(f"   PASS: Block types match expected sequence")
    
    # Check heading levels
    expected_levels = EXPECTED_RESULTS["structured_text"]["heading_levels"]
    actual_levels = [block["metadata"]["level"] for block in blocks if block["type"] == "heading"]
    
    if len(actual_levels) != len(expected_levels):
        print(f"   FAIL: Heading level count mismatch! Expected {len(expected_levels)}, got {len(actual_levels)}")
        all_tests_passed = False
    else:
        levels_match = all(a == e for a, e in zip(actual_levels, expected_levels))
        if not levels_match:
            print(f"   FAIL: Heading levels don't match expected sequence")
            print(f"    Expected: {expected_levels}")
            print(f"    Got: {actual_levels}")
            all_tests_passed = False
        else:
            print(f"   PASS: Heading levels match expected sequence")
    
    # Test 2: Parse table data
    print("\n2. Testing table data parsing:")
    print("----------------------------")
    
    test_table = [
        ["Name", "Age", "Occupation"],
        ["John", "30", "Engineer"],
        ["Alice", "25", "Designer"]
    ]
    
    table_result = parse_table_data(test_table)
    
    # Display results
    if table_result and "headers" in table_result and "rows" in table_result:
        print(f"Parsed table with {len(table_result['headers'])} headers and {len(table_result['rows'])} rows")
        print(f"  - Headers: {', '.join(table_result['headers'])}")
        for i, row in enumerate(table_result['rows']):
            print(f"  - Row {i+1}: {', '.join(row)}")
    else:
        print("Table parsing failed")
        all_tests_passed = False
    
    # VALIDATION - Table Data Parsing
    print("\n" Validating table data parsing:")
    print("------------------------------")
    
    # Check header count
    expected_header_count = EXPECTED_RESULTS["table_data"]["header_count"]
    actual_header_count = len(table_result["headers"])
    if actual_header_count != expected_header_count:
        print(f"   FAIL: Header count mismatch! Expected {expected_header_count}, got {actual_header_count}")
        all_tests_passed = False
    else:
        print(f"   PASS: Header count matches expected ({expected_header_count})")
    
    # Check row count
    expected_row_count = EXPECTED_RESULTS["table_data"]["row_count"]
    actual_row_count = len(table_result["rows"])
    if actual_row_count != expected_row_count:
        print(f"   FAIL: Row count mismatch! Expected {expected_row_count}, got {actual_row_count}")
        all_tests_passed = False
    else:
        print(f"   PASS: Row count matches expected ({expected_row_count})")
    
    # Test 3: Extract metadata
    print("\n3. Testing metadata extraction:")
    print("-----------------------------")
    
    test_metadata_text = """# Test Document
Author: John Doe
Date: 2025-04-20
Version: 1.0.0

This is the content of the document.
"""
    
    metadata = extract_metadata(test_metadata_text)
    
    # Display results
    if metadata:
        print(f"Extracted {len(metadata)} metadata items:")
        for key, value in metadata.items():
            print(f"  - {key}: {value}")
    else:
        print("No metadata extracted")
        all_tests_passed = False
    
    # VALIDATION - Metadata Extraction
    print("\n" Validating metadata extraction:")
    print("------------------------------")
    
    # Check metadata keys
    expected_keys = set(EXPECTED_RESULTS["metadata"]["keys"])
    actual_keys = set(metadata.keys())
    
    missing_keys = expected_keys - actual_keys
    if missing_keys:
        print(f"   FAIL: Missing expected metadata keys: {', '.join(missing_keys)}")
        all_tests_passed = False
    else:
        print(f"   PASS: All expected metadata keys are present")
    
    # Check title value
    expected_title = EXPECTED_RESULTS["metadata"]["title_value"]
    actual_title = metadata.get("title", "")
    if actual_title != expected_title:
        print(f"   FAIL: Title value mismatch! Expected '{expected_title}', got '{actual_title}'")
        all_tests_passed = False
    else:
        print(f"   PASS: Title value matches expected ('{expected_title}')")
    
    # Test 4: Convert to document nodes
    print("\n4. Testing conversion to document nodes:")
    print("-------------------------------------")
    
    document_nodes = convert_to_document_nodes(blocks)
    
    # Display results
    if document_nodes:
        print(f"Converted {len(document_nodes)} blocks to document nodes")
        for i, node in enumerate(document_nodes):
            print(f"  - Node {i+1}: {node['type']}")
    else:
        print("No document nodes created")
        all_tests_passed = False
    
    # VALIDATION - Document Node Conversion
    print("\n" Validating document node conversion:")
    print("-----------------------------------")
    
    if len(document_nodes) == len(blocks):
        print(f"   PASS: Document node count matches block count ({len(blocks)})")
    else:
        print(f"   FAIL: Document node count mismatch! Expected {len(blocks)}, got {len(document_nodes)}")
        all_tests_passed = False
    
    # Check node types match original block types
    node_types = [node["type"] for node in document_nodes]
    block_types = [block["type"] for block in blocks]
    
    if node_types == block_types:
        print(f"   PASS: Document node types match original block types")
    else:
        print(f"   FAIL: Document node types don't match original block types")
        print(f"    Original: {block_types}")
        print(f"    Converted: {node_types}")
        all_tests_passed = False
    
    # FINAL VALIDATION - All tests
    if all_tests_passed:
        print("\n ALL VALIDATION CHECKS PASSED - VERIFICATION COMPLETE!")
        sys.exit(0)
    else:
        print("\nL VALIDATION FAILED - Results don't match expected output")
        sys.exit(1)