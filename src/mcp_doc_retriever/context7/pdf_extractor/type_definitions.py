"""
Common type definitions for PDF extractor.

This module provides shared type definitions and aliases used across the PDF
extraction modules to ensure consistent typing and data validation. It defines
TypedDict classes and type aliases for structured data representation of PDF content.

Links to third-party package documentation:
    - typing: https://docs.python.org/3/library/typing.html
    - TypedDict: https://peps.python.org/pep-0589/

Example input/output:
    Input: Raw data from PDF extraction
    >>> raw_table = {
    ...     "page": 1,
    ...     "data": [["cell"]],
    ...     "accuracy": 0.95
    ... }
    >>> table_data: TableData = raw_table  # Type validation
    
    Output: Structured data with type safety
    >>> assert isinstance(table_data["page"], int)
    >>> assert isinstance(table_data["data"], list)
"""
import sys
# Use more modern typing syntax for Python 3.9+
from typing import TypedDict, Any, Union, Tuple, List, cast, Sequence

# Basic dictionary types
JsonDict = dict[str, Any]

# Common value types
MetadataValue = Union[str, int, float, list[str], dict[str, Any]]
NodeMetadata = dict[str, MetadataValue]

# Table types
class TableData(TypedDict, total=False):
    """Table data with row/column content and metadata."""
    page: int
    data: list[list[str]]
    accuracy: float
    bbox: Union[Tuple[float, float, float, float], list[float]]
    rows: int
    cols: int
    id: str

# Document types
class DocumentNode(TypedDict, total=False):
    """Document node with content and metadata."""
    type: str
    content: str
    metadata: NodeMetadata 
    children: list['DocumentNode']

# Result types
class ExtractedData(TypedDict):
    """Container for extraction results."""
    status: str
    content: Union[list[TableData], str]
    metadata: JsonDict

# Type aliases
TableList = list[TableData]
DocumentList = list[DocumentNode]

def ensure_node_metadata(data: dict[str, Any]) -> NodeMetadata:
    """Convert raw dictionary to proper metadata format."""
    result = {}
    for k, v in data.items():
        if isinstance(v, (str, int, float, list, dict)):
            result[k] = v
    return result

if __name__ == "__main__":
    print("Testing type definitions...")
    
    # Define expected results for validation
    EXPECTED_RESULTS = {
        "table_validation": True,
        "node_validation": True,
        "metadata_fields": 5,
        "result_validation": True
    }
    
    validation_passed = True
    actual_results = {}
    
    # Test TableData
    print("Testing TableData...")
    table: TableData = {
        "page": 1,
        "data": [["header"], ["cell"]],
        "accuracy": 95.5,
        "bbox": (0.0, 0.0, 100.0, 100.0),
        "rows": 2,
        "cols": 1,
        "id": "table_1"
    }
    
    try:
        assert isinstance(table["page"], int), "page should be int"
        assert isinstance(table["data"], list), "data should be list"
        actual_results["table_validation"] = True
        print("✓ TableData validated")
    except AssertionError as e:
        actual_results["table_validation"] = False
        validation_passed = False
        print(f"✗ TableData validation failed: {e}")
    
    # Test DocumentNode
    print("Testing DocumentNode...")
    node: DocumentNode = {
        "type": "text",
        "content": "Sample text",
        "metadata": {"source": "test"},
        "children": []
    }
    
    try:
        assert isinstance(node["type"], str), "type should be str"
        assert isinstance(node["metadata"], dict), "metadata should be dict"
        actual_results["node_validation"] = True
        print("✓ DocumentNode validated")
    except AssertionError as e:
        actual_results["node_validation"] = False
        validation_passed = False
        print(f"✗ DocumentNode validation failed: {e}")
    
    # Test metadata conversion
    print("Testing metadata conversion...")
    metadata = ensure_node_metadata({
        "str_field": "text",
        "int_field": 42,
        "float_field": 3.14,
        "list_field": ["a", "b"],
        "dict_field": {"key": "value"},
        "invalid_field": lambda x: x  # Should be filtered out
    })
    
    try:
        assert "invalid_field" not in metadata, "Invalid field should be filtered"
        assert len(metadata) == 5, f"Should have 5 valid fields, got {len(metadata)}"
        actual_results["metadata_fields"] = len(metadata)
        print("✓ Metadata conversion validated")
    except AssertionError as e:
        actual_results["metadata_fields"] = len(metadata) if "metadata" in locals() else 0
        validation_passed = False
        print(f"✗ Metadata conversion validation failed: {e}")
    
    # Test extraction results
    print("Testing ExtractedData...")
    result: ExtractedData = {
        "status": "success",
        "content": [table],
        "metadata": {"source": "test"}
    }
    
    try:
        assert isinstance(result["content"], list), "content should be list"
        actual_results["result_validation"] = True
        print("✓ ExtractedData validated")
    except AssertionError as e:
        actual_results["result_validation"] = False
        validation_passed = False
        print(f"✗ ExtractedData validation failed: {e}")
    
    # Final validation check
    print("\nValidation Results:")
    for key, expected in EXPECTED_RESULTS.items():
        actual = actual_results.get(key)
        match = actual == expected
        print(f"  {key}: {'✓' if match else '✗'} Expected: {expected}, Got: {actual}")
        if not match:
            validation_passed = False
    
    if validation_passed:
        print("\n✅ VALIDATION COMPLETE - All results match expected values")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED - Results don't match expected values")
        print(f"Expected: {EXPECTED_RESULTS}")
        print(f"Got: {actual_results}")
        sys.exit(1)
