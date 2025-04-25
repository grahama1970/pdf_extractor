# ArangoDB Integration for PDF Extractor

## Overview

This module provides integration between the pdf_extractor project and ArangoDB for storing and querying extracted PDF content. The integration enables efficient storage, retrieval, and querying of PDF extraction results.

## Features

- Store PDF extraction results in ArangoDB
- Query by document type (heading, paragraph, table)
- Text search across extraction results
- Structured storage with indexing for optimal performance

## Requirements

- ArangoDB server (running locally or remotely)
- Python 3.11+
- python-arango package
- Environment variables for ArangoDB connection

## Directory Structure

```
/pdf_extractor/arangodb/
├── connection.py       - ArangoDB connection management
├── pdf_integration.py  - PDF extraction integration
├── simple_integration.py - Simple integration test
├── pdf_arango_example.py - Comprehensive example
└── README.md           - This documentation
```

## Configuration

Set the following environment variables for ArangoDB connection:

```bash
export ARANGO_HOST="http://localhost:8529"
export ARANGO_USER="root"
export ARANGO_PASSWORD="your_password"
export ARANGO_DB="pdf_extractor"
```

Alternatively, the code will use default values if these are not set.

## Basic Usage

### Connecting to ArangoDB

```python
from pdf_extractor.arangodb.connection import get_db

# Connect to ArangoDB
db = get_db()
```

### Storing PDF Extraction Results

```python
from pdf_extractor.arangodb.pdf_arango_example import store_pdf_extraction

# Example extraction result
extraction_result = {
    "type": "heading",
    "level": 1,
    "text": "Introduction",
    "page": 1,
    "token_count": 1,
    "file_path": "sample.pdf",
    "extraction_date": "2025-04-21T12:00:00.000000",
    "source": "marker"
}

# Store the result
stored_count = store_pdf_extraction(db, [extraction_result])
```

### Querying by Document Type

```python
from pdf_extractor.arangodb.pdf_arango_example import query_by_type

# Query for headings
headings = query_by_type(db, "heading")

# Process results
for heading in headings:
    print(f"Heading: {heading['text']} (Page {heading['page']})")
```

### Text Search

```python
from pdf_extractor.arangodb.pdf_arango_example import text_search

# Search for text containing "introduction"
results = text_search(db, "introduction")

# Process results
for doc in results:
    print(f"{doc['type']}: {doc['text'][:50]}...")
```

## Document Schema

PDF extraction results are stored with the following schema:

```json
{
  "type": "heading|paragraph|table|metadata",
  "text": "Extracted text content",
  "page": 1,
  "token_count": 15,
  "file_path": "sample.pdf",
  "extraction_date": "2025-04-21T12:00:00.000000",
  "source": "marker|camelot|qwen",
  
  // Type-specific fields
  "level": 1,                     // For headings
  "caption": "Table 1",           // For tables
  "headers": ["Col1", "Col2"],    // For tables
  "rows": [["Val1", "Val2"]]      // For tables
}
```

## Indexes

The following indexes are created for optimal performance:

- Hash index on `type` field for fast filtering by document type
- Hash index on `file_path` field for filtering by PDF file
- Skiplist index on `page` field for range queries and sorting
- Fulltext index on `text` field for text search

## Validation

The ArangoDB integration includes validation to ensure:

1. Required fields are present in documents
2. Type-specific fields are present (e.g., `level` for headings)
3. Successful database operations

For comprehensive validation, run:

```bash
python -m pdf_extractor.arangodb.simple_integration
```

## Examples

For a complete working example, see:

```bash
python -m pdf_extractor.arangodb.pdf_arango_example
```

This example demonstrates:
- Connecting to ArangoDB
- Storing mock PDF extraction results
- Querying by document type
- Performing text search
- Cleaning up test data

## Troubleshooting

1. **Connection Issues**: Ensure ArangoDB is running and accessible at the configured host
2. **Authentication Issues**: Verify username and password are correct
3. **Missing Indexes**: The code will attempt to create indexes automatically

## References

- ArangoDB Documentation: https://www.arangodb.com/docs/
- python-arango: https://python-arango.readthedocs.io/

## Comprehensive Validation

This module includes comprehensive validation functions that verify actual results against expected results, following the requirements in VALIDATION_REQUIREMENTS.md.

The validation functions:

1. **Connection Validation**: Verifies that the connection to ArangoDB is successful and the database name matches expectations.

2. **Collection Setup Validation**: Validates that the collection is properly created with all required indexes.

3. **Document Storage Validation**: Ensures that documents are stored correctly and can be retrieved with expected values.

4. **Query by Type Validation**: Verifies that querying by document type returns the expected documents.

5. **Text Search Validation**: Confirms that text search functionality returns the expected documents.

To run the comprehensive validation:

```bash
python -m pdf_extractor.arangodb.validate_functions
```

This validation script will:
- Create a test collection
- Insert test documents
- Validate each function against expected results
- Report any mismatches between expected and actual values
- Clean up test data

If successful, the script will output "ALL VALIDATIONS PASSED", confirming that the ArangoDB integration is working correctly.
