# ARANGODB INTEGRATION
> Guidelines for integrating pdf_extractor with ArangoDB

## Overview

The pdf_extractor project uses ArangoDB for storing and querying extracted PDF content. This integration enables powerful semantic, BM25, keyword, and hybrid querying capabilities essential for effective document retrieval.

## Integration Components

The core ArangoDB integration components can be found in:
```
/Users/robert/Documents/dev/workspace/experiments/mcp-doc-retriver/src/mcp_doc_retriever/arangodb
```

These components provide crucial functionality for:
- Document storage
- Collection management
- Query processing
- Search optimization

## Query Types

The ArangoDB integration supports multiple query types:

1. **Semantic Search**:
   - Utilizes vector embeddings
   - Supports similarity-based content retrieval
   - Enables context-aware document discovery

2. **BM25 Search**:
   - Implements the BM25 ranking algorithm
   - Provides keyword-based relevance scoring
   - Supports traditional document retrieval

3. **Keyword Search**:
   - Offers exact and partial keyword matching
   - Enables targeted content retrieval
   - Supports Boolean operators

4. **Hybrid Search**:
   - Combines semantic and keyword approaches
   - Balances precision and recall
   - Provides optimal search results

## Implementation Requirements

When implementing the ArangoDB integration:

1. **Document Structure**:
   - Store extracted JSON content in ArangoDB collections
   - Maintain section hierarchy in the document structure
   - Include metadata for efficient filtering

2. **Connection Management**:
   - Implement proper connection pooling
   - Add robust error handling for database operations
   - Include reconnection logic for resilience

3. **Query Construction**:
   - Build AQL queries programmatically
   - Support filtering by document properties
   - Enable pagination and result limiting

4. **Result Processing**:
   - Parse and normalize query results
   - Maintain document structure in responses
   - Include relevance scoring where appropriate

## Code Example

```python
from arango import ArangoClient

# Initialize ArangoDB client
client = ArangoClient(hosts="http://localhost:8529")
db = client.db("pdf_extractor", username="root", password="password")

# Store document
collection = db.collection("pdf_documents")
doc = {
    "type": "heading",
    "level": 1,
    "text": "Introduction",
    "page": 1,
    "token_count": 1,
    "file_path": "sample.pdf",
    "extraction_date": "2025-04-21T12:00:00.000000",
    "source": "marker"
}
collection.insert(doc)

# Query document
aql = """
FOR doc IN pdf_documents
    FILTER doc.type == @type AND doc.level == @level
    RETURN doc
"""
cursor = db.aql.execute(aql, bind_vars={"type": "heading", "level": 1})
results = [doc for doc in cursor]
```

## Best Practices

1. **Connection Management**:
   - Use a single client instance throughout the application
   - Implement proper connection closing
   - Add logging for database operations

2. **Error Handling**:
   - Catch and properly handle database exceptions
   - Add retry logic for transient errors
   - Log detailed error information

3. **Performance Optimization**:
   - Create appropriate indexes for frequent queries
   - Limit result sets to necessary fields
   - Use batch operations for bulk insertions

4. **Security Considerations**:
   - Store credentials in environment variables
   - Use least privilege database accounts
   - Implement proper input sanitization

## Validation Considerations

When validating the ArangoDB integration:

1. Verify document insertion with precise expected outputs
2. Test each query type with known document sets
3. Validate error handling with connection issues
4. Confirm performance with large document collections

## Debugging Tips

1. Enable verbose logging in ArangoDB client
2. Verify AQL queries separately in ArangoDB web interface
3. Check collection and document structure in ArangoDB
4. Monitor memory usage during large document operations
