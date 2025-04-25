#!/usr/bin/env python3
"""
PDF Extractor ArangoDB Integration Example

This script demonstrates how to use the ArangoDB integration with the PDF extractor.
It shows how to store PDF extraction results in ArangoDB and query them.
"""

import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import connection module
from pdf_extractor.arangodb_borked.connection import get_db

# Collection name for PDF content
PDF_COLLECTION_NAME = "pdf_documents"

def store_pdf_extraction(db, extraction_results):
    """
    Store PDF extraction results in ArangoDB.
    
    Args:
        db: ArangoDB database connection
        extraction_results: List of extraction results
        
    Returns:
        Number of items stored
    """
    try:
        # Get collection
        if not db.has_collection(PDF_COLLECTION_NAME):
            collection = db.create_collection(PDF_COLLECTION_NAME)
            logger.info(f"Created collection: {PDF_COLLECTION_NAME}")
        else:
            collection = db.collection(PDF_COLLECTION_NAME)
            logger.info(f"Using existing collection: {PDF_COLLECTION_NAME}")
        
        # Store extraction results
        stored_count = 0
        for item in extraction_results:
            # Add timestamp if not present
            if "extraction_date" not in item:
                item["extraction_date"] = datetime.now().isoformat()
                
            # Store item
            collection.insert(item)
            stored_count += 1
            
        logger.info(f"Stored {stored_count} PDF extraction items")
        return stored_count
    except Exception as e:
        logger.error(f"Failed to store PDF extraction results: {e}")
        return 0

def query_by_type(db, doc_type, limit=10):
    """
    Query PDF extraction results by type.
    
    Args:
        db: ArangoDB database connection
        doc_type: Document type (heading, paragraph, table)
        limit: Maximum number of results to return
        
    Returns:
        List of matching documents
    """
    try:
        # Build AQL query
        aql = f"""
        FOR doc IN {PDF_COLLECTION_NAME}
            FILTER doc.type == @doc_type
            SORT doc.page ASC
            LIMIT @limit
            RETURN doc
        """
        
        # Execute query
        cursor = db.aql.execute(aql, bind_vars={
            "doc_type": doc_type,
            "limit": limit
        })
        
        # Collect results
        results = [doc for doc in cursor]
        logger.info(f"Found {len(results)} documents of type '{doc_type}'")
        
        return results
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []

def text_search(db, search_text, limit=10):
    """
    Perform text search on PDF extraction results.
    
    Args:
        db: ArangoDB database connection
        search_text: Text to search for
        limit: Maximum number of results to return
        
    Returns:
        List of matching documents
    """
    try:
        # Build AQL query using CONTAINS
        aql = f"""
        FOR doc IN {PDF_COLLECTION_NAME}
            FILTER CONTAINS(doc.text, @search_text)
            SORT doc.page ASC
            LIMIT @limit
            RETURN doc
        """
        
        # Execute query
        cursor = db.aql.execute(aql, bind_vars={
            "search_text": search_text,
            "limit": limit
        })
        
        # Collect results
        results = [doc for doc in cursor]
        logger.info(f"Found {len(results)} documents matching '{search_text}'")
        
        return results
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        return []

def run_example():
    """Run the ArangoDB integration example"""
    logger.info("=== PDF Extractor ArangoDB Integration Example ===")
    
    # Connect to ArangoDB
    db = get_db()
    if not db:
        logger.error("Failed to connect to ArangoDB")
        return False
        
    logger.info(f"Connected to ArangoDB database: {db.name}")
    
    # Create mock PDF extraction results
    test_id = datetime.now().strftime("%Y%m%d%H%M%S")
    extraction_results = [
        {
            "_key": f"heading_{test_id}_1",
            "type": "heading",
            "level": 1,
            "text": "ArangoDB Integration",
            "page": 1,
            "token_count": 2,
            "file_path": "example.pdf",
            "source": "example"
        },
        {
            "_key": f"paragraph_{test_id}_1",
            "type": "paragraph",
            "text": "This is an example paragraph demonstrating ArangoDB integration with the PDF extractor.",
            "page": 1,
            "token_count": 12,
            "file_path": "example.pdf",
            "source": "example"
        },
        {
            "_key": f"heading_{test_id}_2",
            "type": "heading",
            "level": 2,
            "text": "Query Types",
            "page": 2,
            "token_count": 2,
            "file_path": "example.pdf",
            "source": "example"
        },
        {
            "_key": f"paragraph_{test_id}_2",
            "type": "paragraph",
            "text": "ArangoDB supports various query types including text search and filtering.",
            "page": 2,
            "token_count": 10,
            "file_path": "example.pdf",
            "source": "example"
        },
        {
            "_key": f"table_{test_id}_1",
            "type": "table",
            "caption": "Query Types",
            "headers": ["Type", "Description"],
            "rows": [
                ["Text Search", "Full-text search using BM25"],
                ["Filter", "Filter by document properties"]
            ],
            "page": 3,
            "token_count": 20,
            "file_path": "example.pdf",
            "source": "example"
        }
    ]
    
    # Store extraction results
    stored_count = store_pdf_extraction(db, extraction_results)
    if stored_count == 0:
        logger.error("Failed to store extraction results")
        return False
        
    # Query by type
    heading_results = query_by_type(db, "heading")
    if heading_results:
        logger.info("Heading results:")
        for i, doc in enumerate(heading_results[:3], 1):
            logger.info(f"  {i}. {doc.get('text', 'No text')} (Page {doc.get('page', 'unknown')})")
    
    # Text search
    search_results = text_search(db, "integration")
    if search_results:
        logger.info("Search results:")
        for i, doc in enumerate(search_results[:3], 1):
            logger.info(f"  {i}. {doc.get('text', 'No text')[:50]}...")
    
    # Clean up test data
    try:
        collection = db.collection(PDF_COLLECTION_NAME)
        for item in extraction_results:
            collection.delete(item["_key"])
        logger.info("Cleaned up test data")
    except Exception as e:
        logger.warning(f"Failed to clean up test data: {e}")
    
    logger.info("=== Example Complete ===")
    return True

if __name__ == "__main__":
    success = run_example()
    sys.exit(0 if success else 1)
