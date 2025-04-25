#!/usr/bin/env python3
"""
Validation Script for PDF Extractor ArangoDB Integration

This script performs comprehensive validation of the ArangoDB integration
for the PDF extractor, following the requirements in VALIDATION_REQUIREMENTS.md.
It tests each function individually and compares actual results against expected outputs.

Usage:
    env ARANGO_PASSWORD="your_password" uv run validate_pdf_integration.py
"""

import sys
import os
import logging
import json
from datetime import datetime
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import integration module
try:
    from pdf_extractor.arangodb_borked.connection import get_db, create_collections
    from pdf_extractor.arangodb_borked.pdf_integration import (
        setup_pdf_collection,
        store_pdf_content,
        query_pdf_content,
        get_pdf_content_stats,
        find_headings_with_content
    )
except ImportError as e:
    logger.error(f"Failed to import integration modules: {e}")
    sys.exit(1)

# Test fixture directory
FIXTURE_DIR = Path('src/test_fixtures')
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

# Test collection name - use a dedicated test collection
TEST_COLLECTION_NAME = f"pdf_validation_test_{uuid.uuid4().hex[:8]}"

def create_test_fixture():
    """
    Create a test fixture for validation.
    
    Returns:
        Dictionary with test fixture data
    """
    # Generate a unique ID for this test run
    test_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Create test data
    fixture = {
        "documents": [
            # Document 1: Metadata
            {
                "_key": f"meta_{test_id}",
                "type": "metadata",
                "filename": "test_doc.pdf",
                "page_count": 5,
                "extraction_date": datetime.now().isoformat(),
                "file_path": "test_doc.pdf",
                "source": "validation_test"
            },
            # Document 2: Heading 1
            {
                "_key": f"h1_{test_id}",
                "type": "heading",
                "level": 1,
                "text": "Introduction to ArangoDB",
                "page": 1,
                "token_count": 3,
                "file_path": "test_doc.pdf",
                "extraction_date": datetime.now().isoformat(),
                "source": "validation_test"
            },
            # Document 3: Paragraph 1
            {
                "_key": f"p1_{test_id}",
                "type": "paragraph",
                "text": "ArangoDB is a multi-model database that supports key-value, document, and graph data models.",
                "page": 1,
                "token_count": 14,
                "file_path": "test_doc.pdf",
                "extraction_date": datetime.now().isoformat(),
                "source": "validation_test"
            },
            # Document 4: Heading 2
            {
                "_key": f"h2_{test_id}",
                "type": "heading",
                "level": 2,
                "text": "Query Types",
                "page": 2,
                "token_count": 2,
                "file_path": "test_doc.pdf",
                "extraction_date": datetime.now().isoformat(),
                "source": "validation_test"
            },
            # Document 5: Paragraph 2
            {
                "_key": f"p2_{test_id}",
                "type": "paragraph",
                "text": "ArangoDB supports multiple query types including keyword, semantic, BM25, and hybrid search.",
                "page": 2,
                "token_count": 13,
                "file_path": "test_doc.pdf",
                "extraction_date": datetime.now().isoformat(),
                "source": "validation_test"
            },
            # Document 6: Table 1
            {
                "_key": f"t1_{test_id}",
                "type": "table",
                "caption": "Query Type Comparison",
                "headers": ["Type", "Use Case", "Performance"],
                "rows": [
                    ["Keyword", "Exact matching", "Fast"],
                    ["Semantic", "Meaning-based", "Medium"],
                    ["Hybrid", "Combined approach", "Variable"]
                ],
                "page": 3,
                "token_count": 25,
                "file_path": "test_doc.pdf",
                "extraction_date": datetime.now().isoformat(),
                "source": "validation_test"
            }
        ],
        "test_id": test_id
    }
    
    # Save fixture to file
    fixture_path = FIXTURE_DIR / f"arangodb_test_{test_id}.json"
    with open(fixture_path, 'w') as f:
        json.dump(fixture, f, indent=2)
        
    logger.info(f"Created test fixture with ID {test_id}")
    logger.info(f"Saved to {fixture_path}")
    
    return fixture

def setup_test_collection(db):
    """
    Set up a test collection for validation.
    
    Args:
        db: ArangoDB database connection
        
    Returns:
        Collection object if successful, None otherwise
    """
    logger.info(f"Setting up test collection: {TEST_COLLECTION_NAME}")
    
    try:
        # Create test collection and indexes
        collections = create_collections(db, [TEST_COLLECTION_NAME])
        
        if TEST_COLLECTION_NAME not in collections:
            logger.error(f"Failed to create test collection: {TEST_COLLECTION_NAME}")
            return None
            
        collection = collections[TEST_COLLECTION_NAME]
        
        # Create indexes
        collection.add_hash_index(fields=["type"], unique=False)
        collection.add_hash_index(fields=["file_path"], unique=False)
        collection.add_skiplist_index(fields=["page"], unique=False)
        collection.add_fulltext_index(fields=["text"], min_length=3)
        
        logger.info(f"Test collection set up: {TEST_COLLECTION_NAME}")
        return collection
    except Exception as e:
        logger.error(f"Failed to set up test collection: {e}")
        return None

def validate_collection_setup(db):
    """
    Validate the collection setup function.
    
    Args:
        db: ArangoDB database connection
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    try:
        # Call the function
        collection = setup_pdf_collection(db)
        
        # Check if collection was created
        if not collection:
            validation_failures["collection_creation"] = {
                "expected": "Valid collection object",
                "actual": "None"
            }
            return False, validation_failures
            
        # Check if indexes exist
        indexes = collection.indexes()
        index_types = set(idx.get("type") for idx in indexes)
        
        # Verify hash indexes
        if "hash" not in index_types:
            validation_failures["hash_indexes"] = {
                "expected": "Hash indexes on type and file_path",
                "actual": f"Index types: {index_types}"
            }
            
        # Verify fulltext index
        if "fulltext" not in index_types:
            validation_failures["fulltext_index"] = {
                "expected": "Fulltext index on text field",
                "actual": f"Index types: {index_types}"
            }
            
        # Return validation result
        validation_passed = len(validation_failures) == 0
        return validation_passed, validation_failures
    except Exception as e:
        validation_failures["unexpected_error"] = {
            "expected": "No errors",
            "actual": f"Exception: {str(e)}"
        }
        return False, validation_failures

def validate_document_storage(collection, fixture):
    """
    Validate the document storage function.
    
    Args:
        collection: ArangoDB collection
        fixture: Test fixture data
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    try:
        # Store the test documents
        documents = fixture["documents"]
        stored_count = store_pdf_content(collection, documents)
        
        # Check if all documents were stored
        if stored_count != len(documents):
            validation_failures["storage_count"] = {
                "expected": len(documents),
                "actual": stored_count
            }
            
        # Verify each document was stored correctly
        for doc in documents:
            key = doc["_key"]
            stored_doc = collection.get(key)
            
            if not stored_doc:
                validation_failures[f"missing_doc_{key}"] = {
                    "expected": f"Document with key {key}",
                    "actual": "Not found"
                }
                continue
                
            # Verify key fields
            for field in ["type", "text", "page", "file_path"]:
                if field in doc and doc[field] != stored_doc.get(field):
                    validation_failures[f"field_mismatch_{key}_{field}"] = {
                        "expected": doc[field],
                        "actual": stored_doc.get(field)
                    }
        
        # Return validation result
        validation_passed = len(validation_failures) == 0
        return validation_passed, validation_failures
    except Exception as e:
        validation_failures["unexpected_error"] = {
            "expected": "No errors",
            "actual": f"Exception: {str(e)}"
        }
        return False, validation_failures

def validate_query_by_type(db, collection_name, fixture):
    """
    Validate querying by document type.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        fixture: Test fixture data
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    try:
        # Find heading documents
        heading_docs = [doc for doc in fixture["documents"] if doc["type"] == "heading"]
        heading_keys = [doc["_key"] for doc in heading_docs]
        
        # Execute query
        results = query_pdf_content(db, collection_name, doc_type="heading")
        
        # Check if we got the correct documents
        result_keys = [doc["_key"] for doc in results]
        missing_keys = set(heading_keys) - set(result_keys)
        
        if missing_keys:
            validation_failures["missing_headings"] = {
                "expected": f"Keys {list(missing_keys)}",
                "actual": f"Not found in results: {result_keys}"
            }
        
        # Return validation result
        validation_passed = len(validation_failures) == 0
        return validation_passed, validation_failures
    except Exception as e:
        validation_failures["unexpected_error"] = {
            "expected": "No errors",
            "actual": f"Exception: {str(e)}"
        }
        return False, validation_failures

def validate_text_search(db, collection_name, fixture):
    """
    Validate text search functionality.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        fixture: Test fixture data
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    try:
        # Find documents with "ArangoDB" in the text
        text_docs = [doc for doc in fixture["documents"] if "ArangoDB" in doc.get("text", "")]
        text_keys = [doc["_key"] for doc in text_docs]
        
        # Execute query
        results = query_pdf_content(db, collection_name, search_text="ArangoDB")
        
        # Check if we got at least all expected documents
        # Note: Fulltext search might sometimes find additional matches
        result_keys = [doc["_key"] for doc in results]
        missing_keys = set(text_keys) - set(result_keys)
        
        if missing_keys:
            validation_failures["missing_text_matches"] = {
                "expected": f"At least keys {list(text_keys)}",
                "actual": f"Missing keys: {list(missing_keys)}"
            }
        
        # Return validation result
        validation_passed = len(validation_failures) == 0
        return validation_passed, validation_failures
    except Exception as e:
        validation_failures["unexpected_error"] = {
            "expected": "No errors",
            "actual": f"Exception: {str(e)}"
        }
        return False, validation_failures

def validate_combined_query(db, collection_name, fixture):
    """
    Validate combined query functionality.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        fixture: Test fixture data
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    try:
        # Find paragraph documents on page 2
        target_docs = [
            doc for doc in fixture["documents"] 
            if doc["type"] == "paragraph" and doc["page"] == 2
        ]
        target_keys = [doc["_key"] for doc in target_docs]
        
        # Execute query
        results = query_pdf_content(
            db, 
            collection_name, 
            doc_type="paragraph", 
            page=2
        )
        
        # Check if we got the expected documents
        result_keys = [doc["_key"] for doc in results]
        missing_keys = set(target_keys) - set(result_keys)
        
        if missing_keys:
            validation_failures["missing_combined_matches"] = {
                "expected": f"Keys {list(target_keys)}",
                "actual": f"Missing keys: {list(missing_keys)}"
            }
        
        # Return validation result
        validation_passed = len(validation_failures) == 0
        return validation_passed, validation_failures
    except Exception as e:
        validation_failures["unexpected_error"] = {
            "expected": "No errors",
            "actual": f"Exception: {str(e)}"
        }
        return False, validation_failures

def validate_stats(db, collection_name, fixture):
    """
    Validate stats functionality.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        fixture: Test fixture data
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    try:
        # Calculate expected stats
        expected_total = len(fixture["documents"])
        
        # Calculate type counts
        type_counts = {}
        for doc in fixture["documents"]:
            doc_type = doc["type"]
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        # Get stats from function
        stats = get_pdf_content_stats(db, collection_name)
        
        # Validate total count
        if stats["total_documents"] != expected_total:
            validation_failures["total_documents"] = {
                "expected": expected_total,
                "actual": stats["total_documents"]
            }
        
        # Validate type counts
        for doc_type, expected_count in type_counts.items():
            actual_count = stats["type_counts"].get(doc_type, 0)
            if actual_count != expected_count:
                validation_failures[f"type_count_{doc_type}"] = {
                    "expected": expected_count,
                    "actual": actual_count
                }
        
        # Return validation result
        validation_passed = len(validation_failures) == 0
        return validation_passed, validation_failures
    except Exception as e:
        validation_failures["unexpected_error"] = {
            "expected": "No errors",
            "actual": f"Exception: {str(e)}"
        }
        return False, validation_failures

def validate_headings_content(db, collection_name, fixture):
    """
    Validate finding headings with content functionality.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        fixture: Test fixture data
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    try:
        # Get all headings
        heading_docs = [doc for doc in fixture["documents"] if doc["type"] == "heading"]
        
        # Execute query
        results = find_headings_with_content(db, collection_name)
        
        # Check if we got all headings
        if len(results) != len(heading_docs):
            validation_failures["heading_count"] = {
                "expected": len(heading_docs),
                "actual": len(results)
            }
        
        # Check if each heading has content
        for result in results:
            heading = result["heading"]
            content = result["content"]
            
            if not content:
                validation_failures[f"content_for_{heading['_key']}"] = {
                    "expected": "At least one content item",
                    "actual": "No content found"
                }
        
        # Return validation result
        validation_passed = len(validation_failures) == 0
        return validation_passed, validation_failures
    except Exception as e:
        validation_failures["unexpected_error"] = {
            "expected": "No errors",
            "actual": f"Exception: {str(e)}"
        }
        return False, validation_failures

def cleanup_test_collection(db, collection_name):
    """
    Clean up the test collection.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if db.has_collection(collection_name):
            db.delete_collection(collection_name)
            logger.info(f"Deleted test collection: {collection_name}")
            return True
        return True
    except Exception as e:
        logger.error(f"Failed to clean up test collection: {e}")
        return False

def report_validation_results(validation_results):
    """
    Report validation results.
    
    Args:
        validation_results: Dictionary of validation results
        
    Returns:
        True if all passed, False otherwise
    """
    all_passed = True
    
    for test_name, (passed, failures) in validation_results.items():
        if passed:
            logger.info(f"✅ {test_name} validation passed")
        else:
            all_passed = False
            logger.error(f"❌ {test_name} validation failed:")
            for field, details in failures.items():
                logger.error(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
    
    if all_passed:
        logger.info("✅ All validations passed")
    else:
        logger.error("❌ One or more validations failed")
    
    return all_passed

def main():
    """
    Main validation function.
    """
    logger.info("=== Starting PDF Extractor ArangoDB Integration Validation ===")
    
    # Connect to ArangoDB
    try:
        db = get_db()
        if not db:
            logger.error("Failed to connect to ArangoDB")
            return False
            
        logger.info(f"Connected to ArangoDB database: {db.name}")
    except Exception as e:
        logger.error(f"Failed to connect to ArangoDB: {e}")
        return False
    
    # Create test fixture
    fixture = create_test_fixture()
    
    # Set up test collection
    collection = setup_test_collection(db)
    if not collection:
        logger.error("Failed to set up test collection")
        return False
    
    # Run validation tests
    validation_results = {}
    
    # Validate collection setup
    logger.info("Validating collection setup...")
    validation_results["collection_setup"] = validate_collection_setup(db)
    
    # Validate document storage
    logger.info("Validating document storage...")
    validation_results["document_storage"] = validate_document_storage(collection, fixture)
    
    # Validate query by type
    logger.info("Validating query by type...")
    validation_results["query_by_type"] = validate_query_by_type(db, TEST_COLLECTION_NAME, fixture)
    
    # Validate text search
    logger.info("Validating text search...")
    validation_results["text_search"] = validate_text_search(db, TEST_COLLECTION_NAME, fixture)
    
    # Validate combined query
    logger.info("Validating combined query...")
    validation_results["combined_query"] = validate_combined_query(db, TEST_COLLECTION_NAME, fixture)
    
    # Validate stats
    logger.info("Validating stats...")
    validation_results["stats"] = validate_stats(db, TEST_COLLECTION_NAME, fixture)
    
    # Validate headings with content
    logger.info("Validating headings with content...")
    validation_results["headings_with_content"] = validate_headings_content(db, TEST_COLLECTION_NAME, fixture)
    
    # Clean up
    cleanup_test_collection(db, TEST_COLLECTION_NAME)
    
    # Report results
    return report_validation_results(validation_results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
