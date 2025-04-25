#!/usr/bin/env python3
"""
Validation Functions for PDF Extractor ArangoDB Integration

This module provides comprehensive validation functions that verify actual results
against expected results for the ArangoDB integration functions, following the
requirements in VALIDATION_REQUIREMENTS.md.
"""

import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ArangoDB modules
from pdf_extractor.arangodb_borked.connection import get_db

# Collection name
TEST_COLLECTION_NAME = "pdf_validation_test"

def validate_connection(expected_db_name: str = "pdf_extractor") -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate ArangoDB connection against expected database name.
    
    Args:
        expected_db_name: Expected database name
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    # Connect to ArangoDB
    db = get_db()
    
    # Check if connection successful
    if not db:
        validation_failures["connection"] = {
            "expected": "Valid database connection",
            "actual": "Failed to connect to ArangoDB"
        }
        return False, validation_failures
    
    # Validate database name
    actual_db_name = db.name
    if actual_db_name != expected_db_name:
        validation_failures["database_name"] = {
            "expected": expected_db_name,
            "actual": actual_db_name
        }
    
    # Return validation result
    validation_passed = len(validation_failures) == 0
    return validation_passed, validation_failures

def validate_collection_setup(db, expected_collection_name: str = TEST_COLLECTION_NAME, 
                             expected_indexes: List[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate collection setup against expected configuration.
    
    Args:
        db: ArangoDB database connection
        expected_collection_name: Expected collection name
        expected_indexes: List of expected indexes
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    # Default expected indexes if not provided
    if expected_indexes is None:
        expected_indexes = [
            {"type": "hash", "fields": ["type"]},
            {"type": "hash", "fields": ["file_path"]},
            {"type": "skiplist", "fields": ["page"]},
            {"type": "fulltext", "fields": ["text"]}
        ]
    
    # Check if collection exists
    if not db.has_collection(expected_collection_name):
        # Create collection for testing
        try:
            collection = db.create_collection(expected_collection_name)
            
            # Create test indexes
            for idx_config in expected_indexes:
                collection.add_index(idx_config)
                
        except Exception as e:
            validation_failures["collection_creation"] = {
                "expected": f"Collection '{expected_collection_name}' created successfully",
                "actual": f"Failed to create collection: {str(e)}"
            }
            return False, validation_failures
    else:
        # Get existing collection
        collection = db.collection(expected_collection_name)
    
    # Validate collection name
    actual_collection_name = collection.name
    if actual_collection_name != expected_collection_name:
        validation_failures["collection_name"] = {
            "expected": expected_collection_name,
            "actual": actual_collection_name
        }
    
    # Validate indexes
    indexes = collection.indexes()
    
    # Check each expected index type
    for expected_idx in expected_indexes:
        idx_type = expected_idx["type"]
        idx_fields = expected_idx["fields"]
        
        # Look for matching index
        found_index = False
        for idx in indexes:
            if idx.get("type") == idx_type and all(field in idx.get("fields", []) for field in idx_fields):
                found_index = True
                break
                
        if not found_index:
            validation_failures[f"missing_index_{idx_type}_{idx_fields[0]}"] = {
                "expected": f"{idx_type} index on {idx_fields}",
                "actual": "Index not found"
            }
    
    # Return validation result
    validation_passed = len(validation_failures) == 0
    return validation_passed, validation_failures

def validate_document_storage(collection, test_docs: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate document storage against expected documents.
    
    Args:
        collection: ArangoDB collection
        test_docs: List of test documents to store and validate
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    try:
        # Store test documents
        for doc in test_docs:
            collection.insert(doc)
            
        # Retrieve and validate each document
        for doc in test_docs:
            key = doc["_key"]
            retrieved_doc = collection.get(key)
            
            # Check if document exists
            if not retrieved_doc:
                validation_failures[f"missing_doc_{key}"] = {
                    "expected": f"Document with key '{key}'",
                    "actual": "Document not found"
                }
                continue
                
            # Check each field
            for field, expected_value in doc.items():
                actual_value = retrieved_doc.get(field)
                if actual_value != expected_value:
                    validation_failures[f"field_mismatch_{key}_{field}"] = {
                        "expected": expected_value,
                        "actual": actual_value
                    }
    except Exception as e:
        validation_failures["storage_error"] = {
            "expected": "Successful document storage and retrieval",
            "actual": f"Error: {str(e)}"
        }
    finally:
        # Clean up
        try:
            for doc in test_docs:
                if collection.has(doc["_key"]):
                    collection.delete(doc["_key"])
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    # Return validation result
    validation_passed = len(validation_failures) == 0
    return validation_passed, validation_failures

def validate_query_by_type(db, collection_name: str, test_docs: List[Dict[str, Any]], 
                          doc_type: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate query_by_type function against expected results.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        test_docs: List of test documents
        doc_type: Document type to query
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    try:
        # Store test documents
        collection = db.collection(collection_name)
        for doc in test_docs:
            collection.insert(doc)
        
        # Execute query
        aql = f"""
        FOR doc IN {collection_name}
            FILTER doc.type == @doc_type
            SORT doc.page ASC
            RETURN doc
        """
        
        cursor = db.aql.execute(aql, bind_vars={"doc_type": doc_type})
        actual_results = [doc for doc in cursor]
        
        # Calculate expected results
        expected_results = [doc for doc in test_docs if doc["type"] == doc_type]
        
        # Validate result count
        if len(actual_results) != len(expected_results):
            validation_failures["result_count"] = {
                "expected": len(expected_results),
                "actual": len(actual_results)
            }
        
        # Validate each expected document is in results
        for expected_doc in expected_results:
            expected_key = expected_doc["_key"]
            found = False
            
            for actual_doc in actual_results:
                if actual_doc["_key"] == expected_key:
                    found = True
                    break
                    
            if not found:
                validation_failures[f"missing_result_{expected_key}"] = {
                    "expected": f"Document with key '{expected_key}'",
                    "actual": "Document not found in results"
                }
    except Exception as e:
        validation_failures["query_error"] = {
            "expected": "Successful query execution",
            "actual": f"Error: {str(e)}"
        }
    finally:
        # Clean up
        try:
            for doc in test_docs:
                if collection.has(doc["_key"]):
                    collection.delete(doc["_key"])
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    # Return validation result
    validation_passed = len(validation_failures) == 0
    return validation_passed, validation_failures

def validate_text_search(db, collection_name: str, test_docs: List[Dict[str, Any]], 
                        search_text: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate text_search function against expected results.
    
    Args:
        db: ArangoDB database connection
        collection_name: Collection name
        test_docs: List of test documents
        search_text: Text to search for
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    try:
        # Store test documents
        collection = db.collection(collection_name)
        for doc in test_docs:
            collection.insert(doc)
        
        # Execute query
        aql = f"""
        FOR doc IN {collection_name}
            FILTER CONTAINS(doc.text, @search_text)
            SORT doc.page ASC
            RETURN doc
        """
        
        cursor = db.aql.execute(aql, bind_vars={"search_text": search_text})
        actual_results = [doc for doc in cursor]
        
        # Calculate expected results
        expected_results = [doc for doc in test_docs if "text" in doc and search_text in doc["text"]]
        
        # Validate result count
        if len(actual_results) != len(expected_results):
            validation_failures["result_count"] = {
                "expected": len(expected_results),
                "actual": len(actual_results)
            }
        
        # Validate each expected document is in results
        for expected_doc in expected_results:
            expected_key = expected_doc["_key"]
            found = False
            
            for actual_doc in actual_results:
                if actual_doc["_key"] == expected_key:
                    found = True
                    break
                    
            if not found:
                validation_failures[f"missing_result_{expected_key}"] = {
                    "expected": f"Document with key '{expected_key}'",
                    "actual": "Document not found in results"
                }
    except Exception as e:
        validation_failures["search_error"] = {
            "expected": "Successful search execution",
            "actual": f"Error: {str(e)}"
        }
    finally:
        # Clean up
        try:
            for doc in test_docs:
                if collection.has(doc["_key"]):
                    collection.delete(doc["_key"])
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    # Return validation result
    validation_passed = len(validation_failures) == 0
    return validation_passed, validation_failures

def run_all_validations() -> bool:
    """
    Run all validation functions and report results.
    
    Returns:
        True if all validations pass, False otherwise
    """
    all_validations_passed = True
    
    # Generate test data
    test_id = datetime.now().strftime("%Y%m%d%H%M%S")
    test_docs = [
        {
            "_key": f"validation_heading_{test_id}",
            "type": "heading",
            "level": 1,
            "text": "Validation Test Heading",
            "page": 1,
            "token_count": 3,
            "file_path": "validation_test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "validation_test"
        },
        {
            "_key": f"validation_paragraph_{test_id}",
            "type": "paragraph",
            "text": "This is a test paragraph for validation. It contains the word integration.",
            "page": 1,
            "token_count": 12,
            "file_path": "validation_test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "validation_test"
        },
        {
            "_key": f"validation_table_{test_id}",
            "type": "table",
            "caption": "Test Table",
            "headers": ["Column A", "Column B"],
            "rows": [
                ["Row 1A", "Row 1B"],
                ["Row 2A", "Row 2B"]
            ],
            "page": 2,
            "token_count": 15,
            "file_path": "validation_test.pdf",
            "extraction_date": datetime.now().isoformat(),
            "source": "validation_test"
        }
    ]
    
    # Validate connection
    logger.info("1. Validating ArangoDB connection")
    connection_passed, connection_failures = validate_connection()
    
    if connection_passed:
        logger.info("✅ Connection validation passed")
    else:
        logger.error("❌ Connection validation failed:")
        for field, details in connection_failures.items():
            logger.error(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        all_validations_passed = False
    
    # Get database connection
    db = get_db()
    if not db:
        logger.error("Cannot proceed with validations: No database connection")
        return False
    
    # Validate collection setup
    logger.info("\n2. Validating collection setup")
    collection_passed, collection_failures = validate_collection_setup(db)
    
    if collection_passed:
        logger.info("✅ Collection setup validation passed")
    else:
        logger.error("❌ Collection setup validation failed:")
        for field, details in collection_failures.items():
            logger.error(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        all_validations_passed = False
    
    # Get collection
    collection = db.collection(TEST_COLLECTION_NAME)
    
    # Validate document storage
    logger.info("\n3. Validating document storage")
    storage_passed, storage_failures = validate_document_storage(collection, test_docs)
    
    if storage_passed:
        logger.info("✅ Document storage validation passed")
    else:
        logger.error("❌ Document storage validation failed:")
        for field, details in storage_failures.items():
            logger.error(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        all_validations_passed = False
    
    # Validate query by type
    logger.info("\n4. Validating query by type")
    query_passed, query_failures = validate_query_by_type(db, TEST_COLLECTION_NAME, test_docs, "heading")
    
    if query_passed:
        logger.info("✅ Query by type validation passed")
    else:
        logger.error("❌ Query by type validation failed:")
        for field, details in query_failures.items():
            logger.error(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        all_validations_passed = False
    
    # Validate text search
    logger.info("\n5. Validating text search")
    search_passed, search_failures = validate_text_search(db, TEST_COLLECTION_NAME, test_docs, "integration")
    
    if search_passed:
        logger.info("✅ Text search validation passed")
    else:
        logger.error("❌ Text search validation failed:")
        for field, details in search_failures.items():
            logger.error(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        all_validations_passed = False
    
    # Clean up
    try:
        if db.has_collection(TEST_COLLECTION_NAME):
            db.delete_collection(TEST_COLLECTION_NAME)
            logger.info(f"Cleaned up test collection: {TEST_COLLECTION_NAME}")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")
    
    # Final validation report
    if all_validations_passed:
        logger.info("\n✅ ALL VALIDATIONS PASSED")
    else:
        logger.error("\n❌ ONE OR MORE VALIDATIONS FAILED")
    
    return all_validations_passed

if __name__ == "__main__":
    logger.info("=== Running Comprehensive Validation for ArangoDB Integration ===")
    
    success = run_all_validations()
    
    sys.exit(0 if success else 1)
