#!/usr/bin/env python3
"""
Modified version of crud.py for testing - just the main validation section for now
"""

import logging
import uuid
import sys
import json
import os
from datetime import datetime, timezone

# Configure logging for standalone execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import CRUD functions and connection module
try:
    from pdf_extractor.arangodb_borked.connection import get_db, create_collections
    from pdf_extractor.arangodb_borked.crud import (
        insert_lesson,
        get_lesson,
        update_lesson,
        delete_lesson,
        get_lessons_by_tag,
        search_lessons,
        semantic_search,
        ensure_vector_index
    )
except ImportError:
    # Add project root to path if needed
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from pdf_extractor.arangodb_borked.connection import get_db, create_collections
    from pdf_extractor.arangodb_borked.crud import (
        insert_lesson,
        get_lesson,
        update_lesson,
        delete_lesson,
        get_lessons_by_tag,
        search_lessons,
        semantic_search,
        ensure_vector_index
    )

def main():
    logger.info("Starting CRUD validation test")
    
    # Setup test fixtures - expected results
    test_uuid = str(uuid.uuid4())[:8]  # Generate unique ID for this test run
    
    test_lesson = {
        "_key": f"test_lesson_{test_uuid}",
        "problem": "Table extraction failed for complex merged cells",
        "solution": "Implemented custom cell detection algorithm",
        "tags": ["table-extraction", "bug-fix", "merged-cells", test_uuid],
        "author": "Test Author"
    }
    
    expected_lesson = test_lesson.copy()
    
    test_update = {
        "solution": "Updated: Implemented improved cell detection algorithm with better boundary recognition",
        "tags": ["table-extraction", "bug-fix", "merged-cells", "improved", test_uuid]
    }
    
    expected_updated_lesson = expected_lesson.copy()
    expected_updated_lesson.update(test_update)
    
    # Track validation failures
    validation_failures = {}
    
    try:
        # Connect to database
        db = get_db()
        collections_dict = create_collections(db)
        
        # Get the lessons_learned collection directly from the database
        collection = db.collection('lessons_learned')
        if not collection:
            logger.error("lessons_learned collection not available")
            validation_failures["setup"] = {
                "expected": "lessons_learned collection available",
                "actual": "collection not available"
            }
            return False
            
        logger.info(f"Successfully accessed collection: lessons_learned (count: {collection.count()})")
        
        # Step 1: Insert a test lesson
        logger.info("Test step 1: Insert lesson")
        insert_result = insert_lesson(collection, test_lesson)
        
        if not insert_result:
            validation_failures["insert"] = {
                "expected": "Insert operation success",
                "actual": "Insert operation failed"
            }
        elif insert_result.get("_key") != test_lesson["_key"]:
            validation_failures["insert_key"] = {
                "expected": test_lesson["_key"],
                "actual": insert_result.get("_key")
            }
        else:
            logger.info("Insert operation succeeded")
        
        # Step 2: Get the inserted lesson
        logger.info("Test step 2: Get lesson")
        get_result = get_lesson(collection, test_lesson["_key"])
        
        if not get_result:
            validation_failures["get"] = {
                "expected": "Get operation success",
                "actual": "Get operation failed"
            }
        else:
            # Validate retrieval (ignore created_at and embedding which are added dynamically)
            for field in ["_key", "problem", "solution", "tags", "author"]:
                if get_result.get(field) != test_lesson.get(field):
                    validation_failures[f"get_{field}"] = {
                        "expected": test_lesson.get(field),
                        "actual": get_result.get(field)
                    }
            
            if "created_at" not in get_result:
                validation_failures["get_created_at"] = {
                    "expected": "created_at timestamp",
                    "actual": "field missing"
                }
                
            logger.info("Get operation succeeded")
        
        # Step 3: Update the lesson
        logger.info("Test step 3: Update lesson")
        update_result = update_lesson(collection, test_lesson["_key"], test_update)
        
        if not update_result:
            validation_failures["update"] = {
                "expected": "Update operation success",
                "actual": "Update operation failed"
            }
        else:
            # Verify update by getting the document again
            updated_doc = get_lesson(collection, test_lesson["_key"])
            
            if not updated_doc:
                validation_failures["update_verification"] = {
                    "expected": "Document exists after update",
                    "actual": "Document not found after update"
                }
            else:
                # Check updated fields
                for field, value in test_update.items():
                    if updated_doc.get(field) != value:
                        validation_failures[f"update_{field}"] = {
                            "expected": value,
                            "actual": updated_doc.get(field)
                        }
                
                if "updated_at" not in updated_doc:
                    validation_failures["update_timestamp"] = {
                        "expected": "updated_at timestamp",
                        "actual": "field missing"
                    }
                
                logger.info("Update operation succeeded")
        
        # Step 4: Search by tag
        logger.info("Test step 4: Search by tag")
        tag_search_result = get_lessons_by_tag(collection, test_uuid)
        
        if not tag_search_result:
            validation_failures["tag_search"] = {
                "expected": "At least one document with test tag",
                "actual": "No documents found"
            }
        elif len(tag_search_result) != 1:
            validation_failures["tag_search_count"] = {
                "expected": 1,
                "actual": len(tag_search_result)
            }
        elif tag_search_result[0].get("_key") != test_lesson["_key"]:
            validation_failures["tag_search_key"] = {
                "expected": test_lesson["_key"],
                "actual": tag_search_result[0].get("_key")
            }
        else:
            logger.info("Tag search succeeded")
        
        # Step 5: Text search
        logger.info("Test step 5: Text search")
        text_search_result = search_lessons(collection, "improved")
        
        if not text_search_result:
            validation_failures["text_search"] = {
                "expected": "At least one document with 'improved'",
                "actual": "No documents found"
            }
        elif not any(doc.get("_key") == test_lesson["_key"] for doc in text_search_result):
            validation_failures["text_search_match"] = {
                "expected": f"Document with key {test_lesson['_key']} in results",
                "actual": "Test document not found in search results"
            }
        else:
            logger.info("Text search succeeded")
        
        # Step 6: Delete the test lesson
        logger.info("Test step 6: Delete lesson")
        delete_result = delete_lesson(collection, test_lesson["_key"])
        
        if not delete_result:
            validation_failures["delete"] = {
                "expected": "Delete operation success",
                "actual": "Delete operation failed"
            }
        else:
            # Verify deletion
            verify_deleted = get_lesson(collection, test_lesson["_key"])
            if verify_deleted is not None:
                validation_failures["delete_verification"] = {
                    "expected": "Document not found after deletion",
                    "actual": "Document still exists after deletion"
                }
            else:
                logger.info("Delete operation succeeded")
        
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        validation_failures["unexpected_error"] = {
            "expected": "No exceptions",
            "actual": str(e)
        }
        return False
    
    # Final validation report
    validation_passed = len(validation_failures) == 0
    
    if validation_passed:
        logger.info("✅ VALIDATION COMPLETE - All CRUD operations match expected values")
        return True
    else:
        logger.error("❌ VALIDATION FAILED - CRUD operations don't match expected values")
        logger.error("FAILURE DETAILS:")
        for field, details in validation_failures.items():
            logger.error(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        logger.error(f"Total errors: {len(validation_failures)} fields mismatched")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
