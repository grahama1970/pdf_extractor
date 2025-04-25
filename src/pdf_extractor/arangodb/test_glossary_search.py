"""
Test the glossary search functionality.

This script tests the search_glossary function from the pdf_extractor.arangodb.search_api.glossary module
by performing a glossary search on test text and validating the results.
"""

import sys
import os
import uuid
from typing import Dict, List, Any, Optional
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:HH:mm:ss} | {level:<7} | {message}"
)

# Import necessary modules
try:
    from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
    from pdf_extractor.arangodb.search_api.glossary import search_glossary
except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}")
    logger.error("Please ensure the script is run from the project root or the necessary paths are in PYTHONPATH.")
    sys.exit(1)

def setup_test_collection(db):
    """Set up a test collection for storing glossary terms."""
    test_id = str(uuid.uuid4())[:6]
    test_collection = f"test_glossary_{test_id}"
    
    try:
        # Create test collection
        if not db.has_collection(test_collection):
            db.create_collection(test_collection)
            logger.info(f"Created test collection '{test_collection}'")
        
        # Create test glossary terms
        collection = db.collection(test_collection)
        
        test_terms = [
            {
                "_key": f"term1_{test_id}",
                "type": "glossary",
                "term": "Machine Learning",
                "definition": "A subset of artificial intelligence that enables systems to learn and improve from experience."
            },
            {
                "_key": f"term2_{test_id}",
                "type": "glossary",
                "term": "Natural Language Processing",
                "definition": "A field of AI that focuses on the interaction between computers and human language."
            },
            {
                "_key": f"term3_{test_id}",
                "type": "glossary",
                "term": "Neural Network",
                "definition": "A computing system designed to simulate the way the human brain analyzes and processes information."
            }
        ]
        
        for term in test_terms:
            collection.insert(term)
        logger.info(f"Inserted {len(test_terms)} test glossary terms")
        
        return test_collection
    
    except Exception as e:
        logger.error(f"Error setting up test collection: {e}")
        return None

def cleanup_test_resources(db, collection_name):
    """Clean up test resources after testing."""
    try:
        # Delete collection if it exists
        if collection_name and db.has_collection(collection_name):
            db.delete_collection(collection_name)
            logger.info(f"Deleted test collection '{collection_name}'")
            
        return True
    except Exception as e:
        logger.error(f"Error cleaning up test resources: {e}")
        return False

def test_glossary_search():
    """
    Test the glossary search functionality.
    
    Expected behavior:
    1. Test with default glossary
    2. Test with custom glossary
    3. Test with database glossary
    
    Returns:
        bool: True if test passes, False otherwise
    """
    logger.info("Testing glossary search...")
    test_collection = None
    all_tests_passed = True
    
    try:
        # Connect to ArangoDB for database glossary test
        logger.info("Connecting to ArangoDB...")
        client = connect_arango()
        if not client:
            logger.error("Failed to connect to ArangoDB")
            return False
        
        # Get database
        logger.info("Getting database...")
        db = ensure_database(client)
        if not db:
            logger.error("Failed to ensure database exists")
            return False
        
        # Test 1: Default glossary search
        logger.info("Test 1: Default glossary search")
        test_text_1 = "Airplane Parts and Flight Path are important concepts in aviation."
        
        matches_1 = search_glossary(test_text_1)
        
        if not matches_1 or len(matches_1) < 2:
            logger.error(f"Expected at least 2 matches, got {len(matches_1)}")
            all_tests_passed = False
        else:
            # Check that we have the expected terms
            found_terms = [match["term"] for match in matches_1]
            expected_terms = ["Airplane Parts", "Flight Path"]
            
            if not all(term in found_terms for term in expected_terms):
                logger.error(f"Missing expected terms. Found: {found_terms}")
                all_tests_passed = False
            else:
                logger.success("✅ Default glossary test: PASSED")
        
        # Test 2: Custom glossary search
        logger.info("Test 2: Custom glossary search")
        custom_glossary = {
            "Deep Learning": "A subset of machine learning that uses neural networks with many layers.",
            "Convolutional Network": "A type of neural network commonly used for image recognition.",
            "Recurrent Network": "A type of neural network designed for sequential data.",
        }
        
        test_text_2 = "Deep Learning and Convolutional Networks have revolutionized computer vision."
        
        matches_2 = search_glossary(test_text_2, glossary=custom_glossary, similarity_threshold=90.0)
        
        if not matches_2 or len(matches_2) < 1:
            logger.error(f"Expected at least 1 match with custom glossary, got {len(matches_2)}")
            all_tests_passed = False
        else:
            # Check that we found Deep Learning
            found_terms = [match["term"] for match in matches_2]
            if "Deep Learning" not in found_terms:
                logger.error(f"Expected to find 'Deep Learning', found: {found_terms}")
                all_tests_passed = False
            else:
                logger.success("✅ Custom glossary test: PASSED")
        
        # Test 3: Database glossary search
        logger.info("Test 3: Database glossary search")
        
        # Setup test collection with glossary terms
        test_collection = setup_test_collection(db)
        if not test_collection:
            logger.error("Failed to set up test collection")
            all_tests_passed = False
        else:
            # Test search with database glossary
            test_text_3 = "Natural Language Processing and Machine Learning are key components of modern AI."
            
            matches_3 = search_glossary(
                test_text_3, 
                similarity_threshold=90.0,
                db=db,
                collection_name=test_collection
            )
            
            if not matches_3 or len(matches_3) < 2:
                logger.error(f"Expected at least 2 matches with database glossary, got {len(matches_3)}")
                all_tests_passed = False
            else:
                # Check that we found the expected terms
                found_terms = [match["term"] for match in matches_3]
                expected_terms = ["Machine Learning", "Natural Language Processing"]
                
                if not all(term in found_terms for term in expected_terms):
                    logger.error(f"Missing expected terms. Found: {found_terms}")
                    all_tests_passed = False
                else:
                    logger.success("✅ Database glossary test: PASSED")
        
        return all_tests_passed
    
    except Exception as e:
        logger.exception(f"Error during glossary search test: {e}")
        return False
    
    finally:
        # Clean up test resources
        if test_collection:
            logger.info("Cleaning up test resources...")
            cleanup_test_resources(db, test_collection)

if __name__ == "__main__":
    logger.info("Starting glossary search test...")
    
    if test_glossary_search():
        logger.success("✅ Glossary search works!")
        sys.exit(0)
    else:
        logger.error("❌ Glossary search test FAILED")
        sys.exit(1)
