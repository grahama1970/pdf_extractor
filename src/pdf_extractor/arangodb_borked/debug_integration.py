#!/usr/bin/env python3
"""
Debug PDF Extractor ArangoDB Integration
"""

import sys
import traceback
import logging
from pdf_extractor.arangodb_borked.connection import get_db
from pdf_extractor.arangodb_borked.pdf_integration import setup_pdf_collection

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_setup_collection():
    """Debug the setup_pdf_collection function"""
    try:
        # Connect to ArangoDB
        db = get_db()
        if not db:
            logger.error("Failed to connect to ArangoDB")
            return False
            
        logger.info(f"Connected to ArangoDB database: {db.name}")
        
        # Try setting up the collection and capture the full traceback
        try:
            collection = setup_pdf_collection(db)
            if collection:
                logger.info(f"Successfully set up collection: {collection.name}")
                return True
            else:
                logger.error("setup_pdf_collection returned None")
                return False
        except Exception as e:
            logger.error("Error in setup_pdf_collection:")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Exception message: {str(e)}")
            logger.error("Traceback:")
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("=== Debugging PDF Extractor ArangoDB Integration ===")
    
    success = debug_setup_collection()
    
    if success:
        logger.info("✅ Debug successful")
        sys.exit(0)
    else:
        logger.error("❌ Debug failed")
        sys.exit(1)
