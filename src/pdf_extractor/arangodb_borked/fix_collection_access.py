#!/usr/bin/env python3
"""
Quick fix script to test collection access in ArangoDB
"""

import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import connection module
try:
    from pdf_extractor.arangodb_borked.connection import get_db, create_collections
except ImportError:
    # Add project root to path if needed
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from pdf_extractor.arangodb_borked.connection import get_db, create_collections

def main():
    try:
        # Connect to database
        db = get_db()
        logger.info(f"Successfully connected to database: {db.name}")
        
        # Check collection access with create_collections
        collections_dict = create_collections(db)
        logger.info(f"Collections returned from create_collections: {collections_dict.keys() if collections_dict else 'None'}")
        
        # Direct access to collection
        if db.has_collection('lessons_learned'):
            collection = db.collection('lessons_learned')
            logger.info(f"Successfully accessed collection: lessons_learned (count: {collection.count()})")
            
            # List all collections
            all_collections = [c['name'] for c in db.collections()]
            logger.info(f"All collections in database: {all_collections}")
            
            return True
        else:
            logger.error("Collection 'lessons_learned' does not exist")
            return False
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ Collection access test passed")
        sys.exit(0)
    else:
        logger.error("❌ Collection access test failed")
        sys.exit(1)
