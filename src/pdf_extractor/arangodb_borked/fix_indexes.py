#!/usr/bin/env python3
"""
Fix for ArangoDB index creation
"""

import sys
import logging
from pdf_extractor.arangodb_borked.connection import get_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_pdf_collection_indexes(db, collection_name='pdf_documents'):
    """Create indexes for the PDF collection using updated API"""
    try:
        # Get the collection
        if not db.has_collection(collection_name):
            logger.error(f"Collection {collection_name} does not exist")
            return False
            
        collection = db.collection(collection_name)
        
        # Create type index
        collection.add_index({
            'type': 'hash',
            'fields': ['type'],
            'unique': False
        })
        logger.info("Created hash index on 'type' field")
        
        # Create file_path index
        collection.add_index({
            'type': 'hash',
            'fields': ['file_path'],
            'unique': False
        })
        logger.info("Created hash index on 'file_path' field")
        
        # Create page index
        collection.add_index({
            'type': 'skiplist',
            'fields': ['page'],
            'unique': False
        })
        logger.info("Created skiplist index on 'page' field")
        
        # Create fulltext index
        collection.add_index({
            'type': 'fulltext',
            'fields': ['text'],
            'minLength': 3
        })
        logger.info("Created fulltext index on 'text' field")
        
        logger.info(f"Successfully created indexes for collection: {collection_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create indexes: {e}")
        return False

if __name__ == "__main__":
    # Connect to ArangoDB
    db = get_db()
    if not db:
        logger.error("Failed to connect to ArangoDB")
        sys.exit(1)
        
    # Create indexes
    if create_pdf_collection_indexes(db):
        logger.info("✅ Successfully created indexes")
        sys.exit(0)
    else:
        logger.error("❌ Failed to create indexes")
        sys.exit(1)
EOF"
