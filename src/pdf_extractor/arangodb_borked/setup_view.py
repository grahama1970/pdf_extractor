#!/usr/bin/env python3

import os
import sys
import logging
from pdf_extractor.arangodb_borked.connection import get_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Collection and view names
PDF_COLLECTION_NAME = os.getenv('PDF_COLLECTION_NAME', 'pdf_documents')
PDF_VIEW_NAME = os.getenv('PDF_VIEW_NAME', 'pdf_search_view')

def setup_search_view(db):
    '''Set up an ArangoSearch view for BM25 searches'''
    try:
        # Check if view exists
        try:
            if hasattr(db, 'has_view') and db.has_view(PDF_VIEW_NAME):
                logger.info(f'Using existing view: {PDF_VIEW_NAME}')
                return db.view(PDF_VIEW_NAME)
        except Exception:
            pass
        
        # Create ArangoSearch view with BM25 configuration
        if hasattr(db, 'create_arangosearch_view'):
            view = db.create_arangosearch_view(
                name=PDF_VIEW_NAME,
                properties={
                    'links': {
                        PDF_COLLECTION_NAME: {
                            'includeAllFields': False,
                            'fields': {
                                'text': {
                                    'analyzers': ['text_en']
                                },
                                'type': {},
                                'file_path': {},
                                'page': {}
                            },
                            'analyzers': ['identity', 'text_en']
                        }
                    },
                    'commitIntervalMsec': 1000
                }
            )
            logger.info(f'Created ArangoSearch view: {PDF_VIEW_NAME}')
            return view
        else:
            logger.warning('ArangoDB does not support ArangoSearch views')
            return None
    
    except Exception as e:
        logger.error(f'Error setting up search view: {e}')
        return None

if __name__ == '__main__':
    try:
        # Connect to database
        db = get_db()
        logger.info('Connected to ArangoDB')
        
        # Setup view
        view = setup_search_view(db)
        if view:
            print(f'✅ Successfully set up ArangoSearch view: {PDF_VIEW_NAME}')
            sys.exit(0)
        else:
            print(f'❌ Failed to set up ArangoSearch view: {PDF_VIEW_NAME}')
            sys.exit(1)
    
    except Exception as e:
        logger.error(f'Error: {e}')
        sys.exit(1)
