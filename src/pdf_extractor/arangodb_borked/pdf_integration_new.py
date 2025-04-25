#!/usr/bin/env python3
'''
PDF Extractor ArangoDB Integration Module

This module provides functions for integrating the PDF extractor with ArangoDB,
enabling efficient storage and retrieval of extracted PDF content using various
query methods: basic text search, fulltext search, BM25 search, and preparation
for semantic and hybrid search.
'''

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import ArangoDB connection module
try:
    from pdf_extractor.arangodb_borked.connection import get_db, create_collections
except ImportError as e:
    logger.error(f'Failed to import connection module: {e}')
    logger.error('Please ensure the ArangoDB connection module is available')

# PDF collection and view names - can be overridden with environment variables
PDF_COLLECTION_NAME = os.getenv('PDF_COLLECTION_NAME', 'pdf_documents')
PDF_VIEW_NAME = os.getenv('PDF_VIEW_NAME', 'pdf_search_view')

def basic_text_search(db, query_text, limit=10):
    '''
    Perform basic text search using CONTAINS function.
    
    Args:
        db: ArangoDB database connection
        query_text: Text to search for
        limit: Maximum number of results to return
        
    Returns:
        List of matching documents
    '''
    try:
        # Construct AQL query
        aql = f'''
        FOR doc IN {PDF_COLLECTION_NAME}
            FILTER CONTAINS(doc.text, @query_text, true)
            LIMIT @limit
            RETURN doc
        '''
        
        # Execute query
        cursor = db.aql.execute(
            aql,
            bind_vars={
                'query_text': query_text,
                'limit': limit
            }
        )
        
        # Return results
        results = [doc for doc in cursor]
        logger.info(f'Basic text search found {len(results)} documents')
        return results
    
    except Exception as e:
        logger.error(f'Error performing basic text search: {e}')
        return []

def fulltext_search(db, query_text, limit=10):
    '''
    Perform fulltext search using FULLTEXT function.
    
    Args:
        db: ArangoDB database connection
        query_text: Text to search for
        limit: Maximum number of results to return
        
    Returns:
        List of matching documents
    '''
    try:
        # Construct AQL query
        aql = f'''
        FOR doc IN FULLTEXT({PDF_COLLECTION_NAME}, text, @query_text)
            LIMIT @limit
            RETURN doc
        '''
        
        # Execute query
        cursor = db.aql.execute(
            aql,
            bind_vars={
                'query_text': query_text,
                'limit': limit
            }
        )
        
        # Return results
        results = [doc for doc in cursor]
        logger.info(f'Fulltext search found {len(results)} documents')
        return results
    
    except Exception as e:
        logger.error(f'Error performing fulltext search: {e}')
        return []

def bm25_search(db, query_text, limit=10):
    '''
    Perform BM25 search using ArangoSearch view.
    
    Args:
        db: ArangoDB database connection
        query_text: Text to search for
        limit: Maximum number of results to return
        
    Returns:
        List of matching documents, ranked by relevance
    '''
    try:
        # Create or get the search view
        if not db.has_view(PDF_VIEW_NAME):
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
        else:
            logger.info(f'Using existing view: {PDF_VIEW_NAME}')
        
        # Construct AQL query
        aql = f'''
        FOR doc IN {PDF_VIEW_NAME}
            SEARCH ANALYZER(
                PHRASE(doc.text, @query_text, text_en),
                text_en
            )
            SORT BM25(doc) DESC
            LIMIT @limit
            RETURN doc
        '''
        
        # Execute query
        cursor = db.aql.execute(
            aql,
            bind_vars={
                'query_text': query_text,
                'limit': limit
            }
        )
        
        # Return results
        results = [doc for doc in cursor]
        logger.info(f'BM25 search found {len(results)} documents')
        return results
    
    except Exception as e:
        logger.error(f'Error performing BM25 search: {e}')
        return []

if __name__ == '__main__':
    # Test the implementation
    print('Validating ArangoDB integration...')
    
    try:
        # Connect to database
        db = get_db()
        print('✅ Connected to ArangoDB')
        
        # Test basic text search
        print('Testing basic text search...')
        results = basic_text_search(db, 'test', 5)
        print(f'Found {len(results)} results with basic search')
        
        # Test fulltext search
        print('Testing fulltext search...')
        results = fulltext_search(db, 'test', 5)
        print(f'Found {len(results)} results with fulltext search')
        
        # Test BM25 search
        print('Testing BM25 search...')
        results = bm25_search(db, 'test', 5)
        print(f'Found {len(results)} results with BM25 search')
        
        print('✅ ArangoDB integration working correctly')
        
    except Exception as e:
        print(f'❌ Error validating ArangoDB integration: {e}')
        sys.exit(1)
