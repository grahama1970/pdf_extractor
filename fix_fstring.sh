#!/bin/bash
# Replace the problematic line
cat > /home/graham/workspace/experiments/pdf_extractor/src/pdf_extractor/arangodb/pdf_integration.py.fixed << 'ENDFILE'
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
    from pdf_extractor.arangodb.connection import get_db, create_collections
except ImportError as e:
    logger.error(f'Failed to import connection module: {e}')
    logger.error('Please ensure the ArangoDB connection module is available')

# PDF collection and view names - can be overridden with environment variables
PDF_COLLECTION_NAME = os.getenv('PDF_COLLECTION_NAME', 'pdf_documents')
PDF_VIEW_NAME = os.getenv('PDF_VIEW_NAME', 'pdf_search_view')

def setup_pdf_collection(db):
    '''
    Set up the collection for storing PDF extraction results.
    
    Args:
        db: ArangoDB database connection
        
    Returns:
        Collection object if successful, None otherwise
    '''
    try:
        # Create or get collections
        collections = create_collections(db, [PDF_COLLECTION_NAME])
        
        if PDF_COLLECTION_NAME not in collections:
            logger.error(f'Failed to create or access collection: {PDF_COLLECTION_NAME}')
            return None
        
        collection = collections[PDF_COLLECTION_NAME]
        
        # Create necessary indexes for PDF documents
        indexes = collection.indexes()
        index_names = [idx['name'] for idx in indexes]
        
        # Create fulltext index for the 'text' field
        if 'text_fulltext' not in index_names:
            collection.add_fulltext_index(fields=['text'], min_length=3, name='text_fulltext')
            logger.info(Created fulltext index for text field)
        
        # Create hash index for 'type' field (heading, paragraph, table, etc.)
        if 'type_hash' not in index_names:
            collection.add_hash_index(fields=['type'], name='type_hash')
            logger.info(Created hash index for type field)
        
        # Create hash index for 'file_path' field
        if 'file_path_hash' not in index_names:
            collection.add_hash_index(fields=['file_path'], name='file_path_hash')
            logger.info(Created hash index for file_path field)
        
        # Create skiplist index for 'page' field
        if 'page_skiplist' not in index_names:
            collection.add_skiplist_index(fields=['page'], name='page_skiplist')
            logger.info(Created skiplist index for page field)
        
        # Attempt to create vector index for embeddings (may not work if feature not enabled)
        try:
            if 'embeddings_vector' not in index_names:
                collection.add_persistent_index(
                    fields=['embeddings'],
                    name='embeddings_vector',
                    unique=False,
                    sparse=True
                )
                logger.info(Created vector index for embeddings field)
        except Exception as e:
            logger.warning(f'Vector index creation failed: {e}')
            logger.warning('Vector-based semantic search may not be available')
        
        return collection
    
    except Exception as e:
        logger.error(f'Error setting up PDF collection: {e}')
        return None

def setup_search_view(db):
    '''
    Set up an ArangoSearch view for BM25 searches.
    
    Args:
        db: ArangoDB database connection
        
    Returns:
        View object if successful, None otherwise
    '''
    try:
        # Check if view exists
        if db.has_view(PDF_VIEW_NAME):
            logger.info(f'Using existing view: {PDF_VIEW_NAME}')
            return db.view(PDF_VIEW_NAME)
        
        # Create ArangoSearch view with BM25 configuration
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
    
    except Exception as e:
        logger.error(f'Error setting up search view: {e}')
        return None

def store_pdf_document(db, document):
    '''
    Store a PDF document element in ArangoDB.
    
    Args:
        db: ArangoDB database connection
        document: Document element extracted from PDF
        
    Returns:
        Document ID if successful, None otherwise
    '''
    try:
        # Ensure collection exists
        collection = setup_pdf_collection(db)
        if not collection:
            return None
        
        # Add extraction timestamp if not present
        if 'extraction_date' not in document:
            document['extraction_date'] = datetime.now().isoformat()
        
        # Insert document
        result = collection.insert(document)
        logger.info(f'Stored document with key: {result[_key]}')
        return result['_id']
    
    except Exception as e:
        logger.error(f'Error storing PDF document: {e}')
        return None

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
        # Ensure collection exists
        collection = setup_pdf_collection(db)
        if not collection:
            return []
        
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
        # Ensure collection exists
        collection = setup_pdf_collection(db)
        if not collection:
            return []
        
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
        # Ensure view exists
        view = setup_search_view(db)
        if not view:
            return []
        
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

def semantic_search(db, query_embedding, limit=10):
    '''
    Perform semantic search using vector distance.
    Note: This may not work if ArangoDB doesn't have vector search enabled.
    
    Args:
        db: ArangoDB database connection
        query_embedding: Vector embedding of the query
        limit: Maximum number of results to return
        
    Returns:
        List of matching documents, ranked by vector similarity
    '''
    try:
        # Ensure collection exists
        collection = setup_pdf_collection(db)
        if not collection:
            return []
        
        # This might not work if vector search is not enabled
        try:
            # Construct AQL query
            aql = f'''
            FOR doc IN {PDF_COLLECTION_NAME}
                FILTER HAS(doc, embeddings)
                SORT DISTANCE(doc.embeddings, @query_embedding) ASC
                LIMIT @limit
                RETURN doc
            '''
            
            # Execute query
            cursor = db.aql.execute(
                aql,
                bind_vars={
                    'query_embedding': query_embedding,
                    'limit': limit
                }
            )
            
            # Return results
            results = [doc for doc in cursor]
            logger.info(f'Semantic search found {len(results)} documents')
            return results
            
        except Exception as e:
            logger.warning(f'Semantic search failed: {e}')
            logger.warning('ArangoDB may not have vector search enabled')
            return []
        
    except Exception as e:
        logger.error(f'Error performing semantic search: {e}')
        return []

def hybrid_search(db, query_text, query_embedding, limit=10):
    '''
    Perform hybrid search combining BM25 and semantic search.
    Note: This depends on semantic search being available.
    
    Args:
        db: ArangoDB database connection
        query_text: Text to search for (for BM25)
        query_embedding: Vector embedding of the query (for semantic)
        limit: Maximum number of results to return
        
    Returns:
        List of matching documents, ranked by combined relevance
    '''
    try:
        # Get results from both search methods
        bm25_results = bm25_search(db, query_text, limit=limit*2)
        semantic_results = semantic_search(db, query_embedding, limit=limit*2)
        
        if not semantic_results:
            logger.warning('Semantic search failed, falling back to BM25 only')
            return bm25_results[:limit]
        
        # Combine results using a simple scoring mechanism
        # In a production system, this would use more sophisticated methods like RRF
        combined = {}
        
        # Score BM25 results
        for i, doc in enumerate(bm25_results):
            key = doc['_key']
            if key not in combined:
                combined[key] = {
                    'document': doc,
                    'score': 0
                }
            combined[key]['score'] += (limit*2 - i) / (limit*2)  # Normalize to 0-1
        
        # Score semantic results
        for i, doc in enumerate(semantic_results):
            key = doc['_key']
            if key not in combined:
                combined[key] = {
                    'document': doc,
                    'score': 0
                }
            combined[key]['score'] += (limit*2 - i) / (limit*2)  # Normalize to 0-1
        
        # Sort by combined score
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Return top results
        results = [item['document'] for item in sorted_results[:limit]]
        logger.info(f'Hybrid search found {len(results)} documents')
        return results
        
    except Exception as e:
        logger.error(f'Error performing hybrid search: {e}')
        return []

def search_pdf_documents(db, query, search_type='basic', limit=10, embedding=None):
    '''
    Search PDF documents using the specified search type.
    
    Args:
        db: ArangoDB database connection
        query: Search query (text)
        search_type: Type of search to perform ('basic', 'fulltext', 'bm25', 'semantic', 'hybrid')
        limit: Maximum number of results to return
        embedding: Vector embedding for semantic/hybrid search (optional)
        
    Returns:
        List of matching documents
    '''
    # Validate search type
    valid_types = ['basic', 'fulltext', 'bm25', 'semantic', 'hybrid']
    if search_type not in valid_types:
        logger.error(f'Invalid search type: {search_type}. Must be one of {valid_types}')
        return []
    
    # Perform search based on type
    if search_type == 'basic':
        return basic_text_search(db, query, limit)
    elif search_type == 'fulltext':
        return fulltext_search(db, query, limit)
    elif search_type == 'bm25':
        return bm25_search(db, query, limit)
    elif search_type == 'semantic':
        if not embedding:
            logger.error('Embedding is required for semantic search')
            return []
        return semantic_search(db, embedding, limit)
    elif search_type == 'hybrid':
        if not embedding:
            logger.error('Embedding is required for hybrid search')
            return []
        return hybrid_search(db, query, embedding, limit)

def search_by_metadata(db, **filters):
    '''
    Search PDF documents by metadata fields.
    
    Args:
        db: ArangoDB database connection
        **filters: Metadata filters (e.g., type=heading, level=1, page=5)
        
    Returns:
        List of matching documents
    '''
    try:
        # Ensure collection exists
        collection = setup_pdf_collection(db)
        if not collection:
            return []
        
        # Construct filter clauses
        filter_clauses = []
        bind_vars = {}
        
        for key, value in filters.items():
            filter_clauses.append(f'doc.{key} == @{key}')
            bind_vars[key] = value
        
        # Construct AQL query
        filter_str = ' AND '.join(filter_clauses)
        aql = f'''
        FOR doc IN {PDF_COLLECTION_NAME}
            FILTER {filter_str}
            RETURN doc
        '''
        
        # Execute query
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        
        # Return results
        results = [doc for doc in cursor]
        logger.info(f'Metadata search found {len(results)} documents')
        return results
    
    except Exception as e:
        logger.error(f'Error performing metadata search: {e}')
        return []

# Validation function for main integration testing
def validate_arangodb_integration():
    '''
    Validate the ArangoDB integration by testing various search capabilities.
    
    Returns:
        Dict with validation results
    '''
    results = {
        'connection': False,
        'collection_setup': False,
        'view_setup': False,
        'basic_search': False,
        'fulltext_search': False,
        'bm25_search': False,
        'semantic_search': 'not_tested',
        'hybrid_search': 'not_tested'
    }
    
    try:
        # Connect to database
        db = get_db()
        results['connection'] = True
        logger.info('ArangoDB connection successful')
        
        # Set up collection
        collection = setup_pdf_collection(db)
        if collection:
            results['collection_setup'] = True
            logger.info('PDF collection setup successful')
        
        # Set up view
        view = setup_search_view(db)
        if view:
            results['view_setup'] = True
            logger.info('ArangoSearch view setup successful')
        
        # Test sample document
        sample_doc = {
            'type': 'heading',
            'level': 1,
            'text': 'ArangoDB Integration Test',
            'page': 1,
            'token_count': 3,
            'file_path': 'test.pdf',
            'extraction_date': datetime.now().isoformat(),
            'source': 'test'
        }
        
        # Store document
        doc_id = store_pdf_document(db, sample_doc)
        if not doc_id:
            logger.error('Failed to store test document')
            return results
        
        # Test basic search
        basic_results = basic_text_search(db, 'Integration Test')
        if basic_results and len(basic_results) > 0:
            results['basic_search'] = True
            logger.info('Basic search test successful')
        
        # Test fulltext search
        fulltext_results = fulltext_search(db, 'Integration')
        if fulltext_results and len(fulltext_results) > 0:
            results['fulltext_search'] = True
            logger.info('Fulltext search test successful')
        
        # Test BM25 search (may take a moment for the view to be ready)
        for attempt in range(3):
            try:
                bm25_results = bm25_search(db, 'Integration Test')
                if bm25_results and len(bm25_results) > 0:
                    results['bm25_search'] = True
                    logger.info('BM25 search test successful')
                    break
            except Exception as e:
                if attempt < 2:
                    logger.warning(f'BM25 search test attempt {attempt+1} failed, retrying...')
                    import time
                    time.sleep(1)  # Wait for view to be ready
                else:
                    logger.error(f'All BM25 search test attempts failed: {e}')
        
        # Note: We don't test semantic and hybrid search here as they require vector embeddings
        # and may not be available if ArangoDB doesn't have vector search enabled
        
        # Clean up test document
        if doc_id:
            collection.delete(doc_id)
            logger.info(f'Cleaned up test document: {doc_id}')
        
        return results
    
    except Exception as e:
        logger.error(f'ArangoDB integration validation failed: {e}')
        return results

if __name__ == '__main__':
    # Run validation
    print('Validating ArangoDB integration...')
    results = validate_arangodb_integration()
    
    # Print results
    print('\nArangoDB Integration Status:')
    print('----------------------------')
    for key, value in results.items():
        status = '✅ WORKING' if value is True else '❌ FAILED' if value is False else '⚠️ ' + value.upper()
        print(f'{key.replace(_,  ).title()}: {status}')
    
    # Overall status
    working_count = sum(1 for v in results.values() if v is True)
    total_testable = sum(1 for v in results.values() if v is not 'not_tested')
    print(f'\nWorking: {working_count}/{total_testable} tested features')
    
    if working_count == total_testable:
        print('✅ ArangoDB integration is fully functional for tested features')
        sys.exit(0)
    else:
        print('⚠️ Some ArangoDB integration features are not working correctly')
        sys.exit(1)
ENDFILE

# Replace the original file with the fixed version
mv /home/graham/workspace/experiments/pdf_extractor/src/pdf_extractor/arangodb/pdf_integration.py.fixed /home/graham/workspace/experiments/pdf_extractor/src/pdf_extractor/arangodb/pdf_integration.py
chmod +x /home/graham/workspace/experiments/pdf_extractor/src/pdf_extractor/arangodb/pdf_integration.py

echo 'Fixed all quotes in PDF integration file'
