#!/usr/bin/env python3
'''
ArangoDB Search Functions for PDF Extractor

Implements three search types:
1. Basic text search (CONTAINS)
2. Fulltext search (FULLTEXT)
3. BM25 search (ArangoSearch view)
'''

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PDF collection and view names
PDF_COLLECTION_NAME = os.getenv('PDF_COLLECTION_NAME', 'pdf_documents')
PDF_VIEW_NAME = os.getenv('PDF_VIEW_NAME', 'pdf_search_view')

def basic_text_search(db, query_text, limit=10):
    '''Perform basic text search using CONTAINS function'''
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
    '''Perform fulltext search using FULLTEXT function'''
    try:
        # Construct AQL query
        aql = f'''
        FOR doc IN FULLTEXT({PDF_COLLECTION_NAME}, "text", @query_text)
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
    '''Perform BM25 search using ArangoSearch view'''
    try:
        # Ensure view exists
        try:
            if hasattr(db, 'has_view') and db.has_view(PDF_VIEW_NAME):
                pass  # View exists
            else:
                # Create view
                db.create_arangosearch_view(
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
        except Exception as e:
            logger.warning(f'View error: {e}, falling back to fulltext search')
            return fulltext_search(db, query_text, limit)
        
        # Construct AQL query
        aql = f'''
        FOR doc IN {PDF_VIEW_NAME}
            SEARCH ANALYZER(
                PHRASE(doc.text, @query_text, "text_en"),
                "text_en"
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
        # Fall back to fulltext search
        logger.info('Falling back to fulltext search')
        return fulltext_search(db, query_text, limit)

if __name__ == '__main__':
    print('ArangoDB search functions module')
    print('1. Basic text search (CONTAINS)')
    print('2. Fulltext search (FULLTEXT)')
    print('3. BM25 search (ArangoSearch view)')

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
        # This might not work if vector search is not enabled
        try:
            # Construct AQL query
            aql = f'''
            FOR doc IN {PDF_COLLECTION_NAME}
                FILTER HAS(doc, "embeddings")
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
