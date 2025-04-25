#!/usr/bin/env python3
'''ArangoDB Integration for PDF Extractor'''

import sys
from pdf_extractor.arangodb_borked.connection import get_db, create_collections
from pdf_extractor.arangodb_borked.search_functions import basic_text_search, fulltext_search, bm25_search, semantic_search, hybrid_search

def search_pdf_documents(db, query, search_type='basic', limit=10, embedding=None):
    '''Search PDF documents using the specified search type'''
    # Validate search type
    valid_types = ['basic', 'fulltext', 'bm25', 'semantic', 'hybrid']
    if search_type not in valid_types:
        print(f'Invalid search type: {search_type}. Must be one of {valid_types}')
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
            print('Embedding is required for semantic search')
            return []
        return semantic_search(db, embedding, limit)
    elif search_type == 'hybrid':
        if not embedding:
            print('Embedding is required for hybrid search')
            return []
        return hybrid_search(db, query, embedding, limit)

def test_arangodb_integration():
    '''Test the ArangoDB integration'''
    # Connect to database
    try:
        db = get_db()
        print('✅ Connected to ArangoDB')
        
        # Create or get collections
        collections = create_collections(db, ['pdf_documents'])
        if 'pdf_documents' not in collections:
            print('❌ Failed to create or access collection')
            return False
        
        # Test basic search
        results = basic_text_search(db, 'test', 5)
        print(f'Basic search found {len(results)} documents')
        
        # Test fulltext search
        try:
            results = fulltext_search(db, 'test', 5)
            print(f'Fulltext search found {len(results)} documents')
        except Exception as e:
            print(f'❌ Fulltext search error: {e}')
        
        # Test BM25 search
        try:
            results = bm25_search(db, 'test', 5)
            print(f'BM25 search found {len(results)} documents')
        except Exception as e:
            print(f'❌ BM25 search error: {e}')
        
        return True
    
    except Exception as e:
        print(f'❌ Error testing ArangoDB integration: {e}')
        return False

if __name__ == '__main__':
    print('Testing ArangoDB integration...')
    success = test_arangodb_integration()
    
    if success:
        print('✅ ArangoDB integration is working')
        sys.exit(0)
    else:
        print('❌ ArangoDB integration test failed')
        sys.exit(1)
