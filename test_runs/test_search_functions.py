#!/usr/bin/env python3
# test_search_functions.py - Test script for search functions

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path('.').absolute()))

from src.pdf_extractor.arangodb.search_api.search_functions import search_messages, unified_search
from src.pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database

def test_unified_search():
    print('Testing unified search...')
    
    # Connect to ArangoDB
    client = connect_arango()
    if not client:
        print('Failed to connect to ArangoDB')
        return False
    
    db = ensure_database(client)
    if not db:
        print('Failed to ensure database exists')
        return False
    
    # Test unified search
    query = 'test'
    results = unified_search(db, query, top_n=5)
    
    print(f'=== Unified Search ===')
    print(f'Query: "{query}"')
    print(f'Results: {len(results.get("results", []))}')
    print(f'Collections searched: {results.get("collections_searched", [])}')
    
    # Print results
    for i, result in enumerate(results.get('results', [])[:3], 1):
        collection = result.get('collection', 'unknown')
        score = result.get('score', 0)
        doc_id = result.get('_id', 'unknown')
        content = result.get('content', '')
        content_preview = content[:50] + '...' if len(content) > 50 else content
        
        print(f'{i}. [{collection}] Score: {score:.4f}')
        print(f'   ID: {doc_id}')
        print(f'   Content: {content_preview}')
    
    return True

def test_message_search():
    print('\nTesting message search...')
    
    # Connect to ArangoDB
    client = connect_arango()
    if not client:
        print('Failed to connect to ArangoDB')
        return False
    
    db = ensure_database(client)
    if not db:
        print('Failed to ensure database exists')
        return False
    
    # Test message search
    query = 'test'
    results = search_messages(db, query, top_n=5)
    
    print(f'=== Message Search ===')
    print(f'Query: "{query}"')
    print(f'Results: {len(results.get("results", []))}')
    
    # Print results
    for i, result in enumerate(results.get('results', [])[:3], 1):
        message_type = result.get('message_type', 'unknown')
        score = result.get('score', 0)
        msg_id = result.get('_id', 'unknown')
        content = result.get('content', '')
        content_preview = content[:50] + '...' if len(content) > 50 else content
        
        print(f'{i}. [{message_type}] Score: {score:.4f}')
        print(f'   ID: {msg_id}')
        print(f'   Content: {content_preview}')
    
    return True

if __name__ == '__main__':
    unified_result = test_unified_search()
    message_result = test_message_search()
    
    if unified_result and message_result:
        print('\n✅ All tests passed')
        sys.exit(0)
    else:
        print('\n❌ Some tests failed')
        sys.exit(1)
