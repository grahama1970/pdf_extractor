#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path('.').absolute()))

from src.pdf_extractor.arangodb.search_api.message_search import unified_search
from src.pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database

def test_search():
    print('Testing search functionality...')
    
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
    results = unified_search(db, query)
    
    print(f'Search Query: {query}')
    print(f'Results: {len(results.get("results", []))}')
    
    # Print first few results
    for i, result in enumerate(results.get('results', [])[:3], 1):
        collection = result.get('collection', 'unknown')
        score = result.get('score', 0)
        doc_id = result.get('_id', 'unknown')
        content_preview = result.get('content', '')[:50] + '...' if result.get('content') else ''
        
        print(f'{i}. [{collection}] Score: {score:.4f}')
        print(f'   ID: {doc_id}')
        print(f'   Content: {content_preview}')
    
    return True

if __name__ == '__main__':
    success = test_search()
    sys.exit(0 if success else 1)
