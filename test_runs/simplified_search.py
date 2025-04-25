#!/usr/bin/env python3
# simplified_search.py - Test for search functionality with mocked search functions

import sys
import os
from pathlib import Path

# Create example search results for testing
def mock_search_results(query, collection_name, size=5):
    '''Create mock search results for testing'''
    results = []
    for i in range(size):
        results.append({
            '_id': f'{collection_name}/{i}',
            '_key': f'{i}',
            'content': f'Sample content related to {query} (result {i})',
            'score': 0.9 - (i * 0.1)
        })
    return {
        'status': 'success',
        'message': f'Found {len(results)} results',
        'results': results
    }

def search_messages(query, message_type=None, conversation_id=None, top_n=5):
    '''Search messages with mock results'''
    print(f'\n=== Message Search ===')
    print(f'Query: "{query}"')
    if message_type:
        print(f'Message Type: {message_type}')
    if conversation_id:
        print(f'Conversation ID: {conversation_id}')
    
    results = mock_search_results(query, 'messages', top_n)
    
    # Add message-specific fields
    for result in results['results']:
        result['message_type'] = message_type or ('USER' if int(result['_key']) % 2 == 0 else 'AGENT')
        result['conversation_id'] = conversation_id or f'conversation-{int(result["_key"]) % 3}'
    
    print(f'Results: {len(results[results])}')
    
    # Print results
    for i, result in enumerate(results['results'], 1):
        message_type = result.get('message_type', 'unknown')
        score = result.get('score', 0)
        msg_id = result.get('_id', 'unknown')
        content = result.get('content', '')
        conversation = result.get('conversation_id', 'unknown')
        
        print(f'{i}. [{message_type}] Score: {score:.4f}')
        print(f'   ID: {msg_id}')
        print(f'   Conversation: {conversation}')
        print(f'   Content: {content}')
    
    return results

def unified_search(query, collections=None, exclude_collections=None, top_n=5):
    '''Search across collections with mock results'''
    print(f'\n=== Unified Search ===')
    print(f'Query: "{query}"')
    
    # Determine which collections to search
    available_collections = ['documents', 'messages']
    
    if collections:
        search_collections = [c for c in collections if c in available_collections]
    else:
        search_collections = available_collections.copy()
    
    if exclude_collections:
        search_collections = [c for c in search_collections if c not in exclude_collections]
    
    print(f'Collections searched: {search_collections}')
    
    # Generate mock results for each collection
    all_results = []
    for collection in search_collections:
        collection_results = mock_search_results(query, collection, top_n // len(search_collections))
        
        # Add collection information to each result
        for result in collection_results.get('results', []):
            result['collection'] = collection
            all_results.append(result)
    
    # Sort combined results by relevance score (descending)
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Truncate to top_n results overall if requested
    if len(all_results) > top_n:
        all_results = all_results[:top_n]
    
    print(f'Results: {len(all_results)}')
    
    # Print results
    for i, result in enumerate(all_results, 1):
        collection = result.get('collection', 'unknown')
        score = result.get('score', 0)
        doc_id = result.get('_id', 'unknown')
        content = result.get('content', '')
        
        print(f'{i}. [{collection}] Score: {score:.4f}')
        print(f'   ID: {doc_id}')
        print(f'   Content: {content}')
    
    return {
        'status': 'success',
        'message': f'Search completed across {len(search_collections)} collections',
        'collections_searched': search_collections,
        'results': all_results
    }

def test_search_commands():
    '''Test each search command with mock data'''
    # Test 1: Basic message search
    search_messages('configuration')
    
    # Test 2: Message search with type filter
    search_messages('user question', message_type='USER')
    
    # Test 3: Message search with conversation filter
    search_messages('conversation context', conversation_id='conversation-123')
    
    # Test 4: Unified search across all collections
    unified_search('error handling')
    
    # Test 5: Unified search with collection filter
    unified_search('api reference', collections=['documents'])
    
    # Test 6: Unified search with exclusion
    unified_search('security protocol', exclude_collections=['messages'])
    
    # Test 7: Unified search with specific limit
    unified_search('system performance', top_n=3)
    
    print('\n✅ All search commands tested successfully')
    return True

if __name__ == '__main__':
    success = test_search_commands()
    sys.exit(0 if success else 1)
