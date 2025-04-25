#!/usr/bin/env python3
import sys
import uuid
from datetime import datetime, timezone
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from loguru import logger

logger.remove()
logger.add(sys.stderr, level='INFO')

def test_simple_crud():
    # Connect to ArangoDB
    client = connect_arango()
    if not client:
        print('Failed to connect to ArangoDB')
        sys.exit(1)
    
    db = ensure_database(client)
    if not db:
        print('Failed to ensure database exists')
        sys.exit(1)
    
    collection_name = 'claude_message_history'
    
    try:
        # Create test document
        test_id = str(uuid.uuid4())
        test_doc = {
            '_key': test_id,
            'conversation_id': str(uuid.uuid4()),
            'message_type': 'SYSTEM',
            'content': 'This is a test message for CRUD validation',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metadata': {'test': True}
        }
        
        # Get the collection
        collection = db.collection(collection_name)
        
        # Insert document
        print(f'Creating test document with key: {test_id}')
        meta = collection.insert(test_doc, return_new=True)
        print(f'Document created: {meta[_key]}')
        
        # Get document
        retrieved = collection.get(test_id)
        print(f'Document retrieved: {retrieved[_key]}')
        
        # Update document
        update_doc = {'_key': test_id, 'content': 'Updated test content'}
        updated = collection.update(update_doc, return_new=True)
        print(f'Document updated: {updated[new][_key]}')
        print(f'  New content: {updated[new][content]}')
        
        # Delete document
        collection.delete(test_id)
        print(f'Document deleted: {test_id}')
        
        print('\nBasic CRUD operations completed successfully')
        return True
    except Exception as e:
        print(f'Error during CRUD operations: {e}')
        return False
    
if __name__ == '__main__':
    test_simple_crud()
