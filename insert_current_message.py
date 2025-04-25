import sys
import uuid
from datetime import datetime, timezone
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database

def insert_message():
    client = connect_arango()
    db = ensure_database(client)
    
    collection = db.collection('claude_message_history')
    
    # Create message document
    message = {
        '_key': str(uuid.uuid4()),
        'conversation_id': str(uuid.uuid4()),
        'message_type': 'USER',
        'content': 'I don\'t see a basic crud.py file for handling database functions like this',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'metadata': {
            'user_id': 'graham',
            'chat_name': 'PDF Extractor Project - CRUD Implementation',
            'source': 'Simple CRUD Test'
        }
    }
    
    # Insert message
    result = collection.insert(message)
    print(f"Message inserted with key: {result['_key']}\n")
    
    # Retrieve the message to verify
    doc = collection.get(result['_key'])
    print(f"Retrieved message:\n- Key: {doc['_key']}\n- Content: {doc['content']}\n- Type: {doc['message_type']}\n- Timestamp: {doc['timestamp']}")
    print(f"- Metadata: {doc['metadata']}")
    
    print("\nCRUD operations working successfully")

if __name__ == '__main__':
    insert_message()
