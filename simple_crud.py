
import sys
import uuid
from datetime import datetime, timezone
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database

def main():
    # Connect to ArangoDB
    client = connect_arango()
    db = ensure_database(client)
    
    # Create test document
    test_id = str(uuid.uuid4())
    test_doc = {
        "_key": test_id,
        "conversation_id": str(uuid.uuid4()),
        "message_type": "SYSTEM",
        "content": "This is a test message for CRUD validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {"test": True}
    }
    
    # Get the collection
    collection = db.collection("claude_message_history")
    
    # Insert document
    print(f"Creating test document with key: {test_id}")
    collection.insert(test_doc)
    print("Document created")
    
    # Get document
    retrieved = collection.get(test_id)
    print("Document retrieved")
    
    # Update document
    update_doc = {"_key": test_id, "content": "Updated test content"}
    collection.update(update_doc)
    print("Document updated")
    
    # Delete document
    collection.delete(test_id)
    print("Document deleted")
    
    print("
Basic CRUD operations completed successfully")

if __name__ == "__main__":
    main()

