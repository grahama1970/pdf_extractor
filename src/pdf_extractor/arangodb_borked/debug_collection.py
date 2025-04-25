#!/usr/bin/env python3
"""
Debug script for ArangoDB collection access
"""

import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Manual import to be explicit
        from arango import ArangoClient
        
        # Direct database connection
        host = 'http://localhost:8529'
        username = 'root'
        password = 'openSesame'
        db_name = 'pdf_extractor'
        
        print(f"Connecting to ArangoDB at {host} with username {username} and database {db_name}")
        
        # Initialize client
        client = ArangoClient(hosts=host)
        
        # Connect to system database
        sys_db = client.db('_system', username=username, password=password)
        print("Connected to _system database")
        
        # Check if our database exists
        if not sys_db.has_database(db_name):
            print(f"Database {db_name} does not exist, creating it")
            sys_db.create_database(db_name)
        else:
            print(f"Database {db_name} already exists")
        
        # Connect to our database
        db = client.db(db_name, username=username, password=password)
        print(f"Connected to database {db_name}")
        
        # List all collections
        all_collections = [c['name'] for c in db.collections()]
        print(f"All collections in database: {all_collections}")
        
        # Check for our collection
        collection_name = 'lessons_learned'
        if not db.has_collection(collection_name):
            print(f"Collection {collection_name} does not exist, creating it")
            collection = db.create_collection(collection_name)
        else:
            print(f"Collection {collection_name} already exists")
            collection = db.collection(collection_name)
            
        print(f"Collection info: {collection}")
        
        # Test the collection by inserting a document
        test_doc = {'_key': 'test_doc', 'test': 'value'}
        try:
            print("Trying to insert test document...")
            result = collection.insert(test_doc, overwrite=True)
            print(f"Test document inserted: {result}")
            
            # Retrieve the document
            print("Trying to retrieve test document...")
            doc = collection.get('test_doc')
            print(f"Retrieved test document: {doc}")
            
            # Create index for testing
            print("Creating test index on the collection...")
            result = collection.add_persistent_index(fields=['test'], unique=False)
            print(f"Index created: {result}")
            
            # Delete the test document
            print("Trying to delete test document...")
            collection.delete('test_doc')
            print("Test document deleted")
            
            print("Collection test operations completed successfully")
            return True
        except Exception as e:
            print(f"Error during collection operations: {e}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ ArangoDB collection test passed")
        sys.exit(0)
    else:
        print("❌ ArangoDB collection test failed")
        sys.exit(1)
