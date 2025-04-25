#!/usr/bin/env python3
import os
import logging
from arango import ArangoClient
from arango.exceptions import ServerConnectionError, DatabaseCreateError, CollectionCreateError

logger = logging.getLogger(__name__)

def get_db(host=None, username=None, password=None, db_name=None):
    # Get connection parameters from environment variables if not provided
    host = host or os.environ.get('ARANGO_HOST', 'http://localhost:8529')
    username = username or os.environ.get('ARANGO_USER', 'root')
    password = password or os.environ.get('ARANGO_PASSWORD', 'openSesame')
    db_name = db_name or os.environ.get('ARANGO_DB', 'pdf_extractor')
    
    try:
        # Initialize client
        client = ArangoClient(hosts=host)
        
        # Connect to system database to create our database if it doesn't exist
        sys_db = client.db('_system', username=username, password=password)
        
        # Create database if it doesn't exist
        if not sys_db.has_database(db_name):
            sys_db.create_database(db_name)
            logger.info('Created database: ' + db_name)
        
        # Connect to the database
        db = client.db(db_name, username=username, password=password)
        logger.info('Connected to database: ' + db_name)
        
        return db
        
    except ServerConnectionError as e:
        logger.error('Failed to connect to ArangoDB server: ' + str(e))
        raise
    except DatabaseCreateError as e:
        logger.error('Failed to create database: ' + str(e))
        raise

def create_collections(db, collections=None):
    collections = collections or ['lessons_learned']
    collection_objects = {}
    
    for collection_name in collections:
        try:
            # Check if collection exists
            if db.has_collection(collection_name):
                collection = db.collection(collection_name)
                logger.info('Using existing collection: ' + collection_name)
            else:
                # Create collection
                collection = db.create_collection(collection_name)
                logger.info('Created collection: ' + collection_name)
                
                # Create indexes for improved search performance
                if collection_name == 'lessons_learned':
                    try:
                        # Index for tags array
                        if 'tags' in collection.indexes():
                            logger.info('Tags index already exists')
                        else:
                            collection.add_hash_index(fields=['tags[*]'], unique=False)
                            logger.info('Created tags index')
                            
                        # Index for author field
                        if 'author' in collection.indexes():
                            logger.info('Author index already exists')
                        else:
                            collection.add_hash_index(fields=['author'], unique=False)
                            logger.info('Created author index')
                            
                        # Full-text index for problem and solution
                        collection.add_fulltext_index(fields=['problem'], min_length=3, name='problem_fulltext')
                        collection.add_fulltext_index(fields=['solution'], min_length=3, name='solution_fulltext')
                        logger.info('Created fulltext indexes for problem and solution')
                    except Exception as e:
                        logger.warning(f'Error creating indexes: {e}. Continuing without all indexes.')
            
            collection_objects[collection_name] = collection
            
        except CollectionCreateError as e:
            logger.error('Failed to create collection: ' + str(e))
            raise
            
    return collection_objects

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test connection and collection creation
    try:
        db = get_db()
        collections = create_collections(db)
        print("Successfully connected to ArangoDB and created collections")
    except Exception as e:
        print("Error: " + str(e))
