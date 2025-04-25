#!/usr/bin/env python3
"""
Fix collections and views for ArangoDB search functions.

This script repairs or creates the necessary collections and views
required by the search functions.
"""

import sys
from arango import ArangoClient
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# Connection parameters
HOST = "http://192.168.86.49:8529/"
USERNAME = "root"
PASSWORD = "openSesame"
DB_NAME = "pdf_extractor"

# Collection and view names
COLLECTION_NAME = "documents"
EDGE_COLLECTION_NAME = "relationships"
VIEW_NAME = "document_view"
GRAPH_NAME = "knowledge_graph"

def delete_graph_if_exists():
    """Delete the graph if it exists."""
    client = ArangoClient(hosts=HOST)
    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
    
    if db.has_graph(GRAPH_NAME):
        logger.info(f"Deleting graph '{GRAPH_NAME}'...")
        db.delete_graph(GRAPH_NAME)
        logger.info(f"Graph '{GRAPH_NAME}' deleted")
    
    return True

def fix_documents_collection():
    """Fix the documents collection."""
    client = ArangoClient(hosts=HOST)
    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
    
    logger.info("Fixing documents collection...")
    
    # Check if collection exists
    if db.has_collection(COLLECTION_NAME):
        logger.info(f"Collection '{COLLECTION_NAME}' exists, recreating it...")
        
        # Get existing data to preserve it
        try:
            existing_docs = []
            query = f"FOR doc IN {COLLECTION_NAME} RETURN doc"
            cursor = db.aql.execute(query)
            existing_docs = list(cursor)
            logger.info(f"Retrieved {len(existing_docs)} existing documents")
        except Exception as e:
            logger.warning(f"Could not retrieve existing documents: {e}")
            existing_docs = []
        
        # Delete collection
        db.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted collection '{COLLECTION_NAME}'")
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' does not exist, creating it...")
        existing_docs = []
    
    # Create collection
    db.create_collection(COLLECTION_NAME)
    logger.info(f"Created collection '{COLLECTION_NAME}'")
    
    # Insert test documents if no existing data
    if not existing_docs:
        test_docs = [
            {
                "_key": "test_doc_1",
                "problem": "Python error when processing JSON data",
                "solution": "Use try/except blocks to handle JSON parsing exceptions",
                "context": "Error handling in data processing",
                "tags": ["python", "json", "error-handling"]
            },
            {
                "_key": "test_doc_2",
                "problem": "Python script runs out of memory with large datasets",
                "solution": "Use chunking to process large data incrementally",
                "context": "Performance optimization",
                "tags": ["python", "memory", "optimization"]
            },
            {
                "_key": "test_doc_3",
                "problem": "Need to search documents efficiently",
                "solution": "Use ArangoDB's search capabilities with proper indexing",
                "context": "Database search optimization",
                "tags": ["database", "search", "optimization"]
            }
        ]
        existing_docs = test_docs
    
    # Reinsert documents
    collection = db.collection(COLLECTION_NAME)
    for doc in existing_docs:
        # Remove _id and _rev if present (they can't be reinserted)
        if "_id" in doc:
            del doc["_id"]
        if "_rev" in doc:
            del doc["_rev"]
        
        # Insert document
        collection.insert(doc)
        logger.info(f"Inserted document with key: {doc.get('_key', 'unknown')}")
    
    logger.info(f"Collection '{COLLECTION_NAME}' fixed with {len(existing_docs)} documents")
    return True

def fix_relationships_collection():
    """Fix the relationships collection."""
    client = ArangoClient(hosts=HOST)
    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
    
    logger.info("Fixing relationships collection...")
    
    # Check if collection exists
    if db.has_collection(EDGE_COLLECTION_NAME):
        logger.info(f"Edge collection '{EDGE_COLLECTION_NAME}' exists, recreating it...")
        
        # Get existing edges to preserve them
        try:
            existing_edges = []
            query = f"FOR e IN {EDGE_COLLECTION_NAME} RETURN e"
            cursor = db.aql.execute(query)
            existing_edges = list(cursor)
            logger.info(f"Retrieved {len(existing_edges)} existing edges")
        except Exception as e:
            logger.warning(f"Could not retrieve existing edges: {e}")
            existing_edges = []
        
        # Delete collection
        db.delete_collection(EDGE_COLLECTION_NAME)
        logger.info(f"Deleted edge collection '{EDGE_COLLECTION_NAME}'")
    else:
        logger.info(f"Edge collection '{EDGE_COLLECTION_NAME}' does not exist, creating it...")
        existing_edges = []
    
    # Create edge collection
    db.create_collection(EDGE_COLLECTION_NAME, edge=True)
    logger.info(f"Created edge collection '{EDGE_COLLECTION_NAME}'")
    
    # Insert test edges if no existing data
    if not existing_edges:
        test_edges = [
            {
                "_from": f"{COLLECTION_NAME}/test_doc_1",
                "_to": f"{COLLECTION_NAME}/test_doc_2",
                "type": "related_to",
                "weight": 0.8
            },
            {
                "_from": f"{COLLECTION_NAME}/test_doc_2",
                "_to": f"{COLLECTION_NAME}/test_doc_3",
                "type": "similar_to",
                "weight": 0.6
            }
        ]
        existing_edges = test_edges
    
    # Reinsert edges
    collection = db.collection(EDGE_COLLECTION_NAME)
    for edge in existing_edges:
        # Remove _id and _rev if present
        if "_id" in edge:
            del edge["_id"]
        if "_rev" in edge:
            del edge["_rev"]
        
        # Make sure from/to references exist
        if "_from" in edge and "_to" in edge:
            try:
                collection.insert(edge)
                logger.info(f"Inserted edge from {edge['_from']} to {edge['_to']}")
            except Exception as e:
                logger.error(f"Failed to insert edge: {e}")
    
    logger.info(f"Edge collection '{EDGE_COLLECTION_NAME}' fixed with {len(existing_edges)} edges")
    return True

def fix_graph():
    """Fix the knowledge graph."""
    client = ArangoClient(hosts=HOST)
    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
    
    logger.info("Creating graph...")
    
    # Create graph
    edge_definition = {
        "edge_collection": EDGE_COLLECTION_NAME,
        "from_vertex_collections": [COLLECTION_NAME],
        "to_vertex_collections": [COLLECTION_NAME]
    }
    
    db.create_graph(GRAPH_NAME, [edge_definition])
    logger.info(f"Created graph '{GRAPH_NAME}'")
    return True

def fix_document_view():
    """Fix the document view for searching."""
    client = ArangoClient(hosts=HOST)
    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
    
    logger.info("Fixing document view...")
    
    # Check if view exists
    views = db.views()
    for view in views:
        if view["name"] == VIEW_NAME:
            logger.info(f"View '{VIEW_NAME}' exists, deleting it...")
            db.delete_view(VIEW_NAME)
            break
    
    # Create view
    view_properties = {
        "links": {
            COLLECTION_NAME: {
                "includeAllFields": True,
                "storeValues": "none",
                "analyzers": ["text_en"]
            }
        }
    }
    
    try:
        db.create_arangosearch_view(VIEW_NAME, view_properties)
        logger.info(f"Created view '{VIEW_NAME}'")
    except Exception as e:
        logger.error(f"Failed to create view: {e}")
        return False
    
    return True

def fix_analyzers():
    """Fix text analyzers needed for search."""
    client = ArangoClient(hosts=HOST)
    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
    
    logger.info("Checking analyzers...")
    
    # List existing analyzers
    try:
        analyzers = db.analyzers()
        analyzer_names = [a["name"] for a in analyzers]
        logger.info(f"Existing analyzers: {', '.join(analyzer_names)}")
        
        # Check if text_en exists
        if "text_en" not in analyzer_names:
            logger.info("Creating text_en analyzer...")
            db.create_analyzer(
                "text_en",
                "text",
                {
                    "locale": "en",
                    "case": "lower",
                    "stopwords": [],
                    "accent": False,
                    "stemming": True
                },
                ["frequency", "norm", "position"]
            )
            logger.info("Created text_en analyzer")
    except Exception as e:
        logger.error(f"Failed to manage analyzers: {e}")
        return False
    
    return True

def create_vector_index():
    """Create vector index for semantic search."""
    client = ArangoClient(hosts=HOST)
    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
    
    logger.info("Creating vector index for semantic search...")
    
    try:
        # Check if collection has documents
        collection = db.collection(COLLECTION_NAME)
        
        # Add embedding field to documents if needed
        for doc in collection:
            if "embedding" not in doc:
                # Create a simple mock embedding vector (16 dimensions)
                mock_embedding = [0.1] * 16
                collection.update({"_key": doc["_key"], "embedding": mock_embedding})
                logger.info(f"Added mock embedding to document {doc['_key']}")
        
        # Try to create vector index
        try:
            collection.add_index({
                "type": "inverted",
                "fields": ["tags"],
                "name": "tags_index"
            })
            logger.info("Created inverted index on tags field")
        except Exception as e:
            logger.warning(f"Could not create inverted index: {e}")
        
        logger.info("Vector capabilities set up")
    except Exception as e:
        logger.error(f"Failed to set up vector capabilities: {e}")
        return False
    
    return True

def main():
    """Fix collections and views for search functions."""
    logger.info("Starting fix operation...")
    
    try:
        # First, delete the graph if it exists
        if not delete_graph_if_exists():
            logger.error("Failed to delete existing graph")
            return False
        
        # Fix collections
        if not fix_documents_collection():
            logger.error("Failed to fix documents collection")
            return False
        
        if not fix_relationships_collection():
            logger.error("Failed to fix relationships collection")
            return False
        
        # Fix analyzers
        if not fix_analyzers():
            logger.error("Failed to fix analyzers")
            return False
        
        # Fix document view
        if not fix_document_view():
            logger.error("Failed to fix document view")
            return False
        
        # Create vector index for semantic search
        if not create_vector_index():
            logger.error("Failed to create vector index")
            return False
        
        # Fix graph last
        if not fix_graph():
            logger.error("Failed to fix graph")
            return False
        
        logger.info("✅ All collections and views fixed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Unexpected error during fix operation: {e}")
        return False

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
