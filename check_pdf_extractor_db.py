#!/usr/bin/env python3
"""
Check the pdf_extractor database and its collections.
This script will connect to the ArangoDB instance and check
if the pdf_extractor database has the necessary collections.
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

# Required collections and views
REQUIRED_COLLECTIONS = ["documents", "relationships", "messages"]
REQUIRED_VIEWS = ["document_view"]

def check_database():
    """Check if the pdf_extractor database exists and has the required collections."""
    logger.info(f"Connecting to ArangoDB at {HOST}")
    
    try:
        # Initialize client
        client = ArangoClient(hosts=HOST)
        
        # Connect to _system database
        sys_db = client.db("_system", username=USERNAME, password=PASSWORD)
        
        # Check if pdf_extractor database exists
        databases = sys_db.databases()
        if DB_NAME not in databases:
            logger.error(f"❌ Database '{DB_NAME}' does not exist")
            logger.info(f"Available databases: {', '.join(databases)}")
            return False
        
        logger.info(f"✅ Database '{DB_NAME}' exists")
        
        # Connect to pdf_extractor database
        db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
        
        # Check collections
        collections = [c['name'] for c in db.collections() if not c['name'].startswith('_')]
        logger.info(f"Collections in {DB_NAME} database: {', '.join(collections) if collections else 'No collections'}")
        
        # Check required collections
        missing_collections = [c for c in REQUIRED_COLLECTIONS if c
