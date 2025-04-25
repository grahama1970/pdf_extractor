#!/usr/bin/env python3
"""
Simple script to verify connection to ArangoDB.
This will attempt to connect to the ArangoDB instance and
report success or failure.
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

def test_connection():
    """Test connection to ArangoDB."""
    logger.info(f"Attempting to connect to ArangoDB at {HOST}")
    
    try:
        # Initialize client
        client = ArangoClient(hosts=HOST)
        
        # Connect to _system database
        sys_db = client.db("_system", username=USERNAME, password=PASSWORD)
        
        # Try a simple operation to verify connection
        version = sys_db.version()
        logger.info(f"✅ Connected successfully to ArangoDB version: {version}")
        
        # List all databases
        databases = sys_db.databases()
        logger.info(f"Available databases: {', '.join(databases)}")
        
        # List all collections in _system database
        collections = [c['name'] for c in sys_db.collections() if not c['name'].startswith('_')]
        logger.info(f"Collections in _system database: {', '.join(collections) if collections else 'No collections'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to ArangoDB: {e}")
        return False

if __name__ == "__main__":
    if test_connection():
        sys.exit(0)
    else:
        sys.exit(1)
