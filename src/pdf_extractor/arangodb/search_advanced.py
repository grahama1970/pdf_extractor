# search_advanced.py
from src.pdf_extractor.arangodb.search_api.bm25 import search_bm25 
from src.pdf_extractor.arangodb.search_api.semantic import search_semantic
from src.pdf_extractor.arangodb.search_api.hybrid import hybrid_search

# Main validation
if __name__ == "__main__":
    import sys
    import os
    from loguru import logger
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
    
    logger.info("Running ArangoDB search validation")
    
    # Connect to ArangoDB
    client = connect_arango()
    if not client:
        logger.error("Validation failed: Could not connect to ArangoDB")
        sys.exit(1)
    
    # Get database
    db = ensure_database(client)
    if not db:
        logger.error("Validation failed: Could not ensure database")
        sys.exit(1)
    
    # Run test search
    try:
        results = search_bm25(db, "test query", 0.01, 3)
        logger.info(f"Found {len(results.get("results", []))} results in test search")
        logger.success("✅ Search API validation complete!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Search API validation failed: {str(e)}")
        sys.exit(1)

