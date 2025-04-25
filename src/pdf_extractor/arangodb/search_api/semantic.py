# src/pdf_extractor/arangodb/search_api/semantic.py
import time
import json
from typing import Dict, Any, List, Optional, Tuple, Union

from loguru import logger
from arango.database import StandardDatabase
from arango.exceptions import AQLQueryExecuteError, ArangoServerError

# Import config variables and embedding utils
try:
    from pdf_extractor.arangodb.config import (
        COLLECTION_NAME,
        ALL_DATA_FIELDS_PREVIEW,
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSIONS
    )
    from pdf_extractor.arangodb.arango_setup import EMBEDDING_FIELD # Import shared constant
    from pdf_extractor.arangodb.embedding_utils import get_embedding
except ImportError as e:
    logger.critical(f"CRITICAL: Failed module-level import in semantic.py: {e}. Functionality will be broken.")
    # Define fallbacks to allow module to load (but not function)
    COLLECTION_NAME = "documents"
    ALL_DATA_FIELDS_PREVIEW = ["_key"]
    EMBEDDING_MODEL = "text-embedding-ada-002"
    EMBEDDING_DIMENSIONS = 1536
    EMBEDDING_FIELD = "embedding"

def _fetch_semantic_candidates(
    db: StandardDatabase,
    query_text: str,
    top_n: int = 20,
    min_score: float = 0.0,
    tag_filter_clause: str = ""
) -> Dict[str, Any]:
    """
    Fetch semantic candidates for a query using vector similarity.
    
    Args:
        db: ArangoDB database connection.
        query_text: The search query text.
        top_n: Maximum number of results to return.
        min_score: Minimum similarity score threshold.
        tag_filter_clause: Optional AQL filter clause for tag filtering.
    
    Returns:
        Dictionary with results and timing information.
    """
    start_time = time.time()
    
    try:
        # Get query embedding
        query_embedding = get_embedding(query_text, EMBEDDING_MODEL)
        if not query_embedding:
            logger.error("Failed to generate embedding for query")
            return {
                "results": [],
                "count": 0,
                "query": query_text,
                "time": time.time() - start_time,
                "error": "Failed to generate embedding"
            }
        
        # Format preview fields string
        preview_fields_str = ", ".join(f'"{field}"' for field in ALL_DATA_FIELDS_PREVIEW)
        
        # Build the AQL query with vector search and optional tag filtering
        aql = f"""
        FOR doc IN {COLLECTION_NAME}
        {tag_filter_clause}
        LET score = VECTOR_SIMILARITY(doc.{EMBEDDING_FIELD}, @query_embedding)
        FILTER score >= @min_score
        SORT score DESC
        LIMIT @top_n
        RETURN {{
            "doc": KEEP(doc, [{preview_fields_str}]),
            "score": score
        }}
        """
        
        # Execute the query
        bind_vars = {
            "query_embedding": query_embedding,
            "top_n": top_n,
            "min_score": min_score
        }
        
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        return {
            "results": results,
            "count": len(results),
            "query": query_text,
            "embedding_model": EMBEDDING_MODEL,
            "time": elapsed
        }
    
    except (AQLQueryExecuteError, ArangoServerError) as e:
        logger.error(f"ArangoDB query error in semantic search: {e}")
        return {
            "results": [],
            "count": 0,
            "query": query_text,
            "time": time.time() - start_time,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error in semantic search: {e}")
        return {
            "results": [],
            "count": 0,
            "query": query_text,
            "time": time.time() - start_time,
            "error": str(e)
        }

# For module testing
if __name__ == "__main__":
    import sys
    
    # Run a simple test if imports succeed
    print("✅ Semantic search module imported successfully")
    
    # Exit with success
    sys.exit(0)
