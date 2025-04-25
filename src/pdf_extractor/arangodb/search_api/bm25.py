# src/pdf_extractor/arangodb/search_api/bm25.py
import time
from typing import Dict, Any, List, Optional

from loguru import logger
from arango.database import StandardDatabase
from arango.exceptions import AQLQueryExecuteError, ArangoServerError

# Import config variables
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME,
    SEARCH_FIELDS,
    ALL_DATA_FIELDS_PREVIEW,
    TEXT_ANALYZER,
    VIEW_NAME,
)

def _fetch_bm25_candidates(
    db: StandardDatabase,
    query_text: str,
    top_n: int = 20,
    min_score: float = 0.0,
    tag_filter_clause: str = ""
) -> Dict[str, Any]:
    """
    Fetch BM25 candidates for a query.
    
    Args:
        db: ArangoDB database connection.
        query_text: The search query text.
        top_n: Maximum number of results to return.
        min_score: Minimum BM25 score threshold.
        tag_filter_clause: Optional AQL filter clause for tag filtering.
    
    Returns:
        Dictionary with results and timing information.
    """
    start_time = time.time()
    
    try:
        # Construct AQL query with BM25 scoring
        fields_str = ", ".join(f'"{field}"' for field in SEARCH_FIELDS)
        preview_fields_str = ", ".join(f'"{field}"' for field in ALL_DATA_FIELDS_PREVIEW)
        
        # Build the AQL query with optional tag filtering
        aql = f"""
        FOR doc IN {VIEW_NAME}
        SEARCH ANALYZER(TOKENS(@query, "text_en") ALL IN doc, "text_en")
        {tag_filter_clause}
        LET score = BM25(doc)
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
            "query": query_text,
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
            "time": elapsed
        }
    
    except (AQLQueryExecuteError, ArangoServerError) as e:
        logger.error(f"ArangoDB query error: {e}")
        return {
            "results": [],
            "count": 0,
            "query": query_text,
            "time": time.time() - start_time,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error in BM25 search: {e}")
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
    print("✅ BM25 search module imported successfully")
    
    # Exit with success
    sys.exit(0)
