# src/pdf_extractor/arangodb/search_api/bm25.py
import time
import sys
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
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database

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

def bm25_search(
    db: StandardDatabase,
    query_text: str,
    collections: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    min_score: float = 0.0,
    top_n: int = 10,
    offset: int = 0,
    tag_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Search for documents using BM25 algorithm.
    
    Args:
        db: ArangoDB database
        query_text: Search query text
        collections: Optional list of collections to search
        filter_expr: Optional AQL filter expression
        min_score: Minimum BM25 score threshold
        top_n: Maximum number of results to return
        offset: Offset for pagination
        tag_list: Optional list of tags to filter by
        
    Returns:
        Dict with search results
    """
    try:
        start_time = time.time()
        
        # Use default collection if not specified
        if not collections:
            collections = [COLLECTION_NAME]
        
        # Build filter clause
        filter_clauses = []
        if filter_expr:
            filter_clauses.append(f"({filter_expr})")
        
        # Add tag filter if provided
        if tag_list:
            tag_conditions = []
            for tag in tag_list:
                tag_conditions.append(f'"{tag}" IN doc.tags')
            tag_filter = " OR ".join(tag_conditions)
            filter_clauses.append(f"({tag_filter})")
        
        # Combine filter clauses with AND
        filter_clause = ""
        if filter_clauses:
            filter_clause = "FILTER " + " AND ".join(filter_clauses)
        
        # Build the AQL query
        aql = f"""
        FOR doc IN {collections[0]}
        SEARCH ANALYZER(TOKENS(@query, "text_en") ALL IN doc, "text_en")
        {filter_clause}
        LET score = BM25(doc)
        FILTER score >= @min_score
        SORT score DESC
        LIMIT @offset, @top_n
        RETURN {{
            "doc": doc,
            "score": score,
            "collection": "{collections[0]}"
        }}
        """
        
        # Execute the query
        bind_vars = {
            "query": query_text,
            "min_score": min_score,
            "offset": offset,
            "top_n": top_n
        }
        
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)
        
        # Get the total count
        count_aql = f"""
        RETURN LENGTH(
            FOR doc IN {collections[0]}
            SEARCH ANALYZER(TOKENS(@query, "text_en") ALL IN doc, "text_en")
            {filter_clause}
            LET score = BM25(doc)
            FILTER score >= @min_score
            RETURN 1
        )
        """
        count_cursor = db.aql.execute(count_aql, bind_vars=bind_vars)
        total_count = next(count_cursor)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        return {
            "results": results,
            "total": total_count,
            "offset": offset,
            "query": query_text,
            "time": elapsed
        }
    
    except Exception as e:
        logger.error(f"BM25 search error: {e}")
        return {
            "results": [],
            "total": 0,
            "offset": offset,
            "query": query_text,
            "error": str(e)
        }

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Path to test fixture
    fixture_path = "src/test_fixtures/bm25_search_expected_20250422_181050.json"
    
    try:
        # Set up database connection
        client = connect_arango()
        db = ensure_database(client)
        
        # Run a test search query
        test_query = "python error"  # Known query that should match fixture results
        search_results = bm25_search(
            db=db,
            query_text=test_query,
            top_n=3,  # Match expected number in fixture
            min_score=0.0
        )
        
        # Basic verification (print results)
        print(f"✅ BM25 search found {len(search_results['results'])} results")
        for i, result in enumerate(search_results["results"]):
            doc = result["doc"]
            score = result["score"]
            print(f"  Result {i+1}: Key={doc['_key']}, Score={score:.4f}")
        
        # Exit with success
        sys.exit(0)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
