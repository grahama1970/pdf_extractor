# src/pdf_extractor/arangodb/search_api/semantic.py
import time
import json
import sys
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
    from pdf_extractor.arangodb.arango_setup import EMBEDDING_FIELD, connect_arango, ensure_database
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

def semantic_search(
    db: StandardDatabase,
    query: Union[str, List[float]],
    collections: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    min_score: float = 0.7,
    top_n: int = 10,
    tag_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Search for documents using semantic vector similarity.
    
    Args:
        db: ArangoDB database
        query: Search query text or embedding vector
        collections: Optional list of collections to search
        filter_expr: Optional AQL filter expression
        min_score: Minimum similarity score threshold (0-1)
        top_n: Maximum number of results to return
        tag_list: Optional list of tags to filter by
        
    Returns:
        Dict with search results
    """
    try:
        start_time = time.time()
        
        # Use default collection if not specified
        if not collections:
            collections = [COLLECTION_NAME]
        
        # Get query embedding if string is provided
        query_embedding = query
        if isinstance(query, str):
            query_embedding = get_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return {
                    "results": [],
                    "total": 0,
                    "query": query,
                    "error": "Failed to generate embedding"
                }
        
        # Build the filter clause
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
        {filter_clause}
        LET similarity = VECTOR_SIMILARITY(doc.{EMBEDDING_FIELD}, @query_embedding)
        FILTER similarity >= @min_score
        SORT similarity DESC
        LIMIT @top_n
        RETURN {{
            "doc": doc,
            "similarity_score": similarity,
            "collection": "{collections[0]}"
        }}
        """
        
        # Execute the query
        bind_vars = {
            "query_embedding": query_embedding,
            "min_score": min_score,
            "top_n": top_n
        }
        
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)
        
        # Get the total count
        count_aql = f"""
        RETURN LENGTH(
            FOR doc IN {collections[0]}
            {filter_clause}
            LET similarity = VECTOR_SIMILARITY(doc.{EMBEDDING_FIELD}, @query_embedding)
            FILTER similarity >= @min_score
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
            "query": query if isinstance(query, str) else "vector query",
            "time": elapsed
        }
    
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return {
            "results": [],
            "total": 0,
            "query": query if isinstance(query, str) else "vector query",
            "error": str(e)
        }

def validate_semantic_search(search_results: Dict[str, Any], fixture_path: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate semantic search results against known good fixture data.
    
    Args:
        search_results: The results returned from semantic_search
        fixture_path: Path to the fixture file containing expected results
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    # Load fixture data
    try:
        with open(fixture_path, "r") as f:
            expected_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load fixture data: {e}")
        return False, {"fixture_loading_error": {"expected": "Valid JSON file", "actual": str(e)}}
    
    # Track all validation failures
    validation_failures = {}
    
    # Structural validation
    if "results" not in search_results:
        validation_failures["missing_results"] = {
            "expected": "Results field present",
            "actual": "Results field missing"
        }
        return False, validation_failures
        
    # Validate result count
    if len(search_results["results"]) != len(expected_data["results"]):
        validation_failures["result_count"] = {
            "expected": len(expected_data["results"]),
            "actual": len(search_results["results"])
        }
    
    # Validate total count
    if search_results.get("total") != expected_data.get("total"):
        validation_failures["total_count"] = {
            "expected": expected_data.get("total"),
            "actual": search_results.get("total")
        }
    
    # Content validation - compare actual results with expected
    for i, (expected_result, actual_result) in enumerate(
        zip(expected_data["results"], search_results["results"])
    ):
        # Check structure of each result
        if "doc" not in actual_result:
            validation_failures[f"result_{i}_missing_doc"] = {
                "expected": "doc field present",
                "actual": "doc field missing"
            }
            continue
            
        # Check document fields
        for key in expected_result["doc"]:
            if key not in actual_result["doc"]:
                validation_failures[f"result_{i}_doc_missing_{key}"] = {
                    "expected": f"{key} field present",
                    "actual": f"{key} field missing"
                }
            elif actual_result["doc"][key] != expected_result["doc"][key]:
                validation_failures[f"result_{i}_doc_{key}"] = {
                    "expected": expected_result["doc"][key],
                    "actual": actual_result["doc"][key]
                }
        
        # Check similarity score field - might be called different names
        expected_score = expected_result.get("similarity_score", expected_result.get("score"))
        actual_score = actual_result.get("similarity_score", actual_result.get("score"))
        
        # For floating point scores, use approximate comparison
        if expected_score is not None and actual_score is not None:
            # Allow small differences in scores due to floating point precision
            if abs(expected_score - actual_score) > 0.01:
                validation_failures[f"result_{i}_score"] = {
                    "expected": expected_score,
                    "actual": actual_score
                }
        elif expected_score != actual_score:
            validation_failures[f"result_{i}_score"] = {
                "expected": expected_score,
                "actual": actual_score
            }
    
    return len(validation_failures) == 0, validation_failures

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Path to test fixture
    fixture_path = "src/test_fixtures/semantic_search_expected_20250422_181101.json"
    
    try:
        # Set up database connection
        client = connect_arango()
        db = ensure_database(client)
        
        # Run a test search query
        test_query = "python error handling"  # Known query that should match fixture results
        search_results = semantic_search(
            db=db,
            query=test_query,
            top_n=10,
            min_score=0.6
        )
        
        # Validate the results
        validation_passed, validation_failures = validate_semantic_search(search_results, fixture_path)
        
        # Report validation status
        if validation_passed:
            print("✅ VALIDATION COMPLETE - All semantic search results match expected values")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED - Semantic search results don't match expected values") 
            print(f"FAILURE DETAILS:")
            for field, details in validation_failures.items():
                print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
            print(f"Total errors: {len(validation_failures)} fields mismatched")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
