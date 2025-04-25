# src/pdf_extractor/arangodb/search_api/hybrid.py
import json
import time
import sys
from typing import List, Dict, Any, Optional, Tuple, Union
from loguru import logger
from arango.database import StandardDatabase
from arango.exceptions import AQLQueryExecuteError, ArangoServerError

# Import config variables and embedding utils
# --- Configuration and Imports ---
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME,
    SEARCH_FIELDS,
    ALL_DATA_FIELDS_PREVIEW,
    TEXT_ANALYZER,
    VIEW_NAME,
)
from pdf_extractor.arangodb.embedding_utils import get_embedding
from pdf_extractor.arangodb.search_api.bm25 import _fetch_bm25_candidates
from pdf_extractor.arangodb.search_api.semantic import _fetch_semantic_candidates
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database

def validate_search_params(query_text, top_n, initial_k):
    """Validate search parameters and set reasonable defaults."""
    if not query_text:
        query_text = ""  # Allow empty query
    
    if top_n <= 0:
        top_n = 5  # Default to 5 results
    
    if initial_k <= 0:
        initial_k = 20  # Default to 20 candidates
    
    return query_text, top_n, initial_k

def hybrid_search(
    db: StandardDatabase,
    query_text: str,
    collections: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    top_n: int = 5,
    initial_k: int = 20,
    bm25_threshold: float = 0.01,
    similarity_threshold: float = 0.70,
    tag_filters: Optional[List[str]] = None,
    rrf_k: int = 60,
) -> Dict[str, Any]:
    """
    Performs hybrid search by combining BM25 and Semantic search results
    using Reciprocal Rank Fusion (RRF) for re-ranking.

    Args:
        db: ArangoDB database connection.
        query_text: The user's search query.
        collections: Optional list of collections to search in.
        filter_expr: Optional AQL filter expression.
        top_n: The final number of ranked results to return.
        initial_k: Number of results to initially fetch from BM25 and Semantic searches.
        bm25_threshold: Minimum BM25 score for initial candidates.
        similarity_threshold: Minimum similarity score for initial candidates.
        tag_filters: Optional list of tags to filter results.
        rrf_k: Constant used in the RRF calculation (default 60).

    Returns:
        A dictionary containing the ranked 'results', 'total' unique documents found,
        and the 'query' for reference.
    """
    logger.info(f"Hybrid search for: '{query_text}'")
    if tag_filters:
        logger.info(f"Filtering by tags: {tag_filters}")
    
    # Validate search parameters
    query_text, top_n, initial_k = validate_search_params(
        query_text, top_n, initial_k
    )
    
    # Create a structured query for tag filtering
    tag_filter_clause = ""
    if tag_filters and len(tag_filters) > 0:
        tag_conditions = []
        for tag in tag_filters:
            tag_conditions.append(f"POSITION(doc.tags, {json.dumps(tag)}) != false")
        
        if tag_conditions:
            tag_filter_clause = f" FILTER {' AND '.join(tag_conditions)}"
    
    # Add filter expression to tag_filter_clause if provided
    if filter_expr:
        if tag_filter_clause:
            tag_filter_clause += f" AND ({filter_expr})"
        else:
            tag_filter_clause = f" FILTER {filter_expr}"
    
    try:
        # Get candidates from BM25 search
        bm25_results = _fetch_bm25_candidates(
            db, 
            query_text, 
            initial_k, 
            bm25_threshold,
            tag_filter_clause
        )
        bm25_time = bm25_results.get("time", 0)
        bm25_candidates = bm25_results.get("results", [])
        logger.debug(f"BM25 found {len(bm25_candidates)} candidates in {bm25_time:.4f}s")
        
        # Get candidates from semantic search
        semantic_results = _fetch_semantic_candidates(
            db, 
            query_text, 
            initial_k, 
            similarity_threshold,
            tag_filter_clause
        )
        semantic_time = semantic_results.get("time", 0)
        semantic_candidates = semantic_results.get("results", [])
        logger.debug(f"Semantic found {len(semantic_candidates)} candidates in {semantic_time:.4f}s")
        
        # Combine candidates using Reciprocal Rank Fusion (RRF)
        combined_results = reciprocal_rank_fusion(
            bm25_candidates, semantic_candidates, rrf_k
        )
        
        # Limit to top_n results
        final_results = combined_results[:top_n]
        
        # Enhance results with collection information if provided
        if collections and len(collections) > 0:
            collection_name = collections[0]
            for result in final_results:
                result["collection"] = collection_name
        
        return {
            "results": final_results,
            "count": len(final_results),
            "total": len(combined_results),
            "query": query_text,
            "bm25_time": bm25_time,
            "semantic_time": semantic_time,
            "tag_filters": tag_filters
        }
    
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        return {
            "results": [],
            "count": 0,
            "total": 0,
            "query": query_text,
            "error": str(e)
        }

def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combines multiple result lists using Reciprocal Rank Fusion.
    
    Args:
        bm25_results: Results from BM25 search.
        semantic_results: Results from semantic search.
        k: Constant for the RRF formula (default: 60).
    
    Returns:
        A combined list of results, sorted by RRF score.
    """
    # Create a dictionary to track document keys and their rankings
    doc_scores = {}
    
    # Process BM25 results
    for rank, result in enumerate(bm25_results, 1):
        doc_key = result.get("doc", {}).get("_key", "")
        if not doc_key:
            continue
        
        # Initialize if not seen before
        if doc_key not in doc_scores:
            doc_scores[doc_key] = {
                "doc": result.get("doc", {}),
                "bm25_rank": rank,
                "bm25_score": result.get("score", 0),
                "semantic_rank": len(semantic_results) + 1,  # Default to worst possible rank
                "semantic_score": 0,
                "rrf_score": 0
            }
        else:
            # Update BM25 rank info
            doc_scores[doc_key]["bm25_rank"] = rank
            doc_scores[doc_key]["bm25_score"] = result.get("score", 0)
    
    # Process semantic results
    for rank, result in enumerate(semantic_results, 1):
        doc_key = result.get("doc", {}).get("_key", "")
        if not doc_key:
            continue
        
        # Initialize if not seen before
        if doc_key not in doc_scores:
            doc_scores[doc_key] = {
                "doc": result.get("doc", {}),
                "bm25_rank": len(bm25_results) + 1,  # Default to worst possible rank
                "bm25_score": 0,
                "semantic_rank": rank,
                "semantic_score": result.get("score", 0),
                "rrf_score": 0
            }
        else:
            # Update semantic rank info
            doc_scores[doc_key]["semantic_rank"] = rank
            doc_scores[doc_key]["semantic_score"] = result.get("score", 0)
    
    # Calculate RRF scores
    for doc_key, scores in doc_scores.items():
        bm25_rrf = 1 / (k + scores["bm25_rank"])
        semantic_rrf = 1 / (k + scores["semantic_rank"])
        scores["rrf_score"] = bm25_rrf + semantic_rrf
    
    # Convert to list and sort by RRF score (descending)
    result_list = [v for k, v in doc_scores.items()]
    result_list.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    return result_list

def validate_hybrid_search(
    search_results: Dict[str, Any], 
    fixture_path: str
) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate hybrid search results against known good fixture data.
    
    Args:
        search_results: The results returned from hybrid_search
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
    if "count" in expected_data and search_results.get("count") != expected_data.get("count"):
        validation_failures["result_count"] = {
            "expected": expected_data.get("count"),
            "actual": search_results.get("count")
        }
    
    # Validate total count
    if "total" in expected_data and search_results.get("total") != expected_data.get("total"):
        validation_failures["total_count"] = {
            "expected": expected_data.get("total"),
            "actual": search_results.get("total")
        }
    
    # Validate the actual result data
    expected_results = expected_data.get("results", [])
    actual_results = search_results.get("results", [])
    
    # First check if we have the same number of results
    if len(expected_results) != len(actual_results):
        validation_failures["results_length"] = {
            "expected": len(expected_results),
            "actual": len(actual_results)
        }
    
    # Now validate individual results
    min_length = min(len(expected_results), len(actual_results))
    for i in range(min_length):
        expected_result = expected_results[i]
        actual_result = actual_results[i]
        
        # Check document keys match
        expected_key = expected_result.get("doc", {}).get("_key", "")
        actual_key = actual_result.get("doc", {}).get("_key", "")
        
        if expected_key != actual_key:
            validation_failures[f"result_{i}_doc_key"] = {
                "expected": expected_key,
                "actual": actual_key
            }
        
        # Check RRF scores approximately match (floating point comparison)
        if "rrf_score" in expected_result and "rrf_score" in actual_result:
            expected_score = expected_result["rrf_score"]
            actual_score = actual_result["rrf_score"]
            
            # Allow small differences in scores due to floating point precision
            if abs(expected_score - actual_score) > 0.01:
                validation_failures[f"result_{i}_rrf_score"] = {
                    "expected": expected_score,
                    "actual": actual_score
                }
        
        # Check document content
        for key in expected_result.get("doc", {}):
            if key not in actual_result.get("doc", {}):
                validation_failures[f"result_{i}_doc_missing_{key}"] = {
                    "expected": f"{key} field present",
                    "actual": f"{key} field missing"
                }
            elif actual_result["doc"][key] != expected_result["doc"][key]:
                validation_failures[f"result_{i}_doc_{key}"] = {
                    "expected": expected_result["doc"][key],
                    "actual": actual_result["doc"][key]
                }
    
    return len(validation_failures) == 0, validation_failures

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Path to test fixture
    fixture_path = "src/test_fixtures/hybrid_search_expected_20250422_181117.json"
    
    try:
        # Set up database connection
        client = connect_arango()
        db = ensure_database(client)
        
        # Run a test hybrid search
        test_query = "python error"  # Known query that should match fixture results
        search_results = hybrid_search(
            db=db,
            query_text=test_query,
            top_n=3
        )
        
        # Validate the results
        validation_passed, validation_failures = validate_hybrid_search(search_results, fixture_path)
        
        # Report validation status
        if validation_passed:
            print("✅ VALIDATION COMPLETE - All hybrid search results match expected values")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED - Hybrid search results don't match expected values") 
            print(f"FAILURE DETAILS:")
            for field, details in validation_failures.items():
                print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
            print(f"Total errors: {len(validation_failures)} fields mismatched")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
