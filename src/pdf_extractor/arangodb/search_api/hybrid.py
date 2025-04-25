from typing import List, Dict, Any, Optional, Tuple, Union
from arango.database import StandardDatabase
from arango.exceptions import AQLQueryExecuteError, ArangoServerError

# Import config variables and embedding utils
# --- Configuration and Imports ---
from pdf_extractor.arangodb.config import (
    SEARCH_FIELDS,
    ALL_DATA_FIELDS_PREVIEW,
    TEXT_ANALYZER,
    TAG_ANALYZER,
    VIEW_NAME,
    GRAPH_NAME,
)
from pdf_extractor.arangodb.embedding_utils import get_embedding
from pdf_extractor.arangodb.search_api.bm25 import _fetch_bm25_candidates
from pdf_extractor.arangodb.search_api.semantic import _fetch_semantic_candidates
from pdf_extractor.arangodb.search_api.utils import validate_search_params

def hybrid_search(
    db: StandardDatabase,
    query_text: str,
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
        
        return {
            "results": final_results,
            "count": len(final_results),
            "total_candidates": len(combined_results),
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
            "total_candidates": 0,
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

# For module testing
if __name__ == "__main__":
    import json
    import sys
    
    # Run a simple test if imports succeed
    print("✅ Hybrid search module imported successfully")
    
    # Exit with success
    sys.exit(0)
