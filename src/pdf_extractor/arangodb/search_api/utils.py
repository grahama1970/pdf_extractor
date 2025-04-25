"""
Utils for search API validation and common operations.
"""

from typing import List, Dict, Any, Optional

def validate_search_params(
    search_text: Optional[str] = None,
    bm25_threshold: Optional[float] = None,
    top_n: Optional[int] = None,
    offset: Optional[int] = None,
    tags: Optional[List[str]] = None,
    semantic_threshold: Optional[float] = None,
    initial_k: Optional[int] = None,
) -> None:
    """
    Validates input parameters for search functions.
    
    Args:
        search_text: The search text for keyword search
        bm25_threshold: BM25 score threshold (0.0-1.0)
        top_n: Number of results to return
        offset: Pagination offset
        tags: List of tags to filter by
        semantic_threshold: Similarity threshold (0.0-1.0)
        initial_k: Initial number of candidates for hybrid search
        
    Raises:
        ValueError: If any parameters are invalid
    """
    # Validate text search when provided
    if search_text is not None and not isinstance(search_text, str):
        raise ValueError("search_text must be a string")
    
    # Validate BM25 threshold when provided
    if bm25_threshold is not None:
        if not isinstance(bm25_threshold, (int, float)):
            raise ValueError("bm25_threshold must be a number")
        if bm25_threshold < 0 or bm25_threshold > 1:
            raise ValueError("bm25_threshold must be between 0 and 1")
    
    # Validate semantic threshold when provided
    if semantic_threshold is not None:
        if not isinstance(semantic_threshold, (int, float)):
            raise ValueError("semantic_threshold must be a number")
        if semantic_threshold < 0 or semantic_threshold > 1:
            raise ValueError("semantic_threshold must be between 0 and 1")
    
    # Validate top_n when provided
    if top_n is not None:
        if not isinstance(top_n, int):
            raise ValueError("top_n must be an integer")
        if top_n < 1:
            raise ValueError("top_n must be greater than 0")
    
    # Validate offset when provided
    if offset is not None:
        if not isinstance(offset, int):
            raise ValueError("offset must be an integer")
        if offset < 0:
            raise ValueError("offset must be greater than or equal to 0")
    
    # Validate tags when provided
    if tags is not None:
        if not isinstance(tags, list):
            raise ValueError("tags must be a list")
        if not all(isinstance(tag, str) for tag in tags):
            raise ValueError("each tag in tags must be a string")
    
    # Validate initial_k when provided
    if initial_k is not None:
        if not isinstance(initial_k, int):
            raise ValueError("initial_k must be an integer")
        if initial_k < 1:
            raise ValueError("initial_k must be greater than 0")
