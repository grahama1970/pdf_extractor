#!/usr/bin/env python
"""
Script to fix the hybrid.py file to pass the view_name parameter to helper functions.
"""
import sys
import re
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:HH:mm:ss} | {level:<7} | {message}"
)

def fix_hybrid_helpers():
    """Update the hybrid.py file to pass the view_name parameter to helper functions"""
    try:
        file_path = "src/pdf_extractor/arangodb/search_api/hybrid.py"
        with open(file_path, "r") as f:
            content = f.read()
        
        # Find and fix the BM25 fetch call
        bm25_problem_line = "bm25_candidates_raw = _fetch_bm25_candidates(\n                db, query_text, bm25_threshold, initial_k, tags, search_uuid\n            )"
        bm25_fixed_line = "bm25_candidates_raw = _fetch_bm25_candidates(\n                db, query_text, bm25_threshold, initial_k, tags, search_uuid, view_name\n            )"
        
        # Find and fix the semantic fetch call
        semantic_problem_line = "semantic_candidates_raw = _fetch_semantic_candidates(\n                db, query_embedding, similarity_threshold, initial_k, tags, search_uuid\n            )"
        semantic_fixed_line = "semantic_candidates_raw = _fetch_semantic_candidates(\n                db, query_embedding, similarity_threshold, initial_k, tags, search_uuid, view_name\n            )"
        
        # Update the _fetch_bm25_candidates function signature
        bm25_fetch_sig_problem = "def _fetch_bm25_candidates(\n    db: StandardDatabase,\n    search_text: str,\n    threshold: float,\n    limit: int,\n    tags: Optional[List[str]],\n    parent_search_id: str,"
        bm25_fetch_sig_fixed = "def _fetch_bm25_candidates(\n    db: StandardDatabase,\n    search_text: str,\n    threshold: float,\n    limit: int,\n    tags: Optional[List[str]],\n    parent_search_id: str,\n    view_name: str = VIEW_NAME,"
        
        # Update the _fetch_semantic_candidates function signature (assuming it exists)
        semantic_fetch_sig_problem = "def _fetch_semantic_candidates(\n    db: StandardDatabase,\n    query_embedding: List[float],\n    threshold: float,\n    limit: int,\n    tags: Optional[List[str]],\n    parent_search_id: str,"
        semantic_fetch_sig_fixed = "def _fetch_semantic_candidates(\n    db: StandardDatabase,\n    query_embedding: List[float],\n    threshold: float,\n    limit: int,\n    tags: Optional[List[str]],\n    parent_search_id: str,\n    view_name: str = VIEW_NAME,"
        
        # Fix the view name usage with regex to get actual view name constants
        def replace_view_in_query(match):
            query_text = match.group(0)
            # Replace any hardcoded view name with the view_name parameter
            query_text = re.sub(r'FOR doc IN ([A-Za-z0-9_]+)', r'FOR doc IN \1', query_text)
            view_in_query = re.search(r'FOR doc IN ([A-Za-z0-9_]+)', query_text)
            if view_in_query:
                view_constant = view_in_query.group(1)
                query_text = query_text.replace(f'FOR doc IN {view_constant}', 'FOR doc IN {view_name}')
            return query_text
        
        # Apply the replacements
        if bm25_problem_line in content:
            content = content.replace(bm25_problem_line, bm25_fixed_line)
            logger.info("Fixed BM25 fetch call")
        else:
            logger.warning("Could not find BM25 fetch call to fix")
            
        if semantic_problem_line in content:
            content = content.replace(semantic_problem_line, semantic_fixed_line)
            logger.info("Fixed semantic fetch call")
        else:
            logger.warning("Could not find semantic fetch call to fix")
            
        if bm25_fetch_sig_problem in content:
            content = content.replace(bm25_fetch_sig_problem, bm25_fetch_sig_fixed)
            logger.info("Fixed BM25 fetch signature")
        else:
            logger.warning("Could not find BM25 fetch signature to fix")
            
        if semantic_fetch_sig_problem in content:
            content = content.replace(semantic_fetch_sig_problem, semantic_fetch_sig_fixed)
            logger.info("Fixed semantic fetch signature")
        else:
            logger.warning("Could not find semantic fetch signature to fix")
            
        # Use regex to fix the AQL query in _fetch_bm25_candidates
        content = re.sub(r'FOR doc IN [A-Za-z0-9_]+\s+SEARCH', r'FOR doc IN {view_name}\n        SEARCH', content)
        logger.info("Fixed AQL query in BM25 fetch")
        
        # Write back the fixed content
        with open(file_path, "w") as f:
            f.write(content)
        
        logger.success(f"Successfully fixed the hybrid.py file to use custom view names")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing hybrid.py: {e}")
        return False

if __name__ == "__main__":
    logger.info("Fixing hybrid.py to use custom view names...")
    success = fix_hybrid_helpers()
    sys.exit(0 if success else 1)
