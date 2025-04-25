# src/pdf_extractor/arangodb/search_api/hybrid.py
import uuid
from typing import List, Dict, Any, Optional

from loguru import logger
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

def search_hybrid(
    db: StandardDatabase,
    query_text: str,
    view_name: str = VIEW_NAME,
    top_n: int = 10,  # Final number of results to return
    initial_k: int = 20,  # Number of candidates to fetch from each search type
    bm25_threshold: float = 0.01,  # Lower threshold to capture more potential candidates
    similarity_threshold: float = 0.70,  # Lower threshold to capture more potential candidates
    tags: Optional[List[str]] = None,
    rrf_k: int = 60,  # Constant for Reciprocal Rank Fusion (common default)
) -> Dict[str, Any]:
    