#!/usr/bin/env python3
# src/pdf_extractor/arangodb/search_api/message_search.py

import sys
from typing import Dict, Any, List, Optional, Union
from loguru import logger

from arango.database import StandardDatabase

from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME as DOC_COLLECTION_NAME,
    VIEW_NAME
)
from pdf_extractor.arangodb.message_history_config import (
    MESSAGE_COLLECTION_NAME
)

# Import search functions
from pdf_extractor.arangodb.search_api.hybrid import hybrid_search
from pdf_extractor.arangodb.search_api.bm25 import bm25_search
from pdf_extractor.arangodb.search_api.semantic import semantic_search

def search_messages(
    db: StandardDatabase,
    query: str,
    search_type: str = 'hybrid',
    top_n: int = 5,
    conversation_id: Optional[str] = None,
    message_type: Optional[str] = None,
    date_range: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    