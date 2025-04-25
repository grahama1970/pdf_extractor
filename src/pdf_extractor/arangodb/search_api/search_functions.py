#!/usr/bin/env python3
# search_functions.py - Implementation of search functions

import sys
from typing import Dict, Any, List, Optional, Union
from loguru import logger

from arango.database import StandardDatabase

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
    '''
    Search for messages based on the query.
    '''
    try:
        # Build filter conditions
        filter_conditions = []
        
        if conversation_id:
            filter_conditions.append(f"doc.conversation_id == '{conversation_id}'")
        
        if message_type:
            filter_conditions.append(f"doc.message_type == '{message_type}'")
        
        # Convert filter conditions to AQL filter string
        filter_string = " AND ".join(filter_conditions) if filter_conditions else ""
        
        # Select the appropriate search function
        if search_type.lower() == 'bm25':
            search_func = bm25_search
        elif search_type.lower() == 'semantic':
            search_func = semantic_search
        else:  # default to hybrid
            search_func = hybrid_search
        
        # Perform the search
        results = search_func(
            db=db,
            query=query,
            collection_name=MESSAGE_COLLECTION_NAME,
            filter_string=filter_string,
            top_n=top_n
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Failed to search messages: {e}")
        return {"status": "error", "message": str(e), "results": []}

def unified_search(
    db: StandardDatabase,
    query: str,
    search_type: str = 'hybrid',
    collections: List[str] = None,
    exclude_collections: List[str] = None,
    top_n: int = 5
) -> Dict[str, Any]:
    '''
    Search across multiple collections.
    '''
    try:
        # Determine which collections to search
        available_collections = [DOC_COLLECTION_NAME, MESSAGE_COLLECTION_NAME]
        
        if collections:
            search_collections = [c for c in collections if c in available_collections]
        else:
            search_collections = available_collections.copy()
        
        if exclude_collections:
            search_collections = [c for c in search_collections if c not in exclude_collections]
        
        if not search_collections:
            return {"status": "error", "message": "No valid collections to search", "results": []}
        
        # Select the appropriate search function
        if search_type.lower() == 'bm25':
            search_func = bm25_search
        elif search_type.lower() == 'semantic':
            search_func = semantic_search
        else:  # default to hybrid
            search_func = hybrid_search
        
        # Perform search on each collection
        all_results = []
        
        for collection_name in search_collections:
            # Perform search
            collection_results = search_func(
                db=db,
                query=query,
                collection_name=collection_name,
                filter_string="",
                top_n=top_n
            )
            
            # Add collection information to each result
            for result in collection_results.get('results', []):
                result['collection'] = collection_name
                all_results.append(result)
        
        # Sort combined results by relevance score (descending)
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Truncate to top_n results overall if requested
        if len(all_results) > top_n:
            all_results = all_results[:top_n]
        
        return {
            "status": "success",
            "message": f"Search completed across {len(search_collections)} collections",
            "collections_searched": search_collections,
            "results": all_results
        }
    
    except Exception as e:
        logger.error(f"Failed to perform unified search: {e}")
        return {"status": "error", "message": str(e), "results": []}
