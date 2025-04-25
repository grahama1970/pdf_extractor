"""
Keyword Search Module for PDF Extractor ArangoDB Integration.

This module provides functionality for performing keyword searches with fuzzy matching
using ArangoDB and RapidFuzz.

Third-Party Package Documentation:
- python-arango: https://python-driver.arangodb.com/
- rapidfuzz: https://rapidfuzz.github.io/RapidFuzz/

Sample Input:
Search term and database connection details

Expected Output:
List of documents matching the keyword search with fuzzy matching
"""
import sys
import os
from typing import List, Dict, Any, Optional
import re
import rapidfuzz
from loguru import logger

from arango.database import StandardDatabase
from arango.cursor import Cursor # Import Cursor for type checking
from rapidfuzz import fuzz, process

# Import config variables
from pdf_extractor.arangodb.config import (
    VIEW_NAME,
    COLLECTION_NAME,
    TEXT_ANALYZER
)

def search_keyword(
    db: StandardDatabase,
    search_term: str,
    similarity_threshold: float = 97.0,
    top_n: int = 10,
    view_name: str = VIEW_NAME, tags: Optional[List[str]] = None,
    collection_name: str = COLLECTION_NAME,
) -> Dict[str, Any]:
    """
    Perform a keyword search with fuzzy matching.
    
    Args:
        db: ArangoDB database connection
        search_term: The keyword to search for
        similarity_threshold: Minimum similarity score (0-100) for fuzzy matching
        top_n: Maximum number of results to return
        view_name: Name of the ArangoDB search view
        collection_name: Name of the collection
        
    Returns:
        Dictionary containing results and metadata
        
    Raises:
        ValueError: If search_term is empty
        Exception: For any other errors
    """
    if not search_term or search_term.strip() == "":
        raise ValueError("Search term cannot be empty")
    
    # Clean search term
    search_term = search_term.strip()
    
    # AQL query with bind parameters
    aql_query = f"""
    FOR doc IN {view_name}
      SEARCH ANALYZER(doc.problem LIKE @search_pattern OR 
                    doc.solution LIKE @search_pattern OR 
                    doc.context LIKE @search_pattern, 
                    "{TEXT_ANALYZER}")
      SORT BM25(doc) DESC
      LIMIT @top_n
      RETURN {{ 
        doc: KEEP(doc, "_key", "_id", "problem", "solution", "context", "tags")
      }}
    """
    
    # Bind parameters: Use a simple pattern without word boundaries
    # The word matching will be done with rapidfuzz instead
    bind_vars = {
        "search_pattern": f"%{search_term}%",
        "top_n": top_n
    }
    
    try:
        # Execute AQL query
        cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
        # Iterate over the cursor correctly, adding checks for safety
        # Iterate over the cursor correctly, adding explicit type check
        initial_results = []
        if isinstance(cursor, Cursor):
            try:
                initial_results = [doc for doc in cursor]
            except Exception as e: # Catch potential errors during iteration
                logger.error(f"Error iterating over cursor results: {e}", exc_info=True)
                raise # Re-raise to signal failure
        elif cursor is None:
             logger.warning("db.aql.execute returned None, expected a cursor.")
        else:
             # Log if it's an unexpected type (like AsyncJob/BatchJob in sync context)
             logger.error(f"db.aql.execute returned unexpected type: {type(cursor)}. Expected Cursor.")
             # Decide how to handle - raise error?
             raise TypeError(f"Unexpected cursor type: {type(cursor)}")


        # Filter results using rapidfuzz for whole word matching
        filtered_results = []
        for item in initial_results:
            doc = item.get("doc", {})
            
            # Combine searchable text
            text = " ".join([
                str(doc.get("problem", "")),
                str(doc.get("solution", "")),
                str(doc.get("context", ""))
            ]).lower()
            
            # Extract whole words from the text
            words = re.findall(r'\b\w+\b', text)
            
            # Use rapidfuzz to find words with similarity to search_term
            matches = process.extract(
                search_term.lower(),
                words,
                scorer=fuzz.ratio,
                score_cutoff=similarity_threshold
            )
            
            if matches:
                # Add the match and its similarity score
                best_match = matches[0]  # tuple of (match, score)
                item["keyword_score"] = best_match[1] / 100.0  # convert to 0-1 scale
                filtered_results.append(item)
        
        # Sort results by keyword_score (highest first)
        filtered_results.sort(key=lambda x: x.get("keyword_score", 0), reverse=True)
        
        # Limit to top_n
        filtered_results = filtered_results[:top_n]
        
        # Create result object
        result = {
            "results": filtered_results,
            "total": len(filtered_results),
            "search_term": search_term,
            "similarity_threshold": similarity_threshold
        }
        
        logger.info(f"Keyword search for '{search_term}' found {len(filtered_results)} results")
        return result
    
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level:<7} | {message}"
    )
    
    # Connect to ArangoDB
    client = connect_arango()
    if client is None:
        logger.error("Failed to connect to ArangoDB. Exiting.")
        sys.exit(1)
    db = ensure_database(client)
    if db is None:
        logger.error("Failed to ensure database. Exiting.")
        sys.exit(1)

    # Perform search
    search_term = "python"  # Example search term
    try:
        results = search_keyword(db, search_term)
        
        # Output results
        logger.info(f"Found {results['total']} matching documents:")
        for i, item in enumerate(results["results"]):
            doc = item.get("doc", {})
            score = item.get("keyword_score", 0)
            logger.info(f"{i+1}. Key: {doc.get('_key')} (Score: {score:.2f})")
            logger.info(f"   Problem: {doc.get('problem', '')[:50]}...")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
