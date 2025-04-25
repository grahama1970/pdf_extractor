# validate_hybrid.py
import sys
import os
import json
from loguru import logger

from pdf_extractor.arangodb.search_api.hybrid import search_hybrid as hybrid_search
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.search_api.validation import (
    create_fixture_dir, 
    create_fixture, 
    load_fixture, 
    validate_search_results, 
    report_validation_results
)

if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}"
    )
    
    logger.info("Running Hybrid search validation")
    
    # Create fixtures directory
    fixtures_dir = create_fixture_dir()
    
    # Define test parameters
    query_text = "Python exception best practices"
    top_n = 5
    initial_k = 10
    bm25_threshold = 0.01
    similarity_threshold = 0.70
    filter_tags = []
    rrf_k = 60
    
    params = {
        "query_text": query_text,
        "top_n": top_n,
        "initial_k": initial_k,
        "bm25_threshold": bm25_threshold,
        "similarity_threshold": similarity_threshold,
        "filter_tags": filter_tags,
        "rrf_k": rrf_k
    }
    
    # Define fixture path
    fixture_path = os.path.join(fixtures_dir, "hybrid_test_fixture.json")
    
    # Connect to ArangoDB
    client = connect_arango()
    if not client:
        logger.error("Validation failed: Could not connect to ArangoDB")
        sys.exit(1)
    
    # Get database
    db = ensure_database(client)
    if not db:
        logger.error("Validation failed: Could not ensure database")
        sys.exit(1)
    
    # Check if fixture exists
    if not os.path.exists(fixture_path):
        logger.warning(f"Fixture not found at {fixture_path}. Creating from current results.")
        
        # Execute search
        results = hybrid_search(
            db=db, 
            query_text=query_text, 
            view_name="lessons_view",
            top_n=top_n, 
            initial_k=initial_k, 
            bm25_threshold=bm25_threshold, 
            similarity_threshold=similarity_threshold, 
            tags=filter_tags, 
            rrf_k=rrf_k
        )
        
        # Create fixture
        if create_fixture(results, params, fixture_path):
            logger.info(f"Created new fixture with {len(results.get('results', []))} results")
            logger.info("Run validation again with the new fixture to perform actual validation")
            sys.exit(0)
        else:
            logger.error("Failed to create fixture")
            sys.exit(1)
    
    # Load fixture
    fixture_data = load_fixture(fixture_path)
    if not fixture_data:
        logger.error(f"Failed to load fixture from {fixture_path}")
        sys.exit(1)
    
    expected_results = fixture_data.get("results", {})
    
    # Execute search with parameters from fixture
    fixture_params = fixture_data.get("params", params)
    
    results = hybrid_search(
        db=db, 
        query_text=fixture_params.get("query_text", query_text),
        view_name="lessons_view",
        top_n=fixture_params.get("top_n", top_n),
        initial_k=fixture_params.get("initial_k", initial_k),
        bm25_threshold=fixture_params.get("bm25_threshold", bm25_threshold),
        similarity_threshold=fixture_params.get("similarity_threshold", similarity_threshold),
        tags=fixture_params.get("filter_tags", filter_tags),
        rrf_k=fixture_params.get("rrf_k", rrf_k)
    )
    
    # Validate results
    validation_failures = validate_search_results(
        results, 
        expected_results,
        "rrf_score"
    )
    
    # Report validation status
    if not report_validation_results(validation_failures):
        sys.exit(1)
    
    sys.exit(0)
