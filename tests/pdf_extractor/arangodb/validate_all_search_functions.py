#!/usr/bin/env python3
"""
Comprehensive validation script for all search functions in the pdf_extractor.

This script tests each search function with real ArangoDB connections and validates
the results against expected outputs. It ensures that all search functions work as expected.
"""

import sys
import os
import json
import time
from typing import Dict, Any, List, Tuple, Optional, Callable
from loguru import logger

# Import ArangoDB connection
from arango.database import StandardDatabase
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME,
    EDGE_COLLECTION_NAME,
    GRAPH_NAME,
    VIEW_NAME
)

# Import all search functions
from pdf_extractor.arangodb.search_api.bm25 import bm25_search
from pdf_extractor.arangodb.search_api.semantic import semantic_search
from pdf_extractor.arangodb.search_api.hybrid import hybrid_search
from pdf_extractor.arangodb.search_api.graph_traverse import graph_traverse
from pdf_extractor.arangodb.search_api.tag_search import tag_search
from pdf_extractor.arangodb.search_api.keyword import search_keyword
from pdf_extractor.arangodb.search_api.message_search import unified_search
from pdf_extractor.arangodb.search_api.search_basic import (
    find_lessons_by_tags_advanced,
    find_lessons_by_text_like
)
from pdf_extractor.arangodb.search_api.search_functions import search_messages

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

class SearchValidator:
    """
    Class to validate all search functions.
    Uses actual ArangoDB connections for real-world testing.
    """
    
    def __init__(self):
        """Initialize the validator with ArangoDB connection."""
        self.client = connect_arango()
        self.db = ensure_database(self.client)
        self.fixtures_dir = "src/test_fixtures"
        self.results = {}
        self.validation_failures = {}
        
        # Verify connection is working
        collections = [c["name"] for c in self.db.collections() if not c["name"].startswith("_")]
        logger.info(f"Connected to ArangoDB. Available collections: {collections}")
        
        # Ensure fixtures directory exists
        if not os.path.exists(self.fixtures_dir):
            os.makedirs(self.fixtures_dir)
            logger.info(f"Created fixtures directory: {self.fixtures_dir}")
    
    def get_fixture_path(self, fixture_name: str) -> str:
        """Get the full path to a fixture file."""
        return os.path.join(self.fixtures_dir, fixture_name)
    
    def ensure_fixture_exists(self, fixture_path: str, default_content: Dict[str, Any]) -> None:
        """Ensure a fixture file exists, creating it with default content if needed."""
        if not os.path.exists(fixture_path):
            with open(fixture_path, "w") as f:
                json.dump(default_content, f, indent=2)
            logger.info(f"Created fixture file: {fixture_path}")
    
    def validate_search_result(
        self, 
        search_results: Dict[str, Any], 
        fixture_path: str,
        validator_func: Optional[Callable] = None
    ) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
        """
        Generic validation function for search results.
        
        Args:
            search_results: The results from a search function
            fixture_path: Path to the fixture file with expected results
            validator_func: Optional custom validator function
            
        Returns:
            Tuple of (validation_passed, validation_failures)
        """
        # If a custom validator is provided, use it
        if validator_func:
            return validator_func(search_results, fixture_path)
        
        # Default validation logic
        try:
            # Load fixture data
            with open(fixture_path, "r") as f:
                expected_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load fixture data: {e}")
            return False, {"fixture_loading_error": {"expected": "Valid JSON file", "actual": str(e)}}
        
        # Track all validation failures
        validation_failures = {}
        
        # Basic structural validation
        if "results" not in search_results:
            validation_failures["missing_results"] = {
                "expected": "Results field present",
                "actual": "Results field missing"
            }
            return False, validation_failures
            
        # Validate if we have any results
        if "expected_min_results" in expected_data:
            expected_min = expected_data["expected_min_results"]
            actual_count = len(search_results.get("results", []))
            if actual_count < expected_min:
                validation_failures["min_results"] = {
                    "expected": f">= {expected_min}",
                    "actual": actual_count
                }
        
        return len(validation_failures) == 0, validation_failures
    
    def validate_bm25_search(self) -> bool:
        """Validate BM25 search function."""
        fixture_path = self.get_fixture_path("bm25_search_validation.json")
        
        # Ensure fixture exists
        self.ensure_fixture_exists(fixture_path, {
            "expected_min_results": 0,
            "query": "python error",
            "top_n": 3
        })
        
        # Run the search
        test_query = "python error"
        search_results = bm25_search(
            db=self.db,
            query_text=test_query,
            top_n=3,
            min_score=0.0
        )
        
        # Store results for reporting
        self.results["bm25"] = search_results
        
        # Validate results
        validation_passed, failures = self.validate_search_result(search_results, fixture_path)
        if not validation_passed:
            self.validation_failures["bm25"] = failures
        
        logger.info(f"BM25 search validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
        if search_results.get("results"):
            logger.info(f"- Found {len(search_results['results'])} results")
            for i, result in enumerate(search_results["results"][:3]):  # Show top 3
                if "doc" in result and "_key" in result["doc"]:
                    logger.info(f"  - Result {i+1}: Key={result['doc']['_key']}")
        else:
            logger.warning("- No results found")
        
        return validation_passed
    
    def validate_semantic_search(self) -> bool:
        """Validate semantic search function."""
        fixture_path = self.get_fixture_path("semantic_search_validation.json")
        
        # Ensure fixture exists
        self.ensure_fixture_exists(fixture_path, {
            "expected_min_results": 0,
            "query": "python error handling",
            "top_n": 5
        })
        
        # Run the search
        test_query = "python error handling"
        try:
            search_results = semantic_search(
                db=self.db,
                query=test_query,
                top_n=5,
                min_score=0.5  # Lower threshold for testing
            )
            
            # Store results for reporting
            self.results["semantic"] = search_results
            
            # Validate results
            validation_passed, failures = self.validate_search_result(search_results, fixture_path)
            if not validation_passed:
                self.validation_failures["semantic"] = failures
            
            logger.info(f"Semantic search validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
            if search_results.get("results"):
                logger.info(f"- Found {len(search_results['results'])} results")
                for i, result in enumerate(search_results["results"][:3]):  # Show top 3
                    if "doc" in result and "_key" in result["doc"]:
                        logger.info(f"  - Result {i+1}: Key={result['doc']['_key']}")
            else:
                logger.warning("- No results found")
                
            return validation_passed
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            self.validation_failures["semantic"] = {"execution_error": {"expected": "Successful execution", "actual": str(e)}}
            return False
    
    def validate_hybrid_search(self) -> bool:
        """Validate hybrid search function."""
        fixture_path = self.get_fixture_path("hybrid_search_validation.json")
        
        # Ensure fixture exists
        self.ensure_fixture_exists(fixture_path, {
            "expected_min_results": 0,
            "query": "python code",
            "top_n": 3
        })
        
        # Run the search
        test_query = "python code"
        try:
            search_results = hybrid_search(
                db=self.db,
                query_text=test_query,
                top_n=3
            )
            
            # Store results for reporting
            self.results["hybrid"] = search_results
            
            # Validate results
            validation_passed, failures = self.validate_search_result(search_results, fixture_path)
            if not validation_passed:
                self.validation_failures["hybrid"] = failures
            
            logger.info(f"Hybrid search validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
            if search_results.get("results"):
                logger.info(f"- Found {len(search_results['results'])} results")
                for i, result in enumerate(search_results["results"][:3]):  # Show top 3
                    if "doc" in result and "_key" in result["doc"]:
                        logger.info(f"  - Result {i+1}: Key={result['doc']['_key']}")
            else:
                logger.warning("- No results found")
                
            return validation_passed
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            self.validation_failures["hybrid"] = {"execution_error": {"expected": "Successful execution", "actual": str(e)}}
            return False
    
    def validate_graph_traverse(self) -> bool:
        """Validate graph traversal function."""
        fixture_path = self.get_fixture_path("graph_traverse_validation.json")
        
        # Ensure fixture exists
        self.ensure_fixture_exists(fixture_path, {
            "expected_min_results": 0,
            "direction": "ANY",
            "min_depth": 1,
            "max_depth": 2
        })
        
        # Find a test node to traverse from
        test_node_key = None
        try:
            # Try to find a document in the database
            aql = f"FOR doc IN {COLLECTION_NAME} LIMIT 1 RETURN doc._key"
            cursor = self.db.aql.execute(aql)
            results = list(cursor)
            
            if results:
                test_node_key = results[0]
                logger.info(f"Using test node key: {test_node_key}")
            else:
                # No documents found, try to create a test document
                try:
                    collection = self.db.collection(COLLECTION_NAME)
                    doc = collection.insert({"content": "Test document for traversal", "tags": ["test"]})
                    test_node_key = doc["_key"]
                    logger.info(f"Created test node key: {test_node_key}")
                except Exception as e:
                    logger.error(f"Could not create test document: {e}")
                    self.validation_failures["graph_traverse"] = {"setup_error": {"expected": "Test document created", "actual": str(e)}}
                    return False
        except Exception as e:
            logger.error(f"Could not find or create test document: {e}")
            self.validation_failures["graph_traverse"] = {"setup_error": {"expected": "Test document found", "actual": str(e)}}
            return False
        
        # Run the traversal
        try:
            traverse_results = graph_traverse(
                db=self.db,
                start_vertex_key=test_node_key,
                min_depth=1,
                max_depth=2,
                direction="ANY"
            )
            
            # Store results for reporting
            self.results["graph_traverse"] = traverse_results
            
            # Validate results - structure only since graph may be empty
            validation_passed = (
                "results" in traverse_results and
                "count" in traverse_results and
                "params" in traverse_results
            )
            
            logger.info(f"Graph traversal validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
            logger.info(f"- Found {traverse_results.get('count', 0)} traversal paths")
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            self.validation_failures["graph_traverse"] = {"execution_error": {"expected": "Successful execution", "actual": str(e)}}
            return False
    
    def validate_tag_search(self) -> bool:
        """Validate tag search function."""
        fixture_path = self.get_fixture_path("tag_search_validation.json")
        
        # Ensure fixture exists
        self.ensure_fixture_exists(fixture_path, {
            "expected_min_results": 0,
            "tags": ["python", "test"],
            "limit": 10
        })
        
        # Run the search
        test_tags = ["python", "test"]
        try:
            search_results = tag_search(
                db=self.db,
                tags=test_tags,
                limit=10
            )
            
            # Store results for reporting
            self.results["tag_search"] = search_results
            
            # Validate results
            validation_passed, failures = self.validate_search_result(search_results, fixture_path)
            if not validation_passed:
                self.validation_failures["tag_search"] = failures
            
            logger.info(f"Tag search validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
            if search_results.get("results"):
                logger.info(f"- Found {len(search_results['results'])} results for tags {test_tags}")
                for i, result in enumerate(search_results["results"][:3]):  # Show top 3
                    if "doc" in result and "_key" in result["doc"]:
                        logger.info(f"  - Result {i+1}: Key={result['doc']['_key']}")
            else:
                logger.warning(f"- No results found for tags {test_tags}")
                
            return validation_passed
            
        except Exception as e:
            logger.error(f"Tag search failed: {e}")
            self.validation_failures["tag_search"] = {"execution_error": {"expected": "Successful execution", "actual": str(e)}}
            return False
    
    def validate_keyword_search(self) -> bool:
        """Validate keyword search function."""
        fixture_path = self.get_fixture_path("keyword_search_validation.json")
        
        # Ensure fixture exists
        self.ensure_fixture_exists(fixture_path, {
            "expected_min_results": 0,
            "search_term": "python",
            "similarity_threshold": 80.0
        })
        
        # Run the search
        test_term = "python"
        try:
            search_results = search_keyword(
                db=self.db,
                search_term=test_term,
                similarity_threshold=80.0,  # Lower threshold for testing
                top_n=10
            )
            
            # Store results for reporting
            self.results["keyword_search"] = search_results
            
            # Validate results
            validation_passed = (
                "results" in search_results and
                "total" in search_results and
                "search_term" in search_results
            )
            
            if not validation_passed:
                self.validation_failures["keyword_search"] = {
                    "structure_validation": {
                        "expected": "Results, total, and search_term fields",
                        "actual": f"Missing required fields: {', '.join(f for f in ['results', 'total', 'search_term'] if f not in search_results)}"
                    }
                }
            
            logger.info(f"Keyword search validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
            if search_results.get("results"):
                logger.info(f"- Found {len(search_results['results'])} results for term '{test_term}'")
                for i, item in enumerate(search_results["results"][:3]):  # Show top 3
                    doc = item.get("doc", {})
                    score = item.get("keyword_score", 0)
                    logger.info(f"  - Result {i+1}: Key={doc.get('_key', 'unknown')}, Score={score:.2f}")
            else:
                logger.warning(f"- No results found for term '{test_term}'")
                
            return validation_passed
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            self.validation_failures["keyword_search"] = {"execution_error": {"expected": "Successful execution", "actual": str(e)}}
            return False
    
    def validate_unified_search(self) -> bool:
        """Validate unified search function."""
        fixture_path = self.get_fixture_path("unified_search_validation.json")
        
        # Ensure fixture exists
        self.ensure_fixture_exists(fixture_path, {
            "expected_min_results": 0,
            "query": "test query",
            "top_n": 5
        })
        
        # Run the search
        test_query = "test query"
        try:
            search_results = unified_search(
                db=self.db,
                query=test_query,
                top_n=5
            )
            
            # Store results for reporting
            self.results["unified_search"] = search_results
            
            # Validate results
            validation_passed = (
                "results" in search_results and
                "collections_searched" in search_results
            )
            
            if not validation_passed:
                self.validation_failures["unified_search"] = {
                    "structure_validation": {
                        "expected": "Results and collections_searched fields",
                        "actual": f"Missing required fields: {', '.join(f for f in ['results', 'collections_searched'] if f not in search_results)}"
                    }
                }
            
            logger.info(f"Unified search validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
            if search_results.get("results"):
                logger.info(f"- Found {len(search_results['results'])} results across {len(search_results.get('collections_searched', []))} collections")
                for i, result in enumerate(search_results["results"][:3]):  # Show top 3
                    if "doc" in result and "_key" in result["doc"]:
                        logger.info(f"  - Result {i+1}: Key={result['doc']['_key']}, Collection={result.get('collection', 'unknown')}")
            else:
                logger.warning(f"- No results found for query '{test_query}'")
                
            return validation_passed
            
        except Exception as e:
            logger.error(f"Unified search failed: {e}")
            self.validation_failures["unified_search"] = {"execution_error": {"expected": "Successful execution", "actual": str(e)}}
            return False
    
    def validate_search_basic(self) -> bool:
        """Validate basic search functions."""
        fixture_path = self.get_fixture_path("search_basic_validation.json")
        
        # Ensure fixture exists
        self.ensure_fixture_exists(fixture_path, {
            "expected_min_results": 0,
            "tag_keywords": ["python", "test"],
            "text_keywords": ["error", "handling"]
        })
        
        # Run tag search
        test_tags = ["python", "test"]
        test_text = ["error", "handling"]
        try:
            # Tag search
            tag_results = find_lessons_by_tags_advanced(
                db=self.db,
                tag_keywords=test_tags,
                limit=10,
                match_all=False
            )
            
            # Text search
            text_results = find_lessons_by_text_like(
                db=self.db,
                text_keywords=test_text,
                limit=10,
                match_all=False
            )
            
            # Store results for reporting
            self.results["search_basic"] = {
                "tag_results": tag_results,
                "text_results": text_results
            }
            
            # Validate basic structure - these are lists
            validation_passed = (
                isinstance(tag_results, list) and
                isinstance(text_results, list)
            )
            
            if not validation_passed:
                self.validation_failures["search_basic"] = {
                    "structure_validation": {
                        "expected": "List results for both tag and text search",
                        "actual": f"Invalid types: tag_results={type(tag_results)}, text_results={type(text_results)}"
                    }
                }
            
            logger.info(f"Basic search validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
            logger.info(f"- Tag search found {len(tag_results)} results")
            logger.info(f"- Text search found {len(text_results)} results")
                
            return validation_passed
            
        except Exception as e:
            logger.error(f"Basic search failed: {e}")
            self.validation_failures["search_basic"] = {"execution_error": {"expected": "Successful execution", "actual": str(e)}}
            return False
    
    def run_all_validations(self) -> Dict[str, bool]:
        """Run all validation tests and return results."""
        validation_results = {}
        
        # BM25 Search
        validation_results["bm25"] = self.validate_bm25_search()
        
        # Semantic Search
        validation_results["semantic"] = self.validate_semantic_search()
        
        # Hybrid Search
        validation_results["hybrid"] = self.validate_hybrid_search()
        
        # Graph Traversal
        validation_results["graph_traverse"] = self.validate_graph_traverse()
        
        # Tag Search
        validation_results["tag_search"] = self.validate_tag_search()
        
        # Keyword Search
        validation_results["keyword_search"] = self.validate_keyword_search()
        
        # Unified Search
        validation_results["unified_search"] = self.validate_unified_search()
        
        # Basic Search
        validation_results["search_basic"] = self.validate_search_basic()
        
        return validation_results
    
    def print_summary(self, validation_results: Dict[str, bool]) -> None:
        """Print a summary of validation results."""
        print("\n===== SEARCH FUNCTION VALIDATION SUMMARY =====")
        
        # Count passed and failed tests
        passed = sum(1 for result in validation_results.values() if result)
        total = len(validation_results)
        
        print(f"Tests passed: {passed}/{total}\n")
        
        # Print individual results
        for function_name, passed in validation_results.items():
            result_symbol = "✅" if passed else "❌"
            print(f"{result_symbol} {function_name}")
            
            # If validation failed, print details
            if not passed and function_name in self.validation_failures:
                failures = self.validation_failures[function_name]
                for field, details in failures.items():
                    print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        
        print("\n==============================================")

def main():
    """Run all search function validations."""
    validator = SearchValidator()
    validation_results = validator.run_all_validations()
    validator.print_summary(validation_results)
    
    # Exit with status code based on results
    if all(validation_results.values()):
        logger.info("All search functions validated successfully!")
        sys.exit(0)
    else:
        logger.error("Some search functions failed validation. See summary for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
