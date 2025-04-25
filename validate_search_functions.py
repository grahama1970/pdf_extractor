#!/usr/bin/env python3
"""
Validate search functions with real ArangoDB connection.

This script tests each search function using the real ArangoDB database
and outputs validation results. It does NOT mock any database operations.
"""

import sys
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

# Import ArangoDB client
from arango import ArangoClient
from arango.database import StandardDatabase

# Import search functions
from pdf_extractor.arangodb.search_api.bm25 import bm25_search
from pdf_extractor.arangodb.search_api.semantic import semantic_search
from pdf_extractor.arangodb.search_api.hybrid import hybrid_search
from pdf_extractor.arangodb.search_api.graph_traverse import graph_traverse
from pdf_extractor.arangodb.search_api.tag_search import tag_search
from pdf_extractor.arangodb.search_api.keyword import search_keyword
from pdf_extractor.arangodb.search_api.message_search import unified_search

# Connection parameters
HOST = "http://192.168.86.49:8529/"
USERNAME = "root"
PASSWORD = "openSesame"
DB_NAME = "pdf_extractor"

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

class SearchValidator:
    """Validate search functions with real ArangoDB connection."""
    
    def __init__(self):
        """Initialize with ArangoDB connection."""
        # Connect to ArangoDB
        self.client = ArangoClient(hosts=HOST)
        self.db = self.client.db(DB_NAME, username=USERNAME, password=PASSWORD)
        
        # Get collection names
        self.collections = [c["name"] for c in self.db.collections() if not c["name"].startswith("_")]
        logger.info(f"Connected to ArangoDB database '{DB_NAME}'")
        logger.info(f"Available collections: {', '.join(self.collections)}")
        
        # Default collection and document collection
        self.doc_collection = "documents"
        if self.doc_collection not in self.collections:
            logger.warning(f"Collection '{self.doc_collection}' not found! Some tests may fail.")
        
        # Find a test document for traversal
        self.test_doc_key = self._get_test_document_key()
        
        # Results and failures tracking
        self.results = {}
        self.failures = {}
    
    def _get_test_document_key(self) -> Optional[str]:
        """Get a test document key for traversal tests."""
        if self.doc_collection not in self.collections:
            return None
        
        try:
            # Find any document in the collection
            aql = f"FOR doc IN {self.doc_collection} LIMIT 1 RETURN doc._key"
            cursor = self.db.aql.execute(aql)
            results = list(cursor)
            
            if results:
                doc_key = results[0]
                logger.info(f"Found test document key: {doc_key}")
                return doc_key
        except Exception as e:
            logger.error(f"Error finding test document: {e}")
        
        return None
    
    def validate_bm25_search(self) -> bool:
        """Validate BM25 search function."""
        logger.info("Testing BM25 search...")
        
        try:
            # Run BM25 search with a test query
            results = bm25_search(
                db=self.db,
                query_text="python",
                top_n=5
            )
            
            # Check if the function executed without errors
            execution_success = "error" not in results
            
            # Log results
            if "results" in results:
                result_count = len(results["results"])
                logger.info(f"BM25 search returned {result_count} results")
                
                # Show some results if available
                if result_count > 0:
                    for i, item in enumerate(results["results"][:3]):
                        doc = item.get("doc", {})
                        score = item.get("score", 0)
                        logger.info(f"  Result {i+1}: Key={doc.get('_key', 'unknown')}, Score={score}")
            
            # Store results for reporting
            self.results["bm25"] = {
                "success": execution_success,
                "result_count": len(results.get("results", [])),
                "error": results.get("error", None)
            }
            
            return execution_success
        
        except Exception as e:
            logger.error(f"BM25 search validation error: {e}")
            self.results["bm25"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def validate_semantic_search(self) -> bool:
        """Validate semantic search function."""
        logger.info("Testing semantic search...")
        
        try:
            # Run semantic search with a test query
            results = semantic_search(
                db=self.db,
                query="python error handling",
                top_n=5,
                min_score=0.5  # Lower for testing
            )
            
            # Check if the function executed without errors
            execution_success = "error" not in results
            
            # Log results
            if "results" in results:
                result_count = len(results["results"])
                logger.info(f"Semantic search returned {result_count} results")
                
                # Show some results if available
                if result_count > 0:
                    for i, item in enumerate(results["results"][:3]):
                        doc = item.get("doc", {})
                        score = item.get("similarity_score", item.get("score", 0))
                        logger.info(f"  Result {i+1}: Key={doc.get('_key', 'unknown')}, Score={score}")
            
            # Store results for reporting
            self.results["semantic"] = {
                "success": execution_success,
                "result_count": len(results.get("results", [])),
                "error": results.get("error", None)
            }
            
            return execution_success
        
        except Exception as e:
            logger.error(f"Semantic search validation error: {e}")
            self.results["semantic"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def validate_hybrid_search(self) -> bool:
        """Validate hybrid search function."""
        logger.info("Testing hybrid search...")
        
        try:
            # Run hybrid search with a test query
            results = hybrid_search(
                db=self.db,
                query_text="python code",
                top_n=5
            )
            
            # Check if the function executed without errors
            execution_success = "error" not in results
            
            # Log results
            if "results" in results:
                result_count = len(results["results"])
                logger.info(f"Hybrid search returned {result_count} results")
                
                # Show some results if available
                if result_count > 0:
                    for i, item in enumerate(results["results"][:3]):
                        doc = item.get("doc", {})
                        rrf_score = item.get("rrf_score", 0)
                        logger.info(f"  Result {i+1}: Key={doc.get('_key', 'unknown')}, RRF Score={rrf_score}")
            
            # Store results for reporting
            self.results["hybrid"] = {
                "success": execution_success,
                "result_count": len(results.get("results", [])),
                "error": results.get("error", None)
            }
            
            return execution_success
        
        except Exception as e:
            logger.error(f"Hybrid search validation error: {e}")
            self.results["hybrid"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def validate_graph_traverse(self) -> bool:
        """Validate graph traversal function."""
        logger.info("Testing graph traversal...")
        
        if not self.test_doc_key:
            logger.error("No test document found for traversal")
            self.results["graph_traverse"] = {
                "success": False,
                "error": "No test document found"
            }
            return False
        
        try:
            # Run graph traversal with the test document
            results = graph_traverse(
                db=self.db,
                start_vertex_key=self.test_doc_key,
                min_depth=1,
                max_depth=2,
                direction="ANY"
            )
            
            # Check if the function executed without errors
            execution_success = "error" not in results
            
            # Log results
            if "results" in results:
                result_count = len(results["results"])
                logger.info(f"Graph traversal returned {result_count} paths")
                
                # Show some results if available
                if result_count > 0:
                    for i, path in enumerate(results["results"][:3]):
                        if "vertex" in path and "_key" in path["vertex"]:
                            logger.info(f"  Path {i+1}: Target vertex={path['vertex']['_key']}")
            
            # Store results for reporting
            self.results["graph_traverse"] = {
                "success": execution_success,
                "result_count": len(results.get("results", [])),
                "error": results.get("error", None)
            }
            
            return execution_success
        
        except Exception as e:
            logger.error(f"Graph traversal validation error: {e}")
            self.results["graph_traverse"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def validate_tag_search(self) -> bool:
        """Validate tag search function."""
        logger.info("Testing tag search...")
        
        try:
            # Run tag search with test tags
            results = tag_search(
                db=self.db,
                tags=["python", "json"],
                limit=10
            )
            
            # Check if the function executed without errors
            execution_success = "error" not in results
            
            # Log results
            if "results" in results:
                result_count = len(results["results"])
                logger.info(f"Tag search returned {result_count} results")
                
                # Show some results if available
                if result_count > 0:
                    for i, item in enumerate(results["results"][:3]):
                        doc = item.get("doc", {})
                        logger.info(f"  Result {i+1}: Key={doc.get('_key', 'unknown')}")
            
            # Store results for reporting
            self.results["tag_search"] = {
                "success": execution_success,
                "result_count": len(results.get("results", [])),
                "error": results.get("error", None)
            }
            
            return execution_success
        
        except Exception as e:
            logger.error(f"Tag search validation error: {e}")
            self.results["tag_search"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def validate_keyword_search(self) -> bool:
        """Validate keyword search function."""
        logger.info("Testing keyword search...")
        
        try:
            # Run keyword search with a test term
            results = search_keyword(
                db=self.db,
                search_term="python",
                similarity_threshold=80.0,  # Lower for testing
                top_n=10
            )
            
            # Check if the results have expected structure
            execution_success = isinstance(results, dict) and "results" in results
            
            # Log results
            if execution_success and "results" in results:
                result_count = len(results["results"])
                logger.info(f"Keyword search returned {result_count} results")
                
                # Show some results if available
                if result_count > 0:
                    for i, item in enumerate(results["results"][:3]):
                        doc = item.get("doc", {})
                        score = item.get("keyword_score", 0)
                        logger.info(f"  Result {i+1}: Key={doc.get('_key', 'unknown')}, Score={score}")
            
            # Store results for reporting
            self.results["keyword_search"] = {
                "success": execution_success,
                "result_count": len(results.get("results", [])) if execution_success else 0,
                "error": None if execution_success else "Unexpected result structure"
            }
            
            return execution_success
        
        except Exception as e:
            logger.error(f"Keyword search validation error: {e}")
            self.results["keyword_search"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def validate_unified_search(self) -> bool:
        """Validate unified search function."""
        logger.info("Testing unified search...")
        
        try:
            # Run unified search with a test query
            results = unified_search(
                db=self.db,
                query="test query",
                top_n=5
            )
            
            # Check if the function executed without errors
            execution_success = "error" not in results
            
            # Log results
            if "results" in results:
                result_count = len(results["results"])
                logger.info(f"Unified search returned {result_count} results")
                
                # Show collections searched
                if "collections_searched" in results:
                    logger.info(f"  Collections searched: {', '.join(results['collections_searched'])}")
                
                # Show some results if available
                if result_count > 0:
                    for i, item in enumerate(results["results"][:3]):
                        doc = item.get("doc", {})
                        collection = item.get("collection", "unknown")
                        logger.info(f"  Result {i+1}: Key={doc.get('_key', 'unknown')}, Collection={collection}")
            
            # Store results for reporting
            self.results["unified_search"] = {
                "success": execution_success,
                "result_count": len(results.get("results", [])),
                "error": results.get("error", None)
            }
            
            return execution_success
        
        except Exception as e:
            logger.error(f"Unified search validation error: {e}")
            self.results["unified_search"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def run_all_validations(self) -> bool:
        """Run all validation tests."""
        validations = [
            ("BM25 Search", self.validate_bm25_search),
            ("Semantic Search", self.validate_semantic_search),
            ("Hybrid Search", self.validate_hybrid_search),
            ("Graph Traversal", self.validate_graph_traverse),
            ("Tag Search", self.validate_tag_search),
            ("Keyword Search", self.validate_keyword_search),
            ("Unified Search", self.validate_unified_search)
        ]
        
        all_passed = True
        
        for name, validation_func in validations:
            logger.info(f"\n=== Testing {name} ===")
            try:
                passed = validation_func()
                all_passed = all_passed and passed
                logger.info(f"{name}: {'✅ PASSED' if passed else '❌ FAILED'}")
            except Exception as e:
                all_passed = False
                logger.error(f"{name}: ❌ FAILED - Unexpected error: {e}")
        
        return all_passed
    
    def print_summary(self) -> None:
        """Print a summary of validation results."""
        print("\n===== SEARCH FUNCTION VALIDATION SUMMARY =====")
        
        # Count passed validations
        passed = sum(1 for r in self.results.values() if r.get("success", False))
        total = len(self.results)
        
        print(f"Tests passed: {passed}/{total}\n")
        
        # Print individual results
        for name, result in self.results.items():
            success = result.get("success", False)
            symbol = "✅" if success else "❌"
            error = result.get("error", None)
            result_count = result.get("result_count", 0)
            
            print(f"{symbol} {name.replace('_', ' ').title()} - {result_count} results")
            if error:
                print(f"   Error: {error}")
        
        print("\n==============================================")

def main():
    """Run search function validations."""
    logger.info("Starting search function validation...")
    
    validator = SearchValidator()
    all_passed = validator.run_all_validations()
    validator.print_summary()
    
    if all_passed:
        logger.info("✅ All search functions validated successfully")
        sys.exit(0)
    else:
        logger.error("❌ Some search functions failed validation")
        sys.exit(1)

if __name__ == "__main__":
    main()
