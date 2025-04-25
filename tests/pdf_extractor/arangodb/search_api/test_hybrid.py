#!/usr/bin/env python3
# tests/pdf_extractor/arangodb/search_api/test_hybrid.py

import sys
import json
from typing import Dict, Any, Tuple
import unittest
from loguru import logger

from arango.database import StandardDatabase

from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.search_api.hybrid import hybrid_search

class TestHybridSearch(unittest.TestCase):
    """Test class for hybrid search functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Configure logging
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        
        # Connect to ArangoDB
        cls.client = connect_arango()
        cls.db = ensure_database(cls.client)
        
        # Path to test fixture
        cls.fixture_path = "src/test_fixtures/hybrid_search_expected_20250422_181117.json"
    
    def validate_hybrid_search(self, search_results: Dict[str, Any], fixture_path: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
        """
        Validate hybrid search results against known good fixture data.
        
        Args:
            search_results: The results returned from hybrid_search
            fixture_path: Path to the fixture file containing expected results
            
        Returns:
            Tuple of (validation_passed, validation_failures)
        """
        # Load fixture data
        try:
            with open(fixture_path, "r") as f:
                expected_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load fixture data: {e}")
            return False, {"fixture_loading_error": {"expected": "Valid JSON file", "actual": str(e)}}
        
        # Track all validation failures
        validation_failures = {}
        
        # Structural validation
        if "results" not in search_results:
            validation_failures["missing_results"] = {
                "expected": "Results field present",
                "actual": "Results field missing"
            }
            return False, validation_failures
        
        # Validate result count
        if "count" in expected_data and search_results.get("count") != expected_data.get("count"):
            validation_failures["result_count"] = {
                "expected": expected_data.get("count"),
                "actual": search_results.get("count")
            }
        
        # Validate total count
        if "total" in expected_data and search_results.get("total") != expected_data.get("total"):
            validation_failures["total_count"] = {
                "expected": expected_data.get("total"),
                "actual": search_results.get("total")
            }
        
        # Validate the actual result data
        expected_results = expected_data.get("results", [])
        actual_results = search_results.get("results", [])
        
        # First check if we have the same number of results
        if len(expected_results) != len(actual_results):
            validation_failures["results_length"] = {
                "expected": len(expected_results),
                "actual": len(actual_results)
            }
        
        # Now validate individual results
        min_length = min(len(expected_results), len(actual_results))
        for i in range(min_length):
            expected_result = expected_results[i]
            actual_result = actual_results[i]
            
            # Check document keys match
            expected_key = expected_result.get("doc", {}).get("_key", "")
            actual_key = actual_result.get("doc", {}).get("_key", "")
            
            if expected_key != actual_key:
                validation_failures[f"result_{i}_doc_key"] = {
                    "expected": expected_key,
                    "actual": actual_key
                }
            
            # Check RRF scores approximately match (floating point comparison)
            if "rrf_score" in expected_result and "rrf_score" in actual_result:
                expected_score = expected_result["rrf_score"]
                actual_score = actual_result["rrf_score"]
                
                # Allow small differences in scores due to floating point precision
                if abs(expected_score - actual_score) > 0.01:
                    validation_failures[f"result_{i}_rrf_score"] = {
                        "expected": expected_score,
                        "actual": actual_score
                    }
            
            # Check document content
            for key in expected_result.get("doc", {}):
                if key not in actual_result.get("doc", {}):
                    validation_failures[f"result_{i}_doc_missing_{key}"] = {
                        "expected": f"{key} field present",
                        "actual": f"{key} field missing"
                    }
                elif actual_result["doc"][key] != expected_result["doc"][key]:
                    validation_failures[f"result_{i}_doc_{key}"] = {
                        "expected": expected_result["doc"][key],
                        "actual": actual_result["doc"][key]
                    }
        
        return len(validation_failures) == 0, validation_failures
    
    def test_hybrid_search(self):
        """Test hybrid search function."""
        # Run a test hybrid search
        test_query = "python error"  # Known query that should match fixture results
        search_results = hybrid_search(
            db=self.db,
            query_text=test_query,
            top_n=3
        )
        
        # Validate the results
        validation_passed, validation_failures = self.validate_hybrid_search(search_results, self.fixture_path)
        
        # Check if validation passed
        self.assertTrue(validation_passed, f"Hybrid search validation failed: {validation_failures}")
        
        # Additional specific test assertions
        self.assertIn("results", search_results, "Results field should be present")
        self.assertIn("total", search_results, "Total field should be present")
        self.assertGreaterEqual(len(search_results["results"]), 1, "Should have at least one result")

if __name__ == "__main__":
    unittest.main()
