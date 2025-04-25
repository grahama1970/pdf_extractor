#!/usr/bin/env python3
# tests/pdf_extractor/arangodb/search_api/test_bm25.py

import sys
import json
from typing import Dict, Any, Tuple
import unittest
from loguru import logger

from arango.database import StandardDatabase

from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.search_api.bm25 import bm25_search

class TestBM25Search(unittest.TestCase):
    """Test class for BM25 search functionality."""
    
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
        cls.fixture_path = "src/test_fixtures/bm25_search_expected_20250422_181050.json"
    
    def validate_bm25_search(self, search_results: Dict[str, Any], fixture_path: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
        """
        Validate BM25 search results against known good fixture data.
        
        Args:
            search_results: The results returned from bm25_search
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
        if len(search_results["results"]) != len(expected_data["results"]):
            validation_failures["result_count"] = {
                "expected": len(expected_data["results"]),
                "actual": len(search_results["results"])
            }
        
        # Validate total count
        if search_results.get("total") != expected_data.get("total"):
            validation_failures["total_count"] = {
                "expected": expected_data.get("total"),
                "actual": search_results.get("total")
            }
        
        # Validate offset
        if search_results.get("offset") != expected_data.get("offset"):
            validation_failures["offset"] = {
                "expected": expected_data.get("offset"),
                "actual": search_results.get("offset")
            }
            
        # Content validation - compare actual results with expected
        for i, (expected_result, actual_result) in enumerate(
            zip(expected_data["results"], search_results["results"])
        ):
            # Check structure of each result
            if "doc" not in actual_result:
                validation_failures[f"result_{i}_missing_doc"] = {
                    "expected": "doc field present",
                    "actual": "doc field missing"
                }
                continue
                
            # Check document fields
            for key in expected_result["doc"]:
                if key not in actual_result["doc"]:
                    validation_failures[f"result_{i}_doc_missing_{key}"] = {
                        "expected": f"{key} field present",
                        "actual": f"{key} field missing"
                    }
                elif actual_result["doc"][key] != expected_result["doc"][key]:
                    validation_failures[f"result_{i}_doc_{key}"] = {
                        "expected": expected_result["doc"][key],
                        "actual": actual_result["doc"][key]
                    }
            
            # Check score field - might be called 'score' or 'bm25_score'
            expected_score = expected_result.get("bm25_score", expected_result.get("score"))
            actual_score = actual_result.get("bm25_score", actual_result.get("score"))
            
            # For floating point scores, use approximate comparison
            if expected_score is not None and actual_score is not None:
                # Allow small differences in scores due to floating point precision
                if abs(expected_score - actual_score) > 0.01:
                    validation_failures[f"result_{i}_score"] = {
                        "expected": expected_score,
                        "actual": actual_score
                    }
            elif expected_score != actual_score:
                validation_failures[f"result_{i}_score"] = {
                    "expected": expected_score,
                    "actual": actual_score
                }
        
        return len(validation_failures) == 0, validation_failures
    
    def test_bm25_search(self):
        """Test BM25 search function."""
        # Run a test search query
        test_query = "python error"  # Known query that should match fixture results
        search_results = bm25_search(
            db=self.db,
            query_text=test_query,
            top_n=3,  # Match expected number in fixture
            min_score=0.0
        )
        
        # Validate the results
        validation_passed, validation_failures = self.validate_bm25_search(search_results, self.fixture_path)
        
        # Check if validation passed
        self.assertTrue(validation_passed, f"BM25 search validation failed: {validation_failures}")
        
        # Additional specific test assertions
        self.assertIn("results", search_results, "Results field should be present")
        self.assertIn("total", search_results, "Total field should be present")
        self.assertGreaterEqual(len(search_results["results"]), 1, "Should have at least one result")

if __name__ == "__main__":
    unittest.main()
