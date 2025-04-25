"""
Validation Utilities for PDF Extractor ArangoDB Integration.

This module provides utility functions for validating search results against expected outputs.
It includes functions for loading and saving fixtures, comparing results, and reporting validation outcomes.

Third-Party Package Documentation:
- loguru: https://github.com/Delgan/loguru

Sample Input:
Expected search results and actual search results dictionaries

Expected Output:
Validation report indicating if results match expectations and details of any discrepancies
"""
import os
import json
import sys
from typing import Dict, Any, List, Tuple
from pathlib import Path
from loguru import logger

# Path to fixtures
FIXTURES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "test_fixtures")
)

def ensure_fixtures_dir():
    """Ensure fixtures directory exists."""
    Path(FIXTURES_DIR).mkdir(parents=True, exist_ok=True)

def save_fixture(fixture_name: str, data: Dict) -> str:
    """
    Save test data to a fixture file.
    
    Args:
        fixture_name: Name of the fixture file (without extension)
        data: Data to save
        
    Returns:
        Path to the saved fixture file
    """
    ensure_fixtures_dir()
    fixture_path = os.path.join(FIXTURES_DIR, f"{fixture_name}.json")
    with open(fixture_path, "w") as f:
        json.dump(data, f, indent=2)
    return fixture_path

def load_fixture(fixture_name: str) -> Dict:
    """
    Load test data from a fixture file.
    
    Args:
        fixture_name: Name of the fixture file (without extension)
        
    Returns:
        Loaded data, or empty dict if file doesn't exist
    """
    fixture_path = os.path.join(FIXTURES_DIR, f"{fixture_name}.json")
    if not os.path.exists(fixture_path):
        return {}
    with open(fixture_path, "r") as f:
        return json.load(f)

def compare_results(expected: Dict, actual: Dict) -> Tuple[bool, List[Dict]]:
    """
    Compare expected and actual results with field-by-field comparison.
    
    Args:
        expected: Expected results
        actual: Actual results
        
    Returns:
        Tuple of (passed, validation_failures)
    """
    validation_failures = []
    
    # Check total count
    if expected.get("total", 0) != actual.get("total", 0):
        validation_failures.append({
            "field": "total",
            "expected": expected.get("total", 0),
            "actual": actual.get("total", 0)
        })
    
    # Check number of results
    expected_results = expected.get("results", [])
    actual_results = actual.get("results", [])
    if len(expected_results) != len(actual_results):
        validation_failures.append({
            "field": "results_count",
            "expected": len(expected_results),
            "actual": len(actual_results)
        })
    
    # Check first result key fields (if any results exist)
    if expected_results and actual_results:
        expected_keys = set(expected_results[0].keys())
        actual_keys = set(actual_results[0].keys())
        if expected_keys != actual_keys:
            validation_failures.append({
                "field": "result_fields",
                "expected": list(expected_keys),
                "actual": list(actual_keys)
            })
        
        # Compare first result content for each field
        common_keys = expected_keys.intersection(actual_keys)
        for key in common_keys:
            if expected_results[0][key] != actual_results[0][key]:
                if key == "doc":
                    # Special handling for document field which is usually a dict
                    if isinstance(expected_results[0][key], dict) and isinstance(actual_results[0][key], dict):
                        # Check _key field which should always be present
                        if expected_results[0][key].get("_key") != actual_results[0][key].get("_key"):
                            validation_failures.append({
                                "field": f"first_result.{key}._key",
                                "expected": expected_results[0][key].get("_key"),
                                "actual": actual_results[0][key].get("_key")
                            })
                else:
                    # For non-document fields like scores
                    validation_failures.append({
                        "field": f"first_result.{key}",
                        "expected": expected_results[0][key],
                        "actual": actual_results[0][key]
                    })
    
    return len(validation_failures) == 0, validation_failures

def report_validation(passed: bool, failures: List[Dict], name: str):
    """
    Report validation results with detailed error reporting.
    
    Args:
        passed: Whether validation passed
        failures: List of validation failures
        name: Name of the validation for reporting
        
    Returns:
        True if passed, False otherwise
    """
    if passed:
        logger.info(f"✅ {name} validation PASSED!")
        return True
    else:
        logger.error(f"❌ {name} validation FAILED!")
        logger.error(f"FAILURE DETAILS:")
        for failure in failures:
            logger.error(f"  - {failure['field']}: Expected: {failure['expected']}, Got: {failure['actual']}")
        logger.error(f"Total errors: {len(failures)} fields mismatched")
        return False
