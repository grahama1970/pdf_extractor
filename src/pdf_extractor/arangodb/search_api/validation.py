# validation.py
import sys
import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger

def create_fixture_dir():
    """Create fixtures directory if it doesn't exist"""
    fixtures_dir = os.path.join(
        Path(__file__).parent.parent.parent.parent.parent,
        "test_fixtures",
        "arangodb"
    )
    os.makedirs(fixtures_dir, exist_ok=True)
    return fixtures_dir

def validate_search_results(
    results,
    expected,
    score_field="bm25_score"
):
    """
    Validate search results against expected data
    """
    validation_failures = {}
    
    # Structural validation
    if len(results.get("results", [])) != len(expected.get("results", [])):
        validation_failures["result_count"] = {
            "expected": len(expected.get("results", [])),
            "actual": len(results.get("results", []))
        }
    
    # Metadata validation
    for key in ["total", "offset", "limit"]:
        if results.get(key) != expected.get(key):
            validation_failures[f"metadata_{key}"] = {
                "expected": expected.get(key),
                "actual": results.get(key)
            }
    
    # Content validation
    for i, (actual_result, expected_result) in enumerate(
        zip(results.get("results", []), expected.get("results", []))
    ):
        # Skip if either is None
        if not actual_result or not expected_result:
            continue
            
        # Check score with tolerance for floating point
        actual_score = actual_result.get(score_field, 0.0)
        expected_score = expected_result.get(score_field, 0.0)
        
        if abs(actual_score - expected_score) > 0.001:
            validation_failures[f"result_{i}_{score_field}"] = {
                "expected": expected_score,
                "actual": actual_score
            }
        
        # Check document key
        actual_key = actual_result.get("doc", {}).get("_key", "")
        expected_key = expected_result.get("doc", {}).get("_key", "")
        
        if actual_key != expected_key:
            validation_failures[f"result_{i}_doc_key"] = {
                "expected": expected_key,
                "actual": actual_key
            }
    
    return validation_failures

def create_fixture(
    results,
    params,
    filepath
):
    """
    Create a test fixture file with results and params
    """
    fixture_data = {
        "params": params,
        "results": results
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(fixture_data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to create fixture: {str(e)}")
        return False

def load_fixture(filepath):
    """
    Load a test fixture file
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load fixture: {str(e)}")
        return None

def report_validation_results(validation_failures):
    """
    Report validation results according to VALIDATION_REQUIREMENTS.md
    """
    if validation_failures:
        logger.error("❌ VALIDATION FAILED - Results don't match expected values")
        logger.error(f"FAILURE DETAILS:")
        for field, details in validation_failures.items():
            logger.error(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        logger.error(f"Total errors: {len(validation_failures)} fields mismatched")
        return False
    else:
        logger.success("✅ VALIDATION COMPLETE - All results match expected values")
        return True
