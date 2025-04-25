"""
Validation reporting utilities.

This module provides functions for reporting validation results in a
standardized format as specified in VALIDATION_REQUIREMENTS.md.


Links:
  - loguru: https://loguru.readthedocs.io/en/stable/

Sample Input (report_validation_results):
- validation_passed: False
- validation_failures: {"field_A": {"expected": 10, "actual": 12}}

Sample Output (report_validation_results - logs):
  ERROR: ❌ VALIDATION FAILED - Results don't match expected values
  ERROR: FAILURE DETAILS:
  ERROR:   - field_A: Expected: 10, Got: 12
  ERROR: Total errors: 1 fields mismatched

Sample Input (calculate_validation_accuracy):
- result_counts: {"matching_chars": 950, "total_chars": 1000}

Sample Output (calculate_validation_accuracy):
- {"text_accuracy": 95.0, "overall_accuracy": 95.0}
"""

from typing import Dict, Any, Tuple, Optional
import sys
from loguru import logger


def report_validation_results(
    validation_passed: bool,
    validation_failures: Dict[str, Dict[str, Any]],
    exit_on_failure: bool = False,
    log_level: str = "INFO"
) -> int:
    """
    Report validation results in the format expected by VALIDATION_REQUIREMENTS.md.
    
    Follows the exact formatting specified in the requirements:
    - Success: "✅ VALIDATION COMPLETE - All PDF extraction results match expected values"
    - Failure: "❌ VALIDATION FAILED - Results don't match expected values"
      followed by detailed error information
    
    Args:
        validation_passed: Boolean indicating if validation passed
        validation_failures: Dictionary of validation failures
        exit_on_failure: If True, exit the program with code 1 on failure
        log_level: Logging level to use for output ("INFO", "ERROR", etc.)
        
    Returns:
        System exit code (0 for success, 1 for failure)
    """
    if validation_passed:
        message = "✅ VALIDATION COMPLETE - All PDF extraction results match expected values"
        if log_level.upper() == "ERROR":
            logger.error(message)
        else:
            logger.info(message)
        return 0
    else:
        logger.error("❌ VALIDATION FAILED - Results don't match expected values")
        logger.error("FAILURE DETAILS:")
        for field, details in validation_failures.items():
            logger.error(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        logger.error(f"Total errors: {len(validation_failures)} fields mismatched")
        
        if exit_on_failure:
            sys.exit(1)
        
        return 1


def calculate_validation_accuracy(result_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate validation accuracy metrics.
    
    Implements the accuracy measurement requirements from VALIDATION_REQUIREMENTS.md.
    
    Args:
        result_counts: Dictionary with counts of matches and totals
            Example: {
                "matching_chars": 450,
                "total_chars": 500,
                "matching_elements": 9,
                "total_elements": 10,
                "matching_cells": 18,
                "total_cells": 20
            }
            
    Returns:
        Dictionary of accuracy percentages
    """
    accuracy_metrics = {}
    
    # Text extraction accuracy
    if "matching_chars" in result_counts and "total_chars" in result_counts:
        if result_counts["total_chars"] > 0:
            accuracy_metrics["text_accuracy"] = (
                result_counts["matching_chars"] / result_counts["total_chars"] * 100
            )
    
    # Structure extraction accuracy
    if "matching_elements" in result_counts and "total_elements" in result_counts:
        if result_counts["total_elements"] > 0:
            accuracy_metrics["structure_accuracy"] = (
                result_counts["matching_elements"] / result_counts["total_elements"] * 100
            )
    
    # Table extraction accuracy
    if "matching_cells" in result_counts and "total_cells" in result_counts:
        if result_counts["total_cells"] > 0:
            accuracy_metrics["table_accuracy"] = (
                result_counts["matching_cells"] / result_counts["total_cells"] * 100
            )
    
    # Overall accuracy (if applicable)
    if len(accuracy_metrics) > 0:
        accuracy_metrics["overall_accuracy"] = sum(accuracy_metrics.values()) / len(accuracy_metrics)
    
    return accuracy_metrics


def report_accuracy_metrics(accuracy_metrics: Dict[str, float]) -> None:
    """
    Report accuracy metrics in a readable format.
    
    Args:
        accuracy_metrics: Dictionary of accuracy percentages
    """
    logger.info("ACCURACY METRICS:")
    for metric_name, accuracy in accuracy_metrics.items():
        logger.info(f"  - {metric_name}: {accuracy:.2f}%")


def format_validation_error(field: str, expected: Any, actual: Any) -> str:
    """
    Format a validation error message in the standard format.
    
    Args:
        field: Name of the field or aspect that failed validation
        expected: Expected value
        actual: Actual value
        
    Returns:
        Formatted error message
    """
    return f"  - {field}: Expected: {expected}, Got: {actual}"


# --- Main Execution Guard ---
if __name__ == "__main__":
   # Note: report_validation_results is already defined in this file
   # Configure Loguru for verification output
   logger.remove()
   logger.add(sys.stderr, level="INFO") # Use INFO for verification summary

   logger.info("Starting Validation Reporting Utilities Standalone Verification...")

   # --- Define Verification Logic ---
   all_tests_passed = True
   all_failures = {}

   # --- Test Data ---
   SUCCESS_FAILURES = {}
   FAILURE_DETAILS = {
       "field_A": {"expected": 10, "actual": 12},
       "field_B": {"expected": "pass", "actual": "fail"}
   }
   ACCURACY_COUNTS = {
       "matching_chars": 950, "total_chars": 1000,
       "matching_elements": 8, "total_elements": 10,
       "matching_cells": 45, "total_cells": 50
   }
   EXPECTED_ACCURACY = {
       "text_accuracy": 95.0,
       "structure_accuracy": 80.0,
       "table_accuracy": 90.0,
       "overall_accuracy": (95.0 + 80.0 + 90.0) / 3
   }

   # --- Run Verification ---

   # Test 1: report_validation_results (Success Case)
   logger.info("--- Testing report_validation_results (Success) ---")
   try:
       # We expect this to log success and return 0
       exit_code_success = report_validation_results(
           validation_passed=True,
           validation_failures=SUCCESS_FAILURES,
           exit_on_failure=False # Important for testing
       )
       if exit_code_success == 0:
           logger.info("✅ report_validation_results(Success): Returned correct exit code 0.")
       else:
           all_tests_passed = False
           all_failures["report_success_exit_code"] = {"expected": 0, "actual": exit_code_success}
           logger.error(f"❌ report_validation_results(Success): Returned incorrect exit code {exit_code_success}.")
   except Exception as e:
       all_tests_passed = False
       all_failures["report_success_exception"] = {"expected": "Clean run", "actual": f"Exception: {e}"}
       logger.error(f"❌ report_validation_results(Success): Threw unexpected exception: {e}", exc_info=True)

   # Test 2: report_validation_results (Failure Case)
   logger.info("--- Testing report_validation_results (Failure) ---")
   try:
       # We expect this to log failure details and return 1
       exit_code_failure = report_validation_results(
           validation_passed=False,
           validation_failures=FAILURE_DETAILS,
           exit_on_failure=False, # Important for testing
           log_level="ERROR" # Ensure errors are logged as errors
       )
       if exit_code_failure == 1:
           logger.info("✅ report_validation_results(Failure): Returned correct exit code 1.")
       else:
           all_tests_passed = False
           all_failures["report_failure_exit_code"] = {"expected": 1, "actual": exit_code_failure}
           logger.error(f"❌ report_validation_results(Failure): Returned incorrect exit code {exit_code_failure}.")
       # Manual check: Verify the log output contains the expected failure format (cannot automate easily here)
       logger.info("ℹ️ report_validation_results(Failure): Manual check: Verify log output shows '❌ VALIDATION FAILED...' and failure details.")

   except Exception as e:
       all_tests_passed = False
       all_failures["report_failure_exception"] = {"expected": "Clean run", "actual": f"Exception: {e}"}
       logger.error(f"❌ report_validation_results(Failure): Threw unexpected exception: {e}", exc_info=True)

   # Test 3: calculate_validation_accuracy
   logger.info("--- Testing calculate_validation_accuracy ---")
   try:
       calculated_accuracy = calculate_validation_accuracy(ACCURACY_COUNTS)
       # Compare calculated vs expected (allow for small float differences)
       accuracy_match = True
       temp_failures = {}
       for key, expected_val in EXPECTED_ACCURACY.items():
           if key not in calculated_accuracy:
               accuracy_match = False
               temp_failures[f"accuracy_missing_{key}"] = {"expected": expected_val, "actual": "Missing"}
           elif abs(calculated_accuracy[key] - expected_val) > 0.01: # Tolerance for float comparison
               accuracy_match = False
               temp_failures[f"accuracy_mismatch_{key}"] = {"expected": expected_val, "actual": calculated_accuracy[key]}

       if accuracy_match:
           logger.info("✅ calculate_validation_accuracy: Calculated metrics match expected values.")
           # Also test reporting the accuracy
           report_accuracy_metrics(calculated_accuracy)
       else:
           all_tests_passed = False
           all_failures.update(temp_failures)
           logger.error(f"❌ calculate_validation_accuracy: Calculated metrics mismatch. Details: {temp_failures}")

   except Exception as e:
       all_tests_passed = False
       all_failures["calculate_accuracy_exception"] = {"expected": "Clean run", "actual": f"Exception: {e}"}
       logger.error(f"❌ calculate_validation_accuracy: Threw unexpected exception: {e}", exc_info=True)

   # Test 4: format_validation_error
   logger.info("--- Testing format_validation_error ---")
   try:
       formatted_error = format_validation_error("test_field", "expected_val", "actual_val")
       expected_format = "  - test_field: Expected: expected_val, Got: actual_val"
       if formatted_error == expected_format:
            logger.info("✅ format_validation_error: Output matches expected format.")
       else:
            all_tests_passed = False
            all_failures["format_error_output"] = {"expected": expected_format, "actual": formatted_error}
            logger.error(f"❌ format_validation_error: Output mismatch.")

   except Exception as e:
       all_tests_passed = False
       all_failures["format_error_exception"] = {"expected": "Clean run", "actual": f"Exception: {e}"}
       logger.error(f"❌ format_validation_error: Threw unexpected exception: {e}", exc_info=True)


   # --- Final Report ---
   # Use the function from this module itself to report the overall verification status
   final_exit_code = report_validation_results(
       validation_passed=all_tests_passed,
       validation_failures=all_failures,
       exit_on_failure=False # Let sys.exit handle it
   )

   logger.info(f"Validation Reporting Utilities Standalone Verification finished with exit code: {final_exit_code}")
   sys.exit(final_exit_code)


# --- Original __main__ block removed ---
# if __name__ == "__main__":
#     print("This module should be imported, not run directly.")
#     sys.exit(1)