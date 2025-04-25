"""
Base validation utilities for LLM responses.

This module provides core validation functionality including content extraction,
result validation, and basic validation strategies.

Following the Minimalist Architecture principle from CODING_PRINCIPLES.md:
This module uses a functional approach rather than class-based architecture.


Links:
- rapidfuzz: https://github.com/maxbachmann/RapidFuzz
- loguru: https://loguru.readthedocs.io/en/stable/

Sample Input (validate_results):
- actual_result: '```json\n{"field": "value"}\n```'
- expected_fixture: {"field": "value"}

Sample Output (validate_results):
- (True, {})

Sample Input (keyword_validator):
- keyword: "required"

Sample Output (keyword_validator usage):
- validator("This response has the required keyword.") -> True
- validator("This response is missing it.") -> "Response must contain the keyword 'required'"
"""

import json
import regex as re
from loguru import logger
from typing import Any, Dict, List, Tuple, Union, Callable, Optional
from rapidfuzz import process, fuzz # Import rapidfuzz components


def extract_content(response: Any) -> str:
    """
    Extract content string from various response formats.
    
    Handles different response types including dictionaries, strings,
    and structured objects like ModelResponse.
    
    Args:
        response: The response object to extract content from
        
    Returns:
        The extracted content as a string
    """
    # Handle dictionary responses
    if isinstance(response, dict):
        # Try common key patterns
        for key in ["content", "response", "text", "output", "message"]:
            if key in response:
                return response[key]
        
        # Check for LiteLLM format
        if "choices" in response and response["choices"]:
            try:
                choices = response["choices"]
                if isinstance(choices[0], dict):
                    if "message" in choices[0]:
                        if isinstance(choices[0]["message"], dict):
                            if "content" in choices[0]["message"]:
                                return choices[0]["message"]["content"]
            except (IndexError, KeyError, TypeError):
                pass
    
    # Check for ModelResponse objects
    if hasattr(response, "choices"):
        try:
            choices = getattr(response, "choices", [])
            if choices and hasattr(choices[0], "message"):
                message = getattr(choices[0], "message")
                if hasattr(message, "content"):
                    return getattr(message, "content")
        except (IndexError, AttributeError):
            pass
    
    # If it's a string, return it directly
    if isinstance(response, str):
        return response
    
    # Fall back to string representation
    return str(response)


def validate_results(actual_result: Any, expected_fixture: Any) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validates actual results against expected fixture data.
    
    Implements the Core Validation Principle from VALIDATION_REQUIREMENTS.md:
    "Validation must verify ACTUAL RESULTS match EXPECTED RESULTS, not just check for absence of errors!"
    
    Args:
        actual_result: The result to validate
        expected_fixture: The expected data to compare against
        
    Returns:
        Tuple of (validation_passed, validation_failures)
            validation_passed: Boolean indicating if validation passed
            validation_failures: Dictionary of validation failures with expected vs. actual values
    """
    validation_failures = {}
    
    # Extract content from the actual result
    actual_content = extract_content(actual_result)
    
    # Handle different expected fixture types
    if isinstance(expected_fixture, dict):
        # Try to extract JSON from the content if needed
        actual_data = None
        if isinstance(actual_content, str):
            try:
                # First try to parse as direct JSON
                actual_data = json.loads(actual_content)
            except json.JSONDecodeError:
                # Then look for JSON in code blocks
                if "```json" in actual_content and "```" in actual_content.split("```json", 1)[1]:
                    try:
                        json_str = actual_content.split("```json", 1)[1].split("```", 1)[0].strip()
                        actual_data = json.loads(json_str)
                    except (json.JSONDecodeError, IndexError):
                        pass
        
        # If actual data is not a dict after attempts, report error
        if actual_data is None or not isinstance(actual_data, dict):
            validation_failures["format"] = {
                "expected": "JSON dictionary",
                "actual": f"{type(actual_content).__name__}"
            }
            return False, validation_failures
        
        # Compare expected fields
        for key, expected_value in expected_fixture.items():
            if key not in actual_data:
                validation_failures[f"missing_field_{key}"] = {
                    "expected": expected_value,
                    "actual": "FIELD MISSING"
                }
            elif actual_data[key] != expected_value:
                validation_failures[f"field_value_{key}"] = {
                    "expected": expected_value,
                    "actual": actual_data[key]
                }
    else:
        # Text comparison for non-dict expectations
        if actual_content != expected_fixture:
            truncated_expected = str(expected_fixture)
            truncated_actual = str(actual_content)
            
            # Truncate for readability if too long
            if len(truncated_expected) > 100:
                truncated_expected = truncated_expected[:100] + "..."
            if len(truncated_actual) > 100:
                truncated_actual = truncated_actual[:100] + "..."
                
            validation_failures["content"] = {
                "expected": truncated_expected,
                "actual": truncated_actual
            }
    
    return len(validation_failures) == 0, validation_failures


def load_fixture(fixture_path: str) -> Any:
    """
    Load test fixture data from file.
    
    Following VALIDATION_REQUIREMENTS.md guidelines on using pre-processed
    fixtures for validation.
    
    Args:
        fixture_path: Path to the fixture file
        
    Returns:
        The fixture data or None if loading failed
    """
    try:
        with open(fixture_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading fixture from {fixture_path}: {e}")
        return None


def keyword_validator(keyword: str) -> Callable[[Any], Union[bool, str]]:
    """
    Creates a validator that checks if a response contains a specific keyword.
    
    Args:
        keyword: The keyword to search for in the response
        
    Returns:
        A validation function that returns True if valid or an error message string if invalid
    """
    def validate(response: Any) -> Union[bool, str]:
        content = extract_content(response)
        if keyword.lower() in content.lower():
            return True
        return f"Response must contain the keyword '{keyword}'"
    
    return validate


def match_from_list_validator(min_similarity: float = 85.0, delimiter: str = ',') -> Callable:
    """
    Creates a validator that checks if the LLM response best matches an item
    from a list derived from the corpus string, using token set ratio.

    Args:
        min_similarity: Minimum token_set_ratio score (0-100) required for the best match.
        delimiter: The character used to split the corpus_source string into a list of choices.

    Returns:
        A validator function generator.
    """
    def create_validator(corpus: Optional[str] = None) -> Callable[[Any], Union[bool, str]]:
        """
        Creates the actual validator function with the provided corpus string (list of choices).
        """
        def validate(response: Any) -> Union[bool, str]:
            """
            Checks if the response content best matches an item in the corpus list.
            """
            if corpus is None or not isinstance(corpus, str):
                return "Match-from-list validation requires a corpus string (list of choices) to be provided."

            # Parse corpus string into a list of choices
            choices = [item.strip() for item in re.split(f'\\s*{re.escape(delimiter)}\\s*', corpus) if item.strip()] # Use regex split for robustness
            if not choices:
                return "Match-from-list validation: Corpus string did not yield any choices after splitting."

            content = extract_content(response)
            if not content:
                return "Empty response cannot be validated."

            # Find the best match from the choices using token_set_ratio
            # Ensure choices is not empty before calling extractOne
            if not choices:
                 return "Cannot perform match: No valid choices derived from corpus." # Should be caught earlier, but safety check
            
            best_match, score, _ = process.extractOne(content, choices, scorer=fuzz.token_set_ratio)

            logger.debug(f"MatchFromList Val: Comparing '{content}' against choices {choices}. Best match: '{best_match}' with score {score:.1f}% (Threshold: {min_similarity}%)")

            if score >= min_similarity:
                return True
            else:
                return f"Response '{content[:50]}...' did not sufficiently match any item in the list {choices}. Best match '{best_match}' had score {score:.1f}%, threshold {min_similarity}%."

        validate.needs_corpus = True # Still needs the corpus string
        return validate

    return create_validator


# --- Main Execution Guard ---
if __name__ == "__main__":
   import sys
   import os
   # Need to import the reporting function
   try:
       # Adjust relative path based on potential execution context
       from .reporting import report_validation_results
   except ImportError:
       try:
           from reporting import report_validation_results
       except ImportError:
           print("❌ FATAL: Could not import report_validation_results. Ensure it's in the same directory or PYTHONPATH.")
           sys.exit(1)

   # Configure Loguru for verification output
   logger.remove()
   logger.add(sys.stderr, level="INFO") # Use INFO for verification summary

   logger.info("Starting Validation Base Utilities Standalone Verification...")

   # --- Define Verification Logic ---
   all_tests_passed = True
   all_failures = {}

   # --- Test Data ---
   FIXTURE_PATH = "sample_base_fixture.json"
   EXPECTED_FIXTURE_DATA = {
       "expected_field_1": "value1",
       "expected_field_2": 123,
       "expected_field_3": True
   }
   # Test cases for validate_results
   test_cases_validate = [
       # 1. Success: Exact JSON match
       {"id": "validate_json_success", "actual": json.dumps(EXPECTED_FIXTURE_DATA), "expected": EXPECTED_FIXTURE_DATA, "should_pass": True},
       # 2. Success: JSON in code block
       {"id": "validate_json_codeblock_success", "actual": f"```json\n{json.dumps(EXPECTED_FIXTURE_DATA)}\n```", "expected": EXPECTED_FIXTURE_DATA, "should_pass": True},
       # 3. Failure: Mismatched value
       {"id": "validate_json_mismatch_value", "actual": json.dumps({"expected_field_1": "wrong_value", "expected_field_2": 123, "expected_field_3": True}), "expected": EXPECTED_FIXTURE_DATA, "should_pass": False},
       # 4. Failure: Missing field
       {"id": "validate_json_missing_field", "actual": json.dumps({"expected_field_1": "value1", "expected_field_2": 123}), "expected": EXPECTED_FIXTURE_DATA, "should_pass": False},
       # 5. Failure: Not JSON
       {"id": "validate_json_not_json", "actual": "This is just text", "expected": EXPECTED_FIXTURE_DATA, "should_pass": False},
       # 6. Success: Exact text match
       {"id": "validate_text_success", "actual": "Expected text output.", "expected": "Expected text output.", "should_pass": True},
       # 7. Failure: Mismatched text
       {"id": "validate_text_mismatch", "actual": "Actual text output.", "expected": "Expected text output.", "should_pass": False},
       # 8. Success: Keyword validator present
       {"id": "keyword_success", "actual": "The response contains the magic keyword.", "validator": keyword_validator("magic"), "should_pass": True},
       # 9. Failure: Keyword validator absent
       {"id": "keyword_fail", "actual": "The response is missing the word.", "validator": keyword_validator("magic"), "should_pass": False},
       # 10. Success: Match from list validator
       {"id": "match_list_success", "actual": "Option B seems correct", "validator": match_from_list_validator(min_similarity=80)("Option A, Option B, Option C"), "should_pass": True},
       # 11. Failure: Match from list validator (low similarity)
       {"id": "match_list_fail_low_sim", "actual": "Option D is wrong", "validator": match_from_list_validator(min_similarity=80)("Option A, Option B, Option C"), "should_pass": False},
       # 12. Success: Match from list validator (different delimiter)
       {"id": "match_list_success_delim", "actual": "Item 2", "validator": match_from_list_validator(min_similarity=80, delimiter=';')("Item 1; Item 2; Item 3"), "should_pass": True},
       # 13. Failure: Match from list validator (no corpus)
       {"id": "match_list_fail_no_corpus", "actual": "Item 2", "validator": match_from_list_validator(min_similarity=80)(None), "should_pass": False}, # Pass None as corpus
   ]

   # --- Run Verification ---
   logger.info("--- Testing load_fixture ---")
   loaded_data = load_fixture(FIXTURE_PATH)
   if loaded_data == EXPECTED_FIXTURE_DATA:
       logger.info(f"✅ load_fixture: Successfully loaded and matched {FIXTURE_PATH}")
   else:
       all_tests_passed = False
       all_failures["load_fixture_match"] = {"expected": EXPECTED_FIXTURE_DATA, "actual": loaded_data}
       logger.error(f"❌ load_fixture: Failed to load or match {FIXTURE_PATH}")

   loaded_data_fail = load_fixture("non_existent_file.json")
   if loaded_data_fail is None:
       logger.info("✅ load_fixture: Correctly returned None for non-existent file")
   else:
       all_tests_passed = False
       all_failures["load_fixture_nonexistent"] = {"expected": None, "actual": loaded_data_fail}
       logger.error("❌ load_fixture: Did not return None for non-existent file")

   logger.info("--- Testing validate_results and other validators ---")
   for test in test_cases_validate:
       test_id = test["id"]
       logger.debug(f"Running test: {test_id}")
       try:
           if "validator" in test: # Handle specific validator tests
               validator_func = test["validator"]
               validation_result = validator_func(test["actual"])
               passed = validation_result is True
               failures = {} if passed else {test_id: {"expected": True, "actual": validation_result}}
           else: # Handle validate_results tests
               passed, failures = validate_results(test["actual"], test["expected"])

           if passed == test["should_pass"]:
               logger.info(f"✅ {test_id}: Passed as expected (Expected Pass: {test['should_pass']})")
           else:
               all_tests_passed = False
               all_failures.update(failures if failures else {test_id: {"expected": f"Pass={test['should_pass']}", "actual": f"Pass={passed}"}})
               logger.error(f"❌ {test_id}: Failed (Expected Pass: {test['should_pass']}, Got Pass: {passed}) Details: {failures}")
       except Exception as e:
           all_tests_passed = False
           all_failures[f"{test_id}_exception"] = {"expected": "Clean run", "actual": f"Exception: {e}"}
           logger.error(f"❌ {test_id}: Threw unexpected exception: {e}", exc_info=True)


   # --- Report Results ---
   exit_code = report_validation_results(
       validation_passed=all_tests_passed,
       validation_failures=all_failures,
       exit_on_failure=False # Let sys.exit handle it
   )

   # Clean up the sample fixture file
   try:
       os.remove(FIXTURE_PATH)
       logger.info(f"Cleaned up {FIXTURE_PATH}")
   except OSError as e:
       logger.warning(f"Could not clean up {FIXTURE_PATH}: {e}")

   logger.info(f"Validation Base Utilities Standalone Verification finished with exit code: {exit_code}")
   sys.exit(exit_code)

# --- Original __main__ block removed ---
# if __name__ == "__main__":
#     logger.error("This module should be imported, not run directly.")
#     import sys
#     sys.exit(1)