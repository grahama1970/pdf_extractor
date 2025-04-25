"""
JSON-specific validation utilities for LLM responses.

This module provides validators for checking JSON structure and content
in LLM responses.


Links:
  - Python json module: https://docs.python.org/3/library/json.html

Sample Input (json_validator):
  - response: '```json\n{"key": "value"}\n```'

Sample Output (json_validator):
  - True

Sample Input (required_fields_validator factory):
  - required_fields: ["name", "age"]

Sample Input (generated validator):
  - response: '{"name": "Test", "city": "Testville"}'

Sample Output (generated validator):
  - "Missing required fields: age"


"""
import json
from typing import Any, List, Union, Callable, Optional

from pdf_extractor.llm_integration.validation_utils.base import extract_content


def json_validator() -> Callable[[Any], Union[bool, str]]:
    """
    Creates a validator that checks if a response contains valid JSON.
    
    Returns:
        A validation function that returns True if JSON is valid or an error message string if invalid
    """
    def validate(response: Any) -> Union[bool, str]:
        content = extract_content(response)
        
        # Try to find and parse JSON in the content
        try:
            # First try direct JSON parsing
            json.loads(content)
            return True
        except json.JSONDecodeError:
            # Then check for code blocks with JSON
            if "```json" in content and "```" in content.split("```json", 1)[1]:
                try:
                    json_str = content.split("```json", 1)[1].split("```", 1)[0].strip()
                    json.loads(json_str)
                    return True
                except (json.JSONDecodeError, IndexError):
                    pass
            
            # Now check for other code block formats
            if "```" in content:
                blocks = content.split("```")
                # Check each block that might be JSON (odd-indexed blocks are inside ```)
                for i in range(1, len(blocks), 2):
                    if i < len(blocks):
                        # Remove language identifier if present
                        block = blocks[i]
                        if "\n" in block:
                            # Remove language identifier line
                            block = "\n".join(block.split("\n")[1:])
                        
                        try:
                            json.loads(block.strip())
                            return True
                        except json.JSONDecodeError:
                            continue
            
            return "Response must contain valid JSON"
    
    return validate


def required_fields_validator(required_fields: List[str]) -> Callable[[Any], Union[bool, str]]:
    """
    Creates a validator that checks if a JSON response contains all required fields.
    
    Args:
        required_fields: List of field names that must be present in the JSON
        
    Returns:
        A validation function that returns True if all fields are present or an error message if not
    """
    def validate(response: Any) -> Union[bool, str]:
        content = extract_content(response)
        
        # Try to extract JSON from different formats
        json_data = None
        
        # Try direct parsing first
        try:
            json_data = json.loads(content)
        except json.JSONDecodeError:
            # Check for JSON in code blocks
            if "```json" in content and "```" in content.split("```json", 1)[1]:
                try:
                    json_str = content.split("```json", 1)[1].split("```", 1)[0].strip()
                    json_data = json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    pass
            
            # Try other code blocks
            if json_data is None and "```" in content:
                blocks = content.split("```")
                for i in range(1, len(blocks), 2):
                    if i < len(blocks):
                        block = blocks[i]
                        if "\n" in block:
                            block = "\n".join(block.split("\n")[1:])
                        
                        try:
                            json_data = json.loads(block.strip())
                            break
                        except json.JSONDecodeError:
                            continue
        
        # If we couldn't extract JSON, return error
        if json_data is None:
            return "Could not parse JSON from response"
        
        # Check for required fields
        missing_fields = [field for field in required_fields if field not in json_data]
        if missing_fields:
            return f"Missing required fields: {', '.join(missing_fields)}"
        
        return True
    
    return validate


# --- Main Execution Guard ---
if __name__ == "__main__":
   import sys
   import json # Needed for test data
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
   # Import logger here if not already imported globally in the file
   try:
       from loguru import logger
   except ImportError:
       # Basic print fallback if loguru isn't available in this context
       class PrintLogger:
           def info(self, msg): print(f"INFO: {msg}")
           def error(self, msg, **kwargs): print(f"ERROR: {msg}")
           def debug(self, msg): print(f"DEBUG: {msg}")
           def warning(self, msg): print(f"WARNING: {msg}")
           def remove(self): pass
           def add(self, *args, **kwargs): pass
       logger = PrintLogger()

   logger.remove()
   logger.add(sys.stderr, level="INFO") # Use INFO for verification summary

   logger.info("Starting JSON Validators Standalone Verification...")

   # --- Define Verification Logic ---
   all_tests_passed = True
   all_failures = {}

   # --- Test Data ---
   VALID_JSON_OBJ = {"key": "value", "number": 123}
   VALID_JSON_STR = json.dumps(VALID_JSON_OBJ)
   VALID_JSON_CODEBLOCK = f"```json\n{VALID_JSON_STR}\n```"
   VALID_JSON_OTHER_CODEBLOCK = f"```\n{VALID_JSON_STR}\n```"
   INVALID_JSON_STR = '{"key": "value", "number": 123,}' # Trailing comma
   NON_JSON_STR = "This is just plain text."
   REQUIRED_FIELDS = ["key", "number"]
   MISSING_FIELDS_JSON_OBJ = {"key": "value"}
   MISSING_FIELDS_JSON_STR = json.dumps(MISSING_FIELDS_JSON_OBJ)

   test_cases = [
       # json_validator tests
       {"id": "json_valid_direct", "validator": json_validator(), "response": VALID_JSON_STR, "should_pass": True},
       {"id": "json_valid_codeblock", "validator": json_validator(), "response": VALID_JSON_CODEBLOCK, "should_pass": True},
       {"id": "json_valid_other_codeblock", "validator": json_validator(), "response": VALID_JSON_OTHER_CODEBLOCK, "should_pass": True},
       {"id": "json_invalid_direct", "validator": json_validator(), "response": INVALID_JSON_STR, "should_pass": False},
       {"id": "json_invalid_text", "validator": json_validator(), "response": NON_JSON_STR, "should_pass": False},

       # required_fields_validator tests
       {"id": "fields_valid_direct", "validator": required_fields_validator(REQUIRED_FIELDS), "response": VALID_JSON_STR, "should_pass": True},
       {"id": "fields_valid_codeblock", "validator": required_fields_validator(REQUIRED_FIELDS), "response": VALID_JSON_CODEBLOCK, "should_pass": True},
       {"id": "fields_invalid_missing", "validator": required_fields_validator(REQUIRED_FIELDS), "response": MISSING_FIELDS_JSON_STR, "should_pass": False},
       {"id": "fields_invalid_not_json", "validator": required_fields_validator(REQUIRED_FIELDS), "response": NON_JSON_STR, "should_pass": False}, # Expect "Could not parse JSON"
   ]

   # --- Run Verification ---
   logger.info("--- Testing JSON Validators ---")
   for test in test_cases:
       test_id = test["id"]
       logger.debug(f"Running test: {test_id}")
       try:
           validator_func = test["validator"]
           validation_result = validator_func(test["response"])
           passed = validation_result is True
           failures = {} if passed else {test_id: {"expected": True, "actual": validation_result}}

           if passed == test["should_pass"]:
               logger.info(f"✅ {test_id}: Passed as expected (Expected Pass: {test['should_pass']})")
           else:
               all_tests_passed = False
               # Include the specific error message from the validator if it failed when expected
               failure_detail = validation_result if not test['should_pass'] and not passed else f"Pass={passed}"
               current_failure = {test_id: {"expected": f"Pass={test['should_pass']}", "actual": failure_detail}}
               all_failures.update(current_failure)
               logger.error(f"❌ {test_id}: Failed (Expected Pass: {test['should_pass']}, Got Pass: {passed}) Details: {current_failure}")
       except Exception as e:
            all_tests_passed = False
            all_failures[f"{test_id}_exception"] = {"expected": "Clean run or expected failure message", "actual": f"Exception: {e}"}
            logger.error(f"❌ {test_id}: Threw unexpected exception: {e}", exc_info=True)


   # --- Report Results ---
   exit_code = report_validation_results(
       validation_passed=all_tests_passed,
       validation_failures=all_failures,
       exit_on_failure=False # Let sys.exit handle it
   )

   logger.info(f"JSON Validators Standalone Verification finished with exit code: {exit_code}")
   sys.exit(exit_code)

# --- Original __main__ block removed ---
# if __name__ == "__main__":
#     print("This module should be imported, not run directly.")
#     import sys
#     sys.exit(1)