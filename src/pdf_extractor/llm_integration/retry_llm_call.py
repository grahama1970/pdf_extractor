"""
Robust Retry Logic for LLM Calls with Validation.

This module provides the core retry mechanism for LLM calls, leveraging
validation utilities to ensure responses meet requirements before proceeding.

Following the 95/5 rule from CODING_PRINCIPLES.md: This module focuses on the
5% customization built on top of existing LLM call functionality.
"""

import asyncio
from loguru import logger
import sys
from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple, Coroutine, Sequence

# Import validation utilities - Assuming validation_utils is importable
# If it's a local directory, adjust the import path
try:
    # Try relative import if validation_utils is in the same parent directory
    from ..validation_utils.base import extract_content
except ImportError:
    # Fallback if structure is different or running script directly (less ideal)
    try:
        from validation_utils.base import extract_content
    except ImportError:
         # Define a dummy if not found, though this will break functionality
         def extract_content(response: Any) -> str:
              print("WARNING: extract_content dummy function used.")
              return str(response)


class MaxRetriesExceededError(Exception):
    """Raised when the LLM call fails validation after the maximum number of retries."""
    pass


async def retry_llm_call(
    llm_call: Callable[..., Coroutine[Any, Any, Any]], # Type hint for async llm_call
    model: str,
    messages: List[Dict[str, Any]],
    validation_strategies: Sequence[Callable[[Any], Union[bool, str]]], # Use Sequence, Specific Callable hint
    temperature: float = 0.2,
    max_tokens: int = 1000,
    api_base: Optional[str] = None,
    response_format: Optional[Union[str, Type[Any]]] = None, # Specific Type hint
    max_retries: int = 3,
    # Removed corpus parameter
) -> Tuple[Any, int, Optional[Union[bool, List[str]]]]: # Return type includes validation result
    """
    Retry LLM calls until validation passes or max retries is reached.

    Args:
        llm_call: Function that makes the LLM API call
        model: Model name to use
        messages: Initial conversation messages
        validation_strategies: Sequence of ready-to-use validation functions.
        temperature: Temperature for sampling
        max_tokens: Maximum tokens to generate
        api_base: Optional API base URL
        response_format: Optional response format
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (validated response, number of retries needed, validation_result)
        validation_result is True if passed, List[str] of errors if failed, None if exception occurred before validation.

    Raises:
        MaxRetriesExceededError: If validation fails after all retries
    """
    logger.debug(f"Starting LLM retry loop with {len(messages)} messages and {len(validation_strategies)} validators")
    retries = 0
    # Create a copy to avoid modifying the original
    current_messages = messages.copy()
    last_validation_errors: Optional[List[str]] = None # Store errors from the last attempt

    # Removed internal validator instantiation logic

    while retries < max_retries:
        # Add a small delay between retries
        if retries > 0:
            await asyncio.sleep(1.0)

        try:
            # Build parameters for the llm_call
            llm_params = {
                "model": model,
                "messages": current_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Add optional parameters if provided
            if api_base:
                llm_params["api_base"] = api_base
            if response_format:
                llm_params["response_format"] = response_format

            # Create the config for the llm_call
            call_config = {"llm_config": llm_params}

            # Make the call
            logger.debug(f"Attempt {retries + 1}: Calling LLM")
            response = await llm_call(call_config)

            # Apply validation strategies
            validation_errors = []
            for validate in validation_strategies: # Iterate directly
                try:
                    result = validate(response) # Call the validator
                    if result is not True:
                        # Ensure result is a string before appending
                        error_msg = str(result) if not isinstance(result, str) else result
                        validation_errors.append(error_msg)
                except Exception as val_err:
                     # Attempt to get validator name, fallback if needed
                     validator_name = getattr(validate, '__name__', 'Unnamed Validator')
                     logger.error(f"Validation function {validator_name} raised an error: {val_err}", exc_info=True)
                     validation_errors.append(f"Validator {validator_name} failed: {val_err}")

            last_validation_errors = validation_errors # Store errors from this attempt

            # If validation passed, return the response and success status
            if not validation_errors:
                logger.success(f"Attempt {retries + 1}: Validation passed.")
                return response, retries, True # Return True for validation status

            # If validation failed, log and prepare for retry
            logger.warning(f"Attempt {retries + 1}: Validation failed: {last_validation_errors}")

            # Get content from response for feedback
            content = extract_content(response)

            # Add the LLM's response and feedback to the conversation
            current_messages.append({"role": "assistant", "content": content})
            current_messages.append({
                "role": "user",
                "content": f"Your response failed validation with these errors: {', '.join(validation_errors)}. Please correct your response."
            })

        except Exception as e:
            # Log any errors during the LLM call
            logger.error(f"Error during LLM call attempt {retries + 1}: {e}", exc_info=True)
            last_validation_errors = [f"LLM call failed: {e}"] # Store LLM error as validation failure reason
            # Continue to next retry

        retries += 1

    # If we get here, all retries failed
    failure_message = f"Failed to get valid response after {max_retries} attempts. Last errors: {last_validation_errors}"
    logger.error(failure_message)
    # Raise error, including last validation errors in the message
    raise MaxRetriesExceededError(failure_message)
    # The return statement below is unreachable due to the raise.
    # return None, retries, last_validation_errors # type: ignore


# --- Main Execution Guard (Standalone Verification) ---
if __name__ == "__main__":
   import sys
   import asyncio
   # Need reporting for standardized output
   try:
       # Adjust relative path based on potential execution context
       from .validation_utils.reporting import report_validation_results
   except ImportError:
        try:
           # If running from llm_integration directory directly
           sys.path.append(str(Path(__file__).resolve().parent / 'validation_utils'))
           from reporting import report_validation_results
        except ImportError:
           print("❌ FATAL: Could not import report_validation_results.")
           sys.exit(1)

   # Configure Loguru for verification output
   logger.remove()
   logger.add(sys.stderr, level="INFO")

   logger.info("Starting Retry LLM Call Standalone Verification...")

   all_tests_passed = True
   all_failures = {}

   # --- Mock LLM Call Function ---
   mock_call_count = 0
   fail_until_attempt = 2 # Simulate failure on first attempt
   final_valid_response = "Final valid response"
   intermediate_invalid_response = "Invalid intermediate response"

   async def mock_llm_call(config: Dict[str, Any]) -> str:
       """Simulates an LLM call that fails validation initially."""
       global mock_call_count
       mock_call_count += 1
       logger.debug(f"Mock LLM Call: Attempt {mock_call_count}")
       if mock_call_count < fail_until_attempt:
           # Check if feedback is in messages
           last_message = config["llm_config"]["messages"][-1]
           if "failed validation" in last_message.get("content", ""):
                logger.debug("Mock LLM Call: Received validation feedback.")
           return intermediate_invalid_response # Fails validation
       else:
           return final_valid_response # Passes validation

   # --- Mock Validators ---
   def passing_validator(response: Any) -> Union[bool, str]:
       logger.debug("Passing validator called.")
       return True

   def failing_validator(response: Any) -> Union[bool, str]:
       content = extract_content(response)
       logger.debug(f"Failing validator called with content: '{content}'")
       if content == final_valid_response:
           return True
       else:
           return f"Content '{content}' did not match expected '{final_valid_response}'"

   # --- Test Cases ---
   test_cases = [
       {
           "id": "retry_success",
           "llm_call": mock_llm_call,
           "validators": [failing_validator],
           "max_retries": 3,
           "fail_until": 2,
           "expected_retries": 1, # Fails once, succeeds on second attempt (0, 1)
           "expected_result": final_valid_response,
           "expected_validation": True,
           "expect_exception": None,
       },
       {
           "id": "retry_fail_max_retries",
           "llm_call": mock_llm_call,
           "validators": [failing_validator],
           "max_retries": 2,
           "fail_until": 3, # Will always fail within max_retries
           "expected_retries": 2,
           "expected_result": None, # Expect exception
           "expected_validation": [f"Content '{intermediate_invalid_response}' did not match expected '{final_valid_response}'"], # Last error
           "expect_exception": MaxRetriesExceededError,
       },
        {
           "id": "retry_immediate_success",
           "llm_call": mock_llm_call,
           "validators": [passing_validator], # Validator always passes
           "max_retries": 3,
           "fail_until": 1, # Succeeds immediately
           "expected_retries": 0,
           "expected_result": intermediate_invalid_response, # Returns the first response
           "expected_validation": True,
           "expect_exception": None,
       },
   ]

   # --- Run Verification ---
   async def run_all_tests():
       global all_tests_passed, all_failures, mock_call_count, fail_until_attempt

       for test in test_cases:
           test_id = test["id"]
           logger.info(f"--- Running Test: {test_id} ---")
           mock_call_count = 0 # Reset mock counter
           fail_until_attempt = test["fail_until"]
           test_passed = True
           test_failures = {}

           try:
               response, retries, validation_status = await retry_llm_call(
                   llm_call=test["llm_call"],
                   model="test-model",
                   messages=[{"role": "user", "content": "Initial prompt"}],
                   validation_strategies=test["validators"],
                   max_retries=test["max_retries"],
               )

               # Check if an exception was expected but not raised
               if test["expect_exception"] is not None:
                   test_passed = False
                   test_failures["exception_not_raised"] = {"expected": test["expect_exception"].__name__, "actual": "No exception"}
                   logger.error(f"❌ {test_id}: Failed - Expected exception {test['expect_exception'].__name__} was not raised.")

               # Compare results if no exception was expected
               else:
                   if retries != test["expected_retries"]:
                       test_passed = False
                       test_failures["retry_count"] = {"expected": test["expected_retries"], "actual": retries}
                       logger.error(f"❌ {test_id}: Failed - Retry count mismatch.")
                   else:
                        logger.info(f"✅ {test_id}: Retry count matches ({retries}).")

                   actual_result_content = extract_content(response)
                   if actual_result_content != test["expected_result"]:
                        test_passed = False
                        test_failures["result_content"] = {"expected": test["expected_result"], "actual": actual_result_content}
                        logger.error(f"❌ {test_id}: Failed - Result content mismatch.")
                   else:
                        logger.info(f"✅ {test_id}: Result content matches.")

                   if validation_status != test["expected_validation"]:
                        test_passed = False
                        test_failures["validation_status"] = {"expected": test["expected_validation"], "actual": validation_status}
                        logger.error(f"❌ {test_id}: Failed - Validation status mismatch.")
                   else:
                        logger.info(f"✅ {test_id}: Validation status matches.")

           except Exception as e:
               if test["expect_exception"] is None:
                   test_passed = False
                   test_failures["unexpected_exception"] = {"expected": "No exception", "actual": f"{type(e).__name__}: {e}"}
                   logger.error(f"❌ {test_id}: Failed - Unexpected exception raised: {e}", exc_info=True)
               elif not isinstance(e, test["expect_exception"]):
                   test_passed = False
                   test_failures["wrong_exception_type"] = {"expected": test["expect_exception"].__name__, "actual": type(e).__name__}
                   logger.error(f"❌ {test_id}: Failed - Raised wrong exception type. Got {type(e).__name__}, expected {test['expect_exception'].__name__}.", exc_info=True)
               else:
                   # Correct exception was raised, check details if needed (e.g., retry count in message)
                   logger.info(f"✅ {test_id}: Passed - Correctly raised expected exception {type(e).__name__}.")
                   # Optionally check if the error message contains the expected last validation errors
                   if isinstance(e, MaxRetriesExceededError) and isinstance(test["expected_validation"], list):
                       expected_error_substr = "; ".join(test["expected_validation"])
                       if expected_error_substr not in str(e):
                            test_passed = False # Technically passed by raising, but message differs
                            test_failures["exception_message"] = {"expected_contain": expected_error_substr, "actual": str(e)}
                            logger.warning(f"⚠️ {test_id}: Raised correct exception, but message content differs.")


           if not test_passed:
               all_tests_passed = False
               all_failures.update(test_failures)

   # Run the async tests
   try:
       asyncio.run(run_all_tests())
   except Exception as main_err:
       all_tests_passed = False
       all_failures["main_async_run"] = {"expected": "Successful async execution", "actual": f"Failed: {main_err}"}
       logger.error(f"Error running main verification async function: {main_err}", exc_info=True)


   # --- Report Results ---
   exit_code = report_validation_results(
       validation_passed=all_tests_passed,
       validation_failures=all_failures,
       exit_on_failure=False # Let sys.exit handle it
   )

   logger.info(f"Retry LLM Call Standalone Verification finished with exit code: {exit_code}")
   sys.exit(exit_code)

# --- Original __main__ block removed ---
# if __name__ == "__main__":
#     logger.error("This module should be imported, not run directly.")
#     sys.exit(1)