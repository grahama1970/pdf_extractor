"""
# Robust Retry Logic for LLM Calls with Validation

This module provides an asynchronous function `retry_llm_call` that wraps LLM
completion calls (specifically `litellm.acompletion`), adding robust retry logic
and response validation. It attempts the call, validates the response using
provided strategies, and retries with feedback if validation fails, up to a
specified maximum number of attempts.

## Third-Party Packages:
- litellm: https://litellm.vercel.app/docs/ (Used for the underlying LLM call)
- loguru: https://loguru.readthedocs.io/en/stable/ (Used for logging)

## Sample Input (Conceptual):
- llm_call: `litellm.acompletion`
- model: "ollama/mistral" (or any configured model)
- messages: [{"role": "user", "content": "What is the capital of France?"}]
- validation_strategies: [lambda response: "Paris" in response.choices[0].message.content or "Expected 'Paris'"]

## Expected Output (Conceptual):
- Tuple: (response_object, num_retries, validation_status)
  - e.g., (<ModelResponse object>, 0, True) if successful on the first try.
  - Raises MaxRetriesExceededError if validation fails after all retries.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple, Coroutine, Sequence

from loguru import logger
import litellm # Import litellm for the main block example

# --- Configuration ---
# Basic logger setup. Loguru adds a default handler to stderr if none exist.


# --- Utility Functions ---

def _extract_content_from_response(response: Any) -> Optional[str]:
    """
    Safely extracts the message content from a LiteLLM response object.

    Handles potential None values and attribute errors gracefully.
    This is the likely fix for the 'Validation passed but content is None' error.
    """
    try:
        # Standard access pattern for litellm completion/acompletion
        content = response.choices[0].message.content
        if content is None:
            logger.warning("Extracted content is None from response.choices[0].message.content")
            # Attempt to access raw text if available (might vary by provider/response type)
            raw_text = getattr(response, 'text', None)
            if raw_text:
                 logger.warning(f"Falling back to response.text: {raw_text[:100]}...")
                 return str(raw_text)
            return None
        return str(content)
    except (AttributeError, IndexError, TypeError) as e:
        logger.error(f"Error extracting content from response: {e}. Response structure: {response}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error extracting content: {e}. Response: {response}", exc_info=True)
        return None


# --- Custom Exception ---
class MaxRetriesExceededError(Exception):
    """Raised when the LLM call fails validation after the maximum number of retries."""
    pass


# --- Core Retry Function ---
async def retry_llm_call(
    llm_call: Callable[..., Coroutine[Any, Any, Any]],
    model: str,
    messages: List[Dict[str, Any]],
    validation_strategies: Sequence[Callable[[Any], Union[bool, str]]],
    temperature: float = 0.2,
    max_tokens: int = 1000,
    api_base: Optional[str] = None,
    response_format: Optional[Any] = None, # Reverted to Any for broader compatibility
    max_retries: int = 3,
    **kwargs: Any # Allow passing extra kwargs like vertex_project_id, vertex_location
) -> Tuple[Any, int, Union[bool, List[str]]]:
    """
    Retry LLM calls until validation passes or max retries is reached.

    Args:
        llm_call: Async function that makes the LLM API call (e.g., litellm.acompletion).
        model: Model name to use.
        messages: Initial conversation messages.
        validation_strategies: Sequence of validation functions. Each function takes the
                               response object and returns True if valid, or a string
                               error message if invalid.
        temperature: Temperature for sampling.
        max_tokens: Maximum tokens to generate.
        api_base: Optional API base URL (for local models like Ollama).
        response_format: Optional response format (e.g., {"type": "json_object"}).
        max_retries: Maximum number of retry attempts.
        **kwargs: Additional keyword arguments to pass directly to the llm_call function.

    Returns:
        Tuple of (validated response object, number of retries needed, validation_status).
        validation_status is True if passed, List[str] of errors if failed.

    Raises:
        MaxRetriesExceededError: If validation fails after all retries.
    """
    logger.debug(f"Starting LLM retry loop for model '{model}' with {len(messages)} messages, {len(validation_strategies)} validators, {max_retries} retries.")
    retries = 0
    current_messages = messages.copy() # Avoid modifying the original list
    last_validation_errors: Optional[List[str]] = None

    while retries < max_retries:
        if retries > 0:
            logger.info(f"Retrying LLM call (Attempt {retries + 1}/{max_retries})...")
            await asyncio.sleep(1.0 * retries) # Exponential backoff might be better

        try:
            # Prepare parameters for the llm_call
            llm_params = {
                "model": model,
                "messages": current_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs # Pass through extra arguments
            }
            if api_base:
                llm_params["api_base"] = api_base
            if response_format:
                llm_params["response_format"] = response_format

            # Make the actual LLM call
            logger.debug(f"Attempt {retries + 1}: Calling LLM with params: { {k: v for k, v in llm_params.items() if k != 'messages'} }") # Avoid logging full message history
            response = await llm_call(**llm_params)
            logger.debug(f"Attempt {retries + 1}: Received response.") # Log raw response structure if needed: {response}

            # --- Validation Step ---
            validation_errors = []
            for i, validate in enumerate(validation_strategies):
                try:
                    result = validate(response)
                    if result is not True:
                        error_msg = str(result) if isinstance(result, (str, Exception)) else "Unknown validation error"
                        validation_errors.append(f"Validator {i+1} ({getattr(validate, '__name__', 'anon')}): {error_msg}")
                except Exception as val_err:
                    validator_name = getattr(validate, '__name__', f'Unnamed Validator {i+1}')
                    logger.error(f"Validation function {validator_name} raised an error: {val_err}", exc_info=True)
                    validation_errors.append(f"Validator {validator_name} crashed: {val_err}")

            last_validation_errors = validation_errors

            if not validation_errors:
                logger.success(f"Attempt {retries + 1}: Validation passed.")
                return response, retries, True # Success

            # --- Validation Failed: Prepare for Retry ---
            logger.warning(f"Attempt {retries + 1}: Validation failed: {validation_errors}")

            # Extract content for feedback message - Using the corrected function
            content = _extract_content_from_response(response)
            if content is None:
                 # If content extraction fails even after retry, we can't provide feedback
                 logger.error(f"Attempt {retries + 1}: Failed to extract content from response for feedback. Retrying without feedback.")
                 # Optionally, add a generic error message if content is None
                 # current_messages.append({"role": "user", "content": "Your previous response could not be processed. Please try again."})
            else:
                # Add assistant's failed response and user feedback for the next attempt
                current_messages.append({"role": "assistant", "content": content})
                feedback_message = f"Your response failed validation with these errors: {'; '.join(validation_errors)}. Please review the errors and provide a corrected response."
                current_messages.append({"role": "user", "content": feedback_message})
                logger.debug("Added validation feedback to messages for next retry.")

        except Exception as e:
            logger.error(f"Error during LLM call attempt {retries + 1}: {repr(e)}", exc_info=True) # Use repr(e)
            last_validation_errors = [f"LLM call failed: {repr(e)}"] # Use repr(e)
            # Decide whether to retry on LLM call errors (e.g., network issues)
            # For now, we retry on any exception during the call

        retries += 1

    # --- Max Retries Reached ---
    failure_message = f"Failed to get valid LLM response for model '{model}' after {max_retries} attempts. Last errors: {last_validation_errors}"
    logger.error(failure_message)
    raise MaxRetriesExceededError(failure_message)


async def usage_function():
    # Users might need to change this or set environment variables.
    EXAMPLE_MODEL = os.environ.get("EXAMPLE_LLM_MODEL", "openai/gpt-4o-mini") # Default for Ollama
    EXAMPLE_API_BASE = os.environ.get("EXAMPLE_LLM_API_BASE", "https://api.openai.com/v1") # Default OpenAI endpoint

    logger.info(f"--- Running Retry LLM Call Standalone Verification ---")
    logger.info(f"Using Model: {EXAMPLE_MODEL}")
    if "openai" in EXAMPLE_MODEL:
         logger.info(f"Using API Base: {EXAMPLE_API_BASE}")

    # --- Define Simple Validation ---
    EXPECTED_CONTENT_PART = "Paris"
    def simple_validator(response: Any) -> Union[bool, str]:
        """Checks if the expected content part is in the response."""
        content = _extract_content_from_response(response)
        if content and EXPECTED_CONTENT_PART in content:
            logger.debug(f"Simple validator: Found '{EXPECTED_CONTENT_PART}' in content.")
            return True
        else:
            logger.debug(f"Simple validator: Did not find '{EXPECTED_CONTENT_PART}' in content: '{content}'")
            return f"Response content did not contain '{EXPECTED_CONTENT_PART}'"

    # --- Define Expected Results ---
    # We expect the call to succeed, likely on the first try if the model is good.
    EXPECTED_RESULTS = {
        "contains": EXPECTED_CONTENT_PART,
        "max_retries_expected": 0, # Expect success on first try
        "validation_status_expected": True
    }

    # --- Prepare Call Arguments ---
    test_messages = [{"role": "user", "content": "What is the capital of France? Respond concisely."}]
    test_validators = [simple_validator]

    # --- Initialize Status Variables ---
    validation_passed = False
    validation_failures = {}
    actual_response = None
    actual_retries = -1
    actual_validation_status = None
    exit_code = 1 # Default to failure

    # --- Run the Test using asyncio.run directly ---
    try:
        logger.info("Making LLM call using retry_llm_call...")
        # Call asyncio.run directly on the coroutine returned by retry_llm_call
        actual_response, actual_retries, actual_validation_status = (
            await retry_llm_call(
                llm_call=litellm.acompletion, # Use the actual async litellm function
                model=EXAMPLE_MODEL,
                messages=test_messages,
                validation_strategies=test_validators,
                max_retries=2, # Allow one retry for this example
                api_base=EXAMPLE_API_BASE if "ollama" in EXAMPLE_MODEL else None, # Only for Ollama
                temperature=0.1,
            )
        )
        logger.success("LLM call completed.")

        # --- Validate Results ---
        final_content = _extract_content_from_response(actual_response)

        if actual_validation_status != EXPECTED_RESULTS["validation_status_expected"]:
             validation_failures["validation_status"] = {"expected": EXPECTED_RESULTS["validation_status_expected"], "actual": actual_validation_status}
        if final_content is None or EXPECTED_RESULTS["contains"] not in final_content:
             validation_failures["content_check"] = {"expected_contain": EXPECTED_RESULTS["contains"], "actual": final_content}
        # We check <= max_retries_expected as it might succeed earlier than expected
        if actual_retries > EXPECTED_RESULTS["max_retries_expected"]:
             validation_failures["retry_count"] = {"expected": f"<= {EXPECTED_RESULTS['max_retries_expected']}", "actual": actual_retries}

        if not validation_failures:
            validation_passed = True

    except MaxRetriesExceededError as e:
        logger.error(f"LLM call failed after max retries: {e}")
        validation_failures["max_retries_exception"] = {"expected": "Success or fewer retries", "actual": f"MaxRetriesExceededError: {e}"}
    except litellm.exceptions.APIConnectionError as e:
         logger.error(f"LLM Connection Error: {e}. Is the LLM server running/accessible (Model: {EXAMPLE_MODEL}, Base: {EXAMPLE_API_BASE})?", exc_info=True)
         validation_failures["connection_error"] = {"expected": "Successful connection", "actual": f"APIConnectionError: {e}"}
    except Exception as e:
        # Use str(e) for safer logging of arbitrary exception messages
        logger.error(f"An unexpected error occurred during verification: {str(e)}", exc_info=True)
        validation_failures["unexpected_exception"] = {"expected": "No exception", "actual": f"{type(e).__name__}: {str(e)}"}


    # --- Report Validation Status ---
    if validation_passed:
        logger.success("✅ VALIDATION COMPLETE - retry_llm_call produced expected results.")
        print("\n--- Final Response Content ---")
        print(_extract_content_from_response(actual_response))
        print("--------------------------")
        exit_code = 0
    else:
        logger.error("❌ VALIDATION FAILED - Results don't match expected values.")
        print("\n--- Failure Details ---")
        for key, details in validation_failures.items():
            print(f"  - {key}: Expected: {details['expected']}, Got: {details['actual']}")
        print("-----------------------")
        exit_code = 1

    sys.exit(exit_code)


# --- Main Execution Guard (Compliant Standalone Verification) ---
if __name__ == "__main__":
    litellm._turn_on_debug()  # type: ignore
    asyncio.run(usage_function())

    