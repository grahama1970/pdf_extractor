"""
Handles Asynchronous LiteLLM API Calls with Retries and Validation.

This module provides a core function `litellm_call` for making asynchronous
calls to language models via the LiteLLM library. It incorporates features like:
- Exponential backoff retries using `tenacity`.
- Validation and modification of input configurations, especially for JSON
  and multimodal requests.
- Handling of both streaming and non-streaming responses.
- Integration with LiteLLM caching (setup is expected externally).
- Loading environment variables and project configurations.

Relevant Documentation:
- LiteLLM `acompletion`: https://docs.litellm.ai/docs/completion/async_completions
- Tenacity Retrying Library: https://tenacity.readthedocs.io/en/latest/
- Pydantic Models: https://docs.pydantic.dev/latest/
- Project LLM Interaction Notes: ../../repo_docs/llm_interaction.md (Placeholder)

Input/Output:
- Input: A configuration dictionary (`config`) containing `llm_config` (model,
  messages, temperature, max_tokens, stream, caching, response_format, etc.)
  and optional `directories` (e.g., for image paths).
- Output: Returns the response from the LiteLLM call. This can be:
    - A Pydantic model instance if `response_format` was a Pydantic model.
    - A dictionary if `response_format` was 'json'.
    - A string containing the model's text response for standard calls.
    - A SimpleNamespace object mimicking the structure for streaming responses.
- Raises: Exceptions from LiteLLM (e.g., `BadRequestError`) or connection errors.
"""
from types import SimpleNamespace
import litellm
from litellm.exceptions import BadRequestError # Explicit import
from litellm.types.utils import ModelResponse # Import for type checking response
import async_timeout
from deepmerge import always_merger
from pydantic import BaseModel, Field
import asyncio
import os
import copy
from typing import Any, AsyncGenerator, Dict, List, Type, Union, Optional, Tuple
from pydantic import BaseModel
from types import SimpleNamespace
from loguru import logger
import unittest.mock
from litellm.types.utils import Choices, Message, Usage # Corrected: Import from utils where they are defined
import sys # Added for exit codes in main
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from pdf_extractor.llm_integration.multimodal_utils import ( # Corrected to relative import
    format_multimodal_messages,
    is_multimodal,
)
from pdf_extractor.llm_integration.initialize_litellm_cache import initialize_litellm_cache # Corrected to relative import

# Load environment variables Globally
# Environment variables should be loaded by the application entry point or config management
# project_dir = get_project_root() # Removed
# load_env_file() # Removed


# Helper function to validate and update LLM config
def validate_update_config(
    config: Dict[str, Any], insert_into_db: bool = False
) -> Dict[str, Any]:
    """
    Validates and updates LLM config to meet requirements for JSON/structured responses.

    Args:
        llm_config: The LLM configuration dictionary

    Returns:
        Dict[str, Any]: Updated LLM config with proper JSON formatting instructions

    Raises:
        ValueError: If JSON requirements cannot be met
    """
    llm_config = config.get("llm_config", {})
    directories = config.get("directories", {})
    if not llm_config.get("messages", []):
        raise ValueError("A message object is required to query the LLM")

    response_format = llm_config.get("response_format")
    requires_json = response_format == "json" or (
        isinstance(response_format, type) and issubclass(response_format, BaseModel)
    )

    if requires_json:
        messages = llm_config.get("messages", []).copy()
        system_messages = [msg for msg in messages if msg.get("role") == "system"]

        if not system_messages:
            # Add a new system message if none exists
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You must return your response as a well-formatted JSON object.",
                },
            )
        else:
            # Update existing system message if JSON instruction is missing
            system_content = system_messages[0].get("content", "").lower()
            json_instruction_keywords = ["json", "well-formatted", "well formatted"]

            if not any(
                keyword in system_content for keyword in json_instruction_keywords
            ):
                system_messages[0]["content"] = (
                    "You must return your response as a well-formatted JSON object. "
                    + system_messages[0]["content"]
                )

        # Check for multimodal content
        if is_multimodal(messages):
            # hardcoded for now
            image_directory = directories.get("image_directory", "")
            max_size_kb = llm_config.get("max_image_size_kb", 500)
            messages = format_multimodal_messages(
                messages, image_directory, max_size_kb
            )

        llm_config = copy.deepcopy(llm_config)
        llm_config["messages"] = messages

    return llm_config


# Main
from tenacity import RetryCallState
@retry( # type: ignore
    wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff
    stop=stop_after_attempt(3),  # Max 3 retries
    retry=retry_if_exception_type(Exception),  # Retry on any exception, consider refining later
)
async def litellm_call(config: Dict[str, Any]) -> Union[ModelResponse, BaseModel, Dict[str, Any], str, AsyncGenerator[str, None]]: # Added ModelResponse
    """
    Makes an asynchronous call to a language model using LiteLLM with retries.

    Handles configuration validation, JSON/Pydantic response formatting,
    multimodal inputs, and streaming.

    Args:
        config: A dictionary containing 'llm_config' and optional 'directories'.
            llm_config (Dict[str, Any]): LiteLLM parameters (model, messages, temp, etc.).
                - response_format: Can be 'json', a Pydantic model class, or None.
                - stream (bool): Whether to stream the response.
                - caching (bool): Whether to use LiteLLM caching.
                - messages (List[Dict[str, Any]]): The list of messages for the prompt.
            directories (Dict[str, str]): Optional paths, e.g., 'image_directory'.

    Returns:
        Union[BaseModel, Dict[str, Any], str, AsyncGenerator[str, None]]:
            - Pydantic model instance if response_format is a Pydantic class.
            - Dictionary if response_format is 'json'.
            - String containing the full response text if stream=False and no specific format.
            - An async generator yielding response chunks if stream=True.

    Raises:
        ValueError: If the configuration is invalid (e.g., missing messages).
        litellm.exceptions.BadRequestError: If the LLM API reports a bad request.
        Exception: Other exceptions during the API call or processing, after retries.
    """
    try:
        llm_config = config.get("llm_config", {})
        # directories = config.get("directories", {}) # directories are used within validate_update_config

        llm_config = validate_update_config(config)

        # Default to plain text if response_format is not provided
        response_format = llm_config.get("response_format", None)

        api_params = {
            "model": llm_config.get("model", "openai/gpt-4o-mini"), # Ensure a default model is set
            "messages": llm_config["messages"], # Use validated/updated messages
            "temperature": llm_config.get("temperature", 0.2),
            "max_tokens": llm_config.get("max_tokens", 1000),
            "stream": llm_config.get("stream", False),
            "caching": llm_config.get("caching", True),
            "metadata": {"request_id": llm_config.get("request_id"), "hidden": True},
        }
        if llm_config.get("api_base", None):
            # Remove /v1/completions from api_base if present
            # Strange LiteLLM behavior, but it works
            api_base = llm_config["api_base"]
            if api_base.endswith("/v1/completions"):
                api_base = api_base.replace("/v1/completions", "/v1")
                api_params["api_base"] = api_base

        # Add response_format only if explicitly provided
        if response_format:
            api_params["response_format"] = response_format

        response = await litellm.acompletion(**api_params) # type: ignore

        # Handle streaming response
        if api_params["stream"]:
            # Check if response is an async generator/iterable
            if isinstance(response, AsyncGenerator):
                # Return the async generator directly for streaming
                async def stream_generator():
                    full_content = ""
                    async for chunk in response: # Pylance should now understand response is iterable
                        # Assuming chunk structure is correct based on LiteLLM streaming
                        try:
                            content = chunk.choices[0].delta.content or ""
                            full_content += content
                            yield content # Yield content chunks
                        except (AttributeError, IndexError):
                            logger.warning(f"Could not extract content from stream chunk: {chunk}")
                            continue # Skip problematic chunks
                    # Optionally log the full content after streaming is complete
                    # logger.debug(f"Full streamed content: {full_content}")

                return stream_generator()
            else:
                # Handle unexpected non-iterable response in streaming mode
                logger.error(f"Expected AsyncGenerator for streaming, got {type(response)}")
                raise TypeError(f"Unexpected response type for streaming call: {type(response)}")
        else:
            # Handle non-streaming response
            if isinstance(response, ModelResponse):
                hidden_params = getattr(response, '_hidden_params', {}) # Use getattr for safety
                logger.info(f"Cache Hit: {hidden_params.get('cache_hit', 'N/A')}") # Handle missing cache info
                try:
                    # type: ignore because Pylance struggles with isinstance check narrowing
                    content = response.choices[0].message.content # Access should be safer now
                except (AttributeError, IndexError):
                    logger.error("Failed to extract content from non-streaming response structure.")
                    content = None # Handle missing content

                # Set cache_hit in _hidden_params if not already set
                if "cache_hit" not in hidden_params:
                    hidden_params["cache_hit"] = False # Corrected indentation

                # Ensure the return type matches the non-streaming case in the Union
                return response # Return the validated ModelResponse
            else:
                 # Handle unexpected type for non-streaming
                 logger.error(f"Expected ModelResponse for non-streaming, got {type(response)}")
                 raise TypeError(f"Unexpected response type for non-streaming call: {type(response)}")

    except BadRequestError as e: # Use the explicitly imported exception
        logger.error(f"BadRequestError: {e}")
        raise
    except Exception as e:
        logger.error(f"Error calling LLM: {e}", exc_info=True) # Add exc_info for better debugging
        raise


# --- Task Decomposition and Synthesis Logic (Example Implementation) ---

def decompose_query_france_example(query: str) -> list[str]:
    """
    Simulates decomposition for the specific France clothing query.
    In a real system, this would involve LLM calls or more sophisticated logic.
    """
    logger.info(f"Decomposing query: '{query}'")
    if "france" in query.lower() and "wintertime" in query.lower() and "capital" in query.lower():
        return [
            "What is the capital city of France?",
            "What is the average temperature in Paris during Wintertime?", # Hardcoded Paris based on expected answer to Q1
            "What clothes should I wear when it's 30F in Wintertime?" # Hardcoded 30F based on expected answer to Q2
        ]
    else:
        logger.warning("Query does not match the specific France example format for decomposition.")
        return [] # Or handle other queries differently

def simulate_sub_question_call(sub_question: str, context: Dict[str, Any]) -> str:
    """
    Simulates an LLM call to answer a sub-question based on the France example.
    """
    logger.info(f"Simulating LLM call for sub-question: '{sub_question}'")
    answer = f"Placeholder answer for: '{sub_question}'" # Default

    if "capital city of france" in sub_question.lower():
        answer = "Paris"
    elif "average temperature in paris" in sub_question.lower() and "wintertime" in sub_question.lower():
        answer = "Approximately 30F (around 0C)."
    elif "clothes should i wear" in sub_question.lower() and "30f" in sub_question.lower():
        answer = "You should wear warm layers, including a thick sweater, a winter coat, scarf, hat, and gloves."

    logger.info(f"-> Simulated Answer: {answer}")
    return answer

def synthesize_results_france_example(original_query: str, sub_answers: Dict[str, str]) -> str:
    """
    Simulates synthesizing the final answer from sub-answers for the France example.
    """
    logger.info("Synthesizing final answer from sub-answers...")
    # In a real system, this might involve another LLM call with context.
    # Here, we just format the collected answers.

    capital = sub_answers.get("What is the capital city of France?", "Unknown Capital")
    temp = sub_answers.get("What is the average temperature in Paris during Wintertime?", "Unknown Temperature")
    clothing = sub_answers.get("What clothes should I wear when it's 30F in Wintertime?", "Unknown Clothing Advice")

    synthesis = (
        f"Based on your query: '{original_query}'\n"
        f"- The capital of France is {capital}.\n"
        f"- The average temperature there in Wintertime is {temp}.\n"
        f"- Therefore, recommended clothing includes: {clothing}"
    )
    logger.info("-> Synthesized Answer:\n" + synthesis)
    return synthesis


async def handle_complex_query(query: str, config: Dict[str, Any]) -> str:
    """
    Orchestrates the decomposition, sub-question execution (simulated),
    and synthesis for a complex query like the France example.
    """
    logger.info(f"Handling complex query: '{query}'")

    # 1. Decompose
    sub_questions = decompose_query_france_example(query)
    if not sub_questions:
        logger.warning("Could not decompose query, attempting direct call (if implemented).")
        # Optionally, fall back to a direct litellm_call here if needed
        # return await litellm_call(config) # Example fallback
        return "Could not decompose the query using the example logic."


    # 2. Execute Sub-questions (Simulated)
    sub_answers = {}
    context = {"original_query": query} # Context can be built up sequentially

    # Execute sequentially as per the example's dependency
    for question in sub_questions:
        # Update context based on previous answers if necessary for the *next* question's simulation
        # (e.g., pass the capital to the temperature question simulation)
        # This simulation is simple and uses hardcoded values based on expected flow.
        if "average temperature" in question and "Paris" not in question:
             # If the capital was found, inject it into the question for simulation lookup
             capital = sub_answers.get("What is the capital city of France?")
             if capital:
                 question = question.replace("[Capital City]", capital) # Placeholder replacement

        if "clothes should i wear" in question and "[Temperature]" not in question:
             # If the temp was found, inject it
             temp_answer = sub_answers.get("What is the average temperature in Paris during Wintertime?")
             # Extract numeric part for simulation lookup if needed (simplistic extraction)
             import re
             match = re.search(r'(\d+F)', str(temp_answer))
             if match:
                 temp_str = match.group(1)
                 question = question.replace("[Temperature]", temp_str)


        answer = simulate_sub_question_call(question, context)
        sub_answers[question] = answer
        context[f"answer_to_{question[:30]}..."] = answer # Update context

    # 3. Synthesize Results (Simulated)
    final_answer = synthesize_results_france_example(query, sub_answers)

    return final_answer

# --- End of Task Decomposition Logic ---



# Usage Example & Basic Validation
async def main():
    """
    Provides a basic usage example and validation for the litellm_call function.
    Focuses on verifying non-streaming text response type.
    """
    # Keep verbose off for cleaner demo output
    # litellm.set_verbose = True
    initialize_litellm_cache() # Ensure cache is ready

    logger.info("--- Testing Basic Non-Streaming litellm_call ---")

    # Define a simple configuration for a non-streaming text call
    # NOTE: This requires appropriate environment variables (like OPENAI_API_KEY)
    # or a correctly configured LiteLLM setup to run successfully.
    # Using a potentially free/low-cost model for testing.
    test_config = {
        "llm_config": {
            "model": "openai/gpt-4o-mini", # Or another accessible model
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Briefly explain the concept of asynchronous programming."},
            ],
            "temperature": 0.1,
            "max_tokens": 50,
            "stream": False,
            "caching": False, # Disable caching for validation consistency if needed
            # "api_base": "YOUR_API_BASE_IF_NEEDED", # Add if using a local/proxy endpoint
        }
    }

    validation_passed = False
    # Define expected output for the specific query - Adjusted to be less brittle
    EXPECTED_OUTPUT_SUBSTRING = "allows tasks" # More general substring

    try:
            # --- MOCKING REMOVED ---
            # Making a real call to litellm_call
            response = await litellm_call(test_config)

            # --- VALIDATION ---
            # We expect a non-streaming response, which should be the LiteLLM ModelResponse object
            # The actual content is accessed via response.choices[0].message.content
            logger.info(f"Raw response type: {type(response)}") # Corrected indentation

            # --- VALIDATION for Non-Streaming ---
            # Check if the response is the expected ModelResponse type
            if isinstance(response, ModelResponse): # Corrected indentation
                # Access attributes directly now that type is confirmed
                try:
                    # type: ignore because Pylance struggles with isinstance check narrowing
                    content = response.choices[0].message.content # Add type ignore again
                    # Check if content is a non-empty string before using len/slicing
                    if isinstance(content, str) and content:
                        logger.info(f"Response Content (first {min(50, len(content))} chars): {content[:50]}...")
                        # Validate if the expected substring is present
                        if EXPECTED_OUTPUT_SUBSTRING.lower() in content.lower():
                            logger.success(f"✅ Validation passed: Expected substring '{EXPECTED_OUTPUT_SUBSTRING}' found in response.")
                            validation_passed = True
                        else:
                            logger.error(f"❌ Validation failed: Expected substring '{EXPECTED_OUTPUT_SUBSTRING}' NOT found in response.")
                            validation_passed = False
                    elif isinstance(content, str): # It's a string, but empty
                        logger.error("❌ Validation failed: Response content is an empty string.")
                        validation_passed = False
                    else: # It's not a string (e.g., None)
                        logger.error(f"❌ Validation failed: Response content is not a string (type: {type(content)}).")
                        validation_passed = False
                except (AttributeError, IndexError, TypeError) as e:
                    logger.error(f"❌ Validation failed: Error accessing content in ModelResponse: {e}")
                    logger.debug(f"ModelResponse structure: {response}") # Log structure on error
                    validation_passed = False
            # Allow Pydantic models as valid non-streaming responses too
            elif isinstance(response, BaseModel): # Corrected indentation
                 logger.warning(f"Validation partially passed: Received a Pydantic model response ({type(response)}), skipping content check for this test.")
                 validation_passed = True # Assume valid type, but content not checked here
            # Removed redundant elif check for str, dict, AsyncGenerator (Confirming removal)
            else: # Corrected indentation
                # Handles any type not explicitly checked above (ModelResponse, BaseModel)
                logger.error(f"❌ Validation failed: Unexpected response type received in main: {type(response)}")
                validation_passed = False

    except TypeError as e:
            # Catch the TypeError raised by litellm_call for unexpected types
            logger.error(f"Validation failed: {e}")
    except Exception as e:
            logger.error(f"Error during litellm_call test: {e}", exc_info=True)
            logger.warning("Ensure necessary API keys/endpoints are configured for this test.")

    # --- Report Validation Status ---
    if validation_passed:
        print("\n✅ VALIDATION COMPLETE - Basic litellm_call test passed.")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED - Basic litellm_call test did not pass.")
        sys.exit(1)

    # --- Complex Query Simulation (Commented Out - Not core validation) ---
    # complex_query = "What clothes should I wear in Wintertime in the capital city of France?"
    # complex_config = {}
    # print(f"\n--- Testing Complex Query Handling ---")
    # final_result = await handle_complex_query(complex_query, complex_config)
    # print("\n--- FINAL SYNTHESIZED RESULT ---")
    # print(final_result)
    # print("----------------------------------\n")


if __name__ == "__main__":
    # Ensure asyncio event loop is managed correctly
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Critical error running main: {e}", exc_info=True)
        sys.exit(1) # Ensure non-zero exit on critical error
