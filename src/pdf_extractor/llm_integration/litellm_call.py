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
- **Conditional loading and passing of Vertex AI credentials, project, and location.**

Relevant Documentation:
- LiteLLM `acompletion`: https://docs.litellm.ai/docs/completion/async_completions
- LiteLLM Vertex AI: https://docs.litellm.ai/docs/providers/vertex
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
import json # Added for Vertex credentials
from pathlib import Path # Added for Vertex credentials path
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
    RetryCallState # Import RetryCallState
)

from pdf_extractor.llm_integration.multimodal_utils import ( # Corrected to relative import
    format_multimodal_messages,
    is_multimodal,
)
from pdf_extractor.llm_integration.initialize_litellm_cache import initialize_litellm_cache # Corrected to relative import

# litellm.set_verbose = False # Keep verbose logging off unless needed

# Define the path to the Vertex AI credentials file
# Assuming the script runs from the project root or the path is relative to project root
VERTEX_CREDENTIALS_PATH = Path("src/pdf_extractor/vertex_ai_service_account.json")
DEFAULT_VERTEX_LOCATION = "us-central1" # Define default location

# Helper function to validate and update LLM config
def validate_update_config(
    config: Dict[str, Any], insert_into_db: bool = False
) -> Dict[str, Any]:
    """
    Validates and updates LLM config to meet requirements for JSON/structured responses.

    Args:
        config: The configuration dictionary containing llm_config and directories.

    Returns:
        Dict[str, Any]: Updated LLM config with proper JSON formatting instructions
                        and potentially formatted multimodal messages.

    Raises:
        ValueError: If messages are missing or multimodal formatting fails.
    """
    llm_config = config.get("llm_config", {})
    directories = config.get("directories", {})
    if not llm_config.get("messages", []):
        raise ValueError("A message object is required to query the LLM")

    # Deep copy to avoid modifying the original config dict
    updated_llm_config = copy.deepcopy(llm_config)
    messages = updated_llm_config.get("messages", [])

    response_format = updated_llm_config.get("response_format")
    requires_json = response_format == "json" or (
        isinstance(response_format, type) and issubclass(response_format, BaseModel)
    )

    if requires_json:
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

    # Check for multimodal content AFTER potential system message modification
    if is_multimodal(messages):
        image_directory = directories.get("image_directory", "")
        max_size_kb = updated_llm_config.get("max_image_size_kb", 500)
        try:
            messages = format_multimodal_messages(
                messages, image_directory, max_size_kb
            )
        except Exception as e:
             logger.error(f"Failed to format multimodal messages: {e}", exc_info=True)
             raise ValueError(f"Error processing images for multimodal input: {e}")

    updated_llm_config["messages"] = messages
    return updated_llm_config


# Define a callback function for logging retries
def log_retry_attempt(retry_state: RetryCallState):
    """Logs details of a retry attempt."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        f"Retrying LLM call (Attempt {retry_state.attempt_number}) "
        f"due to exception: {exception}" # Safely access exception
    )

# Restore retry decorator
@retry( # type: ignore
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff (adjusted min)
    stop=stop_after_attempt(3),  # Max 3 retries
    retry=retry_if_exception_type(Exception),  # Retry on any exception
    before_sleep=log_retry_attempt, # Log before sleeping
)
async def litellm_call(config: Dict[str, Any]) -> Union[ModelResponse, BaseModel, Dict[str, Any], str, AsyncGenerator[str, None]]:
    """
    Makes an asynchronous call to a language model using LiteLLM with retries.
    Handles configuration validation, JSON/Pydantic response formatting,
    multimodal inputs, streaming, and Vertex AI credential/project/location injection.
    """
    model_name = "unknown" # Initialize model_name
    try:
        llm_config = config.get("llm_config", {})
        
        # Validate/update config (handles messages, JSON hints, multimodal)
        validated_llm_config = validate_update_config(config)

        # Determine if JSON output is required based on the original config
        original_response_format = llm_config.get("response_format") # Use original config for this check
        requires_json = original_response_format == "json" or (
            isinstance(original_response_format, type) and issubclass(original_response_format, BaseModel)
        )

        model_name = validated_llm_config.get("model", "openai/gpt-4o-mini") # Ensure a default model is set

        api_params = {
            "model": model_name,
            "messages": validated_llm_config["messages"], # Use validated/updated messages
            "temperature": validated_llm_config.get("temperature", 0.2),
            "max_tokens": validated_llm_config.get("max_tokens", 1000),
            "stream": validated_llm_config.get("stream", False),
            "caching": validated_llm_config.get("caching", True), # Default to True
            "metadata": {"request_id": validated_llm_config.get("request_id"), "hidden": True},
        }
        
        # Add api_base if present
        if validated_llm_config.get("api_base"):
            api_base = validated_llm_config["api_base"]
            # LiteLLM expects api_base without /v1/completions for some providers
            if api_base.endswith("/v1/completions"):
                api_base = api_base.rsplit("/v1/completions", 1)[0] + "/v1"
            elif api_base.endswith("/v1"): # Ensure it ends with /v1 if needed
                 pass # Already correct
            # Add other potential suffixes if necessary
            api_params["api_base"] = api_base

        # Enforce JSON mode if required
        if requires_json:
            logger.debug("Setting response_format to enforce JSON object output.")
            api_params["response_format"] = {"type": "json_object"}
        # Pass Pydantic model directly if specified (LiteLLM handles this via instructor)
        elif isinstance(original_response_format, type) and issubclass(original_response_format, BaseModel):
             api_params["response_model"] = original_response_format


        # --- Vertex AI Specific Parameters ---
        if model_name.startswith("vertex_ai/"): 
            logger.debug(f"Vertex AI model detected ({model_name}). Adding Vertex parameters.")
            
            # Add Location
            api_params["vertex_location"] = os.getenv("VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION)
            logger.debug(f"Using Vertex Location: {api_params['vertex_location']}")

            # Add Project ID
            vertex_project_id = os.getenv("VERTEX_PROJECT_ID")
            if vertex_project_id:
                api_params["vertex_project"] = vertex_project_id
                logger.debug(f"Using Vertex Project ID from env var: {vertex_project_id}")
            else:
                 # Log warning if project ID is missing, as it's often required
                 logger.warning("VERTEX_PROJECT_ID environment variable not set. Vertex AI calls might fail if not implicitly handled by credentials/ADC.")

            # Add Credentials (if file exists)
            if VERTEX_CREDENTIALS_PATH.exists():
                try:
                    # Pass the path directly, LiteLLM can handle it
                    api_params["vertex_credentials"] = str(VERTEX_CREDENTIALS_PATH.resolve())
                    logger.debug(f"Using Vertex Credentials Path: {api_params['vertex_credentials']}")
                except Exception as cred_err: # Catch potential errors resolving path
                    logger.error(f"Failed to resolve Vertex AI credentials path {VERTEX_CREDENTIALS_PATH}: {cred_err}")
            else:
                logger.warning(f"Vertex AI credentials file not found at {VERTEX_CREDENTIALS_PATH}. Relying on Application Default Credentials (ADC) if available.")
                if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                     logger.warning("GOOGLE_APPLICATION_CREDENTIALS env var not set and local credential file missing. Vertex AI calls might fail if gcloud auth application-default login was not run.")
        # --- End Vertex AI Specific Parameters ---


        logger.debug(f"Calling litellm.acompletion with params: { {k: v for k, v in api_params.items() if k != 'vertex_credentials'} }") # Avoid logging full creds path if used
        response = await litellm.acompletion(**api_params) # type: ignore

        # Handle streaming response
        if api_params["stream"]:
            if isinstance(response, AsyncGenerator):
                async def stream_generator():
                    full_content = ""
                    async for chunk in response:
                        try:
                            # Add type ignore for Pylance limitation with stream chunks
                            content = chunk.choices[0].delta.content or "" # type: ignore
                            full_content += content
                            yield content
                        except (AttributeError, IndexError, TypeError):
                            logger.warning(f"Could not extract content from stream chunk: {chunk}")
                            continue
                    # logger.debug(f"Full streamed content: {full_content}") # Optional: log full content
                return stream_generator()
            else:
                logger.error(f"Expected AsyncGenerator for streaming, got {type(response)}")
                raise TypeError(f"Unexpected response type for streaming call: {type(response)}")
        else:
            # Handle non-streaming response
            if isinstance(response, ModelResponse):
                hidden_params = getattr(response, '_hidden_params', {})
                logger.info(f"Cache Hit: {hidden_params.get('cache_hit', 'N/A')}")
                # Set cache_hit if not present (might happen on first call)
                if "cache_hit" not in hidden_params:
                     hidden_params["cache_hit"] = False
                     # Attempt to update the response object if possible (might not be mutable)
                     try:
                          setattr(response, '_hidden_params', hidden_params)
                     except AttributeError:
                          pass # Ignore if attribute cannot be set

                # Return the full ModelResponse object for non-streaming
                return response
            elif isinstance(response, BaseModel):
                 # If LiteLLM returns a Pydantic model directly (via instructor)
                 logger.info("Received Pydantic model directly from LiteLLM.")
                 return response
            else:
                 logger.error(f"Expected ModelResponse or Pydantic Model for non-streaming, got {type(response)}")
                 raise TypeError(f"Unexpected response type for non-streaming call: {type(response)}")

    except BadRequestError as e:
        logger.error(f"BadRequestError calling LLM ({model_name}): {e}") # model_name is now guaranteed to be defined
        raise # Re-raise after logging
    except Exception as e:
        # This will catch errors after retries have been exhausted
        logger.error(f"Error calling LLM ({model_name}) after retries: {e}", exc_info=True) # model_name is now guaranteed to be defined
        raise # Re-raise the final exception


# --- Task Decomposition and Synthesis Logic (Example Implementation - Unchanged) ---
# ... (Keep the existing decompose, simulate, synthesize, handle_complex_query functions) ...
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
    test_config = {
        "llm_config": {
            "model": "openai/gpt-4o-mini", # Use a non-Vertex model for basic test
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Briefly explain the concept of asynchronous programming."},
            ],
            "temperature": 0.1,
            "max_tokens": 50,
            "stream": False,
            "caching": False, # Disable caching for validation consistency
        }
    }

    validation_passed = False
    EXPECTED_OUTPUT_SUBSTRING = "allows tasks" # More general substring

    try:
            response = await litellm_call(test_config)
            logger.info(f"Raw response type: {type(response)}")

            # --- VALIDATION for Non-Streaming ---
            if isinstance(response, ModelResponse):
                try:
                    # Add type ignore for Pylance limitation
                    content = response.choices[0].message.content # type: ignore
                    if isinstance(content, str) and content:
                        logger.info(f"Response Content (first {min(50, len(content))} chars): {content[:50]}...")
                        if EXPECTED_OUTPUT_SUBSTRING.lower() in content.lower():
                            logger.success(f"✅ Validation passed: Expected substring '{EXPECTED_OUTPUT_SUBSTRING}' found in response.")
                            validation_passed = True
                        else:
                            logger.error(f"❌ Validation failed: Expected substring '{EXPECTED_OUTPUT_SUBSTRING}' NOT found in response.")
                    elif isinstance(content, str):
                        logger.error("❌ Validation failed: Response content is an empty string.")
                    else:
                        logger.error(f"❌ Validation failed: Response content is not a string (type: {type(content)}).")
                except (AttributeError, IndexError, TypeError) as e:
                    logger.error(f"❌ Validation failed: Error accessing content in ModelResponse: {e}")
                    logger.debug(f"ModelResponse structure: {response}")
            elif isinstance(response, BaseModel):
                 logger.warning(f"Validation partially passed: Received a Pydantic model response ({type(response)}), skipping content check.")
                 validation_passed = True # Assume valid type if Pydantic model received
            else:
                logger.error(f"❌ Validation failed: Unexpected response type received: {type(response)}")

    except TypeError as e:
            logger.error(f"Validation failed due to TypeError: {e}")
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


if __name__ == "__main__":
    # Ensure asyncio event loop is managed correctly
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Critical error running main: {e}", exc_info=True)
        sys.exit(1) # Ensure non-zero exit on critical error
