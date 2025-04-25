"""
Description: Utilities for generating text embeddings using OpenAI's API.

Provides asynchronous functions to create embeddings with retry logic
and handles client initialization.

Core Libraries/Concepts:
------------------------
- OpenAI Python Client: https://github.com/openai/openai-python
- Tenacity: For retry logic.
- asyncio: For asynchronous operations.

Key Functions:
--------------
- create_embedding_with_openai: Generates embedding for a given text asynchronously.
- get_openai_client: Lazily initializes the synchronous OpenAI client.
- get_async_openai_client: Lazily initializes the asynchronous OpenAI client.

Sample I/O (create_embedding_with_openai):
------------------------------------------
Input:
  text: "Sample text to embed."
Output:
  {
    "embedding": [0.1, 0.2, ..., -0.05], # List of floats
    "metadata": {
      "embedding_model": "text-embedding-3-small",
      "provider": "openai",
      "duration_seconds": 0.123,
      "usage": {"prompt_tokens": 6, "total_tokens": 6},
      "dimensions": 1536
    }
  }
"""

import os
import time
from typing import List, Dict, Union, Any, Optional
from functools import lru_cache
import asyncio
import warnings

# Use standard library logging
import logging

# Import OpenAI client
from openai import OpenAI, AsyncOpenAI, APIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import litellm # Keep litellm import
import sys # Keep sys import

# Assuming config is accessible or defaults are okay
# from pdf_extractor.arangodb.config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
# Using defaults directly for now - Corrected to ada-002 and 1536
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", 1536)) # Use 1536 as default

# Setup logger (already configured with standard logging)
logger = logging.getLogger(__name__) # Keep standard logger assignment

# --- OpenAI Configuration ---
# Default model, can be overridden by environment variable
DEFAULT_OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002") # Corrected default
# API key is expected to be set as an environment variable OPENAI_API_KEY
# The OpenAI client automatically picks it up.

# --- OpenAI Client Initialization ---
# Use LRU cache to avoid re-creating the client unnecessarily
@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Get or lazily initialize the synchronous OpenAI client."""
    logger.info("Initializing synchronous OpenAI client.")
    try:
        client = OpenAI()
        # Perform a simple test call to ensure the client is configured correctly
        # client.models.list() # This might be too slow/costly for initialization
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        raise

@lru_cache(maxsize=1)
def get_async_openai_client() -> AsyncOpenAI:
    """Get or lazily initialize the asynchronous OpenAI client."""
    logger.info("Initializing asynchronous OpenAI client.")
    try:
        client = AsyncOpenAI()
        # Consider adding a lightweight check if necessary
        return client
    except Exception as e:
        logger.error(f"Failed to initialize AsyncOpenAI client: {e}", exc_info=True)
        raise

# --- Embedding Generation ---

# Define retry mechanism for API calls
retry_decorator = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((APIError, RateLimitError, asyncio.TimeoutError)),
    reraise=True,
)

@retry_decorator # type: ignore
async def create_embedding_with_openai(
   text: str,
    model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
    client: Optional[AsyncOpenAI] = None,
    **kwargs: Any # Allow passing extra arguments like dimensions if needed
) -> Dict[str, Union[List[float], Dict[str, Any]]]:
    """
    Generate an embedding using the OpenAI API (asynchronously).

    Args:
        text: The text content to embed.
        model: The OpenAI embedding model to use.
        client: An optional pre-initialized AsyncOpenAI client.
        **kwargs: Additional arguments for the OpenAI API call (e.g., dimensions).

    Returns:
        A dictionary containing the embedding vector and metadata.
    """
    # The check `not text` already covers empty strings.
    # The type hint `text: str` means Pylance should know it's a string if it passes the initial check.
    # Removing the redundant isinstance check.
    if not text:
         raise ValueError("Input text must be a non-empty string.")
    # Ensure it's actually a string at runtime just in case type hints are ignored/bypassed
    if not isinstance(text, str):
         raise ValueError("Input text must be a string.")


    aclient = client or get_async_openai_client()
    start_time = time.perf_counter()

    try:
        logger.debug(f"Requesting OpenAI embedding for text (length: {len(text)}) using model: {model}")
        response = await aclient.embeddings.create(
            input=[text], # API expects a list of strings
            model=model,
            **kwargs
        )
        end_time = time.perf_counter()
        duration = end_time - start_time

        if response.data and len(response.data) > 0:
            embedding_data = response.data[0]
            embedding = embedding_data.embedding
            usage = response.usage

            logger.debug(f"Successfully generated embedding. Duration: {duration:.4f}s. Usage: {usage.total_tokens} tokens.")

            metadata = {
                "embedding_model": model,
                "provider": "openai",
                "duration_seconds": duration,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "total_tokens": usage.total_tokens,
                },
                "dimensions": len(embedding),
                # Add any extra metadata returned or needed
            }
            if hasattr(embedding_data, 'object'):
                 metadata['object_type'] = embedding_data.object
            if hasattr(embedding_data, 'index'):
                 metadata['index'] = embedding_data.index


            return {"embedding": embedding, "metadata": metadata}
        else:
            logger.error("OpenAI API returned empty data for embedding request.")
            raise ValueError("OpenAI API returned no embedding data.")

    except (APIError, RateLimitError) as e:
        logger.error(f"OpenAI API error during embedding generation: {e}", exc_info=True)
        raise # Re-raise for tenacity to handle retries
    except Exception as e:
        logger.error(f"Unexpected error during OpenAI embedding generation: {e}", exc_info=True)
        raise # Re-raise other unexpected errors




# --- Synchronous Embedding Functions (Copied from mcp_doc_retriever) ---

# Removed incompatible @logger.catch decorator
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> Optional[List[float]]:
    """
    Generates an embedding for the given text using LiteLLM.

    Args:
        text (str): The input text to embed.
        model (str): The embedding model identifier (e.g., "text-embedding-3-small").

    Returns:
        Optional[List[float]]: The embedding vector, or None if an error occurs.
    """
    if not text or not text.strip():
        logger.warning("Cannot generate embedding for empty or whitespace-only text.")
        return None
    try:
        logger.debug(f"Requesting embedding: model='{model}', text='{text[:100]}...'")
        response = litellm.embedding(model=model, input=text)
        embedding = response.data[0]['embedding']

        if EMBEDDING_DIMENSIONS > 0 and len(embedding) != EMBEDDING_DIMENSIONS:
            logger.error(
                f"Embedding dimension mismatch! Expected {EMBEDDING_DIMENSIONS}, got {len(embedding)}. Check model '{model}'."
            )
            # return None # Make it critical?

        logger.debug(f"Generated embedding ({len(embedding)} dims).")
        return embedding
    except Exception as e:
        logger.exception(f"LiteLLM embedding error: model='{model}', error='{e}'")
        return None

def get_text_for_embedding(doc_data: Dict[str, Any]) -> str:
    """
    Combines relevant text fields from a document into a single string
    suitable for generating a representative embedding.

    Args:
        doc_data (Dict[str, Any]): The document data dictionary.

    Returns:
        str: A concatenated string of relevant text fields.
    """
    parts = [
        doc_data.get("problem", ""),
        doc_data.get("solution", ""),
        doc_data.get("context", ""),
    ]
    return "\n".join(filter(None, parts)).strip()

# --- End Synchronous Functions ---
# --- Example Usage ---
async def main():
    """Example usage of the embedding function."""
    test_text = "This is a test sentence for OpenAI embedding."
    print(f"Generating embedding for: '{test_text}' using model {DEFAULT_OPENAI_EMBEDDING_MODEL}")

    try:
        # Ensure OPENAI_API_KEY is set in your environment variables
        if not os.getenv("OPENAI_API_KEY"):
            print("\nWARNING: OPENAI_API_KEY environment variable not set.")
            print("Please set it to run the example.")
            return

        result_dict = await create_embedding_with_openai(test_text)

        print("\n--- OpenAI Embedding Result ---")
        if result_dict and "embedding" in result_dict:
            print(f"Embedding dimension: {len(result_dict['embedding'])}")
            # print(f"Embedding vector (first 10 dims): {result_dict['embedding'][:10]}...") # Uncomment to view part of the vector
        else:
            print("Embedding generation failed or returned empty result.")

        if result_dict and "metadata" in result_dict:
            metadata = result_dict["metadata"]
            # Add type check before iterating
            if isinstance(metadata, dict):
                print("Metadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
            else:
                print(f"Metadata is not a dictionary: {metadata}")

    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")

if __name__ == "__main__":
    # To run this example directly:
    # 1. Make sure you have an OPENAI_API_KEY environment variable set.
    # 2. Run `python -m mcp_litellm.utils.embedding_utils` from the project root directory.
    asyncio.run(main())
