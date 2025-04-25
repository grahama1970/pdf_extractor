"""
Utilities for Handling Multimodal Messages in LiteLLM Calls.

This module provides helper functions specifically designed to process message
lists that may contain multimodal content (like images alongside text) before
sending them to a language model via LiteLLM.

Functions:
- `is_multimodal`: Checks if a list of messages contains image URLs.
- `format_multimodal_messages`: Processes messages, potentially handling image
  URLs or other multimodal elements as needed for the target LLM API.
  (Currently, it primarily formats the structure but doesn't perform
  compression or complex processing).

Relevant Documentation:
- LiteLLM Multimodal Support: https://docs.litellm.ai/docs/providers/openai#openai-multimodal-support
- Project Multimodal Handling Notes: ../../repo_docs/multimodal_handling.md (Placeholder)

Input/Output:
- `is_multimodal`: Input is a list of message dictionaries, output is boolean.
- `format_multimodal_messages`: Input is a list of message dictionaries,
  optional image directory, and max size; output is a processed list of
  message dictionaries formatted for API calls.
"""
from typing import List, Dict, Any
from loguru import logger


###
# Helper Functions
###
def is_multimodal(messages: List[Dict[str, Any]]) -> bool:
    """
    Determine if the messages list contains multimodal content (e.g., images).

    Args:
        messages (List[Dict[str, Any]]): List of message dictionaries.

    Returns:
        bool: True if any message contains multimodal content, False otherwise.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            item.get("type") == "image_url" for item in content
        ):
            return True
    return False


def format_multimodal_messages(
    messages: List[Dict[str, Any]], image_directory: str, max_size_kb: int = 500
) -> List[Dict[str, Any]]:
    """
    Processes a messages list to extract and format content for LLM input.

    Args:
        messages (List[Dict[str, Any]]): List of messages, each containing multimodal content.
        image_directory (str): Directory to store compressed images.
        max_size_kb (int): Maximum size for compressed images in KB.

    Returns:
        List[Dict[str, Any]]: Processed list of content dictionaries.
    """
    if not messages:
        logger.warning("Received empty messages list. Returning an empty content list.")
        return []

    processed_messages = []
    for message in messages:
        if "content" in message and isinstance(message["content"], list):
            processed_content = []
            for item in message["content"]:
                if item.get("type") == "text":
                    processed_content.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image_url":
                    image_url = None # Initialize image_url
                    try:
                        # Extract the URL directly from the nested structure
                        image_url = item["image_url"]["url"]
                        # Format it correctly for the API
                        processed_content.append(
                            {"type": "image_url", "url": image_url}
                        )
                    except (ValueError, KeyError, TypeError) as e: # Combined exceptions
                        logger.error(f"Error processing image url: {image_url} - {e}") # Log image_url if available
                        continue
            processed_messages.append(
                {"role": message.get("role", "user"), "content": processed_content}
            )
        else:
            # If not multimodal content, pass through unchanged
            processed_messages.append(message)
    return processed_messages


# --- Standalone Validation Block --- 

import sys

def main_validation():
    """Performs basic validation checks on the multimodal utility functions."""
    logger.info("--- Running Standalone Validation for multimodal_utils.py ---")
    validation_passed = True
    errors = []

    # Test Data
    text_message = [{"role": "user", "content": "Hello there!"}]
    multimodal_message = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
        ]}
    ]
    malformed_image_message = [
         {"role": "user", "content": [
            {"type": "text", "text": "Describe this image:"},
            {"type": "image_url", "image_url": "not_a_dict"} # Malformed
        ]}
    ]

    # 1. Test is_multimodal
    try:
        assert not is_multimodal(text_message), "is_multimodal failed on text-only message."
        logger.debug("is_multimodal passed for text-only.")
        assert is_multimodal(multimodal_message), "is_multimodal failed on multimodal message."
        logger.debug("is_multimodal passed for multimodal.")
    except AssertionError as e:
        errors.append(f"is_multimodal assertion failed: {e}")
        validation_passed = False
    except Exception as e:
        errors.append(f"Error during is_multimodal test: {e}")
        validation_passed = False

    # 2. Test format_multimodal_messages (Good case)
    try:
        formatted = format_multimodal_messages(multimodal_message, "/tmp/images")
        assert len(formatted) == 1, "format_multimodal_messages should return 1 message."
        assert isinstance(formatted[0]["content"], list), "Formatted content should be a list."
        assert len(formatted[0]["content"]) == 2, "Formatted content should have 2 items."
        assert formatted[0]["content"][0]["type"] == "text", "First item should be text."
        assert formatted[0]["content"][1]["type"] == "image_url", "Second item should be image_url."
        assert "url" in formatted[0]["content"][1], "Image item should have a 'url' key."
        assert formatted[0]["content"][1]["url"] == "data:image/png;base64,iVBOR...", "Image URL mismatch."
        logger.debug("format_multimodal_messages passed for valid multimodal input.")
    except AssertionError as e:
        errors.append(f"format_multimodal_messages (good case) assertion failed: {e}")
        validation_passed = False
    except Exception as e:
        errors.append(f"Error during format_multimodal_messages (good case) test: {e}")
        validation_passed = False

    # 3. Test format_multimodal_messages (Malformed case)
    try:
        formatted_malformed = format_multimodal_messages(malformed_image_message, "/tmp/images")
        # Expecting the malformed image part to be skipped, leaving only the text part
        assert len(formatted_malformed) == 1, "format_multimodal_messages (malformed) should return 1 message."
        assert isinstance(formatted_malformed[0]["content"], list), "Malformed formatted content should be a list."
        assert len(formatted_malformed[0]["content"]) == 1, "Malformed formatted content should have 1 item (text only)."
        assert formatted_malformed[0]["content"][0]["type"] == "text", "Malformed formatted content should contain only text."
        logger.debug("format_multimodal_messages handled malformed image input gracefully.")
    except Exception as e:
        errors.append(f"Error during format_multimodal_messages (malformed case) test: {e}")
        validation_passed = False

    # Report validation status
    if validation_passed:
        logger.success("✅ Standalone validation passed: Multimodal utility functions behave as expected.")
        print("\n✅ VALIDATION COMPLETE - Multimodal utilities verified.")
        sys.exit(0)
    else:
        for error in errors:
            logger.error(f"❌ {error}")
        print("\n❌ VALIDATION FAILED - Multimodal utility verification failed.")
        sys.exit(1)

if __name__ == "__main__":
    main_validation()
