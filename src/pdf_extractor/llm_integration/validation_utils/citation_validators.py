# -*- coding: utf-8 -*-
"""
Citation validation utilities for LLM responses.

This module provides validators for checking if LLM responses properly
cite from a provided corpus using fuzzy matching.

Links:
  - rapidfuzz: https://github.com/maxbachmann/RapidFuzz
  - loguru: https://loguru.readthedocs.io/en/stable/

Sample Input (citation_validator factory):
  '''python
  min_similarity = 90.0
  validator_creator = citation_validator(min_similarity=min_similarity)
  '''

Sample Input (generated validator):
  '''python
  corpus = "The quick brown fox jumps over the lazy dog."
  response = "A quick brown fox jumped."
  validator = validator_creator(corpus)
  result = validator(response)
  '''

Sample Output (generated validator):
  '''python
  # result will be True
  '''
"""
from typing import Any, Callable, Union, Optional
import logging
from loguru import logger
from pathlib import Path # Added for __main__
import sys # Added for __main__

# Import fuzzy matching library
try:
    from rapidfuzz import fuzz
except ImportError:
    logging.error("rapidfuzz not installed. Install with: pip install rapidfuzz")
    raise

from pdf_extractor.llm_integration.validation_utils.base import extract_content


def citation_validator(min_similarity: float = 95.0) -> Callable:
    """
    Creates a validator that checks if content is properly cited from a corpus.

    This validator requires a corpus to be provided either:
    1. At creation time: citation_validator(min_similarity)(corpus)
    2. When used with retry_llm_call via the corpus parameter

    Args:
        min_similarity: Minimum similarity percentage (0-100) required to pass validation

    Returns:
        A validator function generator that checks if content is cited from the corpus
    """
    def create_validator(corpus: Optional[str] = None) -> Callable[[Any], Union[bool, str]]:
        """
        Creates the actual validator function with the provided corpus.

        Args:
            corpus: The text corpus to check against

        Returns:
            A validation function
        """
        def validate(response: Any) -> Union[bool, str]:
            """
            Checks if the response content is sufficiently similar to the corpus.

            Args:
                response: The LLM response to validate

            Returns:
                True if validation passes, or error message string if it fails
            """
            # If no corpus is provided, this validator can't work
            if corpus is None:
                return "Citation validation requires a corpus to be provided"

            # Extract content from the response
            content = extract_content(response)

            if not content or not corpus:
                return "Empty response or corpus cannot be validated for citations"

            # For longer content, we'll check paragraph by paragraph
            # This is more robust than checking the entire content at once
            paragraphs = content.split("\n\n")

            # For very short content, check the whole thing
            if len(paragraphs) <= 1 or len(content) < 100:
                # Use partial_ratio to find if the content exists within the corpus
                content_lower = content.lower()
                corpus_lower = corpus.lower()
                # Use token_set_ratio for better matching when order/extra words differ
                similarity = fuzz.token_set_ratio(content_lower, corpus_lower)
                logger.debug(f"Citation Val (Short): Comparing '{content_lower}' against '{corpus_lower}'. Token Set Ratio: {similarity:.1f}% (Threshold: {min_similarity}%)") # UPDATED LOGGING
                if similarity >= min_similarity:
                    return True
                else:
                    # Update error message slightly for clarity
                    return f"Token set ratio similarity to corpus is {similarity:.1f}%, which is below the required {min_similarity}%"

            # For longer content, check each paragraph and get the highest partial similarity
            max_similarity = 0.0
            for paragraph in paragraphs:
                # Skip very short paragraphs
                if len(paragraph.strip()) < 10:
                    continue

                # Calculate token_set_ratio similarity for this paragraph against the whole corpus
                para_lower = paragraph.lower()
                corpus_lower = corpus.lower()
                similarity = fuzz.token_set_ratio(para_lower, corpus_lower)
                logger.debug(f"Citation Val (Para): Comparing '{para_lower[:50]}...' against corpus. Token Set Ratio: {similarity:.1f}%") # UPDATED LOGGING
                max_similarity = max(max_similarity, similarity)

                # If any paragraph meets the threshold, consider it valid
                if similarity >= min_similarity:
                    return True

            # No paragraph met the threshold
            return f"Maximum token set ratio similarity to corpus is {max_similarity:.1f}%, which is below the required {min_similarity}%"

        # Mark this validator as requiring a corpus
        validate.needs_corpus = True # type: ignore[attr-defined] # Ignore Pylance error for dynamic attribute
        return validate

    return create_validator


def extract_citation_validator(required_phrases: list, min_similarity: float = 90.0) -> Callable:
    """
    Creates a validator that checks if specific required phrases are cited.

    This is a more targeted validator that checks for citation of specific key phrases
    rather than overall content similarity.

    Args:
        required_phrases: List of phrases that must be cited from the corpus
        min_similarity: Minimum similarity percentage required for phrase matching

    Returns:
        A validator function generator
    """
    def create_validator(corpus: Optional[str] = None) -> Callable[[Any], Union[bool, str]]:
        def validate(response: Any) -> Union[bool, str]:
            if corpus is None:
                return "Citation validation requires a corpus to be provided"

            content = extract_content(response)

            if not content:
                return "Empty response cannot be validated for citations"

            # Check each required phrase
            missing_phrases = []
            for phrase in required_phrases:
                # Look for this phrase in the corpus
                if phrase not in corpus:
                    missing_phrases.append(f"'{phrase}' (not found in corpus)")
                    continue

                # Check if the phrase was properly cited
                best_similarity = 0.0
                content_paragraphs = content.split("\n")

                for paragraph in content_paragraphs:
                    similarity = fuzz.partial_ratio(phrase.lower(), paragraph.lower())
                    best_similarity = max(best_similarity, similarity)

                    if similarity >= min_similarity:
                        break

                if best_similarity < min_similarity:
                    missing_phrases.append(f"'{phrase}' (similarity: {best_similarity:.1f}%)")

            if missing_phrases:
                return f"Missing or inadequately cited phrases: {', '.join(missing_phrases)}"

            return True

        validate.needs_corpus = True # type: ignore[attr-defined] # Ignore Pylance error for dynamic attribute
        return validate

    return create_validator


# --- Main Execution Guard ---
if __name__ == "__main__":
   # Need to import the reporting function
   try:
       # Adjust relative path based on potential execution context
       from .reporting import report_validation_results
   except ImportError:
       try:
           # If running from validation_utils directory directly
           sys.path.append(str(Path(__file__).resolve().parent))
           from reporting import report_validation_results
       except ImportError:
           print("❌ FATAL: Could not import report_validation_results. Ensure it's in the same directory or PYTHONPATH.")
           sys.exit(1)

   # Configure Loguru for verification output
   logger.remove()
   logger.add(sys.stderr, level="INFO") # Use INFO for verification summary

   logger.info("Starting Citation Validators Standalone Verification...")

   # --- Define Verification Logic ---
   all_tests_passed = True
   all_failures = {}

   # --- Test Data ---
   SAMPLE_CORPUS = """
   The quick brown fox jumps over the lazy dog. This is the first paragraph.
   It contains common words and follows standard English structure.

   A second paragraph provides additional context. Lorem ipsum dolor sit amet,
   consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et
   dolore magna aliqua. Key phrase one is important.
   """
   REQUIRED_PHRASES = ["quick brown fox", "Key phrase one"]

   test_cases = [
       # citation_validator tests
       {"id": "citation_success_exact", "validator_factory": citation_validator, "args": {"min_similarity": 95.0}, "corpus": SAMPLE_CORPUS, "response": "The quick brown fox jumps over the lazy dog.", "should_pass": True},
       {"id": "citation_success_partial", "validator_factory": citation_validator, "args": {"min_similarity": 85.0}, "corpus": SAMPLE_CORPUS, "response": "A quick brown fox jumped.", "should_pass": True},
       {"id": "citation_fail_low_sim", "validator_factory": citation_validator, "args": {"min_similarity": 95.0}, "corpus": SAMPLE_CORPUS, "response": "A slow red cat sleeps.", "should_pass": False},
       {"id": "citation_fail_no_corpus", "validator_factory": citation_validator, "args": {"min_similarity": 95.0}, "corpus": None, "response": "The quick brown fox.", "should_pass": False}, # Expect failure message
       {"id": "citation_success_long_content", "validator_factory": citation_validator, "args": {"min_similarity": 90.0}, "corpus": SAMPLE_CORPUS, "response": "This response is longer.\n\nIt includes Key phrase one which is important.\n\nMore text here.", "should_pass": True},

       # extract_citation_validator tests
       {"id": "extract_citation_success", "validator_factory": extract_citation_validator, "args": {"required_phrases": REQUIRED_PHRASES, "min_similarity": 90.0}, "corpus": SAMPLE_CORPUS, "response": "The response mentioned the quick brown fox and also Key phrase one.", "should_pass": True},
       {"id": "extract_citation_fail_missing_phrase", "validator_factory": extract_citation_validator, "args": {"required_phrases": REQUIRED_PHRASES, "min_similarity": 90.0}, "corpus": SAMPLE_CORPUS, "response": "The response mentioned the quick brown fox but forgot the other one.", "should_pass": False},
       {"id": "extract_citation_fail_low_sim", "validator_factory": extract_citation_validator, "args": {"required_phrases": REQUIRED_PHRASES, "min_similarity": 95.0}, "corpus": SAMPLE_CORPUS, "response": "The quick brown fox and Key phrase 1.", "should_pass": False}, # Similarity likely too low for "Key phrase 1" vs "Key phrase one" at 95%
       {"id": "extract_citation_fail_phrase_not_in_corpus", "validator_factory": extract_citation_validator, "args": {"required_phrases": ["Nonexistent phrase"], "min_similarity": 90.0}, "corpus": SAMPLE_CORPUS, "response": "This cites something else.", "should_pass": False}, # Expect failure message about phrase not in corpus
       {"id": "extract_citation_fail_no_corpus", "validator_factory": extract_citation_validator, "args": {"required_phrases": REQUIRED_PHRASES, "min_similarity": 90.0}, "corpus": None, "response": "The quick brown fox.", "should_pass": False}, # Expect failure message
   ]

   # --- Run Verification ---
   logger.info("--- Testing Citation Validators ---")
   validation_result = None # Initialize outside loop for except block access
   for test in test_cases:
       test_id = test["id"]
       logger.debug(f"Running test: {test_id}")
       try:
           # Create the validator instance
           validator_factory = test["validator_factory"]
           validator_instance_creator = validator_factory(**test["args"])
           # Provide the corpus to the instance creator
           validator_func = validator_instance_creator(test["corpus"])

           # Run the validation
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
           # Check if failure was expected due to missing corpus
           should_fail_exception = (test.get("corpus") is None and test_id.endswith("_no_corpus")) or \
                                   (test_id == "extract_citation_fail_phrase_not_in_corpus") # Specific cases might return error strings instead of raising

           if test["should_pass"] is False and isinstance(validation_result, str): # If failure expected and validator returned error string
                logger.info(f"✅ {test_id}: Failed as expected with message: '{validation_result}'")
           elif not test["should_pass"] and should_fail_exception:
                logger.info(f"✅ {test_id}: Failed with expected condition (e.g., no corpus): {e}")
           else: # Unexpected exception or failed when should pass
               all_tests_passed = False
               all_failures[f"{test_id}_exception"] = {"expected": "Clean run or expected failure message", "actual": f"Exception: {e}"}
               logger.error(f"❌ {test_id}: Threw unexpected exception or failed incorrectly: {e}", exc_info=True)


   # --- Report Results ---
   exit_code = report_validation_results(
       validation_passed=all_tests_passed,
       validation_failures=all_failures,
       exit_on_failure=False # Let sys.exit handle it
   )

   logger.info(f"Citation Validators Standalone Verification finished with exit code: {exit_code}")
   sys.exit(exit_code)