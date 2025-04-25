# LLM Validation Client Demo
#
# This module demonstrates using the llm_client package to make validated LLM calls.
# It showcases different validation strategies: exact, list, corpus, and fuzzy matching.
#
# NOTE: This script makes REAL LLM calls (using codellama-7b locally by default)
# and requires the corresponding LLM service and Redis (for caching) to be running.
# Ensure API keys/endpoints are configured correctly (e.g., via .env).
#
# ## Third-Party Packages:
# - litellm: https://github.com/BerriAI/litellm
# - rapidfuzz: https://github.com/maxbachmann/RapidFuzz
# - loguru: https://github.com/Delgan/loguru
# - pydantic: https://docs.pydantic.dev/latest/
# - tqdm: https://github.com/tqdm/tqdm
#
# ## Sample Input:
# Predefined test cases for different validation types.
#
# ## Expected Output:
# Log messages showing the progress and results of each validation test call.
# Exits with code 0 on success, 1 on failure.

import asyncio
import sys
import logging
import os
import re # Keep re and os for potential future log processing integration if desired

# Third-party
from loguru import logger
import litellm
from tqdm import tqdm # Keep tqdm for potential future use

# Local
# Assuming the script is run from the project root
from pdf_extractor.llm_client.initialize_litellm_cache import initialize_litellm_cache
from pdf_extractor.llm_client.validation_client import get_llm_response
from pdf_extractor.llm_client.schema_models import QuestionAnswer, CodeResponse

# Configure logger (optional, can rely on validation_client's logger)
logger.remove()
logger.add(
    sink=lambda msg: tqdm.write(msg, end=""),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    filter=lambda record: "Provider List:" not in record["message"]
            and "https://docs.litellm.ai/docs/providers" not in record["message"]
)
logger.add(
    "llm_client_demo.log",
    level="DEBUG",
    rotation="10 MB",
     filter=lambda record: "Provider List:" not in record["message"]
            and "https://docs.litellm.ai/docs/providers" not in record["message"]
)


async def main() -> None:
    """
    Run validation tests for different validation types using the llm_client.
    """
    logger.info("\n========= LLM CLIENT VALIDATION DEMO =========\n")
    overall_success = True
    test_results = {}

    # Silence litellm logs during cache init
    # for name in logging.root.manager.loggerDict:
    #     if name.startswith("litellm"):
    #         logging.getLogger(name).setLevel(logging.CRITICAL)

    # Suppress stdout from cache init if desired (can make debugging harder)
    # orig_stdout = sys.stdout
    # sys.stdout = open(os.devnull, 'w')
    try:
        initialize_litellm_cache()
    except Exception as e:
        logger.error(f"❌ Cache initialization failed: {e}. Demo cannot proceed reliably.")
        sys.exit(1)
    # finally:
        # sys.stdout.close()
        # sys.stdout = orig_stdout

    litellm.enable_json_schema_validation = True # Ensure schema validation is on

    # --- Test 1: Exact Validation ---
    logger.info("\n========== 1. EXACT VALIDATION TEST ==========")
    logger.info("Testing if LLM correctly answers '4' to '2+2'")
    logger.info("============================================\n")
    test_call_exact = {
        "model": "gpt-4o-mini", # Use remote model for testing
        "prompt": "What is 2+2? Provide only the number.",
        # "local": True, # Remove local flag
        "response_schema": QuestionAnswer,
        "validation": {
            "type": "exact",
            "expected": "4",
            "field": "answer"
        },
        "max_validation_retries": 1 # Limit retries for demo speed
    }
    try:
        results_exact = await get_llm_response(test_call_exact)
        # Basic check: Did the call succeed and return a result?
        # More specific validation happens inside get_llm_response and is logged.
        test_results["exact"] = bool(results_exact and results_exact[0][1] is not None)
        if not test_results["exact"]:
             logger.error("❌ Exact validation test appeared to fail based on final response.")
             overall_success = False
    except Exception as e:
        logger.error(f"❌ Exact validation test failed with exception: {e}")
        test_results["exact"] = False
        overall_success = False


    # --- Test 2: List Validation ---
    logger.info("\n========== 2. LIST VALIDATION TEST ==========")
    logger.info("Testing if LLM correctly identifies 'Paris' as the capital of France")
    logger.info("===========================================\n")
    list_test = {
        "model": "gpt-4o-mini", # Use remote model for testing
        "prompt": "What is the capital of France? Provide only the name.",
        # "local": True, # Remove local flag
        "response_schema": QuestionAnswer,
        "validation": {
            "type": "list",
            "allowed_values": ["Madrid", "Berlin", "Paris", "Rome", "London"],
            "expected": "Paris", # Optional: check if it's this specific value in the list
            "field": "answer"
        },
         "max_validation_retries": 1
    }
    try:
        results_list = await get_llm_response(list_test)
        test_results["list"] = bool(results_list and results_list[0][1] is not None)
        if not test_results["list"]:
             logger.error("❌ List validation test appeared to fail based on final response.")
             overall_success = False
    except Exception as e:
        logger.error(f"❌ List validation test failed with exception: {e}")
        test_results["list"] = False
        overall_success = False

    # --- Test 3: Corpus Validation ---
    logger.info("\n========== 3. CORPUS VALIDATION TEST ==========")
    logger.info("Testing if LLM explanation of a qubit matches the quantum computing corpus")
    logger.info("=============================================\n")
    corpus = [
        "Quantum computing uses the principles of quantum mechanics to process information. Unlike classical computers that use bits, quantum computers use quantum bits or qubits.",
        "The power of quantum computing comes from the ability to use quantum phenomena such as superposition and entanglement.",
        "A qubit can exist in multiple states simultaneously due to superposition, unlike a classical bit which can only be 0 or 1.",
        "Quantum entanglement allows qubits to be correlated with each other, no matter how far apart they are physically."
    ]
    corpus_test = {
        "model": "gpt-4o-mini", # Use remote model for testing
        "prompt": "Explain what a qubit is in one sentence.",
        # "local": True, # Remove local flag
        "validation": {
            "type": "corpus",
            "corpus": corpus,
            "threshold": 70 # Lower threshold slightly for demo robustness
        },
         "max_validation_retries": 1
    }
    try:
        results_corpus = await get_llm_response(corpus_test)
        # Corpus validation logs pass/fail internally, check if response was received
        test_results["corpus"] = bool(results_corpus and results_corpus[0][1] is not None)
        if not test_results["corpus"]:
             logger.error("❌ Corpus validation test appeared to fail based on final response.")
             overall_success = False
    except Exception as e:
        logger.error(f"❌ Corpus validation test failed with exception: {e}")
        test_results["corpus"] = False
        overall_success = False

    # --- Test 4: Fuzzy Validation ---
    logger.info("\n========== 4. FUZZY VALIDATION TEST ==========")
    logger.info("Testing if LLM's factorial function contains expected code pattern")
    logger.info("============================================\n")
    fuzzy_test = {
        "model": "gpt-4o-mini", # Use remote model for testing
        "prompt": "Write a Python function to calculate factorial.",
        # "local": True, # Remove local flag
        "response_schema": CodeResponse,
        "validation": {
            "type": "fuzzy",
            "expected": "def factorial(n):", # More specific pattern
            "field": "code",
            "threshold": 80
        },
         "max_validation_retries": 1
    }
    try:
        results_fuzzy = await get_llm_response(fuzzy_test)
        test_results["fuzzy"] = bool(results_fuzzy and results_fuzzy[0][1] is not None)
        if not test_results["fuzzy"]:
             logger.error("❌ Fuzzy validation test appeared to fail based on final response.")
             overall_success = False
    except Exception as e:
        logger.error(f"❌ Fuzzy validation test failed with exception: {e}")
        test_results["fuzzy"] = False
        overall_success = False

    # --- Demo Summary ---
    logger.info("\n========== DEMO SUMMARY ==========")
    logger.info(f"Exact Validation Test:  {'✅ PASSED' if test_results.get('exact') else '❌ FAILED'}")
    logger.info(f"List Validation Test:   {'✅ PASSED' if test_results.get('list') else '❌ FAILED'}")
    logger.info(f"Corpus Validation Test: {'✅ PASSED' if test_results.get('corpus') else '❌ FAILED'}")
    logger.info(f"Fuzzy Validation Test:  {'✅ PASSED' if test_results.get('fuzzy') else '❌ FAILED'}")
    logger.info("================================\n")

    if overall_success:
        logger.success("✅ All LLM client validation demos completed successfully (check logs for details).")
        sys.exit(0)
    else:
        logger.error("❌ Some LLM client validation demos failed (check logs for details).")
        sys.exit(1)


if __name__ == "__main__":
    # Silence lower-level logs if desired
    logging.basicConfig(level=logging.CRITICAL)
    # for name in ["httpx", "httpcore"]:
    #     logging.getLogger(name).setLevel(logging.CRITICAL)
    #     logging.getLogger(name).propagate = False

    asyncio.run(main())