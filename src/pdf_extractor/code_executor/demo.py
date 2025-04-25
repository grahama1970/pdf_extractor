# LLM Validation Demo
#
# This module demonstrates the validation mechanisms supported by the 
# code executor system, using direct testing rather than LLM calls.
#
# ## Third-Party Packages:
# - loguru: https://github.com/Delgan/loguru (v0.7.0)
#
# ## Sample Input:
# Test cases for validation functions
#
# ## Expected Output:
# Log messages showing validation test results for different validation types

# === Standard library ===
import sys
import logging

# === Third-party ===
from loguru import logger

# === Local ===
from pdf_extractor.code_executor.validators.code_validator import validate_code_execution, extract_code_from_text
from pdf_extractor.code_executor.validators.corpus_validator import validate_corpus_match

def run_validation_demo():
    """
    Run validation tests for the different validation mechanisms.
    Using direct function calls rather than LLM integration.
    """
    logger.info("\n========= CODE EXECUTOR VALIDATION DEMO =========\n")
    
    # Silence other loggers
    logging.basicConfig(level=logging.CRITICAL)
    
    # Test data
    factorial_code = """
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)

print(factorial(5))
"""
    
    buggy_code = """
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        # Bug: missing return statement
        n * factorial(n-1)

print(factorial(5))
"""

    # === Code Execution Validation ===
    logger.info("\n========== CODE EXECUTION VALIDATION TEST ==========")
    logger.info("Testing if factorial function executes correctly")
    logger.info("====================================================\n")
    
    # Test with valid code
    valid_result, valid_details = validate_code_execution(
        code=factorial_code,
        task_id="test_valid",
        call_id=1
    )
    
    # Test with buggy code
    invalid_result, invalid_details = validate_code_execution(
        code=buggy_code,
        task_id="test_invalid",
        call_id=2
    )
    
    # === Code Extraction ===
    logger.info("\n========== CODE EXTRACTION TEST ==========")
    logger.info("Testing extraction of code from text")
    logger.info("=========================================\n")
    
    text_with_code = """
Here's a factorial function in Python:

```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)
```

You can use it to calculate 5! = 120.
"""
    
    extracted_code = extract_code_from_text(text_with_code)
    extraction_success = "factorial" in extracted_code
    
    if extraction_success:
        logger.info(f"✅ Code extraction PASSED")
        logger.info(f"Extracted code:\n{extracted_code}")
    else:
        logger.error(f"❌ Code extraction FAILED")
    
    # === Corpus Validation ===
    logger.info("\n========== CORPUS VALIDATION TEST ==========")
    logger.info("Testing text validation against reference corpus")
    logger.info("=============================================\n")
    
    # Example corpus with quantum computing content
    corpus = [
        "Quantum computing uses the principles of quantum mechanics to process information. Unlike classical computers that use bits, quantum computers use quantum bits or qubits.",
        "The power of quantum computing comes from the ability to use quantum phenomena such as superposition and entanglement.",
        "A qubit can exist in multiple states simultaneously due to superposition, unlike a classical bit which can only be 0 or 1.",
        "Quantum entanglement allows qubits to be correlated with each other, no matter how far apart they are physically."
    ]
    
    # Good match
    good_text = "A qubit is a quantum bit that can exist in multiple states simultaneously due to superposition."
    good_valid, good_results = validate_corpus_match(good_text, corpus, 75, "test_good")
    
    # Poor match
    poor_text = "Quantum computers are really fast and powerful machines."
    poor_valid, poor_results = validate_corpus_match(poor_text, corpus, 75, "test_poor")
    
    # Check test results
    tests_passed = valid_result and not invalid_result and extraction_success and good_valid and not poor_valid
    
    # Results summary
    logger.info("\n========== VALIDATION SUMMARY ==========")
    logger.info(f"Valid code execution: {'✅ PASSED' if valid_result else '❌ FAILED'}")
    logger.info(f"Invalid code detection: {'✅ PASSED' if not invalid_result else '❌ FAILED'}")
    logger.info(f"Code extraction: {'✅ PASSED' if extraction_success else '❌ FAILED'}")
    logger.info(f"Corpus validation (good match): {'✅ PASSED' if good_valid else '❌ FAILED'}")
    logger.info(f"Corpus validation (poor match): {'✅ PASSED' if not poor_valid else '❌ FAILED'}")
    
    if tests_passed:
        logger.success("\n✅ All validation tests passed successfully!")
        return 0
    else:
        logger.error("\n❌ Some validation tests failed")
        return 1

if __name__ == "__main__":
    # Silence httpx and httpcore logs
    for name in ["httpx", "httpcore"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).propagate = False

    # Run the demo
    exit_code = run_validation_demo()
    sys.exit(exit_code)