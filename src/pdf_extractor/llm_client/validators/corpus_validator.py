# Corpus Validation Module
#
# This module handles validation of LLM responses against a corpus of text paragraphs
# using fuzzy matching techniques from RapidFuzz.

import re
from typing import Dict, List, Tuple, Any
from loguru import logger
from rapidfuzz import fuzz

from pdf_extractor.llm_client.text_utils import highlight_matching_words

def validate_corpus_match(response_text: str, corpus: List[str], threshold: int = 75, task_id: str = "") -> Tuple[bool, Dict[str, Any]]:
    """
    Validates if a response text exists with high similarity somewhere in the corpus.
    
    This checks if the content of the response aligns with approved content
    in the corpus - similar to ensuring compliance with technical standards,
    precedent law, or regulatory requirements.
    
    Args:
        response_text: The LLM response to validate
        corpus: List of approved reference paragraphs
        threshold: Minimum similarity percentage required
        task_id: Task identifier for logging
        
    Returns:
        (is_valid, results): Validation result and detailed match information
    """
    response_clean = response_text.strip().lower()
    
    # Log what we're validating
    logger.debug(f"[Task {task_id}] Validating response against corpus of {len(corpus)} paragraphs")
    logger.debug(f"[Task {task_id}] Response text: \"{response_text[:100]}...\"")
    
    # Initialize results
    results = {
        "valid": False,
        "best_score": 0,
        "best_match": "",
        "best_method": "",
        "matching_words": [],
        "missing_words": [],
        "match_details": None
    }
    
    # Check against each paragraph in the corpus
    for i, para in enumerate(corpus):
        para_clean = para.strip().lower()
        
        # Try different fuzzy matching methods
        token_set_score = fuzz.token_set_ratio(response_clean, para_clean)
        token_sort_score = fuzz.token_sort_ratio(response_clean, para_clean)
        partial_score = fuzz.partial_ratio(response_clean, para_clean)
        simple_score = fuzz.ratio(response_clean, para_clean)
        
        # Track all scores
        method_scores = {
            "token_set": token_set_score,
            "token_sort": token_sort_score,
            "partial": partial_score,
            "simple": simple_score
        }
        
        # Find best method for this paragraph
        best_method = max(method_scores.items(), key=lambda x: x[1])
        method_name, score = best_method
        
        # Log paragraph comparison
        logger.debug(f"[Task {task_id}] Corpus paragraph {i+1}: {para[:50]}...")
        logger.debug(f"[Task {task_id}]   Scores: {method_scores}")
        
        # If this is the best match so far
        if score > results["best_score"]:
            results["best_score"] = score
            results["best_match"] = para
            results["best_method"] = method_name
            
            # Find matching and missing keywords
            response_words = set(re.findall(r'\b\w+\b', response_clean))
            para_words = set(re.findall(r'\b\w+\b', para_clean))
            
            # Filter out common words
            common_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are'}
            key_response_words = response_words - common_words
            key_para_words = para_words - common_words
            
            results["matching_words"] = sorted(key_response_words & key_para_words)
            results["missing_words"] = sorted(key_para_words - key_response_words)
            
            # Create highlighted versions for visualization
            highlighted_resp, highlighted_para = highlight_matching_words(response_text, para)
            
            # Calculate word overlap for statistics
            results["match_details"] = {
                "highlighted_response": highlighted_resp,
                "highlighted_paragraph": highlighted_para,
                "matching_words_count": len(results["matching_words"]),
                "total_words_paragraph": len(key_para_words),
                "word_overlap_percent": round(len(results["matching_words"]) / len(key_para_words) * 100, 1) 
                                        if key_para_words else 0
            }
    
    # Check if best match exceeds threshold
    results["valid"] = results["best_score"] >= threshold
    
    # Log validation result
    if results["valid"]:
        logger.debug(f"[Task {task_id}] ✅ Corpus validation PASSED: {results['best_score']}% match")
    else:
        logger.debug(f"[Task {task_id}] ❌ Corpus validation FAILED: {results['best_score']}% match (threshold: {threshold}%)")
    
    return results["valid"], results

if __name__ == "__main__":
    import sys
    
    # Test with a good match
    quantum_corpus = [
        "Quantum computing uses the principles of quantum mechanics to process information. Unlike classical computers that use bits, quantum computers use quantum bits or qubits.",
        "The power of quantum computing comes from the ability to use quantum phenomena such as superposition and entanglement.",
        "A qubit can exist in multiple states simultaneously due to superposition, unlike a classical bit which can only be 0 or 1.",
        "Quantum entanglement allows qubits to be correlated with each other, no matter how far apart they are physically."
    ]
    
    good_response = "A qubit is a quantum bit that can exist in multiple states simultaneously due to superposition."
    bad_response = "Quantum computers are really fast and powerful machines."
    
    # Run validation
    good_valid, good_results = validate_corpus_match(good_response, quantum_corpus, 75, "test")
    bad_valid, bad_results = validate_corpus_match(bad_response, quantum_corpus, 75, "test")
    
    # --- Detailed Validation & Tally ---
    tests_passed_count = 0
    tests_failed_count = 0
    total_tests = 2
    validation_failures = {}

    # 1. Validate good response (should be True)
    if good_valid:
        tests_passed_count += 1
        print(f"✅ Test 'good_response': PASSED (Score: {good_results['best_score']:.2f}%)")
    else:
        tests_failed_count += 1
        validation_failures["good_response"] = {"expected": True, "actual": False, "score": good_results['best_score']}
        print(f"❌ Test 'good_response': FAILED (Score: {good_results['best_score']:.2f}%)")

    # 2. Validate bad response (should be False)
    if not bad_valid:
        tests_passed_count += 1
        print(f"✅ Test 'bad_response': PASSED (Detected invalid, Score: {bad_results['best_score']:.2f}%)")
    else:
        tests_failed_count += 1
        validation_failures["bad_response"] = {"expected": False, "actual": True, "score": bad_results['best_score']}
        print(f"❌ Test 'bad_response': FAILED (Incorrectly validated, Score: {bad_results['best_score']:.2f}%)")

    # --- Report validation status ---
    print(f"\n--- Test Summary ---")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {tests_passed_count}")
    print(f"Failed: {tests_failed_count}")
    
    if tests_failed_count == 0:
        print("\n✅ VALIDATION COMPLETE - All corpus validation tests passed.")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED - Some corpus validation tests failed.")
        print("FAILURE DETAILS:")
        for test_name, details in validation_failures.items():
             print(f"  - {test_name}: Expected valid={details['expected']}, Got valid={details['actual']} (Score: {details['score']:.2f}%)")
        sys.exit(1)