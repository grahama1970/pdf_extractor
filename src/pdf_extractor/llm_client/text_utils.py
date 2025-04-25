# Text Utilities for LLM Validation
#
# This module provides text processing utilities for validation of LLM responses.

import re
from typing import Tuple

def highlight_matching_words(text1: str, text2: str) -> Tuple[str, str]:
    """
    Highlight matching words between two texts.
    Returns both texts with HTML-like tags to highlight matching words.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Tuple containing both texts with matching words highlighted
    """
    # Get all words from both texts
    words1 = re.findall(r'\b\w+\b', text1.lower())
    words2 = re.findall(r'\b\w+\b', text2.lower())
    
    # Find matching words
    matching_words = set(words1) & set(words2)
    
    # Create highlighted versions
    highlighted1 = text1
    highlighted2 = text2
    
    for word in matching_words:
        # Don't highlight very common words
        if word in {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}:
            continue
            
        # Case insensitive replace with highlighting
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        highlighted1 = pattern.sub(f"[{word}]", highlighted1)
        highlighted2 = pattern.sub(f"[{word}]", highlighted2)
    
    return highlighted1, highlighted2

if __name__ == "__main__":
    import sys
    
    # Test case 1: Texts with matching words
    text1 = "Quantum computing uses qubits instead of regular bits."
    text2 = "A qubit can exist in multiple states due to quantum principles."
    
    highlighted1, highlighted2 = highlight_matching_words(text1, text2)
    
    print("Original texts:")
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print("\nHighlighted texts:")
    print(f"Text 1: {highlighted1}")
    print(f"Text 2: {highlighted2}")
    
    # --- Detailed Validation & Tally ---
    tests_passed_count = 0
    tests_failed_count = 0
    total_tests = 1
    test_passed = False
    
    # 1. Check if highlighting happened correctly
    # Based on the function logic, only 'quantum' is common between the two texts.
    expected_highlights = ["[quantum]"]
    unexpected_highlights = ["[qubit]", "[qubits]"] # Ensure these are NOT highlighted

    highlight_check_passed = all(hl in highlighted1 and hl in highlighted2 for hl in expected_highlights)
    highlight_check_failed = any(
        uhl in highlighted1 or uhl in highlighted2 for uhl in unexpected_highlights
    )

    if highlight_check_passed and not highlight_check_failed:
        tests_passed_count += 1
        test_passed = True
        print("\n✅ Test 'highlighting': PASSED")
    else:
        tests_failed_count += 1
        print("\n❌ Test 'highlighting': FAILED")
        if not highlight_check_passed:
             print(f"   Expected '{' and '.join(expected_highlights)}' to be highlighted in both texts.")
        if highlight_check_failed:
             print(f"   Unexpected highlights found: found one of {' or '.join(unexpected_highlights)}.")
        print(f"   Got Text 1: {highlighted1}")
        print(f"   Got Text 2: {highlighted2}")

    # --- Report validation status ---
    print(f"\n--- Test Summary ---")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {tests_passed_count}")
    print(f"Failed: {tests_failed_count}")

    if tests_failed_count == 0:
        print("\n✅ VALIDATION COMPLETE - All text utility tests passed.")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED - Some text utility tests failed.")
        sys.exit(1)
