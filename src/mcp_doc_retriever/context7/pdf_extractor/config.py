"""
Configuration settings for PDF extraction and processing.

This module contains all configuration values used across the PDF extraction process,
including model settings, thresholds, and processing parameters.

Links to third-party package documentation:
- Qwen-VL: https://huggingface.co/Qwen/Qwen-VL-Chat
- Camelot: https://camelot-py.readthedocs.io/en/master/
- Tiktoken: https://github.com/openai/tiktoken

Configuration values here are used by the PDF extractor modules to control:
1. Visual-language model settings (Qwen-VL parameters)
2. PDF scanning and OCR quality thresholds
3. Table extraction parameters (using Camelot)
4. Token counting (using Tiktoken)
5. Output file organization
"""
import sys
import json
from pathlib import Path

# Qwen-VL model settings
QWEN_MODEL_NAME = "Qwen/Qwen-VL-Chat"
QWEN_MAX_NEW_TOKENS = 1024
QWEN_PROMPT = """
Please provide a detailed textual description of the image in Markdown format.
Focus on the key visual elements, ensuring any text content is accurately transcribed.
Include layout information if it's a diagram or complex visualization.
End your description with any key insights or observations about the image purpose.
"""

# PDF scanning and quality thresholds
SCANNED_CHECK_MAX_PAGES = 5  # Pages to check for scanned content detection
SCANNED_TEXT_LENGTH_THRESHOLD = 100  # Min character count per page for non-scanned
SCANNED_OCR_CONFIDENCE_THRESHOLD = 80.0  # Minimum OCR confidence percentage
LOW_CONFIDENCE_THRESHOLD = 75.0  # Table extraction confidence threshold

# Model configuration
TIKTOKEN_ENCODING_MODEL = "gpt-4"  # Model to use for token counting

# Directory paths
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_CORRECTIONS_DIR = "corrections"

# Table extraction settings
CAMELOT_DEFAULT_FLAVOR = "lattice"
CAMELOT_LATTICE_LINE_SCALE = 15  # Changed from 40 to 15 for better table extraction
CAMELOT_STREAM_EDGE_TOL = 500
TABLE_CONTEXT_MARGIN = 200.0  # pixels
TABLE_CONTEXT_MAX_LINES = 5
# Convert float to int percentage for merge threshold
TABLE_MERGE_SIMILARITY_THRESHOLD = 80  # percent (0-100)

def get_all_config():
    """Return all configuration values as a dictionary."""
    return {k: v for k, v in globals().items()
            if k.isupper() and not k.startswith('_')}

def verify_config_integrity():
    """Verify all required configuration settings are present and valid."""
    config_dict = get_all_config()
    
    # Define required settings
    required_settings = {
        'QWEN_MODEL_NAME',
        'QWEN_MAX_NEW_TOKENS',
        'QWEN_PROMPT',
        'SCANNED_CHECK_MAX_PAGES',
        'SCANNED_TEXT_LENGTH_THRESHOLD',
        'SCANNED_OCR_CONFIDENCE_THRESHOLD',
        'LOW_CONFIDENCE_THRESHOLD',
        'TIKTOKEN_ENCODING_MODEL',
    }
    
    # Check for missing settings
    missing = required_settings - set(config_dict.keys())
    if missing:
        return False, f"Missing required settings: {missing}"
    
    # Validate setting types
    type_validation = [
        (QWEN_MODEL_NAME, str, "QWEN_MODEL_NAME must be a string"),
        (QWEN_MAX_NEW_TOKENS, int, "QWEN_MAX_NEW_TOKENS must be an integer"),
        (QWEN_PROMPT, str, "QWEN_PROMPT must be a string"),
        (SCANNED_CHECK_MAX_PAGES, int, "SCANNED_CHECK_MAX_PAGES must be an integer"),
        (SCANNED_TEXT_LENGTH_THRESHOLD, int, "SCANNED_TEXT_LENGTH_THRESHOLD must be an integer"),
        (SCANNED_OCR_CONFIDENCE_THRESHOLD, (int, float), "SCANNED_OCR_CONFIDENCE_THRESHOLD must be a number"),
        (LOW_CONFIDENCE_THRESHOLD, (int, float), "LOW_CONFIDENCE_THRESHOLD must be a number"),
    ]
    
    for value, expected_type, error_msg in type_validation:
        if not isinstance(value, expected_type):
            return False, error_msg
    
    return True, "All configuration settings are valid"

if __name__ == "__main__":
    # Define expected results for validation
    EXPECTED_RESULTS = {
        "config_integrity": True,
        "categories": {
            "Model Settings": 3,
            "Quality Thresholds": 4,
            "Table Extraction": 6,
            "Directories": 2
        }
    }
    
    validation_passed = True
    actual_results = {}
    
    print("PDF Extractor Configuration Verification")
    print("======================================")
    
    # Verify config integrity
    integrity_result, integrity_message = verify_config_integrity()
    actual_results["config_integrity"] = integrity_result
    if not integrity_result:
        validation_passed = False
        print(f"✗ Configuration integrity check failed: {integrity_message}")
    else:
        print(f"✓ Configuration integrity check passed: {integrity_message}")
    
    # Group settings by category for clearer output
    config_dict = get_all_config()
    categories = {
        "Model Settings": {
            "QWEN_MODEL_NAME",
            "QWEN_MAX_NEW_TOKENS",
            "TIKTOKEN_ENCODING_MODEL"
        },
        "Quality Thresholds": {
            "SCANNED_CHECK_MAX_PAGES",
            "SCANNED_TEXT_LENGTH_THRESHOLD",
            "SCANNED_OCR_CONFIDENCE_THRESHOLD",
            "LOW_CONFIDENCE_THRESHOLD"
        },
        "Table Extraction": {
            "CAMELOT_DEFAULT_FLAVOR",
            "CAMELOT_LATTICE_LINE_SCALE",
            "CAMELOT_STREAM_EDGE_TOL",
            "TABLE_CONTEXT_MARGIN",
            "TABLE_CONTEXT_MAX_LINES",
            "TABLE_MERGE_SIMILARITY_THRESHOLD"
        },
        "Directories": {
            "DEFAULT_OUTPUT_DIR",
            "DEFAULT_CORRECTIONS_DIR"
        }
    }
    
    # Verify and display settings by category
    actual_category_counts = {}
    for category, settings in categories.items():
        print(f"\n{category}:")
        print("-" * len(category) + ":")
        
        found_settings = 0
        for setting in settings:
            if setting in config_dict:
                found_settings += 1
                value = config_dict[setting]
                # Format multiline strings nicely
                if isinstance(value, str) and "\n" in value:
                    print(f"{setting}:")
                    for line in value.strip().split("\n"):
                        print(f"    {line}")
                else:
                    print(f"{setting}: {value}")
            else:
                print(f"{setting}: NOT FOUND!")
                validation_passed = False
        
        actual_category_counts[category] = found_settings
        if found_settings != len(settings):
            validation_passed = False
    
    actual_results["categories"] = actual_category_counts
    
    # Final validation check
    print("\nValidation Results:")
    
    # Check config integrity
    integrity_match = actual_results["config_integrity"] == EXPECTED_RESULTS["config_integrity"]
    print(f"  config_integrity: {'✓' if integrity_match else '✗'} Expected: {EXPECTED_RESULTS['config_integrity']}, Got: {actual_results['config_integrity']}")
    if not integrity_match:
        validation_passed = False
    
    # Check category counts
    for category, expected_count in EXPECTED_RESULTS["categories"].items():
        actual_count = actual_results["categories"].get(category, 0)
        match = actual_count == expected_count
        print(f"  {category} count: {'✓' if match else '✗'} Expected: {expected_count}, Got: {actual_count}")
        if not match:
            validation_passed = False
    
    if validation_passed:
        print("\n✅ VALIDATION COMPLETE - All results match expected values")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED - Results don't match expected values")
        print(f"Expected: {EXPECTED_RESULTS}")
        print(f"Got: {actual_results}")
        sys.exit(1)
