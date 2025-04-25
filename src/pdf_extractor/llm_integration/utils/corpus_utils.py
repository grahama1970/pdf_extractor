# -*- coding: utf-8 -*-
"""
Corpus Loading Utilities for LLM Integration Engine.

Provides functionality to load text content from various sources like strings,
files (txt, pdf, json, html, md), and directories, handling different encodings
and formats.

Links:
- PyMuPDF (fitz): https://pymupdf.readthedocs.io/en/latest/
- html2text: https://github.com/Alir3z4/html2text
- bleach: https://bleach.readthedocs.io/en/latest/

Sample Input (validation_options dict):
{
    "corpus_source": "./my_corpus_dir",
    "corpus_type": "directory",
    "recursive": True,
    "file_patterns": ["*.txt", "*.md"]
}
or
{
    "corpus_source": "path/to/my_document.pdf",
    "corpus_type": "file"
}
or
{
    "corpus_source": "This is the corpus text.",
    "corpus_type": "string"
}


Sample Output (string or None):
"Content from file1.txt\\n\\n---\\n\\nContent from file2.md"
or
"Content extracted from my_document.pdf..."
or
"This is the corpus text."
or
None (if loading fails)
"""
import glob
import json
import sys
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from loguru import logger

# Optional imports for specific file types
try:
    import fitz # PyMuPDF
except ImportError:
    logger.warning("PyMuPDF (fitz) not installed. PDF corpus loading will not work. Install with: uv add PyMuPDF")
    fitz = None
try:
    import html2text
    import bleach
except ImportError:
    logger.warning("html2text or bleach not installed. HTML corpus loading will not work. Install with: uv add html2text bleach")
    html2text = None
    bleach = None


def load_corpus_from_source(validation_options: Dict[str, Any], task_id: str) -> Optional[str]:
    """
    Loads corpus text based on validation_options configuration.

    Supports loading from:
    - Direct string (`corpus_type`: "string")
    - Single file (`corpus_type`: "file", supports .txt, .md, .json, .pdf, .html)
    - Directory (`corpus_type`: "directory", supports glob patterns and recursion)

    Args:
        validation_options: Dictionary containing corpus loading configuration.
            Expected keys:
            - `corpus_source`: The string, file path, or directory path.
            - `corpus_type`: "string", "file", or "directory".
            - `recursive` (optional, for directory): bool, default False.
            - `file_patterns` (optional, for directory): List[str], default ["*"].
        task_id: Identifier for the task requesting the corpus (for logging).

    Returns:
        The loaded corpus text as a single string, or None if loading failed.
        Multiple files from a directory are joined with "\\n\\n---\\n\\n".
    """
    corpus_source = validation_options.get("corpus_source")
    # Default to 'string' if type not specified or if source looks like non-path text
    default_type = "string" if isinstance(corpus_source, str) and not Path(corpus_source).exists() and not Path(corpus_source).is_dir() else "file"
    corpus_type = validation_options.get("corpus_type", default_type)


    if not corpus_source:
        logger.debug(f"Task {task_id}: No corpus_source provided in validation_options.")
        return None

    loaded_texts: List[str] = []

    if corpus_type == "string":
        if isinstance(corpus_source, str):
            logger.debug(f"Task {task_id}: Using direct string corpus.")
            loaded_texts.append(corpus_source)
        else:
            logger.warning(f"Task {task_id}: corpus_type is 'string' but corpus_source is not a string ({type(corpus_source)}). Skipping.")

    elif corpus_type == "file":
        if not isinstance(corpus_source, str):
            logger.warning(f"Task {task_id}: corpus_type is 'file' but corpus_source is not a string path ({type(corpus_source)}). Skipping.")
            return None

        file_path = Path(corpus_source)
        logger.debug(f"Task {task_id}: Attempting to load corpus from file: {file_path}")
        if not file_path.is_file():
            logger.warning(f"Task {task_id}: Corpus file not found: {file_path}. Skipping.")
            return None

        try:
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                if fitz:
                    doc_text = ""
                    with fitz.open(file_path) as doc:
                        for page in doc:
                            # Use get_text() which is the correct method for PyMuPDF > 1.18.0
                            doc_text += page.get_text("text") + "\n"
                    loaded_texts.append(doc_text)
                    logger.debug(f"Task {task_id}: Successfully extracted text from PDF: {file_path}")
                else:
                    logger.warning(f"Task {task_id}: Cannot load PDF corpus {file_path}, PyMuPDF (fitz) not installed. Skipping.")
            elif suffix == ".json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Represent JSON as string for corpus consistency
                    loaded_texts.append(json.dumps(data, indent=2))
                    logger.debug(f"Task {task_id}: Successfully loaded and stringified JSON: {file_path}")
            elif suffix == ".html" or suffix == ".htm":
                if html2text and bleach:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    text_content = html2text.html2text(html_content)
                    cleaned_text = bleach.clean(text_content, tags=[], strip=True)
                    loaded_texts.append(cleaned_text)
                    logger.debug(f"Task {task_id}: Successfully converted HTML to text: {file_path}")
                else:
                    logger.warning(f"Task {task_id}: Cannot load HTML corpus {file_path}, html2text or bleach not installed. Skipping.")
            elif suffix in [".txt", ".md", ""]: # Treat .txt, .md, and no extension as plain text
                 with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_texts.append(f.read())
                 logger.debug(f"Task {task_id}: Successfully loaded text/markdown file: {file_path}")
            else: # Fallback for other types
                 logger.warning(f"Task {task_id}: Unsupported file type '{suffix}' for corpus file {file_path}. Attempting to read as plain text.")
                 try:
                     # Try reading with utf-8 first, fallback to latin-1 if that fails
                     try:
                         with open(file_path, 'r', encoding='utf-8') as f:
                             loaded_texts.append(f.read())
                     except UnicodeDecodeError:
                         logger.warning(f"Task {task_id}: UTF-8 decoding failed for {file_path}. Trying latin-1.")
                         with open(file_path, 'r', encoding='latin-1') as f:
                             loaded_texts.append(f.read())
                     logger.debug(f"Task {task_id}: Successfully read unsupported file as text: {file_path}")
                 except Exception as read_err:
                     logger.warning(f"Task {task_id}: Failed to read file {file_path} as plain text: {read_err}. Skipping.")

        except Exception as e:
            logger.warning(f"Task {task_id}: Error processing corpus file {file_path}: {e}. Skipping.")

    elif corpus_type == "directory":
        if not isinstance(corpus_source, str):
            logger.warning(f"Task {task_id}: corpus_type is 'directory' but corpus_source is not a string path ({type(corpus_source)}). Skipping.")
            return None

        dir_path = Path(corpus_source)
        logger.debug(f"Task {task_id}: Attempting to load corpus from directory: {dir_path}")
        if not dir_path.is_dir():
            logger.warning(f"Task {task_id}: Corpus directory not found: {dir_path}. Skipping.")
            return None

        recursive = validation_options.get("recursive", False)
        patterns = validation_options.get("file_patterns", ["*"])
        if not isinstance(patterns, list):
            logger.warning(f"Task {task_id}: 'file_patterns' should be a list, got {type(patterns)}. Defaulting to ['*'].")
            patterns = ["*"]

        files_processed_count = 0
        for pattern in patterns:
            glob_method = dir_path.rglob if recursive else dir_path.glob
            try:
                for file_path in glob_method(pattern):
                    if file_path.is_file():
                        # Recursively call this function's file loading part
                        file_options = {"corpus_source": str(file_path), "corpus_type": "file"}
                        # Use a modified task_id for clarity in logs during recursive calls
                        file_task_id = f"{task_id} [from dir:{file_path.name}]"
                        # Recursive call using the public function name
                        file_text = load_corpus_from_source(file_options, file_task_id)
                        if file_text:
                            loaded_texts.append(file_text)
                            files_processed_count += 1
            except Exception as glob_err:
                 logger.warning(f"Task {task_id}: Error during glob operation for pattern '{pattern}' in {dir_path}: {glob_err}")

        logger.debug(f"Task {task_id}: Loaded {files_processed_count} files from directory {dir_path}.")


    else:
        logger.warning(f"Task {task_id}: Unknown corpus_type '{corpus_type}'. Skipping corpus loading.")

    if not loaded_texts:
        logger.warning(f"Task {task_id}: No corpus text could be loaded based on validation_options.")
        return None

    # Combine all loaded texts
    return "\n\n---\n\n".join(loaded_texts) # Separate file contents clearly

# --- Main Execution Guard (Standalone Verification) ---
if __name__ == "__main__":
    # Need reporting for standardized output
    try:
        # Adjust relative path based on potential execution context
        from ..validation_utils.reporting import report_validation_results
    except ImportError:
         try:
            # If running from utils directory directly
            sys.path.append(str(Path(__file__).resolve().parent.parent / 'validation_utils'))
            from reporting import report_validation_results
         except ImportError:
            print("❌ FATAL: Could not import report_validation_results.")
            sys.exit(1)


    # Configure Loguru for verification output
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("Starting Corpus Utils Standalone Verification...")

    all_tests_passed = True
    all_failures = {}
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)

    # --- Create Test Files ---
    test_files = {
        "test.txt": "Plain text content.",
        "test.md": "# Markdown Content\n\nWith a paragraph.",
        "test.json": json.dumps({"json_key": "json_value"}),
        "nested/test_nested.txt": "Nested text content.",
        "other.dat": "Some other data format.",
        # PDF and HTML require external libs, test conditionally
    }
    # Create nested dir
    (temp_dir_path / "nested").mkdir(parents=True, exist_ok=True) # Ensure parent exists
    for fname, content in test_files.items():
        try:
            # Ensure parent directory exists for nested files
            full_path = temp_dir_path / fname
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to create test file {fname}: {e}")
            all_tests_passed = False # Cannot proceed reliably
            all_failures["create_test_file"] = {"expected": "Success", "actual": f"Failed for {fname}: {e}"}

    # --- Test Cases ---
    test_cases = [
        {"id": "string_direct", "options": {"corpus_source": "Direct string.", "corpus_type": "string"}, "expected": "Direct string."},
        {"id": "file_txt", "options": {"corpus_source": str(temp_dir_path / "test.txt"), "corpus_type": "file"}, "expected": test_files["test.txt"]},
        {"id": "file_md", "options": {"corpus_source": str(temp_dir_path / "test.md"), "corpus_type": "file"}, "expected": test_files["test.md"]},
        {"id": "file_json", "options": {"corpus_source": str(temp_dir_path / "test.json"), "corpus_type": "file"}, "expected": json.dumps(json.loads(test_files["test.json"]), indent=2)}, # Expect pretty-printed
        {"id": "file_other_dat", "options": {"corpus_source": str(temp_dir_path / "other.dat"), "corpus_type": "file"}, "expected": test_files["other.dat"]}, # Expect fallback to text
        {"id": "file_nonexistent", "options": {"corpus_source": str(temp_dir_path / "nonexistent.txt"), "corpus_type": "file"}, "expected": None},
        {"id": "dir_flat_all", "options": {"corpus_source": str(temp_dir_path), "corpus_type": "directory"}, "expected_parts": [test_files["test.txt"], test_files["test.md"], json.dumps(json.loads(test_files["test.json"]), indent=2), test_files["other.dat"]]}, # Order might vary
        {"id": "dir_flat_txt", "options": {"corpus_source": str(temp_dir_path), "corpus_type": "directory", "file_patterns": ["*.txt"]}, "expected_parts": [test_files["test.txt"]]},
        {"id": "dir_recursive_txt", "options": {"corpus_source": str(temp_dir_path), "corpus_type": "directory", "recursive": True, "file_patterns": ["*.txt"]}, "expected_parts": [test_files["test.txt"], test_files["nested/test_nested.txt"]]}, # Order might vary
        {"id": "dir_nonexistent", "options": {"corpus_source": str(temp_dir_path / "nosuchdir"), "corpus_type": "directory"}, "expected": None},
        {"id": "invalid_type", "options": {"corpus_source": "abc", "corpus_type": "invalid"}, "expected": None},
        {"id": "no_source", "options": {}, "expected": None},
    ]

    # --- Run Verification ---
    if all_tests_passed: # Only run if test files were created
        logger.info("--- Testing load_corpus_from_source ---")
        for test in test_cases:
            test_id = test["id"]
            logger.debug(f"Running test: {test_id}")
            try:
                # Call the public function name in the test
                actual_result = load_corpus_from_source(test["options"], test_id)

                # Validation logic
                passed = False
                failures = {} # Initialize failures for this test case
                if "expected_parts" in test:
                    # For directory results, check if all expected parts are present
                    if actual_result is None:
                        passed = False
                        failures = {test_id: {"expected": "String containing parts", "actual": None}}
                    else:
                        actual_parts = actual_result.split("\n\n---\n\n")
                        # Use sets for easier comparison regardless of order
                        expected_set = set(test["expected_parts"])
                        actual_set = set(actual_parts)
                        if expected_set == actual_set:
                            passed = True
                        else:
                            passed = False
                            missing_parts = list(expected_set - actual_set)
                            extra_parts = list(actual_set - expected_set)
                            failures = {test_id: {"expected": f"Parts: {sorted(list(expected_set))}", "actual": f"Parts: {sorted(list(actual_set))} (Missing: {missing_parts}, Extra: {extra_parts})"}}
                else:
                    # For single string or None results
                    if actual_result == test["expected"]:
                        passed = True
                    else:
                        passed = False
                        failures = {test_id: {"expected": test["expected"], "actual": actual_result}}

                # Reporting
                if passed:
                    logger.info(f"✅ {test_id}: Passed.")
                else:
                    all_tests_passed = False
                    all_failures.update(failures)
                    logger.error(f"❌ {test_id}: Failed. Details: {failures}")

            except Exception as e:
                all_tests_passed = False
                all_failures[f"{test_id}_exception"] = {"expected": "Clean run", "actual": f"Exception: {e}"}
                logger.error(f"❌ {test_id}: Threw unexpected exception: {e}", exc_info=True)

    # --- Report Results ---
    exit_code = report_validation_results(
        validation_passed=all_tests_passed,
        validation_failures=all_failures,
        exit_on_failure=False
    )

    # --- Cleanup ---
    try:
        temp_dir.cleanup()
        logger.info(f"Cleaned up temporary directory: {temp_dir_path}")
    except Exception as e:
        logger.warning(f"Could not clean up temporary directory {temp_dir_path}: {e}")


    logger.info(f"Corpus Utils Standalone Verification finished with exit code: {exit_code}")
    sys.exit(exit_code)