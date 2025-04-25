#!/usr/bin/env python
"""
Validation Module for ArangoDB Lessons CLI Implementation.

This module implements comprehensive validation for the Lessons CLI
following the requirements in VALIDATION_REQUIREMENTS.md. It verifies
actual results against expected results and provides detailed error
reporting for any mismatches.

Usage:
    ```
    uv run src/pdf_extractor/arangodb/validate_lessons_cli.py
    ```
"""

import os
import sys
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import subprocess
from datetime import datetime
from loguru import logger

# --- Add parent directory to path for module imports ---
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

# --- Import project modules ---
try:
    from pdf_extractor.arangodb.lessons import (
        add_lesson,
        get_lesson,
        update_lesson,
        delete_lesson,
    )
    from pdf_extractor.arangodb.arango_setup import (
        connect_arango,
        ensure_database,
    )
    from pdf_extractor.arangodb.config import (
        COLLECTION_NAME,
    )
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# --- Fixtures Setup ---
FIXTURES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "test_fixtures")
)
EXPECTED_FIXTURE = os.path.join(FIXTURES_DIR, "lessons_cli_expected.json")

# --- Helper Functions ---
def ensure_fixtures_dir():
    """Ensure the fixtures directory exists."""
    Path(FIXTURES_DIR).mkdir(parents=True, exist_ok=True)

def load_or_create_fixture() -> Dict[str, Any]:
    """
    Load expected fixture data or create it if it doesn't exist.
    
    Returns:
        Dict containing expected test data
    """
    ensure_fixtures_dir()
    
    if os.path.exists(EXPECTED_FIXTURE):
        try:
            with open(EXPECTED_FIXTURE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load fixture: {e}")
            return create_fixture()
    else:
        return create_fixture()

def create_fixture() -> Dict[str, Any]:
    """
    Create a new fixture with expected test data.
    
    Returns:
        Dict containing expected test data
    """
    test_fixture = {
        "create_test": {
            "_key": f"test_lesson_{uuid.uuid4().hex[:8]}",
            "project": "pdf_extractor",
            "module": "lessons_cli",
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "author": "Claude",
            "tags": ["test", "validation", "arangodb"],
            "problem": "Implementing CLI for ArangoDB lessons",
            "solution": "Create a comprehensive CLI with validation",
            "lessons": [
                {
                    "category": "Implementation",
                    "title": "CLI Design",
                    "description": "Use typer for CLI implementation",
                    "details": "Typer provides a clean API for building CLI applications",
                    "benefit": "Reduces boilerplate code and improves maintainability"
                }
            ]
        },
        "update_test": {
            "solution": "Updated solution with improved approach",
            "tags": ["test", "validation", "updated"]
        },
        "cli_commands": {
            "add": "uv run src/pdf_extractor/arangodb/lessons_cli.py add --problem \"Test problem\" --solution \"Test solution\" --project \"pdf_extractor\" --module \"test_module\" --tags \"test,cli\" --author \"Tester\"",
            "get": "uv run src/pdf_extractor/arangodb/lessons_cli.py get {key}",
            "list": "uv run src/pdf_extractor/arangodb/lessons_cli.py list",
            "update": "uv run src/pdf_extractor/arangodb/lessons_cli.py update {key} --solution \"Updated solution\"",
            "delete": "uv run src/pdf_extractor/arangodb/lessons_cli.py delete {key} --confirm false"
        }
    }
    
    # Save the fixture
    ensure_fixtures_dir()
    with open(EXPECTED_FIXTURE, "w") as f:
        json.dump(test_fixture, f, indent=2)
        
    logger.info(f"Created new test fixture: {EXPECTED_FIXTURE}")
    return test_fixture

def run_cli_command(command: str) -> Tuple[int, str, str]:
    """
    Run a CLI command and return its exit code, stdout, and stderr.
    
    Args:
        command: Command string to execute
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr

def compare_json_output(expected: Dict[str, Any], actual_json_str: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Compare expected JSON data with actual JSON string output.
    
    Args:
        expected: Expected JSON data
        actual_json_str: Actual JSON string from command output
        
    Returns:
        Tuple of (all_match, list_of_errors)
    """
    validation_failures = []
    
    try:
        # Parse the actual JSON
        actual = json.loads(actual_json_str)
        
        # Check each expected field
        for field, expected_value in expected.items():
            if field not in actual:
                validation_failures.append({
                    "field": field,
                    "expected": expected_value,
                    "actual": "FIELD MISSING"
                })
                continue
            
            actual_value = actual[field]
            
            # Skip _id, _rev fields for validation
            if field in ["_id", "_rev"]:
                continue
                
            # Compare values
            if expected_value != actual_value:
                validation_failures.append({
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value
                })
        
        return len(validation_failures) == 0, validation_failures
        
    except json.JSONDecodeError as e:
        validation_failures.append({
            "error": "JSON parsing error",
            "message": str(e),
            "output": actual_json_str
        })
        return False, validation_failures

# --- Main Validation Function ---
def validate_lessons_cli():
    """
    Validate the Lessons CLI implementation.
    
    Returns:
        True if all validations pass, False otherwise
    """
    logger.info("Starting Lessons CLI validation")
    
    # Setup
    validation_passed = True
    validation_failures = {}
    
    # Load test fixtures
    fixtures = load_or_create_fixture()
    
    # 1. Test the add command
    logger.info("Testing 'add' command...")
    add_command = fixtures["cli_commands"]["add"]
    add_exit_code, add_stdout, add_stderr = run_cli_command(add_command)
    
    if add_exit_code != 0:
        validation_failures["add_command"] = {
            "expected": "Exit code 0",
            "actual": f"Exit code {add_exit_code}",
            "stderr": add_stderr
        }
        validation_passed = False
        logger.error(f"'add' command failed: {add_stderr}")
    else:
        logger.info("'add' command executed successfully")
        
        # Extract the key from stdout for further commands
        try:
            # Find the key in the output
            import re
            key_match = re.search(r"key: (.+?)[\n\"]", add_stdout)
            
            if key_match:
                lesson_key = key_match.group(1)
                logger.info(f"Extracted lesson key: {lesson_key}")
            else:
                # Try to parse JSON from output to find _key
                try:
                    output_json = json.loads(add_stdout.strip())
                    if isinstance(output_json, dict) and "_key" in output_json:
                        lesson_key = output_json["_key"]
                        logger.info(f"Extracted lesson key from JSON: {lesson_key}")
                    else:
                        lesson_key = None
                        logger.error("Could not extract lesson key from JSON")
                except:
                    lesson_key = None
                    logger.error("Could not extract lesson key from output")
            
            if not lesson_key:
                validation_failures["add_command_key"] = {
                    "expected": "Lesson key in output",
                    "actual": "No key found",
                    "stdout": add_stdout
                }
                validation_passed = False
            else:
                # 2. Test the get command
                logger.info(f"Testing 'get' command with key: {lesson_key}")
                get_command = fixtures["cli_commands"]["get"].format(key=lesson_key)
                get_exit_code, get_stdout, get_stderr = run_cli_command(get_command)
                
                if get_exit_code != 0:
                    validation_failures["get_command"] = {
                        "expected": "Exit code 0",
                        "actual": f"Exit code {get_exit_code}",
                        "stderr": get_stderr
                    }
                    validation_passed = False
                    logger.error(f"'get' command failed: {get_stderr}")
                else:
                    logger.info("'get' command executed successfully")
                    
                    # Verify expected fields in the output
                    expected_fields = ["problem", "solution", "project", "module", "tags"]
                    for field in expected_fields:
                        if field not in get_stdout:
                            validation_failures[f"get_command_{field}"] = {
                                "expected": f"Field '{field}' in output",
                                "actual": f"Field '{field}' not found",
                                "stdout": get_stdout[:200] + "..." if len(get_stdout) > 200 else get_stdout
                            }
                            validation_passed = False
                
                # 3. Test the list command
                logger.info("Testing 'list' command...")
                list_command = fixtures["cli_commands"]["list"]
                list_exit_code, list_stdout, list_stderr = run_cli_command(list_command)
                
                if list_exit_code != 0:
                    validation_failures["list_command"] = {
                        "expected": "Exit code 0",
                        "actual": f"Exit code {list_exit_code}",
                        "stderr": list_stderr
                    }
                    validation_passed = False
                    logger.error(f"'list' command failed: {list_stderr}")
                else:
                    logger.info("'list' command executed successfully")
                    
                    # Verify the lesson key appears in the list output
                    if lesson_key not in list_stdout:
                        validation_failures["list_command_key"] = {
                            "expected": f"Lesson key '{lesson_key}' in output",
                            "actual": "Key not found in list output",
                            "stdout": list_stdout[:200] + "..." if len(list_stdout) > 200 else list_stdout
                        }
                        validation_passed = False
                
                # 4. Test the update command
                logger.info(f"Testing 'update' command with key: {lesson_key}")
                update_command = fixtures["cli_commands"]["update"].format(key=lesson_key)
                update_exit_code, update_stdout, update_stderr = run_cli_command(update_command)
                
                if update_exit_code != 0:
                    validation_failures["update_command"] = {
                        "expected": "Exit code 0",
                        "actual": f"Exit code {update_exit_code}",
                        "stderr": update_stderr
                    }
                    validation_passed = False
                    logger.error(f"'update' command failed: {update_stderr}")
                else:
                    logger.info("'update' command executed successfully")
                    
                    # Verify updated solution in the output
                    if "Updated solution" not in update_stdout:
                        validation_failures["update_command_solution"] = {
                            "expected": "Updated solution in output",
                            "actual": "Updated solution not found",
                            "stdout": update_stdout[:200] + "..." if len(update_stdout) > 200 else update_stdout
                        }
                        validation_passed = False
                
                # 5. Test the delete command
                logger.info(f"Testing 'delete' command with key: {lesson_key}")
                delete_command = fixtures["cli_commands"]["delete"].format(key=lesson_key)
                delete_exit_code, delete_stdout, delete_stderr = run_cli_command(delete_command)
                
                if delete_exit_code != 0:
                    validation_failures["delete_command"] = {
                        "expected": "Exit code 0",
                        "actual": f"Exit code {delete_exit_code}",
                        "stderr": delete_stderr
                    }
                    validation_passed = False
                    logger.error(f"'delete' command failed: {delete_stderr}")
                else:
                    logger.info("'delete' command executed successfully")
                    
                    # Verify successful deletion message
                    if "Successfully deleted" not in delete_stdout:
                        validation_failures["delete_command_success"] = {
                            "expected": "Successfully deleted message",
                            "actual": "Success message not found",
                            "stdout": delete_stdout
                        }
                        validation_passed = False
                    
                    # Verify the lesson is actually deleted
                    get_after_delete_command = fixtures["cli_commands"]["get"].format(key=lesson_key)
                    get_after_delete_exit_code, get_after_delete_stdout, get_after_delete_stderr = run_cli_command(get_after_delete_command)
                    
                    if get_after_delete_exit_code == 0:
                        validation_failures["delete_command_verification"] = {
                            "expected": "Non-zero exit code (lesson should be deleted)",
                            "actual": f"Exit code {get_after_delete_exit_code}",
                            "stdout": get_after_delete_stdout
                        }
                        validation_passed = False
                    else:
                        logger.info("Verified lesson deletion")
        except Exception as e:
            validation_failures["command_execution"] = {
                "error": str(e),
                "message": "Failed to execute CLI validation sequence"
            }
            validation_passed = False
            logger.exception("Error during CLI validation")
    
    # Report validation status
    if validation_passed:
        logger.info("✅ VALIDATION COMPLETE - All Lessons CLI tests passed")
        
        # Save validation results to fixture
        fixture_path = os.path.join(FIXTURES_DIR, "lessons_cli_validation_results.json")
        with open(fixture_path, "w") as f:
            results = {
                "validation_passed": True,
                "timestamp": datetime.now().isoformat(),
                "commands_tested": list(fixtures["cli_commands"].keys())
            }
            json.dump(results, f, indent=2)
            
        logger.info(f"Saved validation results to: {fixture_path}")
        return True
    else:
        logger.error("❌ VALIDATION FAILED - Lessons CLI tests failed")
        logger.error(f"FAILURE DETAILS:")
        for field, details in validation_failures.items():
            expected = details.get("expected", "N/A")
            actual = details.get("actual", "N/A")
            logger.error(f"  - {field}: Expected: {expected}, Got: {actual}")
        
        # Save validation failures to fixture
        fixture_path = os.path.join(FIXTURES_DIR, "lessons_cli_validation_failures.json")
        with open(fixture_path, "w") as f:
            results = {
                "validation_passed": False,
                "timestamp": datetime.now().isoformat(),
                "failures": validation_failures
            }
            json.dump(results, f, indent=2)
            
        logger.info(f"Saved validation failures to: {fixture_path}")
        logger.error(f"Total errors: {len(validation_failures)} fields mismatched")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr, 
        level="INFO", 
        format="{time:HH:mm:ss} | {level:<7} | {message}"
    
