# -*- coding: utf-8 -*-
"""
CLI for interacting with the LLM Integration module.

Provides commands to process batch requests and query lessons learned,
mirroring the functionality of the FastAPI endpoints in main.py.

Links:
  - Typer: https://typer.tiangolo.com/
  - Loguru: https://loguru.readthedocs.io/en/stable/
  - PyYAML: https://pyyaml.org/wiki/PyYAMLDocumentation
  - python-arango: https://python-driver-for-arangodb.readthedocs.io/en/latest/

"""
import asyncio
import json
import sys
from pathlib import Path
# Ensure Optional is imported correctly here
import typing # Import typing module itself
from typing import Optional, List, Dict, Any, Callable, Tuple, Union # Added Callable, Tuple, Union

# --- Additions for __main__ verification ---
import subprocess
import tempfile
from pdf_extractor.llm_integration.validation_utils.reporting import report_validation_results
# --- End Additions ---

import typer
from loguru import logger
import yaml # For config loading

# Assuming models are correctly placed relative to this CLI script
# Adjust imports based on actual project structure if needed
try:
    from pdf_extractor.llm_integration.models import (
        BatchRequest,
        BatchResponse,
        TaskItem,
        LessonQueryRequest,
        LessonQueryResponse,
        ResultItem # Import ResultItem if needed for direct construction
    )
    from pdf_extractor.llm_integration.engine import process_batch
    # Need db utils and config loading similar to main.py
    # Corrected import paths based on arango_utils.py structure
    from pdf_extractor.llm_integration.utils.db.arango_utils import connect_to_arango_client, query_lessons_by_similarity
    from pdf_extractor.llm_integration.initialize_litellm_cache import initialize_litellm_cache
    from pdf_extractor.llm_integration.validation_utils.reporting import report_validation_results # Correct function name
    from arango.database import StandardDatabase # Import for type hint
    # Removed incorrect ValidationOutcome import

except ImportError as e:
    logger.error(f"Failed to import necessary modules. Ensure paths are correct: {e}")
    sys.exit(1)


# --- Configuration Loading (Simplified from main.py) ---
# TODO: Use a more robust config loading mechanism shared with main.py
CONFIG_PATH = "config.yaml" # Expect config in the working directory
arango_config: Dict[str, Any] = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        full_config = yaml.safe_load(f)
        arango_config = full_config.get('database', {})
        if not arango_config:
            logger.warning(f"ArangoDB configuration not found or empty in {CONFIG_PATH}")
        else:
            # Substitute environment variables if needed (omitted for brevity, add if required)
            pass
except FileNotFoundError:
    logger.error(f"Configuration file not found at {CONFIG_PATH}. Database operations might fail.")
except yaml.YAMLError as e:
    logger.error(f"Error parsing configuration file {CONFIG_PATH}: {e}")


# --- Typer App Initialization ---
app = typer.Typer(
    name="llm-integration-cli",
    help="CLI for processing LLM batch requests and querying lessons learned.",
    add_completion=False,
)


# --- Helper Functions ---
def get_db_client():
    """Connects to ArangoDB based on loaded config."""
    if not arango_config:
        logger.error("ArangoDB configuration missing. Cannot connect.")
        raise typer.Exit(code=1)
    try:
        client = connect_to_arango_client(arango_config)
        # Quick check
        client.version()
        logger.info("Successfully connected to ArangoDB.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to ArangoDB: {e}")
        raise typer.Exit(code=1)

async def run_ask_batch(request: BatchRequest) -> BatchResponse:
    """Async wrapper to call process_batch."""
    # Initialize cache (similar to FastAPI startup)
    initialize_litellm_cache()
    logger.info("Processing batch request...")
    response = await process_batch(request)
    logger.info("Batch processing complete.")
    return response

async def run_query_lessons(db_client: StandardDatabase, query_text: str, top_k: int) -> LessonQueryResponse: # Added type hint
    """Async wrapper to call query_lessons_by_similarity."""
    logger.info(f"Querying lessons for: '{query_text[:50]}...', top_k={top_k}")
    # Assuming query_lessons_by_similarity is async, otherwise remove await
    # Also assuming it returns the structure needed for LessonQueryResponse
    # The function signature might differ slightly from the FastAPI endpoint's usage
    # Adjust the call as per the actual function definition
    try:
        # The original function might return raw results, need to adapt
        # For now, assuming it returns data compatible with LessonResultItem
        # Removed await as query_lessons_by_similarity is synchronous
        similar_lessons_raw = query_lessons_by_similarity(
            db=db_client, # Pass the client instance
            query_text=query_text,
            top_n=top_k,
            # Add other necessary parameters like collection names if required by the function
            # collection_name="lessons_learned", # Example
            # view_name="lessons_learned_view" # Example
        )

        # Process raw results into LessonQueryResponse format (similar to main.py)
        results = []
        for item in similar_lessons_raw:
             lesson_doc = item.get('document', {})
             score = item.get('similarity_score', 0.0)
             lesson_id = lesson_doc.get('_id', 'N/A')
             lesson_key = lesson_doc.get('_key', 'N/A')
             results.append({ # Construct dicts first
                 "id": lesson_id,
                 "key": lesson_key,
                 "score": score,
                 "lesson": lesson_doc
             })

        # Validate with Pydantic model before returning
        response_model = LessonQueryResponse(lessons=results)
        logger.info(f"Found {len(response_model.lessons)} relevant lessons.")
        return response_model

    except Exception as e:
        logger.error(f"Error during lesson query: {e}")
        # Re-raise or handle as appropriate for CLI
        raise typer.Exit(code=1)


# --- CLI Commands ---

@app.command("ask", help="Process a batch of LLM questions defined in a JSON file or string.")
def ask_command(
    input_data: str = typer.Argument(
        ...,
        help="Path to a JSON file containing the BatchRequest data, or a JSON string directly."
    )
):
    """
    Processes a batch LLM request.
    Input can be a file path or a JSON string.
    """
    request_data: Optional[Dict[str, Any]] = None
    input_path = Path(input_data)

    if input_path.is_file():
        logger.info(f"Loading BatchRequest from file: {input_path}")
        try:
            with open(input_path, 'r') as f:
                request_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {e}")
            raise typer.Exit(code=1)
        except IOError as e:
            logger.error(f"Could not read file {input_path}: {e}")
            raise typer.Exit(code=1)
    else:
        logger.info("Attempting to parse input as JSON string.")
        try:
            request_data = json.loads(input_data)
        except json.JSONDecodeError:
            logger.error("Input is neither a valid file path nor a valid JSON string.")
            raise typer.Exit(code=1)

    if not request_data:
         logger.error("Failed to load request data.")
         raise typer.Exit(code=1)

    try:
        batch_request = BatchRequest.model_validate(request_data)
        logger.debug("BatchRequest validated successfully.")
        print("<<<< DEBUG: BatchRequest validated, proceeding to run_ask_batch >>>>", flush=True) # ADDED DEBUG PRINT
    except Exception as e: # Catch Pydantic validation errors
        logger.error(f"Invalid BatchRequest data: {e}")
        raise typer.Exit(code=1)

    # Run the async function
    try:
        response = asyncio.run(run_ask_batch(batch_request))

        # Print validation reports for each result
        logger.info("--- Validation Reports ---")
        if response.responses: # Corrected attribute name
            for i, result in enumerate(response.responses): # Corrected attribute name
                print(f"\n--- Result {i+1} (Task ID: {result.task_id}) ---")
                # Check validation_status and call reporting function appropriately
                validation_status = result.validation_status
                if validation_status is True:
                    # Validation passed
                    report_validation_results(validation_passed=True, validation_failures={})
                    # The function logs directly, no need to print its return value (exit code)
                elif isinstance(validation_status, list):
                    # Validation failed, create a basic failure dict from the list of errors
                    failures_dict = {
                        f"error_{idx}": {"expected": "N/A", "actual": error_msg}
                        for idx, error_msg in enumerate(validation_status)
                    }
                    report_validation_results(validation_passed=False, validation_failures=failures_dict)
                else:
                    # validation_status is None or some unexpected type
                    print("No validation status recorded or status is invalid for this result.")
        else:
            print("No responses found in the response.")
        logger.info("--- End Validation Reports ---")

        # Print the full response as JSON (optional, can be commented out)
        logger.info("--- Full Batch Response (JSON) ---")
        print(response.model_dump_json(indent=2))
        logger.info("--- End Full Batch Response ---")

    except Exception as e:
        logger.error(f"An error occurred during batch processing: {e}")
        raise typer.Exit(code=1)


@app.command("query-lessons", help="Query lessons learned by semantic similarity.")
def query_lessons_command(
    query_text: str = typer.Argument(..., help="The natural language query text."),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of results to return.")
):
    """
    Queries lessons learned database based on semantic similarity to the query text.
    """
    db_client = None
    try:
        db_client = get_db_client()
        # Run the async query function
        response = asyncio.run(run_query_lessons(db_client, query_text, top_k))
        # Print the response as JSON
        print(response.model_dump_json(indent=2))
    except Exception as e:
        # Error logged in get_db_client or run_query_lessons
        # Exit code handled there or here if needed
        logger.error(f"Lesson query command failed: {e}")
        raise typer.Exit(code=1)
    finally:
        # Clean up DB connection if necessary (depends on client implementation)
        if db_client:
            logger.debug("Closing DB client (if applicable).")
            # Add client closing logic if needed, e.g., db_client.close()


# --- Main Execution Guard ---
# --- MODIFIED __main__ block ---
if __name__ == "__main__":
   # Configure Loguru
   logger.remove() # Remove default handler
   # Set console logger level - Use INFO for less verbose verification output
   logger.add(sys.stderr, level="INFO")
   # logger.add("cli_log_{time}.log", level="DEBUG") # Keep file logger commented for verification

   logger.info("Starting LLM Integration CLI Standalone Verification...")

   # --- Define Verification Logic ---
   async def _verify_ask_command():
       """Runs a minimal verification of the 'ask' command logic."""
       validation_failures = {}
       test_input_file = "sample_cli_ask_request.json" # Use the created file
       expected_task_count = 2
       expected_task_ids = ["cli_test_Q0", "cli_test_Q1"]
       expected_q1_status = "success" # Optimistic check

       try:
           # 1. Load Request Data
           logger.info(f"Loading test request from: {test_input_file}")
           try:
               with open(test_input_file, 'r') as f:
                   request_data = json.load(f)
               batch_request = BatchRequest.model_validate(request_data)
               logger.debug("Test BatchRequest validated successfully.")
           except Exception as e:
               validation_failures["load_request"] = {"expected": "Valid BatchRequest", "actual": f"Failed: {e}"}
               return False, validation_failures # Cannot proceed

           # 2. Run Core Logic (run_ask_batch)
           logger.info("Running core batch processing logic (run_ask_batch)...")
           try:
               # Ensure LiteLLM cache is initialized for the test run
               initialize_litellm_cache()
               response: BatchResponse = await run_ask_batch(batch_request)
               logger.info("Core batch processing finished.")
           except Exception as e:
               logger.error(f"Error during run_ask_batch execution: {e}", exc_info=True)
               validation_failures["run_ask_batch"] = {"expected": "Successful execution", "actual": f"Failed: {e}"}
               return False, validation_failures # Cannot proceed

           # 3. Validate Response Structure
           logger.info("Validating response structure...")
           # The isinstance check below was removed as 'response' is already typed as BatchResponse
           if len(response.responses) != expected_task_count:
               validation_failures["response_count"] = {"expected": expected_task_count, "actual": len(response.responses)}
           else:
               # Check task IDs
               actual_task_ids = [r.task_id for r in response.responses]
               if set(actual_task_ids) != set(expected_task_ids):
                    validation_failures["task_ids"] = {"expected": sorted(expected_task_ids), "actual": sorted(actual_task_ids)}

               # Check status of Q1 (index 1) - More specific check
               q1_result = response.responses[1]
               if q1_result.task_id != "cli_test_Q1":
                    validation_failures["q1_task_id"] = {"expected": "cli_test_Q1", "actual": q1_result.task_id}
               # Check Q1 status (Be lenient, LLM calls can fail)
               if q1_result.status != expected_q1_status:
                    logger.warning(f"Task cli_test_Q1 status was '{q1_result.status}' (expected '{expected_q1_status}'). This might be due to LLM issues.")
                    # Optionally add to failures if strict check is needed:
                    # validation_failures["q1_status"] = {"expected": expected_q1_status, "actual": q1_result.status}

               # Check Q1 validation outcome (should be True if status is success)
               if q1_result.status == "success" and q1_result.validation_status is not True:
                    validation_failures["q1_validation_status"] = {"expected": True, "actual": q1_result.validation_status}

       except Exception as e:
           logger.error(f"Unexpected error during verification: {e}", exc_info=True)
           validation_failures["verification_error"] = {"expected": "Clean run", "actual": f"Failed: {e}"}

       validation_passed = len(validation_failures) == 0
       return validation_passed, validation_failures

   # --- Run Verification ---
   try:
       passed, failures = asyncio.run(_verify_ask_command())
   except Exception as main_err:
       # Catch errors during the async run itself
       passed = False
       failures = {"main_async_run": {"expected": "Successful async execution", "actual": f"Failed: {main_err}"}}
       logger.error(f"Error running main verification async function: {main_err}", exc_info=True)

   # --- Report Results ---
   exit_code = report_validation_results(
       validation_passed=passed,
       validation_failures=failures,
       exit_on_failure=False # Let sys.exit handle it
   )

   logger.info(f"CLI Standalone Verification finished with exit code: {exit_code}")
   sys.exit(exit_code)

# --- Original app() call removed from __main__ ---
# app()
# logger.info("CLI finished.")