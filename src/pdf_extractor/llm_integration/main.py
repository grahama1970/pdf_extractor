# Description: Main FastAPI application for the MCP LiteLLM Service.
#              Handles incoming batch requests, processes them using the engine,
#              and returns the results. Includes startup logic for cache initialization.
# Core Lib Links:
# - FastAPI: https://fastapi.tiangolo.com/
# - Uvicorn: https://www.uvicorn.org/
# Sample I/O: N/A (This is the main application entry point)

import yaml
import os
from typing import List, Dict, Any, Callable, Awaitable, Union # Added Callable, Awaitable, Union
import sys # Added for exit codes in main
import asyncio # Added for asyncio.run
import json # Added for loading sample request
from pathlib import Path # Added for path handling
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from loguru import logger
from fastapi_mcp import FastApiMCP # Changed import
from fastapi.middleware.cors import CORSMiddleware
from arango.client import ArangoClient # Corrected import path
# import uvicorn # Not directly used, but related for running the app

from pdf_extractor.llm_integration.models import (
    BatchRequest,
    BatchResponse,
    LessonQueryRequest,
    LessonQueryResponse,
    LessonResultItem
)
from pdf_extractor.llm_integration.engine import process_batch
from pdf_extractor.llm_integration.initialize_litellm_cache import initialize_litellm_cache
# Use absolute import
from pdf_extractor.llm_integration.utils.db.arango_utils import connect_to_arango_client, query_lessons_by_similarity
# --- Configuration Loading ---
# Load ArangoDB config - adjust path as necessary
# TODO: Use a more robust config loading mechanism (e.g., Pydantic settings)
CONFIG_PATH = "config.yaml" # Expect config in the working directory (/app)
arango_config: Dict[str, Any] = {} # Use lowercase and add type hint
try:
    with open(CONFIG_PATH, 'r') as f:
        full_config = yaml.safe_load(f)
        arango_config = full_config.get('database', {}) # Use lowercase
        if not arango_config: # Use lowercase
            logger.warning(f"ArangoDB configuration not found or empty in {CONFIG_PATH}")
        else:
            # Substitute environment variables in arango_config
            for key, value in arango_config.items(): # Use lowercase
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    env_value = os.environ.get(env_var)
                    if env_value is not None:
                        arango_config[key] = env_value # Use lowercase
                    else:
                        logger.warning(f"Environment variable '{env_var}' not set for database config key '{key}'")
except FileNotFoundError:
    logger.error(f"Configuration file not found at {CONFIG_PATH}") # Keep as error
except yaml.YAMLError as e:
    logger.error(f"Error parsing configuration file {CONFIG_PATH}: {e}")


# --- Dependency for ArangoDB Connection ---
from arango.database import StandardDatabase # Add import

def get_db() -> StandardDatabase: # Correct return type
    """
    Dependency function to get an ArangoDB database connection.
    Handles connection errors.
    """
    if not arango_config: # Use lowercase
        logger.error("ArangoDB connection cannot be established: Configuration missing.")
        raise HTTPException(status_code=500, detail="Database configuration error.")
    try:
        # Assuming connect_to_arango_client returns the db object directly
        db = connect_to_arango_client(arango_config) # Use lowercase
        # Optional: Add a check to ensure the connection is live
        # db.version()
        return db
    except Exception as e:
        logger.error(f"Failed to connect to ArangoDB: {e}")
        raise HTTPException(status_code=503, detail=f"Could not connect to the database: {e}")


# --- FastAPI Application Instance ---
app = FastAPI(
    title="MCP LiteLLM Service",
    version="0.1.0",
    description="A service to process batch requests for LiteLLM calls via MCP and query lessons learned.",
)

# Instantiate the MCP server
mcp_server = FastApiMCP(
    fastapi=app, # Pass the FastAPI app instance here
    name="mcp-litellm-batch-fastapi", # Server name for MCP discovery
    # Automatically describe responses based on FastAPI response_model
    describe_all_responses=True
)
# Mount the MCP server endpoints
mcp_server.mount(mount_path="/mcp") # Specify the mount path here

# Add CORS middleware to allow connections from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-known-frontend.com"], # TODO: restrict to known client origins before production
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)
@app.middleware("http") # type: ignore
async def add_security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]): # Specify Callable signature
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response


@app.on_event("startup") # type: ignore
async def startup_event():
    """
    Initializes the LiteLLM cache upon application startup.
    """
    logger.info("Initializing LiteLLM cache on startup...")
    # Assuming initialize_litellm_cache is synchronous based on example/plan
    # If it were async, it would need an 'await'
    initialize_litellm_cache()
    logger.info("Cache initialization complete.")

@app.post("/ask", response_model=BatchResponse, summary="Process a batch of LLM questions") # type: ignore
async def ask_batch(request: BatchRequest):
    """
    Accepts a batch of questions, processes them concurrently or sequentially
    based on dependencies using the engine, and returns the results.

    Handles potential errors during processing and returns appropriate HTTP responses.
    """
    logger.info(f"Received batch request with {len(request.tasks)} tasks.")
    try:
        # The core logic is delegated to the process_batch function
        response = await process_batch(request)
        logger.info(f"Successfully processed batch request.")
        return response
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        logger.warning(f"HTTP Exception during processing: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        logger.exception(f"Fatal error processing batch request: {e}")
        # Log full error, return generic 500 error to the client
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/", summary="Health check") # type: ignore
async def read_root():
    """
    Provides a basic health check endpoint to confirm the service is running.
    """
    logger.debug("Root health check endpoint accessed.")
    return {"message": "MCP LiteLLM Service is running"}


# --- Lessons Learned Query Endpoint ---

@app.post("/query_lessons", # type: ignore
          response_model=LessonQueryResponse,
          summary="Query Lessons Learned by Semantic Similarity",
          tags=["Lessons Learned"])
async def query_lessons(
    request: LessonQueryRequest,
    db: StandardDatabase = Depends(get_db) # Correct dependency type
):
    """
    Accepts a natural language query and returns the top_k most semantically
    similar lessons learned from the ArangoDB database.

    Requires a vector index on the 'lesson_embedding' field in the
    'lessons_learned' collection.
    """
    logger.info(f"Received lessons query: '{request.query_text[:50]}...', top_k={request.top_k}")
    try:
        # Call the utility function to perform the similarity search
        # This function handles embedding the query text and querying ArangoDB
        similar_lessons_raw: List[Dict[str, Any]] = query_lessons_by_similarity(
            db=db,
            query_text=request.query_text,
            top_n=request.top_k
        )

        # Process results into the response model format
        results = []
        for item in similar_lessons_raw:
            lesson_doc = item.get('document', {})
            score = item.get('similarity_score', 0.0)
            lesson_id = lesson_doc.get('_id', 'N/A')
            lesson_key = lesson_doc.get('_key', 'N/A')

            results.append(LessonResultItem(
                id=lesson_id,
                key=lesson_key,
                score=score,
                lesson=lesson_doc # Include the full lesson document
            ))

        logger.info(f"Returning {len(results)} similar lessons for query.")
        return LessonQueryResponse(lessons=results)

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (e.g., from get_db)
        logger.warning(f"HTTP Exception during lessons query: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during query or processing
        logger.exception(f"Error processing lessons query: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")




# --- Standalone Validation Block --- 

async def main_validation():
    """
    Performs end-to-end validation by processing a sample batch request
    directly through the ask_batch function and verifying the results.
    """
    logger.info("--- Running Standalone Validation for main.py (End-to-End Test) ---")
    overall_validation_passed = True # Renamed for clarity
    errors = []
    batch_response: Optional[BatchResponse] = None

    # --- Sample Request Data with Enhanced Corpus Tests ---
    SAMPLE_REQUEST_DATA_ENHANCED = {
      "tasks": [
        # Original tests (keep for regression)
        {
          "task_id": "json_ok",
          "question": "Respond ONLY with the following JSON, no other text: {\"name\": \"Roo\", \"status\": \"coding\"}",
          "model": "openai/gpt-4o-mini",
          "validation_strategy": "json",
          "validation_options": {"required_fields": ["name", "status"]}
        },
        # New tests for file/directory loading
        {
          "task_id": "cite_txt",
          "question": "Summarize the text corpus content [SOURCE_1]",
          "model": "openai/gpt-4o-mini",
          "validation_strategy": "citation",
          "validation_options": {
            "corpus_source": "test_corpus.txt",
            "corpus_type": "file",
            "min_similarity": 85.0 # Adjust similarity as needed
          }
        },
        {
          "task_id": "cite_md",
          "question": "Summarize the markdown corpus content [SOURCE_1]",
          "model": "openai/gpt-4o-mini",
          "validation_strategy": "citation",
          "validation_options": {
            "corpus_source": "test_corpus.md",
            "corpus_type": "file",
            "min_similarity": 85.0
          }
        },
        {
          "task_id": "cite_json",
          "question": "Summarize the JSON corpus content [SOURCE_1]",
          "model": "openai/gpt-4o-mini",
          "validation_strategy": "citation",
          "validation_options": {
            "corpus_source": "test_corpus.json",
            "corpus_type": "file",
            "min_similarity": 85.0
          }
        },
         {
          "task_id": "cite_pdf",
          "question": "Summarize the PDF corpus content [SOURCE_1]",
          "model": "openai/gpt-4o-mini",
          "validation_strategy": "citation",
          "validation_options": {
            "corpus_source": "test_corpus.pdf",
            "corpus_type": "file",
            "min_similarity": 85.0
          }
        },
        {
          "task_id": "cite_html",
          "question": "Summarize the HTML corpus content [SOURCE_1]",
          "model": "openai/gpt-4o-mini",
          "validation_strategy": "citation",
          "validation_options": {
            "corpus_source": "test_corpus.html",
            "corpus_type": "file",
            "min_similarity": 85.0
          }
        },
        {
          "task_id": "cite_dir_recursive",
          "question": "Summarize the content from the text and nested files [SOURCE_1]",
          "model": "openai/gpt-4o-mini",
          "validation_strategy": "citation",
          "validation_options": {
            "corpus_source": ".", # Current directory
            "corpus_type": "directory",
            "file_patterns": ["test_corpus.txt", "test_corpus_dir/*.txt"],
            "recursive": True,
            "min_similarity": 85.0
          }
        },
         {
          "task_id": "cite_dir_nonrecursive",
          "question": "Summarize the content from the top-level text file only [SOURCE_1]",
          "model": "openai/gpt-4o-mini",
          "validation_strategy": "citation",
          "validation_options": {
            "corpus_source": ".", # Current directory
            "corpus_type": "directory",
            "file_patterns": ["test_corpus.txt"], # Only match top-level txt
            "recursive": False,
            "min_similarity": 85.0
          }
        },
        # Test bad file path
        {
          "task_id": "cite_bad_file",
          "question": "This should fail corpus loading [SOURCE_1]",
          "model": "openai/gpt-4o-mini",
          "validation_strategy": "citation",
          "validation_options": {
            "corpus_source": "non_existent_file.txt",
            "corpus_type": "file",
            "min_similarity": 85.0
          }
        }
      ],
      "max_concurrency": 3 # Increase concurrency slightly
    }


    # --- Expected Outcomes (Updated) ---
    EXPECTED_OUTCOMES = {
        "json_ok": {"status": "success", "validation_status": True},
        # Assuming citation tasks will likely fail similarity check without specific tuning/prompts
        "cite_txt": {"status": "error", "validation_status_is_list": True},
        "cite_md": {"status": "error", "validation_status_is_list": True},
        "cite_json": {"status": "error", "validation_status_is_list": True},
        "cite_pdf": {"status": "error", "validation_status_is_list": True},
        "cite_html": {"status": "error", "validation_status_is_list": True},
        "cite_dir_recursive": {"status": "error", "validation_status_is_list": True},
        "cite_dir_nonrecursive": {"status": "error", "validation_status_is_list": True},
        "cite_bad_file": {"status": "error", "validation_status_is_list": True}, # Expect failure as corpus loading fails
    }

    try:
        # 1. Load Sample Request Data
        batch_request = BatchRequest.model_validate(SAMPLE_REQUEST_DATA_ENHANCED)
        logger.info(f"Validated enhanced sample request data with {len(batch_request.tasks)} tasks.")

        # 2. Initialize Cache (needed by engine)
        initialize_litellm_cache()

        # 3. Process Batch Request directly via ask_batch
        logger.info("Processing sample batch request...")
        # Directly call the endpoint logic function
        batch_response = await ask_batch(batch_request)
        logger.info("Sample batch request processed.")

        # 4. Validate Response against Expected Outcomes
        if not batch_response or not batch_response.responses or len(batch_response.responses) != len(EXPECTED_OUTCOMES):
             errors.append(f"Response validation failed: Response missing, empty, or length mismatch. Got {len(batch_response.responses) if batch_response else 'None'} responses, expected {len(EXPECTED_OUTCOMES)}.")
             overall_validation_passed = False
        else:
            logger.info("Validating individual task results...")
            results_map = {res.task_id: res for res in batch_response.responses}
            for task_id, expected in EXPECTED_OUTCOMES.items():
                if task_id not in results_map:
                    errors.append(f"Task '{task_id}': Missing from response.")
                    overall_validation_passed = False
                    continue

                actual = results_map[task_id]
                task_passed = True

                # Check status
                if actual.status != expected["status"]:
                    errors.append(f"Task '{task_id}': Status mismatch. Expected: {expected['status']}, Got: {actual.status}")
                    task_passed = False

                # Check validation_status type/value
                if "validation_status" in expected:
                    if actual.validation_status != expected["validation_status"]:
                        errors.append(f"Task '{task_id}': Validation status mismatch. Expected: {expected['validation_status']}, Got: {actual.validation_status}")
                        task_passed = False
                elif "validation_status_is_list" in expected:
                     if not isinstance(actual.validation_status, list):
                         errors.append(f"Task '{task_id}': Validation status type mismatch. Expected: list, Got: {type(actual.validation_status).__name__}")
                         task_passed = False

                if task_passed:
                     logger.debug(f"Task '{task_id}': Validation passed.")
                else:
                     overall_validation_passed = False # Mark overall as failed if any task fails

    except FileNotFoundError as e:
        errors.append(f"Setup error: {e}")
        overall_validation_passed = False
    except Exception as e:
        errors.append(f"Runtime error during validation: {e}")
        logger.exception("Detailed error during main_validation:")
        overall_validation_passed = False

    # --- Report Final Validation Status ---
    if overall_validation_passed:
        logger.success("✅ Standalone validation passed: Sample batch processed and results match expected outcomes.")
        print("\n✅ VALIDATION COMPLETE - End-to-end processing verified.")
        sys.exit(0)
    else:
        logger.error("❌ Standalone validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        # Optionally print the full response if it exists
        if batch_response:
             logger.error("Full Response:")
             try:
                 print(batch_response.model_dump_json(indent=2))
             except Exception:
                 logger.error("Could not dump response model to JSON.")

        print("\n❌ VALIDATION FAILED - End-to-end processing verification failed.")
        sys.exit(1)

if __name__ == "__main__":
    # Run the standalone validation function
    try:
        # Note: This validation does NOT run the FastAPI server itself.
        # It only checks the setup defined in this script.
        asyncio.run(main_validation())
    except Exception as e:
        logger.critical(f"Critical error running main_validation: {e}", exc_info=True)
        sys.exit(1)

# Note: To run this application locally, use:
# uvicorn pdf_extractor.llm_integration.main:app --reload --port 8000
# Note: To run this application locally, use:
# uvicorn mcp_litellm.main:app --reload --port 8000