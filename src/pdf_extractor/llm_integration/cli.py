# -*- coding: utf-8 -*-
"""
Simplified CLI for interacting with the LLM Integration module (Version 2).

Provides a command to process agent-driven batch requests, with LLM-based model selection or model override.

Core Functionality:
- Accepts agent-provided JSON BatchRequest (string or file).
- Uses an LLM to select optimal models for tasks based on cost and context length, unless a model override is specified.
- Processes requests using the llm_integration.engine.
- Includes a standalone test execution block (__main__) using Typer's CliRunner.

Links:
  - Typer: https://typer.tiangolo.com/
  - Typer Testing: https://typer.tiangolo.com/tutorial/testing/
  - Loguru: https://loguru.readthedocs.io/en/stable/
  - LiteLLM: https://docs.litellm.ai/docs/
"""
import typer
from typer.testing import CliRunner # Import CliRunner
from loguru import logger
from pathlib import Path
import asyncio
import json
import sys
import os
import tempfile
import copy # Added for deepcopy
from typing import Dict, Any, Optional, List, Tuple
from pydantic import ValidationError
from dotenv import load_dotenv
import litellm
# Ensure ModelResponse is imported if not already
from litellm.types.utils import ModelResponse

try:
    from pdf_extractor.llm_integration.models import BatchRequest, BatchResponse, TaskItem, ResultItem
    from pdf_extractor.llm_integration.engine import process_batch
    from pdf_extractor.llm_integration.initialize_litellm_cache import initialize_litellm_cache
    from pdf_extractor.llm_integration.utils.litellm_utils import model_cost_per_million_tokens
except ImportError as e:
    logger.error(f"Failed to import necessary modules. Ensure paths are correct: {e}")
    sys.exit(1)

load_dotenv()

app = typer.Typer(
    name="llm-integration-cli-v2",
    help="CLI for processing agent-driven LLM batch requests.",
    add_completion=False,
)

# Model Information Structure - Use base model name for Vertex AI
MODEL_INFO = {
    "xai/grok-3-mini-beta": {
        "description": "Small context and fast model for quick responses. Good for coding tasks and debugging.",
        "context_length": "120k tokens",
        "context_tokens": 120_000,
        "cost_func": model_cost_per_million_tokens
    },
    "openai/gpt-4o-mini": {
        "description": "Cheap model for testing and debugging code.",
        "context_length": "128k tokens", # Corrected context length
        "context_tokens": 128_000,
        "cost_func": model_cost_per_million_tokens # Use the util function
    },
    "vertex_ai/gemini-2.5-pro-exp-03-25": { # Use base name
        "description": "Free model with large context for planning and complex coding.",
        "context_length": "1M tokens",
        "context_tokens": 1_000_000,
        # Define a simple lambda returning a string for free models if util func fails
        "cost_func": lambda m: model_cost_per_million_tokens(m) or "$0.00 (Free)" 
    },
}

async def select_models_with_llm(tasks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Queries an LLM to select models for tasks in a single call.
    
    Args:
        tasks: List of task dictionaries with 'task_id', 'question', and optional 'metadata'.
    
    Returns:
        List of JSON objects, e.g., [{"task_id": "t1", "model": "openai/gpt-4o-mini"}, ...].
    """
    if not tasks:
        return []

    # Construct model info string
    model_info_parts = []
    for model, info in MODEL_INFO.items():
         cost_info = info['cost_func'](model)
         # Format cost nicely, handling potential None from util or string from lambda
         cost_str = "N/A"
         if isinstance(cost_info, dict):
              # Format cost using total_cost if available
              total_cost = cost_info.get('total_cost')
              if total_cost is not None:
                   cost_str = f"${total_cost:.2f}/M tokens"
              else: # Fallback if total_cost key is missing
                   in_cost = cost_info.get('input_cost', 0)
                   out_cost = cost_info.get('output_cost', 0)
                   cost_str = f"In: ${in_cost:.2f}/M, Out: ${out_cost:.2f}/M"
         elif isinstance(cost_info, str):
              cost_str = cost_info # Use string directly (e.g., "$0.00 (Free)")
         
         model_info_parts.append(
              f"- {model}: {info['description']}, Cost: {cost_str}, Context: {info['context_length']}"
         )
    model_info_str = "\n".join(model_info_parts)


    # Construct tasks summary
    tasks_summary = "\n".join(
        f"- Task ID: {task.get('task_id', 'unknown')}\n  Question: {task.get('question', '')[:200]}...\n  Metadata: {json.dumps(task.get('metadata', {}), indent=2)}"
        for task in tasks
    )

    # Prompt for model selection - Use base model name in example
    prompt = f"""
You are an agent selecting the best LLM models for tasks based on cost and context length.
Available models:
{model_info_str}

Tasks:
{tasks_summary}

Instructions:
1. Analyze each task's question and metadata.
2. Prioritize low-cost models (e.g., free or cheap) and ensure sufficient context length.
3. Select the most suitable model for each task (e.g., cheap for short tasks, large context for complex tasks).
4. Return a JSON array of objects with 'task_id' and 'model', e.g.:
   [
     {{"task_id": "t1", "model": "openai/gpt-4o-mini"}},
     {{"task_id": "t2", "model": "vertex_ai/gemini-2.5-pro-exp-03-25"}} 
   ]
5. Ensure model IDs match the available models exactly (including prefixes like 'vertex_ai/').

Respond only with the JSON array.
"""
    logger.debug(f"Model selection prompt:\n{prompt}") # Log the prompt
    try:
        # *** Use an accessible model for the selection task itself ***
        selection_model = "openai/gpt-4o-mini" 
        logger.info(f"Using model '{selection_model}' for model selection task.")
        response = await litellm.acompletion(
            model=selection_model, 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0,
        )
        # Safely access content, checking the type first
        result: Optional[str] = None
        if isinstance(response, ModelResponse): # Check if it's a ModelResponse
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                # Add type ignore for Pylance limitation
                if choice and choice.message and choice.message.content: # type: ignore
                    result = choice.message.content # type: ignore
        else:
            logger.warning(f"LLM response was not a ModelResponse, type: {type(response)}")

        # Corrected check and raise
        if result is None:
            raise ValueError("LLM response content is missing or invalid")

        logger.debug(f"Raw LLM response for model selection: {result}")
        # Attempt to parse the JSON, handling potential surrounding text/markdown
        try:
            # Find the start and end of the JSON array
            json_start = result.find('[')
            json_end = result.rfind(']') + 1
            if json_start != -1 and json_end != 0:
                 json_str = result[json_start:json_end]
                 result_json = json.loads(json_str)
            else:
                 # Handle case where response might be just the JSON without markdown
                 try:
                      result_json = json.loads(result)
                      if not isinstance(result_json, list):
                           raise ValueError("LLM returned JSON, but it was not a list")
                 except json.JSONDecodeError:
                      raise ValueError("Could not find or parse JSON array in LLM response.")

        except json.JSONDecodeError as json_err:
             logger.error(f"Failed to decode JSON from LLM response: {json_err}")
             logger.error(f"LLM Raw Response causing error: {result}")
             raise ValueError(f"LLM returned invalid JSON: {json_err}") from json_err


        if not isinstance(result_json, list):
            raise ValueError("LLM returned JSON, but it was not a list")
        
        valid_models = set(MODEL_INFO.keys())
        assignments = []
        for item in result_json:
            task_id = item.get("task_id")
            model = item.get("model")
            if model not in valid_models:
                logger.warning(f"LLM returned invalid model '{model}' for task {task_id}. Valid models: {valid_models}")
                # Ensure task_id is a string for fallback
                fallback_model = fallback_select_model(tasks, str(task_id) if task_id else f"unknown_task_{item}")
                assignments.append({"task_id": task_id, "model": fallback_model})
            else:
                 assignments.append({"task_id": task_id, "model": model})

        logger.info(f"LLM selected models for {len(assignments)} tasks")
        return assignments
    except Exception as e:
        logger.error(f"Failed to select models with LLM: {e}", exc_info=True)
        # Ensure task_id is a string for fallback_select_model and return type matches List[Dict[str, str]]
        return [
            {"task_id": (tid := str(task.get("task_id", f"unknown_task_{i}"))), "model": fallback_select_model(tasks, tid)}
            for i, task in enumerate(tasks)
        ]


def fallback_select_model(tasks: List[Dict[str, Any]], task_id: str) -> str:
    """Fallback heuristic for model selection."""
    for task in tasks:
        if task.get("task_id") == task_id:
            question = task.get("question", "")
            metadata = task.get("metadata", {})
            question_length = len(question.split())
            if question_length < 100:
                return "openai/gpt-4o-mini"
            # Use base model name here
            elif question_length > 1000 or "complex" in metadata.get("description", "").lower():
                logger.warning(f"Fallback selected 'vertex_ai/gemini-2.5-pro-exp-03-25' for task {task_id}. Location/Project must be set via env vars or ADC.")
                return "vertex_ai/gemini-2.5-pro-exp-03-25" 
            else:
                return "xai/grok-3-mini-beta"
    return "xai/grok-3-mini-beta"

# Reverted to original synchronous function
def preprocess_batch_request(request_data: Dict[str, Any], override_model: Optional[str] = None) -> Dict[str, Any]:
    """
    Preprocesses the BatchRequest to assign models for tasks.
    
    Args:
        request_data: Raw BatchRequest dictionary from the agent.
        override_model: If specified, all tasks use this model.
    
    Returns:
        Modified BatchRequest dictionary with model assignments.
    """
    processed_data = copy.deepcopy(request_data) # Work on a copy
    processed_tasks = processed_data.get("tasks", [])

    if override_model:
        # Validate override model
        if override_model not in MODEL_INFO:
            logger.error(f"Invalid override model: {override_model}. Available models: {list(MODEL_INFO.keys())}")
            raise typer.Exit(code=1)
        # Apply override model to all tasks
        for task in processed_tasks:
            task["model"] = override_model
            logger.info(f"Overrode model to {override_model} for task {task.get('task_id')}")
    else:
        # Apply LLM-driven selection for tasks without models
        tasks_needing_models = [task for task in processed_tasks if not task.get("model")]
        if tasks_needing_models:
            # This block now runs correctly when script is executed as main entry point
            # or via subprocess, as it starts its own event loop if none exists.
            logger.debug("Attempting LLM-based model selection using asyncio.run().")
            try:
                 model_assignments = asyncio.run(select_models_with_llm(tasks_needing_models))
            except RuntimeError as e:
                 # This might happen if somehow called from an already running loop elsewhere
                 logger.error(f"RuntimeError during asyncio.run for model selection: {e}. This should not happen in standard CLI execution.")
                 # Fallback: Assign default model to avoid complete failure
                 model_assignments = [{"task_id": task.get("task_id"), "model": "xai/grok-3-mini-beta"} for task in tasks_needing_models]
            # Handle potential failure from the simplified test call
            except Exception as e:
                 logger.error(f"Model selection call failed: {e}. Assigning fallback.") # Updated log message
                 model_assignments = [{"task_id": task.get("task_id"), "model": "xai/grok-3-mini-beta"} for task in tasks_needing_models]


            model_map = {item["task_id"]: item["model"] for item in model_assignments}
            for task in processed_tasks:
                if not task.get("model"):
                    # Use fallback_select_model if task_id not in model_map (e.g., LLM failed)
                    task_id_str = str(task.get("task_id", f"unknown_task_{processed_tasks.index(task)}"))
                    assigned_model = model_map.get(task_id_str)
                    if not assigned_model:
                         logger.warning(f"LLM did not assign model for task {task_id_str}. Using fallback.")
                         assigned_model = fallback_select_model(processed_tasks, task_id_str) # Pass processed_tasks here
                    task["model"] = assigned_model
                    logger.info(f"Assigned model {task['model']} to task {task_id_str}")

    return processed_data # Return the modified copy

def load_input(input_data: str) -> Dict[str, Any]:
    """Load JSON data from a file or string."""
    input_path = Path(input_data)
    if input_path.is_file():
        logger.info(f"Loading BatchRequest from file: {input_path}")
        try:
            with open(input_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {e}")
            raise typer.Exit(code=1)
        except IOError as e:
            logger.error(f"Could not read file {input_path}: {e}")
            raise typer.Exit(code=1)
    else:
        logger.info("Attempting to parse input as JSON string.")
        try:
            return json.loads(input_data)
        except json.JSONDecodeError:
            logger.error("Input is neither a valid file path nor a valid JSON string.")
            raise typer.Exit(code=1)

@app.command("ask", help=(
    "Process a batch of LLM tasks from an agent, with automatic model selection or model override.\n\n"
    "WHEN TO USE: Agents call this to process tasks, similar to querying a specialized tool.\n\n"
    "HOW TO USE: Pass a JSON string or file path with a BatchRequest. If 'model' is unspecified and no override is provided, an LLM selects the best model based on cost and context. Use --model to override with a single model for all tasks."
))
# Reverted to original synchronous function signature for Typer
def ask_command(
    input_data: str = typer.Argument(
        ...,
        help="JSON string or path to a JSON file containing the BatchRequest data."
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        # Correctly handle default value from Typer
        callback=lambda v: v if v is not None else None,
        help="Override model for all tasks (e.g., 'openai/gpt-4o-mini'). If not specified, uses LLM-driven selection for tasks without models."
    )
):
    """
    Processes an agent-driven batch LLM request with internal model selection or override.
    """
    try:
        # 1. Load and Preprocess Input
        request_data = load_input(input_data)
        # Preprocessing is now synchronous
        processed_request_data = preprocess_batch_request(request_data, override_model=model) # Pass original data
        batch_request = BatchRequest.model_validate(processed_request_data)
        logger.info(f"Validated {len(batch_request.tasks)} tasks.")
        logger.debug(f"Validated BatchRequest: {batch_request.model_dump(mode='json')}")

        # 2. Initialize Cache
        initialize_litellm_cache()
        logger.info("LiteLLM cache initialized.")

        # 3. Process Batch (needs to be run in an event loop)
        logger.info("Processing batch request...")
        # Run the async process_batch function using asyncio.run()
        response: BatchResponse = asyncio.run(process_batch(batch_request))
        logger.info("Batch processing complete.")

        # 4. Log and Output Response
        response_json = response.model_dump_json(indent=2)
        logger.info(f"Batch Response:\n{response_json}")
        print(response_json) # Print JSON to stdout for CLI capture

        # 5. Basic Validation (Optional here, main validation in test_usage)
        if not response or not response.responses:
            logger.error("❌ No response or empty response received.")
            # Don't exit here, let test_usage handle validation failure
            # raise typer.Exit(code=1) # Removed exit

        errors = [r for r in response.responses if r.status != 'success']
        if errors:
            logger.warning(f"Found {len(errors)} tasks with errors.")
            for error_item in errors:
                logger.warning(f"  - Task ID: {error_item.task_id}, Status: {error_item.status}, Message: {error_item.error_message}")
        else:
            logger.info("✅ All tasks completed successfully.")

        # Note: No return value needed when run as CLI command

    except ValidationError as e:
        logger.error(f"Invalid BatchRequest data: {e}")
        raise typer.Exit(code=1)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading input data: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Unexpected error during batch processing: {e}", exc_info=True)
        raise typer.Exit(code=1)

@app.command("models-info", help=(
    "Display information about available LLM models.\n\n"
    "AGENT NOTE: Used internally by the ask command to select models."
))
def models_info_command():
    """Prints model details."""
    print("\n--- Available LLM Models ---")
    for model_name in sorted(MODEL_INFO.keys()):
        info = MODEL_INFO[model_name]
        print(f"\nModel: {model_name}")
        print(f"  Description: {info.get('description', 'N/A')}")
        try:
            # Use the cost_func associated with the specific model entry
            cost_info = info['cost_func'](model_name)
            cost_str = "N/A"
            if isinstance(cost_info, dict):
                 total_cost = cost_info.get('total_cost')
                 if total_cost is not None:
                      cost_str = f"${total_cost:.2f}/M tokens"
                 else:
                      in_cost = cost_info.get('input_cost', 0)
                      out_cost = cost_info.get('output_cost', 0)
                      cost_str = f"In: ${in_cost:.2f}/M, Out: ${out_cost:.2f}/M"
            elif isinstance(cost_info, str):
                 cost_str = cost_info
            print(f"  Cost per million tokens: {cost_str}")
        except Exception as e:
            logger.warning(f"Could not retrieve cost for model {model_name}: {e}")
            print("  Cost per million tokens: N/A")
        print(f"  Context length: {info.get('context_length', 'N/A')}")
    print("\n---------------------------")

def create_sample_batch_request_v2() -> Dict[str, Any]:
    """Creates a sample BatchRequest for testing agent-driven input."""
    return {
        "tasks": [
            {
                "task_id": "cli_v2_test_Q0",
                "question": "What is the capital of France?",
                "metadata": {"source": "cli_v2_test"}
                # Model intentionally left out for LLM selection test
            },
            {
                "task_id": "cli_v2_test_Q1",
                # Make question shorter to likely get assigned openai/gpt-4o-mini
                "question": "Explain quantum mechanics simply.", 
                "metadata": {"source": "cli_v2_test"} # Removed "complex" description
                # Model intentionally left out for LLM selection test
            }
        ]
    }

# Removed subprocess helpers

# Refactored test_usage to use CliRunner
def test_usage():
    """Tests the CLI 'ask' command logic using Typer's CliRunner."""
    logger.info("--- Starting Standalone CLI v2 Usage Test (CliRunner) ---")
    runner = CliRunner()
    temp_file_path: Optional[str] = None
    validation_passed = False
    exit_code = 1
    validation_failures = []

    try:
        # 1. Create Sample Input File
        sample_request_data = create_sample_batch_request_v2()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_request_data, f, indent=2)
            temp_file_path = f.name
        logger.info(f"Created temporary sample input file: {temp_file_path}")

        EXPECTED_STATUS = {
            "cli_v2_test_Q0": "success",
            "cli_v2_test_Q1": "success"
        }

        # 2. Test Without Override (LLM Selection via CLI)
        logger.info("Testing 'ask' command via CliRunner (LLM Selection)")
        result_no_override = runner.invoke(app, ["ask", temp_file_path])
        stdout_no_override = result_no_override.stdout
        # stderr_no_override = result_no_override.stderr # Capture stderr if needed

        if result_no_override.exit_code != 0:
            validation_failures.append(f"Without override: CLI command failed with exit code {result_no_override.exit_code}.")
            logger.error(f"Output (No Override):\n{stdout_no_override}") # Log stdout on failure
            # logger.error(f"Stderr (No Override):\n{stderr_no_override}") # Optionally log stderr
        else:
            try:
                response_data_no_override = json.loads(stdout_no_override)
                response_no_override = BatchResponse.model_validate(response_data_no_override)

                if response_no_override and response_no_override.responses and len(response_no_override.responses) == len(EXPECTED_STATUS):
                    results_dict_no_override = {res.task_id: res.status for res in response_no_override.responses}
                    for task_id, expected_status in EXPECTED_STATUS.items():
                        actual_status = results_dict_no_override.get(task_id)
                        if actual_status != expected_status:
                            validation_failures.append(f"Without override, Task '{task_id}': Expected status '{expected_status}', Got '{actual_status}'")
                        logger.info(f"Without override, Task '{task_id}': Status '{actual_status}'")
                    # Check logs (can't easily check stderr from CliRunner, rely on exit code/stdout)
                    logger.info("Without override: Command succeeded, assuming model assignment worked.")
                else:
                     validation_failures.append(f"Without override: Parsed response invalid or missing expected results ({len(EXPECTED_STATUS)}).")

            except json.JSONDecodeError as e:
                validation_failures.append(f"Without override: Failed to decode JSON from stdout: {e}")
                logger.error(f"Invalid JSON received (stdout):\n{stdout_no_override}")
            except ValidationError as e:
                 validation_failures.append(f"Without override: Failed to validate BatchResponse: {e}")


        # 3. Test With Override (via CLI)
        logger.info("Testing 'ask' command via CliRunner with model override")
        override_model = "openai/gpt-4o-mini"
        result_override = runner.invoke(app, ["ask", "--model", override_model, temp_file_path])
        stdout_override = result_override.stdout
        # stderr_override = result_override.stderr # Capture stderr if needed

        if result_override.exit_code != 0:
            validation_failures.append(f"Override test: CLI command failed with exit code {result_override.exit_code}.")
            logger.error(f"Output (Override):\n{stdout_override}") # Log stdout on failure
            # logger.error(f"Stderr (Override):\n{stderr_override}") # Optionally log stderr
        else:
            try:
                response_data_override = json.loads(stdout_override)
                response_override = BatchResponse.model_validate(response_data_override)

                if response_override and response_override.responses and len(response_override.responses) == len(EXPECTED_STATUS):
                    results_dict_override = {res.task_id: res.status for res in response_override.responses}
                    for task_id, expected_status in EXPECTED_STATUS.items():
                        actual_status = results_dict_override.get(task_id)
                        if actual_status != expected_status:
                            validation_failures.append(f"Override test, Task '{task_id}': Expected status '{expected_status}', Got '{actual_status}'")
                        else:
                             logger.info(f"Override test, Task '{task_id}': Status '{actual_status}'")
                    # Check logs (can't easily check stderr from CliRunner, rely on exit code/stdout)
                    logger.info(f"Override test: Command succeeded, assuming override model '{override_model}' was used.")
                else:
                     validation_failures.append(f"Override test: Parsed response invalid or missing expected results ({len(EXPECTED_STATUS)}).")

            except json.JSONDecodeError as e:
                 validation_failures.append(f"Override test: Failed to decode JSON from stdout: {e}")
                 logger.error(f"Invalid JSON received (stdout):\n{stdout_override}")
            except ValidationError as e:
                 validation_failures.append(f"Override test: Failed to validate BatchResponse: {e}")


        # Final Validation Check
        if not validation_failures:
            validation_passed = True
            exit_code = 0
        else:
            logger.error("Test failed: Result validation errors.")
            for failure in validation_failures:
                logger.error(f"  - {failure}")

    except Exception as e:
        logger.error(f"Test failed with unexpected error in test setup or execution: {e}", exc_info=True)
        exit_code = 1
    finally:
        if temp_file_path and Path(temp_file_path).exists():
            try:
                Path(temp_file_path).unlink()
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except OSError as e:
                logger.error(f"Error deleting temporary file {temp_file_path}: {e}")

        if validation_passed:
            print("\n✅ VALIDATION COMPLETE - All results match expected values.")
        else:
            print("\n❌ VALIDATION FAILED - Results don't match expected values.")
            for failure in validation_failures:
                print(f"  - {failure}")

        logger.info("--- Standalone CLI v2 Usage Test (CliRunner) Complete ---")
        sys.exit(exit_code)

if __name__ == "__main__":
    # Setup logging for both CLI execution and test_usage direct calls
    logger.remove()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    # Log to stderr 
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Log level set to: {log_level}") 

    # Check if running the test directly (no args) or via Typer (has args)
    is_running_test = len(sys.argv) == 1

    if not is_running_test:
        logger.debug("CLI arguments detected. Running Typer app...")
        # Typer handles the rest, including calling ask_command
        app()
    else:
        logger.info("No CLI arguments detected. Running standalone test_usage...")
        load_dotenv(override=True)
        # Adjusted API key check - Added VERTEX_PROJECT_ID
        required_keys = ['OPENAI_API_KEY', 'GEMINI_API_KEY', 'VERTEX_PROJECT_ID'] 
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            logger.error(f"Missing required environment variables for test: {', '.join(missing_keys)}")
            sys.exit(1)
        else:
            logger.debug("Required API keys found in environment.")
            # Run the synchronous test_usage function (CliRunner is synchronous)
            test_usage() # test_usage now handles its own sys.exit()