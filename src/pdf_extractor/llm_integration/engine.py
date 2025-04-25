# -*- coding: utf-8 -*-
"""
MCP LiteLLM Service: Asynchronous Execution Engine

Description:
------------
This module implements the core asynchronous engine for processing batches
of LLM requests using LiteLLM. It handles concurrent and sequential execution,
dependency management (using results as context), retries with backoff, and
pluggable validation strategies.

Core Libraries/Concepts:
------------------------
- asyncio: For managing asynchronous tasks and concurrency.
- typing: For precise type hinting.
- loguru: For logging.
- LiteLLM: The underlying library for interacting with various LLM APIs.
- Links:
  - asyncio: https://docs.python.org/3/library/asyncio.html
  - LiteLLM: https://docs.litellm.ai/docs/
  - loguru: https://loguru.readthedocs.io/en/stable/
  - Pydantic: https://docs.pydantic.dev/

Key Components:
---------------
- `process_batch`: Orchestrates batch processing using a DAG-based execution engine.
- Dependency Handling: Builds message history from successful dependency results.
- Concurrency: Executes concurrent tasks in parallel with `asyncio`.
- Error Handling: Captures and logs exceptions during task execution.
- Retries: Uses `retry_llm_call` for robust LLM interactions.
- Validation: Supports pluggable strategies (e.g., Pydantic, JSON, Citation).

Sample I/O (Conceptual):
------------------------
Input (`BatchRequest`):
{
  "tasks": [
    {"task_id": "Q0", "question": "What is the capital of France?", "method": "concurrent", "model": "gpt-4o-mini"},
    {"task_id": "Q1", "question": "What is the weather like there?", "method": "sequential", "dependencies": ["Q0"], "model": "gpt-4o-mini"},
    {"task_id": "Q2", "question": "Is Berlin the capital of Germany?", "method": "concurrent", "model": "gpt-4o-mini"}
  ]
}

Output (`BatchResponse`):
{
  "responses": [
    {"task_id": "Q0", "status": "success", "result": "Paris", "retry_count": 0, ...},
    {"task_id": "Q1", "status": "success", "result": "The weather in Paris is currently sunny...", "retry_count": 0, ...},
    {"task_id": "Q2", "status": "success", "result": "Yes, Berlin is the capital of Germany.", "retry_count": 0, ...}
  ]
}
"""

import asyncio
import json
import sys
import os
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple, Set, Union, AsyncGenerator, Literal, Awaitable
from loguru import logger
from pydantic import BaseModel

# Absolute imports
from pdf_extractor.llm_integration.models import BatchRequest, BatchResponse, TaskItem, ResultItem
from pdf_extractor.llm_integration.validation_utils.base import match_from_list_validator
from pdf_extractor.llm_integration.litellm_call import litellm_call
from litellm.types.utils import ModelResponse
from pdf_extractor.llm_integration.retry_llm_call import retry_llm_call, MaxRetriesExceededError

# Validation and utility imports
try:
    from pdf_extractor.llm_integration.validation_utils.citation_validators import citation_validator, extract_citation_validator
    from pdf_extractor.llm_integration.validation_utils.json_validators import json_validator, required_fields_validator
    from pdf_extractor.llm_integration.utils.corpus_utils import load_corpus_from_source
    from pdf_extractor.llm_integration.utils.dependency_graph import build_dependency_graph, DEFAULT_MAX_DEPTH
except ImportError:
    logger.error("Failed to import validation or corpus utilities. Validation will not be performed.")
    def citation_validator(*args: Any, **kwargs: Any) -> Callable[..., Callable[[Any], Union[bool, str]]]:
        return lambda *a, **kw: lambda r: True
    def extract_citation_validator(*args: Any, **kwargs: Any) -> Callable[..., Callable[[Any], Union[bool, str]]]:
        return lambda *a, **kw: lambda r: True
    def json_validator(*args: Any, **kwargs: Any) -> Callable[[Any], Union[bool, str]]:
        return lambda r: True
    def required_fields_validator(*args: Any, **kwargs: Any) -> Callable[[Any], Union[bool, str]]:
        return lambda r: True
    def load_corpus_from_source(*args: Any, **kwargs: Any) -> Optional[str]:
        return None

def get_pydantic_validator(model_name: str) -> Optional[Callable[[Any], Union[bool, str]]]:
    """Placeholder for Pydantic validator retrieval."""
    logger.warning(f"Pydantic validation requested for {model_name}, but not implemented.")
    return None

# Validation strategy registry
VALIDATOR_REGISTRY: Dict[str, Callable[[TaskItem, Optional[str]], List[Callable[[Any], Union[bool, str]]]]] = {
    "json": lambda task, corpus: [
        json_validator(),
        required_fields_validator(task.validation_options.get("required_fields", []))
        if isinstance(task.validation_options.get("required_fields"), list) else json_validator()
    ],
    "citation": lambda task, corpus: [
        citation_validator(min_similarity=task.validation_options.get("min_similarity", 95.0))(corpus)
        if corpus else lambda r: False
    ],
    "extract_citation": lambda task, corpus: [
        extract_citation_validator(
            required_phrases=task.validation_options.get("required_phrases", []),
            min_similarity=task.validation_options.get("min_similarity", 90.0)
        )(corpus) if corpus and isinstance(task.validation_options.get("required_phrases"), list) else lambda r: False
    ],
    "match_from_list": lambda task, corpus: [
        match_from_list_validator(
            min_similarity=task.validation_options.get("min_similarity", 85.0),
            delimiter=task.validation_options.get("delimiter", ",")
        )(corpus) if corpus else lambda r: False
    ],
    "pydantic": lambda task, corpus: (
        [get_pydantic_validator(task.response_model)] if task.response_model and get_pydantic_validator(task.response_model) else []
    ),
}

def build_message_history(
    task_id: str,
    task_map: Dict[str, TaskItem],
    completed_results: Dict[str, ResultItem],
    original_dependencies: Dict[str, Set[str]]
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Builds message history and corpus from task dependencies."""
    messages: List[Dict[str, Any]] = []
    corpus_parts: List[str] = []
    for dep_id in sorted(original_dependencies.get(task_id, [])):
        if dep_id in completed_results and completed_results[dep_id].status == "success":
            dep_task = task_map.get(dep_id)
            dep_result = completed_results[dep_id].result
            if dep_task and dep_result is not None:
                messages.append({"role": "user", "content": dep_task.question})
                result_content: str
                if isinstance(dep_result, BaseModel):
                    try:
                        result_content = dep_result.model_dump_json()
                    except Exception as json_err:
                        logger.warning(f"Failed to dump Pydantic model for dep {dep_id}: {json_err}")
                        result_content = str(dep_result)
                elif isinstance(dep_result, (dict, list)):
                    try:
                        result_content = json.dumps(dep_result)
                    except Exception as json_err:
                        logger.warning(f"Failed to dump dict/list for dep {dep_id}: {json_err}")
                        result_content = str(dep_result)
                else:
                    result_content = str(dep_result)
                messages.append({"role": "assistant", "content": result_content})
                corpus_parts.append(result_content)
            else:
                logger.warning(f"Task {task_id}: Missing data for dependency {dep_id}")
        elif dep_id in completed_results:
            logger.warning(f"Task {task_id}: Dependency {dep_id} failed (status: {completed_results[dep_id].status})")
        else:
            logger.error(f"Task {task_id}: Dependency {dep_id} not found")
    messages.append({"role": "user", "content": task_map[task_id].question})
    final_corpus = "\n\n---\n\n".join(corpus_parts) if corpus_parts else None
    return messages, final_corpus

def get_validators(task: TaskItem, corpus: Optional[str]) -> List[Callable[[Any], Union[bool, str]]]:
    """Instantiates validators based on task's validation strategy."""
    strategy = task.validation_strategy
    if not strategy:
        return []
    if strategy not in VALIDATOR_REGISTRY:
        logger.warning(f"Task {task.task_id}: Unknown validation strategy '{strategy}'")
        return []
    return VALIDATOR_REGISTRY[strategy](task, corpus)

async def execute_llm_call(
    task: TaskItem,
    messages: List[Dict[str, Any]],
    validators: List[Callable[[Any], Union[bool, str]]]
) -> Tuple[Any, int, Optional[Union[bool, List[str]]]]:
    """Executes LLM call with retry logic."""
    if task.model is None:
        raise ValueError(f"Task {task.task_id} has no model specified")
    try:
        response, retry_count, validation_result = await retry_llm_call(
            llm_call=litellm_call,
            model=task.model,
            messages=messages,
            validation_strategies=validators,
            temperature=task.temperature if task.temperature is not None else 0.2,
            max_tokens=task.max_tokens if task.max_tokens is not None else 1000,
            api_base=task.api_base,
            response_format=task.response_model,
            max_retries=task.max_retries,
        )
        logger.debug(f"Task {task.task_id}: LLM call finished")
        return response, retry_count, validation_result
    except MaxRetriesExceededError as e:
        logger.error(f"Task {task.task_id} failed after max retries: {e}")
        error_str = str(e)
        # Safely extract validation errors without assuming JSON
        if "Last errors: " in error_str:
            errors_part = error_str.split("Last errors: ", 1)[1]
            # Handle cases where errors_part is a list-like string
            if errors_part.startswith('[') and errors_part.endswith(']'):
                # Strip brackets and split by commas, handling single quotes
                errors = [err.strip().strip("'") for err in errors_part[1:-1].split(", ")]
                validation_result = errors if errors else [error_str]
            else:
                validation_result = [errors_part]
        else:
            validation_result = [error_str]
        return None, task.max_retries, validation_result

async def extract_content(task_id: str, response: Any) -> Optional[Union[str, Dict[str, Any], BaseModel]]:
    """Extracts content from LLM response asynchronously."""
    if response is None:
        return None
    if isinstance(response, ModelResponse):
        try:
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            logger.warning(f"Task {task_id}: ModelResponse missing expected choices/message")
            return f"Error: Incomplete ModelResponse structure: {response}"
        except (AttributeError, IndexError, TypeError) as e:
            logger.warning(f"Task {task_id}: Could not extract content from ModelResponse: {e}")
            return f"Error: Could not parse ModelResponse: {response}"
    if isinstance(response, BaseModel):
        return response
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        return response
    if isinstance(response, AsyncGenerator):
        logger.warning(f"Task {task_id}: Received AsyncGenerator, collecting content")
        return "".join([chunk async for chunk in response])
    logger.error(f"Task {task_id}: Unhandled response type: {type(response)}")
    return f"Error: Unhandled response type {type(response)}"

def create_result_item(
    task_id: str,
    content: Optional[Union[str, Dict[str, Any], BaseModel]],
    validation_result: Optional[Union[bool, List[str]]],
    retry_count: int
) -> ResultItem:
    """Creates a ResultItem based on validation and content."""
    status: Literal['success', 'error', 'dependency_failed'] = 'error'
    final_content: Any = content
    error_msg: Optional[str] = None

    if validation_result is True:
        status = 'success'
        if content is None:
            logger.error(f"Task {task_id}: Validation passed but content is None")
            status = 'error'
            error_msg = "Internal error: Validation passed but content is None"
    elif isinstance(validation_result, list):
        error_msg = "; ".join(validation_result)
    else:
        error_msg = "Unknown error during validation or pre-check failed"

    if isinstance(content, str) and content.startswith("Error:"):
        status = 'error'
        val_status_str = f"; Validation Status: {validation_result}" if validation_result is not True and validation_result is not None else ""
        error_msg = f"{content}{val_status_str}"
        final_content = None

    return ResultItem(
        task_id=task_id,
        status=status,
        result=final_content,
        retry_count=retry_count,
        error_message=error_msg,
        validation_status=validation_result
    )

def update_dependents(
    task_id: str,
    completed_results: Dict[str, ResultItem],
    dependents: Dict[str, Set[str]],
    in_degree: Dict[str, int],
    ready_queue: asyncio.Queue[str]
):
    """Updates dependent tasks based on current task's result."""
    if task_id not in completed_results:
        logger.error(f"Task {task_id} not in completed_results. Logic error?")
        return
    current_result = completed_results[task_id]
    for dep_id in dependents.get(task_id, []):
        if dep_id not in in_degree:
            continue
        if current_result.status != "success":
            if dep_id not in completed_results:
                logger.warning(f"Marking task {dep_id} as dependency_failed due to {task_id}")
                completed_results[dep_id] = ResultItem(
                    task_id=dep_id,
                    status="dependency_failed",
                    result=None,
                    error_message=f"Dependency {task_id} failed with status: {current_result.status}",
                    retry_count=0,
                    validation_status=None
                )
                in_degree.pop(dep_id, None)
            continue
        in_degree[dep_id] -= 1
        if in_degree[dep_id] == 0 and dep_id not in completed_results:
            ready_queue.put_nowait(dep_id)

async def run_task(
    task_id: str,
    task_map: Dict[str, TaskItem],
    completed_results: Dict[str, ResultItem],
    original_dependencies: Dict[str, Set[str]],
    dependents: Dict[str, Set[str]],
    in_degree: Dict[str, int],
    ready_queue: asyncio.Queue[str],
    semaphore: asyncio.Semaphore
):
    """Executes a single task and updates dependencies."""
    async with semaphore:
        task = task_map[task_id]
        try:
            # Build message history and corpus
            messages, base_corpus = build_message_history(task_id, task_map, completed_results, original_dependencies)
            logger.debug(f"Task {task_id}: Built {len(messages)} messages")

            # Load additional corpus
            corpus_from_options = load_corpus_from_source(task.validation_options or {}, task_id)
            final_corpus = (
                f"{corpus_from_options}\n\n---\n\n{base_corpus}" if corpus_from_options and base_corpus
                else corpus_from_options or base_corpus
            )
            if final_corpus:
                logger.debug(f"Task {task_id}: Corpus length: {len(final_corpus)} chars")
            else:
                logger.debug(f"Task {task_id}: No corpus available")

            # Check for required corpus
            if task.validation_strategy in ["citation", "extract_citation", "match_from_list"] and not final_corpus:
                logger.error(f"Task {task_id}: Corpus required for '{task.validation_strategy}' but none loaded")
                completed_results[task_id] = ResultItem(
                    task_id=task_id,
                    status="error",
                    result=None,
                    error_message=f"Corpus required for '{task.validation_strategy}' validation but none loaded",
                    retry_count=0,
                    validation_status=[f"Corpus required for '{task.validation_strategy}' validation but none loaded"]
                )
                update_dependents(task_id, completed_results, dependents, in_degree, ready_queue)
                return

            # Get validators
            validators = get_validators(task, final_corpus)
            if not validators and task.validation_strategy:
                logger.warning(f"Task {task_id}: No validators instantiated for strategy '{task.validation_strategy}'")

            # Execute LLM call
            response, retry_count, validation_result = await execute_llm_call(task, messages, validators)

            # Extract content
            content = await extract_content(task_id, response)

            # Create result
            result_item = create_result_item(task_id, content, validation_result, retry_count)
            completed_results[task_id] = result_item

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            validation_result = [str(e)] if isinstance(e, MaxRetriesExceededError) else None
            retry_count = task.max_retries if isinstance(e, MaxRetriesExceededError) else 0
            completed_results[task_id] = ResultItem(
                task_id=task_id,
                status="error",
                result=None,
                error_message=f"Unexpected error: {e}",
                retry_count=retry_count,
                validation_status=validation_result
            )
        update_dependents(task_id, completed_results, dependents, in_degree, ready_queue)

async def schedule_tasks(
    ready_queue: asyncio.Queue[str],
    completed_results: Dict[str, ResultItem],
    task_map: Dict[str, TaskItem],
    original_dependencies: Dict[str, Set[str]],
    dependents: Dict[str, Set[str]],
    in_degree: Dict[str, int],
    semaphore: asyncio.Semaphore
):
    """Schedules and executes tasks from the ready queue."""
    active_tasks: Set[asyncio.Task] = set()
    while not ready_queue.empty():
        task_id = ready_queue.get_nowait()
        if task_id not in completed_results:
            task = asyncio.create_task(
                run_task(task_id, task_map, completed_results, original_dependencies, dependents, in_degree, ready_queue, semaphore),
                name=task_id
            )
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

    logger.debug(f"Scheduling: Starting with {len(active_tasks)} tasks")
    while active_tasks:
        done, _ = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        logger.debug(f"Scheduling: Completed {len(done)} tasks, {len(active_tasks) - len(done)} pending")
        new_tasks_added = 0
        while not ready_queue.empty():
            task_id = ready_queue.get_nowait()
            if task_id not in completed_results and not any(t.get_name() == task_id for t in active_tasks):
                task = asyncio.create_task(
                    run_task(task_id, task_map, completed_results, original_dependencies, dependents, in_degree, ready_queue, semaphore),
                    name=task_id
                )
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)
                new_tasks_added += 1
        if new_tasks_added:
            logger.debug(f"Scheduling: Added {new_tasks_added} new tasks")
    logger.debug("Scheduling: All tasks completed")

async def process_batch(request: BatchRequest) -> BatchResponse:
    """
    Processes a batch of LLM tasks asynchronously using a DAG-based execution engine.

    Args:
        request: The BatchRequest containing tasks with dependencies.

    Returns:
        A BatchResponse containing the results for each task.

    Raises:
        ValueError: If dependency chain exceeds maximum allowed depth
    """
    task_map = {task.task_id: task for task in request.tasks}
    completed_results: Dict[str, ResultItem] = {}

    # Validate dependencies
    for task in request.tasks:
        for dep_id in task.dependencies or []:
            if dep_id not in task_map:
                logger.warning(f"Task {task.task_id} has invalid dependency {dep_id}. Marking as dependency_failed.")
                completed_results[task.task_id] = ResultItem(
                    task_id=task.task_id,
                    status="dependency_failed",
                    result=None,
                    error_message=f"Invalid dependency {dep_id} not found in task map",
                    retry_count=0,
                    validation_status=None
                )

    # Build dependency graph
    try:
        original_dependencies, dependents, in_degree = build_dependency_graph(tasks=request.tasks, max_depth=DEFAULT_MAX_DEPTH)
    except ValueError as e:
        logger.error(f"Dependency graph validation failed: {e}")
        raise

    # Initialize ready queue
    ready_queue: asyncio.Queue[str] = asyncio.Queue()
    for task_id, deg in in_degree.items():
        if deg == 0 and task_id not in completed_results:
            ready_queue.put_nowait(task_id)

    semaphore = asyncio.Semaphore(request.max_concurrency)
    await schedule_tasks(ready_queue, completed_results, task_map, original_dependencies, dependents, in_degree, semaphore)

    for task_id in task_map:
        if task_id not in completed_results:
            logger.error(f"Task {task_id} not completed. Possible cycle or error")
            completed_results[task_id] = ResultItem(
                task_id=task_id,
                status="error",
                result=None,
                error_message="Task did not complete. Unresolved dependencies or cycle detected",
                retry_count=0,
                validation_status=None
            )

    ordered_responses = [completed_results[task.task_id] for task in request.tasks if task.task_id in completed_results]
    if len(ordered_responses) != len(request.tasks):
        logger.error("Mismatch between requested tasks and results")
    return BatchResponse(responses=ordered_responses)

if __name__ == "__main__":
    try:
        from pdf_extractor.llm_integration.validation_utils.reporting import report_validation_results
        from pdf_extractor.llm_integration.models import BatchRequest, TaskItem, BatchResponse
        from pdf_extractor.llm_integration.initialize_litellm_cache import initialize_litellm_cache
    except ImportError as e:
        print(f"❌ FATAL: Could not import required module: {e}")
        sys.exit(1)

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info("Starting LLM Engine Standalone Verification...")

    all_tests_passed = True
    all_failures = {}

    # Test Data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_corpus_file:
        temp_corpus_file.write("This is the corpus text for citation test.")
        temp_corpus_path = temp_corpus_file.name

    SAMPLE_REQUEST = BatchRequest(
        tasks=[
            TaskItem(task_id="T0", question="What is 1+1?", model="gpt-4o-mini", method="concurrent"),
            TaskItem(task_id="T1", question="What is the capital of Test?", model="gpt-4o-mini", method="concurrent"),
            TaskItem(
                task_id="T2",
                question="Summarize the previous answer.",
                model="gpt-4o-mini",
                method="sequential",
                dependencies=["T0"],
                validation_strategy="citation",
                validation_options={"corpus_source": "The previous answer was 2.", "corpus_type": "string", "min_similarity": 80.0}
            ),
            TaskItem(
                task_id="T3",
                question="Extract JSON: {'city': 'Testville', 'country': 'Testland'}",
                model="gpt-4o-mini",
                method="concurrent",
                validation_strategy="json",
                validation_options={"required_fields": ["city", "country"]}
            ),
            TaskItem(
                task_id="T4",
                question="This should fail.",
                model="gpt-4o-mini",
                method="sequential",
                dependencies=["T_NONEXISTENT"]
            ),
            TaskItem(
                task_id="T5",
                question="Is 'corpus text' mentioned?",
                model="gpt-4o-mini",
                method="concurrent",
                validation_strategy="citation",
                validation_options={"corpus_source": temp_corpus_path, "corpus_type": "file", "min_similarity": 60.0}
            ),
        ],
        max_concurrency=2
    )

    EXPECTED_TASK_COUNT = len(SAMPLE_REQUEST.tasks)
    EXPECTED_TASK_IDS = {t.task_id for t in SAMPLE_REQUEST.tasks}

    logger.info("--- Testing process_batch ---")
    try:
        initialize_litellm_cache()
        response = asyncio.run(process_batch(SAMPLE_REQUEST))

        if len(response.responses) != EXPECTED_TASK_COUNT:
            all_tests_passed = False
            all_failures["response_count"] = {"expected": EXPECTED_TASK_COUNT, "actual": len(response.responses)}
            logger.error(f"❌ process_batch: Incorrect number of responses")
        else:
            logger.info(f"✅ process_batch: Returned correct number of responses ({len(response.responses)})")
            actual_task_ids = {r.task_id for r in response.responses}
            if actual_task_ids != EXPECTED_TASK_IDS:
                all_tests_passed = False
                all_failures["task_ids"] = {"expected": sorted(list(EXPECTED_TASK_IDS)), "actual": sorted(list(actual_task_ids))}
                logger.error(f"❌ process_batch: Task IDs do not match")
            else:
                logger.info(f"✅ process_batch: Task IDs match")

            results_map = {r.task_id: r for r in response.responses}
            t4_result = results_map.get("T4")
            if not t4_result or t4_result.status != "dependency_failed":
                all_tests_passed = False
                status = t4_result.status if t4_result else "MISSING"
                all_failures["T4_status"] = {"expected": "dependency_failed", "actual": status}
                logger.error(f"❌ process_batch: Task T4 status incorrect (Expected: dependency_failed, Got: {status})")
            else:
                logger.info("✅ process_batch: Task T4 correctly marked as dependency_failed")

            t5_result = results_map.get("T5")
            if not t5_result:
                all_tests_passed = False
                all_failures["T5_status"] = {"expected": "success/error", "actual": "MISSING"}
                logger.error("❌ process_batch: Task T5 result missing")
            elif t5_result.status not in ["success", "error"]:
                all_tests_passed = False
                all_failures["T5_status"] = {"expected": "success or error", "actual": t5_result.status}
                logger.error(f"❌ process_batch: Task T5 has unexpected status: {t5_result.status}")
            elif t5_result.status == "success":
                logger.info("✅ process_batch: Task T5 (corpus file) likely succeeded")
            else:
                logger.warning(f"ℹ️ process_batch: Task T5 (corpus file) resulted in error: {t5_result.error_message}")

    except ValueError as ve:
        all_tests_passed = False
        all_failures["process_batch_value_error"] = {"expected": "Clean run", "actual": f"ValueError: {ve}"}
        logger.error(f"❌ process_batch: Raised ValueError: {ve}", exc_info=True)
    except Exception as e:
        all_tests_passed = False
        all_failures["process_batch_exception"] = {"expected": "Clean run", "actual": f"Exception: {e}"}
        logger.error(f"❌ process_batch: Threw unexpected exception: {e}", exc_info=True)

    exit_code = report_validation_results(
        validation_passed=all_tests_passed,
        validation_failures=all_failures,
        exit_on_failure=False
    )

    try:
        os.remove(temp_corpus_path)
        logger.info(f"Cleaned up temporary corpus file: {temp_corpus_path}")
    except OSError as e:
        logger.warning(f"Could not clean up temporary corpus file {temp_corpus_path}: {e}")

    logger.info(f"LLM Engine Standalone Verification finished with exit code: {exit_code}")
    sys.exit(exit_code)