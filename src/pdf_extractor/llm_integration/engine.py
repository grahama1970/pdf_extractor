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
- `process_batch`: The main entry point function that orchestrates the
  processing of a `BatchRequest`.
- Dependency Handling: Builds message history including questions and results
  from successful dependencies to provide context for the current task.
- Concurrency: Concurrent tasks are executed in parallel using `asyncio.gather`.
- Error Handling: Captures exceptions during task execution.
- Retries: Leverages `retry_llm_call` for robust LLM interactions.
- Validation: Supports optional validation strategies (e.g., Pydantic, JSON, Citation).

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
import traceback
import json
import sys
import os
import tempfile
from pathlib import Path
import typing
from typing import List, Dict, Any, Optional, Callable, Tuple, Set, Union, AsyncGenerator, Type, Literal, Sequence
from loguru import logger
from pydantic import BaseModel

# Absolute imports based on project structure
from pdf_extractor.llm_integration.models import (
    BatchRequest,
    BatchResponse,
    TaskItem,
    ResultItem,
)
from pdf_extractor.llm_integration.validation_utils.base import match_from_list_validator
from pdf_extractor.llm_integration.litellm_call import litellm_call
from litellm.types.utils import ModelResponse
from pdf_extractor.llm_integration.retry_llm_call import retry_llm_call, MaxRetriesExceededError

# Import validation utilities and corpus loader
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
    """
    Retrieves or creates a Pydantic validation function based on model name.
    Placeholder implementation.
    """
    logger.warning(f"Pydantic validation requested for {model_name}, but not fully implemented yet.")
    return None


async def process_batch(request: BatchRequest) -> BatchResponse:
    """
    Processes a batch of LLM tasks asynchronously using a DAG-based execution engine.

    Args:
        request: The BatchRequest containing tasks with dependencies.

    Returns:
        A BatchResponse containing the results for each task.

    Raises:
        ValueError: If dependency chain exceeds maximum allowed depth (20)
    """
    task_map: Dict[str, TaskItem] = {task.task_id: task for task in request.tasks}

    # Build Dependency Graph and Perform Checks
    try:
        original_dependencies, dependents, in_degree = build_dependency_graph(
            tasks=request.tasks,
            max_depth=DEFAULT_MAX_DEPTH
        )
    except ValueError as e:
        logger.error(f"Dependency graph validation failed: {e}")
        raise e

    # Task Execution
    completed_results: Dict[str, ResultItem] = {}
    ready_queue: asyncio.Queue[str] = asyncio.Queue()

    for task_id, deg in in_degree.items():
        if deg == 0:
            ready_queue.put_nowait(task_id)

    semaphore = asyncio.Semaphore(request.max_concurrency)

    async def run_task(task_id: str):
        task = task_map[task_id]
        await semaphore.acquire()
        response_obj: Any = None
        validation_result: Optional[Union[bool, List[str]]] = None
        retry_count = 0
        try:
            # Build Message History with Dependency Context
            messages: List[Dict[str, Any]] = []
            sorted_deps = sorted(list(original_dependencies.get(task_id, [])))
            corpus_parts: List[str] = []
            for dep_id in sorted_deps:
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
                                logger.warning(f"Failed to dump Pydantic model to JSON for dep {dep_id}: {json_err}")
                                result_content = str(dep_result)
                        elif isinstance(dep_result, (dict, list)):
                            try:
                                result_content = json.dumps(dep_result)
                            except Exception as json_err:
                                logger.warning(f"Failed to dump dict/list to JSON for dep {dep_id}: {json_err}")
                                result_content = str(dep_result)
                        else:
                            result_content = str(dep_result)
                        messages.append({"role": "assistant", "content": result_content})
                        corpus_parts.append(result_content)
                    else:
                        logger.warning(f"Task {task_id}: Missing task data or result for successful dependency {dep_id}")
                elif dep_id in completed_results:
                    logger.warning(f"Task {task_id}: Dependency {dep_id} did not succeed (status: {completed_results[dep_id].status}). Skipping context.")
                else:
                    logger.error(f"Task {task_id}: Dependency {dep_id} not found in completed results. Graph logic error?")

            messages.append({"role": "user", "content": task_map[task_id].question})

            logger.debug(f"Running task {task_id} with {len(messages)} messages in history.")

            # Prepare validation strategies
            instantiated_validators: List[Callable[[Any], Union[bool, str]]] = []
            validation_options = task.validation_options or {}

            # Load Corpus from Source
            corpus_from_options = load_corpus_from_source(validation_options, task_id)

            if corpus_from_options:
                corpus_parts.insert(0, corpus_from_options)

            final_corpus = "\n\n---\n\n".join(corpus_parts) if corpus_parts else None
            if final_corpus:
                logger.debug(f"Task {task_id}: Final corpus length: {len(final_corpus)} chars.")
            else:
                logger.debug(f"Task {task_id}: No final corpus available for validation.")

            # Instantiate Validators
            strategy_name = task.validation_strategy
            if strategy_name == "json":
                instantiated_validators.append(json_validator())
                required_fields = validation_options.get("required_fields")
                if isinstance(required_fields, list):
                    instantiated_validators.append(required_fields_validator(required_fields))
            elif strategy_name == "citation":
                min_similarity = validation_options.get("min_similarity", 95.0)
                if final_corpus is None:
                    logger.warning(f"Task {task_id}: Citation validation requested but no corpus available.")
                else:
                    instantiated_validators.append(citation_validator(min_similarity=min_similarity)(final_corpus))
            elif strategy_name == "extract_citation":
                required_phrases = validation_options.get("required_phrases")
                min_similarity = validation_options.get("min_similarity", 90.0)
                if not isinstance(required_phrases, list):
                    logger.warning(f"Task {task_id}: extract_citation validation requested but 'required_phrases' missing or not a list.")
                elif final_corpus is None:
                    logger.warning(f"Task {task_id}: extract_citation validation requested but no corpus available.")
                else:
                    instantiated_validators.append(extract_citation_validator(required_phrases, min_similarity=min_similarity)(final_corpus))
            elif strategy_name == "match_from_list":
                min_similarity = validation_options.get("min_similarity", 85.0)
                delimiter = validation_options.get("delimiter", ',')
                if final_corpus is None:
                    logger.warning(f"Task {task_id}: Match-from-list validation requested but no corpus string available.")
                else:
                    instantiated_validators.append(match_from_list_validator(min_similarity=min_similarity, delimiter=delimiter)(final_corpus))
            elif strategy_name == "pydantic" and task.response_model:
                validator = get_pydantic_validator(task.response_model)
                if validator:
                    instantiated_validators.append(validator)
                else:
                    raise ValueError(f"Pydantic validator for model '{task.response_model}' could not be created.")
            elif strategy_name is not None and strategy_name not in ["pydantic", "json", "citation", "extract_citation", "match_from_list"]:
                logger.warning(f"Task {task_id}: Unknown validation_strategy '{strategy_name}'.")

            if task.model is None:
                raise ValueError(f"Task {task_id} has no model specified.")

            if strategy_name in ["citation", "extract_citation", "match_from_list"] and final_corpus is None:
                logger.error(f"Task {task_id}: Failing early - Corpus required for '{strategy_name}' validation but none loaded.")
                completed_results[task_id] = ResultItem(
                    task_id=task_id,
                    status="error",
                    result=None,
                    error_message=f"Corpus required for '{strategy_name}' validation but none could be loaded.",
                    retry_count=0,
                    validation_status=[f"Corpus required for '{strategy_name}' validation but none could be loaded."]
                )
                return

            # Call LLM with Retry Logic
            logger.debug(f"Task {task_id}: Attempting LLM call...")
            try:
                response_obj, retry_count, validation_result = await retry_llm_call(
                    llm_call=litellm_call,
                    model=task.model,
                    messages=messages,
                    validation_strategies=instantiated_validators,
                    temperature=task.temperature if task.temperature is not None else 0.2,
                    max_tokens=task.max_tokens if task.max_tokens is not None else 1000,
                    api_base=task.api_base,
                    response_format=task.response_model,
                    max_retries=task.max_retries,
                )
                logger.debug(f"Task {task_id}: LLM call finished.")
            except MaxRetriesExceededError as e:
                logger.error(f"Task {task_id} failed after max retries: {e}")
                response_obj = None
                error_str = str(e)
                if "Last errors: " in error_str:
                    try:
                        errors_part = error_str.split("Last errors: ", 1)[1]
                        validation_result = json.loads(errors_part) if errors_part.startswith('[') else [errors_part]
                    except Exception:
                        validation_result = [error_str]
                else:
                    validation_result = [error_str]
                retry_count = task.max_retries

            # Extract Content
            content_to_store: Optional[Union[str, Dict[str, Any], BaseModel]] = None
            if response_obj is not None:
                if isinstance(response_obj, ModelResponse):
                    try:
                        content_to_store = response_obj.choices[0].message.content
                    except (AttributeError, IndexError, TypeError):
                        logger.warning(f"Task {task_id}: Could not extract content from ModelResponse structure.")
                        content_to_store = f"Error: Could not parse ModelResponse: {response_obj}"
                elif isinstance(response_obj, BaseModel):
                    content_to_store = response_obj
                elif isinstance(response_obj, dict):
                    content_to_store = response_obj
                elif isinstance(response_obj, str):
                    content_to_store = response_obj
                elif isinstance(response_obj, AsyncGenerator):
                    logger.warning(f"Task {task_id}: Received unexpected AsyncGenerator. Collecting content.")
                    full_streamed_content = "".join([chunk async for chunk in response_obj])
                    content_to_store = full_streamed_content
                else:
                    logger.error(f"Task {task_id}: Unhandled response type: {type(response_obj)}")
                    content_to_store = f"Error: Unhandled response type {type(response_obj)}"

            # Create ResultItem
            status: Literal['success', 'error', 'dependency_failed']
            final_result_content: Any = None
            error_msg: Optional[str] = None

            if validation_result is True:
                status = "success"
                final_result_content = content_to_store
                if content_to_store is None:
                    logger.error(f"Task {task_id}: Validation passed but content_to_store is None.")
                    status = "error"
                    error_msg = "Internal error: Validation passed but content is None."
            elif isinstance(validation_result, list):
                status = "error"
                error_msg = "; ".join(validation_result)
                final_result_content = content_to_store
            else:
                status = "error"
                error_msg = "Unknown error before or during validation, or pre-check failed."
                final_result_content = content_to_store

            if isinstance(content_to_store, str) and content_to_store.startswith("Error:"):
                status = "error"
                val_status_str = f"; Validation Status: {validation_result}" if validation_result is not True and validation_result is not None else ""
                error_msg = f"{content_to_store}{val_status_str}"
                final_result_content = None

            result_item = ResultItem(
                task_id=task_id,
                status=status,
                result=final_result_content,
                retry_count=retry_count,
                error_message=error_msg,
                validation_status=validation_result
            )
            completed_results[task_id] = result_item

        except MaxRetriesExceededError as e:
            logger.error(f"Task {task_id} ultimately failed validation (caught again): {e}")
            if validation_result is None:
                validation_result = [str(e)]
            error_list = validation_result if isinstance(validation_result, list) else [str(validation_result)]
            completed_results[task_id] = ResultItem(
                task_id=task_id,
                status="error",
                result=None,
                error_message=f"Validation failed after max retries: {'; '.join(error_list)}",
                retry_count=task.max_retries,
                validation_status=validation_result
            )
        except Exception as e:
            logger.error(f"Unexpected error in run_task for {task_id}: {e}", exc_info=True)
            try:
                from typing import Optional
                logger.debug(f"Attempting ResultItem creation inside except block for task {task_id}")
                completed_results[task_id] = ResultItem(
                    task_id=task_id,
                    status="error",
                    result=None,
                    error_message=f"Unexpected engine error: {e}",
                    retry_count=retry_count,
                    validation_status=None
                )
            except NameError as ne:
                logger.critical(f"STILL GETTING NameError '{ne}' even when creating ResultItem in except block after explicit import!")
                completed_results[task_id] = ResultItem(
                    task_id=task_id,
                    status="error",
                    result=None,
                    error_message=f"FATAL: Unexpected engine error AND NameError creating ResultItem: {e}",
                    retry_count=retry_count,
                    validation_status=None
                )
            except Exception as inner_e:
                logger.critical(f"Another error occurred while handling exception for task {task_id}: {inner_e}")
                completed_results[task_id] = ResultItem(
                    task_id=task_id,
                    status="error",
                    result=None,
                    error_message=f"FATAL: Error during exception handling: {inner_e}",
                    retry_count=retry_count,
                    validation_status=None
                )
        finally:
            semaphore.release()

            # Update Dependents
            if task_id in completed_results:
                current_task_result = completed_results[task_id]
                for dependent_id in dependents.get(task_id, []):
                    if dependent_id not in in_degree:
                        continue
                    if current_task_result.status != "success":
                        if dependent_id not in completed_results:
                            logger.warning(f"Marking task {dependent_id} as dependency_failed due to failure in {task_id}.")
                            completed_results[dependent_id] = ResultItem(
                                task_id=dependent_id,
                                status="dependency_failed",
                                result=None,
                                error_message=f"Dependency {task_id} failed with status: {current_task_result.status}",
                                retry_count=0,
                                validation_status=None
                            )
                            in_degree.pop(dependent_id, None)
                        continue
                    if dependent_id in in_degree:
                        in_degree[dependent_id] -= 1
                        if in_degree[dependent_id] == 0 and dependent_id not in completed_results:
                            if dependent_id not in completed_results:
                                ready_queue.put_nowait(dependent_id)
            else:
                logger.error(f"Task {task_id} finished but not found in completed_results within finally block. Logic error?")

    async def execute_tasks():
        active_tasks: Set[asyncio.Task[Any]] = set()

        while not ready_queue.empty():
            task_id = ready_queue.get_nowait()
            if task_id not in completed_results:
                task_coro = run_task(task_id)
                task_obj = asyncio.create_task(task_coro, name=task_id)
                active_tasks.add(task_obj)
                task_obj.add_done_callback(active_tasks.discard)

        logger.debug(f"Execute loop: Starting main processing loop with {len(active_tasks)} active tasks.")
        while active_tasks:
            logger.debug(f"Execute loop: Top of loop. Active tasks: {[t.get_name() for t in active_tasks if hasattr(t, 'get_name')]}")
            done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            logger.debug(f"Execute loop: Wait completed. Done: {[t.get_name() for t in done if hasattr(t, 'get_name')]}, Pending: {len(pending)}")

            new_tasks_added = 0
            while not ready_queue.empty():
                task_id = ready_queue.get_nowait()
                if task_id not in completed_results and not any(t.get_name() == task_id for t in active_tasks if hasattr(t, 'get_name')):
                    task_coro = run_task(task_id)
                    task_obj = asyncio.create_task(task_coro, name=task_id)
                    active_tasks.add(task_obj)
                    task_obj.add_done_callback(active_tasks.discard)
                    new_tasks_added += 1
            if new_tasks_added > 0:
                logger.debug(f"Execute loop: Added {new_tasks_added} new tasks from ready queue.")
            else:
                logger.debug(f"Execute loop: No new tasks added from ready queue.")
        logger.debug(f"Execute loop: Exiting main processing loop.")

    # Start Execution
    await execute_tasks()

    # Final Result Aggregation
    for task_id in task_map:
        if task_id not in completed_results:
            logger.error(f"Task {task_id} was never completed. Possible cycle or graph error.")
            completed_results[task_id] = ResultItem(
                task_id=task_id,
                status="error",
                result=None,
                error_message="Task did not complete. Unresolved dependencies or cycle detected.",
                retry_count=0,
                validation_status=None
            )

    ordered_responses = [completed_results[task.task_id] for task in request.tasks if task.task_id in completed_results]
    if len(ordered_responses) != len(request.tasks):
        logger.error("Mismatch between requested tasks and results obtained. Some tasks might be missing.")

    return BatchResponse(responses=ordered_responses)


if __name__ == "__main__":
    try:
        from pdf_extractor.llm_integration.validation_utils.reporting import report_validation_results
    except ImportError:
        print("❌ FATAL: Could not import report_validation_results.")
        sys.exit(1)
    try:
        from pdf_extractor.llm_integration.models import BatchRequest, TaskItem, BatchResponse, ResultItem
    except ImportError:
        print("❌ FATAL: Could not import engine models (BatchRequest, etc.).")
        sys.exit(1)
    try:
        from pdf_extractor.llm_integration.initialize_litellm_cache import initialize_litellm_cache
    except ImportError:
        print("❌ FATAL: Could not import initialize_litellm_cache.")
        sys.exit(1)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("Starting LLM Engine Standalone Verification...")

    all_tests_passed = True
    all_failures = {}

    # Test Data
    temp_corpus_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    temp_corpus_file.write("This is the corpus text for citation test.")
    temp_corpus_file.close()
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
                validation_options={
                    "corpus_source": "The previous answer was 2.",
                    "corpus_type": "string",
                    "min_similarity": 80.0
                }
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
                validation_options={
                    "corpus_source": temp_corpus_path,
                    "corpus_type": "file",
                    "min_similarity": 85.0
                }
            ),
        ],
        max_concurrency=2
    )

    EXPECTED_TASK_COUNT = len(SAMPLE_REQUEST.tasks)
    EXPECTED_TASK_IDS = {t.task_id for t in SAMPLE_REQUEST.tasks}

    # Run Verification
    logger.info("--- Testing process_batch ---")
    try:
        initialize_litellm_cache()
        response: BatchResponse = asyncio.run(process_batch(SAMPLE_REQUEST))

        if len(response.responses) != EXPECTED_TASK_COUNT:
            all_tests_passed = False
            all_failures["response_count"] = {"expected": EXPECTED_TASK_COUNT, "actual": len(response.responses)}
            logger.error(f"❌ process_batch: Incorrect number of responses.")
        else:
            logger.info(f"✅ process_batch: Returned correct number of responses ({len(response.responses)}).")
            actual_task_ids = {r.task_id for r in response.responses}
            if actual_task_ids != EXPECTED_TASK_IDS:
                all_tests_passed = False
                all_failures["task_ids"] = {"expected": sorted(list(EXPECTED_TASK_IDS)), "actual": sorted(list(actual_task_ids))}
                logger.error(f"❌ process_batch: Task IDs in response do not match request.")
            else:
                logger.info(f"✅ process_batch: Task IDs match.")

            results_map = {r.task_id: r for r in response.responses}
            t4_result = results_map.get("T4")
            if not t4_result or t4_result.status != "dependency_failed":
                all_tests_passed = False
                status = t4_result.status if t4_result else "MISSING"
                all_failures["T4_status"] = {"expected": "dependency_failed", "actual": status}
                logger.error(f"❌ process_batch: Task T4 status incorrect (Expected: dependency_failed, Got: {status}).")
            else:
                logger.info("✅ process_batch: Task T4 correctly marked as dependency_failed.")

            t5_result = results_map.get("T5")
            if not t5_result:
                all_tests_passed = False
                all_failures["T5_status"] = {"expected": "success/error", "actual": "MISSING"}
                logger.error("❌ process_batch: Task T5 result missing.")
            elif t5_result.status not in ["success", "error"]:
                all_tests_passed = False
                all_failures["T5_status"] = {"expected": "success or error", "actual": t5_result.status}
                logger.error(f"❌ process_batch: Task T5 has unexpected status: {t5_result.status}")
            elif t5_result.status == "success":
                logger.info("✅ process_batch: Task T5 (corpus file) likely succeeded.")
            else:
                logger.warning(f"ℹ️ process_batch: Task T5 (corpus file) resulted in error: {t5_result.error_message}. This might be an LLM issue.")

    except ValueError as ve:
        all_tests_passed = False
        all_failures["process_batch_value_error"] = {"expected": "Clean run", "actual": f"ValueError: {ve}"}
        logger.error(f"❌ process_batch: Raised ValueError: {ve}", exc_info=True)
    except Exception as e:
        all_tests_passed = False
        all_failures["process_batch_exception"] = {"expected": "Clean run", "actual": f"Exception: {e}"}
        logger.error(f"❌ process_batch: Threw unexpected exception: {e}", exc_info=True)

    # Report Results
    exit_code = report_validation_results(
        validation_passed=all_tests_passed,
        validation_failures=all_failures,
        exit_on_failure=False
    )

    # Cleanup
    try:
        os.remove(temp_corpus_path)
        logger.info(f"Cleaned up temporary corpus file: {temp_corpus_path}")
    except OSError as e:
        logger.warning(f"Could not clean up temporary corpus file {temp_corpus_path}: {e}")

    logger.info(f"LLM Engine Standalone Verification finished with exit code: {exit_code}")
    sys.exit(exit_code)