# -*- coding: utf-8 -*-
"""
Dependency Graph Utilities for LLM Integration Engine.

Description:
------------
This module provides functions to build and validate the dependency graph
for a batch of LLM tasks based on explicit and implicit sequential dependencies.
It calculates in-degrees, dependents, and checks for excessive dependency depth.

Links:
  - typing: https://docs.python.org/3/library/typing.html
  - loguru: https://loguru.readthedocs.io/en/stable/

Sample Input (build_dependency_graph):
  - tasks: [
        TaskItem(task_id="T0", ...),
        TaskItem(task_id="T1", dependencies=["T0"], method="sequential", ...),
        TaskItem(task_id="T2", method="sequential", ...)
    ]
  - max_depth: 20

Sample Output (build_dependency_graph):
  - (
        {"T0": set(), "T1": {"T0"}, "T2": set()}, # original_dependencies
        {"T0": {"T1"}, "T1": {"T2"}, "T2": set()}, # dependents (incl. sequential)
        {"T0": 0, "T1": 1, "T2": 1}               # in_degree (incl. sequential)
    )
  - Raises ValueError if max_depth is exceeded.
"""

import sys
from typing import List, Dict, Set, Tuple, Optional
from loguru import logger

# Use absolute import path for TaskItem
try:
   from pdf_extractor.llm_integration.models import TaskItem
except ImportError:
   logger.critical("FATAL: Failed to import TaskItem model from pdf_extractor.llm_integration.models. Dependency graph functionality will fail.")
   # Raise the error to prevent execution with a dummy class
   raise ImportError("Could not import TaskItem model.") from None

# Define a constant for the default max depth
DEFAULT_MAX_DEPTH = 20

def build_dependency_graph(
    tasks: List[TaskItem],
    max_depth: int = DEFAULT_MAX_DEPTH
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, int]]:
    """
    Builds the dependency graph, calculates in-degrees/dependents, and checks depth.

    Handles both explicit dependencies and implicit sequential dependencies.

    Args:
        tasks: A list of TaskItem objects.
        max_depth: The maximum allowed dependency chain depth.

    Returns:
        A tuple containing:
        - original_dependencies: Dict mapping task_id to its explicit dependencies.
        - dependents: Dict mapping task_id to the set of tasks that depend on it (including implicit sequential).
        - in_degree: Dict mapping task_id to its in-degree (number of dependencies, including implicit sequential).

    Raises:
        ValueError: If the dependency chain for any task exceeds max_depth based on *explicit* dependencies.
    """
    task_map: Dict[str, TaskItem] = {task.task_id: task for task in tasks}
    original_dependencies: Dict[str, Set[str]] = {task.task_id: set(task.dependencies) for task in tasks}

    # --- Initial Graph Setup (based on explicit dependencies) ---
    dependents: Dict[str, Set[str]] = {task_id: set() for task_id in task_map}
    in_degree: Dict[str, int] = {task_id: 0 for task_id in task_map}
    for task_id, explicit_deps in original_dependencies.items():
        in_degree[task_id] = len(explicit_deps)
        for dep in explicit_deps:
            if dep in dependents: # Ensure dependency exists in the batch
                dependents[dep].add(task_id)
            else:
                logger.warning(f"Task {task_id} lists explicit dependency '{dep}' which is not present in the batch request.")
                # If strict dependency checking is needed, raise ValueError here
                # raise ValueError(f"Task {task_id} lists dependency '{dep}' which is not in the batch.")

    # --- Add Implicit Sequential Dependencies ---
    # This modifies in_degree and dependents for scheduling purposes
    for i, task in enumerate(tasks):
        # Add dependency on the *previous* task if sequential and no *explicit* dependencies exist
        if task.method == "sequential" and not original_dependencies[task.task_id] and i > 0:
            prev_task_id = tasks[i-1].task_id
            if prev_task_id in task_map: # Ensure previous task exists
               # Check if this implicit dependency (prev_task_id -> task.task_id) already exists
               if task.task_id not in dependents.get(prev_task_id, set()):
                   in_degree[task.task_id] += 1
                   dependents.setdefault(prev_task_id, set()).add(task.task_id)
                   logger.debug(f"Added implicit sequential dependency: {prev_task_id} -> {task.task_id}")

    # --- Depth Check (using original explicit dependencies only) ---
    # This prevents implicit sequential dependencies from causing excessive depth errors
    def calculate_max_depth_recursive(
        task_id: str,
        visited: Optional[Set[str]] = None,
        current_depth: int = 0
    ) -> int:
        """Recursive helper to calculate maximum explicit dependency depth."""
        local_visited = visited if visited is not None else set()
        # Cycle detection or path already visited at greater/equal depth
        if task_id in local_visited:
             # If we revisit a node, it implies a cycle or a path we've already explored.
             # We return the current depth as going further would be redundant or infinite.
             # This check primarily handles cycles.
            return current_depth

        local_visited.add(task_id)
        max_d = current_depth

        # Iterate through *explicit* dependencies only for depth calculation
        for dep in original_dependencies.get(task_id, []):
            if dep in task_map: # Only consider dependencies within the batch
                # Pass a copy of visited set to avoid interference between different branches
                depth = calculate_max_depth_recursive(dep, local_visited.copy(), current_depth + 1)
                if depth > max_d:
                    max_d = depth
            # else: Dependency not in batch - ignore for depth calculation

        return max_d

    logger.debug("Performing dependency depth check...")
    for task_id in task_map:
        calculated_depth = calculate_max_depth_recursive(task_id)
        logger.debug(f"Task {task_id}: Calculated explicit dependency depth = {calculated_depth}")
        if calculated_depth > max_depth:
            error_message = (
                f"Dependency chain for task {task_id} exceeds maximum allowed depth of {max_depth}. "
                f"Found depth: {calculated_depth}. Simplify your task dependencies."
            )
            logger.error(error_message)
            raise ValueError(error_message)
    logger.debug("Dependency depth check passed.")

    return original_dependencies, dependents, in_degree


# --- Main Execution Guard (Standalone Verification) ---
if __name__ == "__main__":
    # Need reporting for standardized output
    try:
        # Adjust relative path based on potential execution context
        from pdf_extractor.llm_integration.validation_utils.reporting import report_validation_results
    except ImportError:
        try:
            # If running from utils directory directly
            sys.path.append(str(Path(__file__).resolve().parent.parent / 'validation_utils'))
            from reporting import report_validation_results
        except ImportError:
            print("❌ FATAL: Could not import report_validation_results.")
            sys.exit(1)

    # Need models for test data
    try:
        from pdf_extractor.llm_integration.models import TaskItem
    except ImportError:
       print("❌ FATAL: Could not import TaskItem model for __main__.")
       # Exit if the real model cannot be imported for tests
       sys.exit(1)

    # Configure Loguru for verification output
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("Starting Dependency Graph Utilities Standalone Verification...")

    all_tests_passed = True
    all_failures = {}

    # --- Test Cases ---
    test_cases = [
        {
            "id": "simple_concurrent",
            "tasks": [
                TaskItem(task_id="T0", question="Q0"),
                TaskItem(task_id="T1", question="Q1"),
            ],
            "expected_orig_deps": {"T0": set(), "T1": set()},
            "expected_deps": {"T0": set(), "T1": set()},
            "expected_in_degree": {"T0": 0, "T1": 0},
            "expect_exception": None,
        },
        {
            "id": "simple_explicit_seq",
            "tasks": [
                TaskItem(task_id="T0", question="Q0"),
                TaskItem(task_id="T1", question="Q1", dependencies=["T0"], method="sequential"),
            ],
            "expected_orig_deps": {"T0": set(), "T1": {"T0"}},
            "expected_deps": {"T0": {"T1"}, "T1": set()},
            "expected_in_degree": {"T0": 0, "T1": 1},
            "expect_exception": None,
        },
        {
            "id": "simple_implicit_seq",
            "tasks": [
                TaskItem(task_id="T0", question="Q0"),
                TaskItem(task_id="T1", question="Q1", method="sequential"), # Implicit dependency on T0
            ],
            "expected_orig_deps": {"T0": set(), "T1": set()},
            "expected_deps": {"T0": {"T1"}, "T1": set()}, # T0 gets T1 as dependent
            "expected_in_degree": {"T0": 0, "T1": 1}, # T1 gets in-degree 1
            "expect_exception": None,
        },
        {
            "id": "mixed_explicit_implicit",
            "tasks": [
                TaskItem(task_id="T0", question="Q0"),
                TaskItem(task_id="T1", question="Q1", dependencies=["T0"]), # Explicit
                TaskItem(task_id="T2", question="Q2", method="sequential"), # Implicit on T1
                TaskItem(task_id="T3", question="Q3", dependencies=["T0"]), # Explicit
                TaskItem(task_id="T4", question="Q4", method="sequential"), # Implicit on T3
            ],
            "expected_orig_deps": {"T0": set(), "T1": {"T0"}, "T2": set(), "T3": {"T0"}, "T4": set()},
            "expected_deps": {"T0": {"T1", "T3"}, "T1": {"T2"}, "T2": set(), "T3": {"T4"}, "T4": set()},
            "expected_in_degree": {"T0": 0, "T1": 1, "T2": 1, "T3": 1, "T4": 1},
            "expect_exception": None,
        },
        {
            "id": "depth_check_pass",
          "tasks": [
              TaskItem(task_id="T0", question="Q0"),
              TaskItem(task_id="T1", question="Q1", dependencies=["T0"]),
              TaskItem(task_id="T2", question="Q2", dependencies=["T1"]),
            ],
            "max_depth": 5,
            "expected_orig_deps": {"T0": set(), "T1": {"T0"}, "T2": {"T1"}},
            "expected_deps": {"T0": {"T1"}, "T1": {"T2"}, "T2": set()},
            "expected_in_degree": {"T0": 0, "T1": 1, "T2": 1},
            "expect_exception": None,
        },
        {
            "id": "depth_check_fail",
          "tasks": [
              TaskItem(task_id="T0", question="Q0"),
              TaskItem(task_id="T1", question="Q1", dependencies=["T0"]),
              TaskItem(task_id="T2", question="Q2", dependencies=["T1"]),
            ],
            "max_depth": 1, # Should fail as depth is 2
            "expect_exception": ValueError,
        },
         {
            "id": "ignore_missing_dep",
            "tasks": [
                TaskItem(task_id="T0", question="Q0"),
                TaskItem(task_id="T1", question="Q1", dependencies=["T_MISSING"]),
            ],
            "expected_orig_deps": {"T0": set(), "T1": {"T_MISSING"}},
            "expected_deps": {"T0": set(), "T1": set()}, # T_MISSING doesn't exist to add T1
            "expected_in_degree": {"T0": 0, "T1": 1}, # In-degree counts explicit deps even if missing
            "expect_exception": None, # Should log warning but not fail by default
        },
        {
            "id": "implicit_seq_overrides_nothing", # Ensure implicit doesn't add if explicit exists
            "tasks": [
                TaskItem(task_id="T0", question="Q0"),
                TaskItem(task_id="T1", question="Q1", dependencies=["T0"], method="sequential"),
            ],
            "expected_orig_deps": {"T0": set(), "T1": {"T0"}},
            "expected_deps": {"T0": {"T1"}, "T1": set()},
            "expected_in_degree": {"T0": 0, "T1": 1}, # Should remain 1
            "expect_exception": None,
        },
    ]

    # --- Run Verification ---
    logger.info("--- Testing build_dependency_graph ---")
    for test in test_cases:
        test_id = test["id"]
        logger.info(f"--- Running Test: {test_id} ---")
        test_passed = True
        test_failures = {}
        max_depth_override = test.get("max_depth", DEFAULT_MAX_DEPTH)

        try:
            orig_deps, deps, in_deg = build_dependency_graph(test["tasks"], max_depth=max_depth_override)

            # Check if exception was expected but not raised
            if test["expect_exception"] is not None:
                test_passed = False
                test_failures["exception_not_raised"] = {"expected": test["expect_exception"].__name__, "actual": "No exception"}
                logger.error(f"❌ {test_id}: Failed - Expected exception {test['expect_exception'].__name__} was not raised.")
            else:
                # Compare graph structures
                if orig_deps != test["expected_orig_deps"]:
                    test_passed = False
                    test_failures["original_dependencies"] = {"expected": test["expected_orig_deps"], "actual": orig_deps}
                if deps != test["expected_deps"]:
                    test_passed = False
                    test_failures["dependents"] = {"expected": test["expected_deps"], "actual": deps}
                if in_deg != test["expected_in_degree"]:
                    test_passed = False
                    test_failures["in_degree"] = {"expected": test["expected_in_degree"], "actual": in_deg}

                if test_passed:
                    logger.info(f"✅ {test_id}: Passed - Graph structures match expected.")
                else:
                    logger.error(f"❌ {test_id}: Failed - Graph structure mismatch. Details: {test_failures}")

        except Exception as e:
            if test["expect_exception"] is None:
                test_passed = False
                test_failures["unexpected_exception"] = {"expected": "No exception", "actual": f"{type(e).__name__}: {e}"}
                logger.error(f"❌ {test_id}: Failed - Unexpected exception raised: {e}", exc_info=True)
            elif not isinstance(e, test["expect_exception"]):
                test_passed = False
                test_failures["wrong_exception_type"] = {"expected": test["expect_exception"].__name__, "actual": type(e).__name__}
                logger.error(f"❌ {test_id}: Failed - Raised wrong exception type. Got {type(e).__name__}, expected {test['expect_exception'].__name__}.", exc_info=True)
            else:
                logger.info(f"✅ {test_id}: Passed - Correctly raised expected exception {type(e).__name__}.")

        if not test_passed:
            all_tests_passed = False
            all_failures.update(test_failures)

    # --- Final Report ---
    exit_code = report_validation_results(
        validation_passed=all_tests_passed,
        validation_failures=all_failures,
        exit_on_failure=False # Let sys.exit handle it
    )

    logger.info(f"Dependency Graph Utilities Standalone Verification finished with exit code: {exit_code}")
    sys.exit(exit_code)