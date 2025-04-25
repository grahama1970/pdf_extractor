# -*- coding: utf-8 -*-
"""
Description:
    Defines Pydantic models for the MCP LiteLLM service request and response structures.
    These models ensure data validation and structure for interactions with the service.

Core Lib Links:
    - Pydantic Documentation: https://docs.pydantic.dev/latest/

Sample I/O:
    N/A (This file defines models, not executable logic)
"""

from typing import List, Optional, Any, Literal, Dict, Union # Added Dict
from pydantic import BaseModel, Field

# ============================================================================ #
#                                Request Models                                #
# ============================================================================ #

class TaskItem(BaseModel):
    """
    Represents a single task in the batch, with dependencies.
    """
    task_id: str = Field(..., description="Unique identifier for this task.")
    dependencies: List[str] = Field(default_factory=list, description="List of task_ids this task depends on.")
    question: str = Field(..., description="The text of the question to be processed by the LLM.")
    model: Optional[str] = Field(
        default='openai/gpt-4o-mini',
        description="The LiteLLM model string to use for this task."
    )
    validation_strategy: Optional[str] = Field(
        default='pydantic',
        description="Strategy for validating the LLM response (e.g., 'pydantic')."
    )
    temperature: Optional[float] = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the LLM."
    )
    max_tokens: Optional[int] = Field(
        default=1000,
        gt=0,
        description="Maximum number of tokens to generate in the response."
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Optional API base URL for the LLM provider."
    )
    response_model: Optional[str] = Field(
        default=None,
        description="Optional Pydantic model name (as string) for structured response parsing."
    )
    method: Literal['sequential', 'concurrent'] = Field(
        default='concurrent',
        description="Execution method ('sequential' or 'concurrent')."
    )
    # Additional LiteLLM parameters can be added here as needed


    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for this task."
    )
    validation_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary containing configuration for validation (e.g., corpus, min_similarity, required_phrases)."
    )

class BatchRequest(BaseModel):
    """
    Represents a batch request containing multiple task items.
    """
    tasks: List[TaskItem] = Field(..., description="A list of tasks to be processed with dependencies.")

    max_concurrency: int = Field(
        default=10,
        gt=0,
        description="Maximum number of concurrent tasks to run."
    )
# ============================================================================ #
#                                Response Models                               #
# ============================================================================ #

class ResultItem(BaseModel):
    """
    Represents the result for a single task item from a batch request.
    """
    task_id: str = Field(..., description="Identifier linking back to the original TaskItem task_id.")
    status: Literal['success', 'error', 'dependency_failed'] = Field(
        ...,
        description="Indicates if the processing was successful, resulted in an error, or failed due to dependency failure."
    )
    result: Optional[Any] = Field(
        default=None,
        description="The processed result from the LLM (can be structured or raw text)."
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Detailed error message if the status is 'error' or 'dependency_failed'."
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score associated with the result, if applicable."
    )
    retry_count: Optional[int] = Field(
        default=0,
        ge=0,
        description="Number of retries attempted for this task."
    )
    validation_status: Optional[Union[bool, List[str]]] = Field(
       default=None,
       description="Result of validation: True if passed, list of error strings if failed, None if not run or errored before validation."
    )


class BatchResponse(BaseModel):
    """
    Represents the batch response containing results for all processed tasks.
    """
    responses: List[ResultItem] = Field(..., description="A list of responses corresponding to the input tasks.")


# ============================================================================ #
#                             Lessons Learned Models                           #
# ============================================================================ #

class LessonQueryRequest(BaseModel):
    """
    Request model for querying lessons learned.
    """
    query_text: str = Field(..., description="The natural language query to search for relevant lessons.")
    top_k: int = Field(default=3, gt=0, description="The maximum number of lessons to return.")


class LessonResultItem(BaseModel):
    """
    Represents a single lesson learned document returned by the query.
    Includes the document content and similarity score.
    """
    id: str = Field(..., description="The unique identifier (_id) of the lesson document in ArangoDB.")
    key: str = Field(..., description="The unique key (_key) of the lesson document in ArangoDB.")
    score: float = Field(..., description="The similarity score between the query and the lesson.")
    lesson: Dict[str, Any] = Field(..., description="The full lesson learned document content.") # Corrected type hint


class LessonQueryResponse(BaseModel):
    """
    Response model for the lessons learned query endpoint.
    Contains a list of relevant lesson documents.
    """
    lessons: List[LessonResultItem] = Field(..., description="A list of relevant lessons learned, sorted by similarity.")


# ============================================================================ #
#                                End of File                                   #
# ============================================================================ #


# --- Standalone Validation Block --- 

import sys
from loguru import logger

def main_validation():
    """Performs basic validation by instantiating models."""
    logger.info("--- Running Standalone Validation for models.py ---")
    validation_passed = True
    errors = []

    try:
        # Test TaskItem
        task = TaskItem(task_id="t1", question="What is 2+2?")
        logger.debug(f"TaskItem instantiated: {task.model_dump()}")

        # Test BatchRequest
        batch_req = BatchRequest(tasks=[task])
        logger.debug(f"BatchRequest instantiated: {batch_req.model_dump()}")

        # Test ResultItem (success)
        res_success = ResultItem(task_id="t1", status="success", result="4")
        logger.debug(f"ResultItem (success) instantiated: {res_success.model_dump()}")

        # Test ResultItem (error)
        res_error = ResultItem(task_id="t2", status="error", error_message="LLM failed")
        logger.debug(f"ResultItem (error) instantiated: {res_error.model_dump()}")

        # Test BatchResponse
        batch_res = BatchResponse(responses=[res_success, res_error])
        logger.debug(f"BatchResponse instantiated: {batch_res.model_dump()}")

        # Test LessonQueryRequest
        lesson_req = LessonQueryRequest(query_text="How to handle errors?")
        logger.debug(f"LessonQueryRequest instantiated: {lesson_req.model_dump()}")

        # Test LessonResultItem
        lesson_res_item = LessonResultItem(id="lessons/123", key="123", score=0.95, lesson={"problem": "X", "solution": "Y"})
        logger.debug(f"LessonResultItem instantiated: {lesson_res_item.model_dump()}")

        # Test LessonQueryResponse
        lesson_res = LessonQueryResponse(lessons=[lesson_res_item])
        logger.debug(f"LessonQueryResponse instantiated: {lesson_res.model_dump()}")

    except Exception as e:
        errors.append(f"Failed to instantiate models: {e}")
        validation_passed = False

    # Report validation status
    if validation_passed:
        logger.success("✅ Standalone validation passed: All models instantiated successfully.")
        print("\n✅ VALIDATION COMPLETE - Basic model instantiation verified.")
        sys.exit(0)
    else:
        for error in errors:
            logger.error(f"❌ {error}")
        print("\n❌ VALIDATION FAILED - Model instantiation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main_validation()
