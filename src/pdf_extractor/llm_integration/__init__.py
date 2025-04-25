"""
MCP LiteLLM Service Package Initialization.

This file marks the directory as a Python package, allowing modules within
to be imported.

Relevant Documentation:
- Python Packages: https://docs.python.org/3/tutorial/modules.html#packages
- LiteLLM: https://docs.litellm.ai/docs/

Input/Output: Not applicable.
"""

from pdf_extractor.llm_integration.litellm_call import litellm_call, handle_complex_query
from pdf_extractor.llm_integration.models import (
    BatchRequest,
    BatchResponse,
    TaskItem,
    ResultItem,
    LessonQueryRequest,
    LessonQueryResponse,
    LessonResultItem
)
from pdf_extractor.llm_integration.parser import substitute_placeholders
from pdf_extractor.llm_integration.retry_llm_call import retry_llm_call

__all__ = [
    'litellm_call',
    'handle_complex_query',
    'BatchRequest',
    'BatchResponse',
    'TaskItem',
    'ResultItem',
    'LessonQueryRequest',
    'LessonQueryResponse',
    'LessonResultItem',
    'substitute_placeholders',
    'retry_llm_call',
]