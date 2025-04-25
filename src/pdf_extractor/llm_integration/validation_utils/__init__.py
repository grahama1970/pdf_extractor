"""
Validation Utilities for LLM Responses.

This package provides validation functions and utilities for LLM responses,
following the principles outlined in VALIDATION_REQUIREMENTS.md.

Modules:
- base: Core validation functionality and content extraction
- json_validators: Validation for JSON structures
- citation_validators: Validation for citations and references to source material
- reporting: Functions for reporting validation results
"""

# Export core functions for easy access
from pdf_extractor.llm_integration.validation_utils.base import (
    extract_content,
    validate_results,
    load_fixture,
    keyword_validator
)

from pdf_extractor.llm_integration.validation_utils.json_validators import (
    json_validator,
    required_fields_validator
)

from pdf_extractor.llm_integration.validation_utils.citation_validators import (
    citation_validator
)

from pdf_extractor.llm_integration.validation_utils.reporting import (
    report_validation_results
)

__all__ = [
    'extract_content',
    'validate_results',
    'load_fixture',
    'keyword_validator',
    'json_validator',
    'required_fields_validator',
    'citation_validator',
    'report_validation_results'
]