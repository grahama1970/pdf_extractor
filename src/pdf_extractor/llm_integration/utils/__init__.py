"""
LiteLLM Utility Functions Package.

This package contains utility modules for file handling, embedding operations,
image processing, NLP tasks using spaCy, and vector operations.

Relevant Documentation:
- spaCy: https://spacy.io/api/doc
- File Operations: https://docs.python.org/3/library/pathlib.html
"""

from pdf_extractor.llm_integration.utils.embedding_utils import create_embedding_with_openai
from pdf_extractor.llm_integration.utils.file_utils import load_text_file, get_project_root, load_env_file
from pdf_extractor.llm_integration.utils.image_utils import process_image_input, compress_image, decode_base64_image, convert_image_to_base64
from pdf_extractor.llm_integration.utils.spacy_utils import get_spacy_model, count_tokens, truncate_text_by_tokens
from pdf_extractor.llm_integration.utils.vector_utils import truncate_vector_for_display, format_embedding_for_debug, get_vector_stats

__all__ = [
    'create_embedding_with_openai',
    'load_text_file',
    'get_project_root',
    'load_env_file',
    'process_image_input',
    'compress_image',
    'decode_base64_image',
    'convert_image_to_base64',
    'get_spacy_model',
    'count_tokens',
    'truncate_text_by_tokens',
    'truncate_vector_for_display',
    'format_embedding_for_debug',
    'get_vector_stats',
]