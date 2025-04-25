"""
Database Utilities Package.

This package provides database interaction utilities, primarily focused on
ArangoDB operations for the LiteLLM service.

Relevant Documentation:
- ArangoDB Python Driver: https://docs.python-arango.com/
"""

from pdf_extractor.llm_integration.utils.db.arango_utils import (
    connect_to_arango_client,
    insert_object,
    handle_relationships,
    get_lessons,
    upsert_lesson,
    update_lesson,
    delete_lesson,
    query_lessons_by_keyword,
    query_lessons_by_concept,
    query_lessons_by_similarity
)

__all__ = [
    'connect_to_arango_client',
    'insert_object',
    'handle_relationships',
    'get_lessons',
    'upsert_lesson',
    'update_lesson',
    'delete_lesson',
    'query_lessons_by_keyword',
    'query_lessons_by_concept',
    'query_lessons_by_similarity'
]