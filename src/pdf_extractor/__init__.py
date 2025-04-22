#!/usr/bin/env python3
"""
PDF Extractor module for converting PDFs to structured JSON objects.

This module provides a complete pipeline for extracting structured data from PDFs,
including tables, headings, paragraphs, and more, with a focus on maintaining
hierarchical structure and accurate extraction.
"""

__version__ = '0.1.0'

from .table_extractor import TableExtractor
from .improved_table_merger import process_and_merge_tables

__all__ = [
    'TableExtractor',
    'process_and_merge_tables',
]
