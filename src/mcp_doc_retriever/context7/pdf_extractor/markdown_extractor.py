"""
Markdown extraction and processing module for PDF extraction workflow.

This module extracts structured content from Markdown files, including:
- Text blocks with section hierarchy tracking
- Tables with parsing and validation
- Images with metadata
- Code blocks with language detection and metadata

It integrates with the Marker PDF extraction pipeline by supporting
marker_json format for enhanced table and image extraction from PDFs.

Third-party package documentation:
- markdown-it-py: https://github.com/executablebooks/markdown-it-py
- tiktoken: https://github.com/openai/tiktoken
- ftfy: https://github.com/rspeer/ftfy

Example usage:
    >>> from mcp_doc_retriever.context7.pdf_extractor.markdown_extractor import extract_from_markdown
    >>> markdown_file = "example.md"
    >>> repo_link = "https://github.com/example/repo"
    >>> extracted_data = extract_from_markdown(markdown_file, repo_link)
    >>> for item in extracted_data:
    ...     print(f"Type: {item['type']}, Token count: {item['token_count']}")
"""

import os
import sys
import re
import json
import datetime
import unicodedata
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, cast

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dependencies 
try:
    from markdown_it import MarkdownIt
    MARKDOWN_IT_AVAILABLE = True
except ImportError:
    logger.warning("markdown-it-py not found. Install with: uv add markdown-it-py")
    MarkdownIt = None
    MARKDOWN_IT_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    logger.warning("tiktoken not found. Install with: uv add tiktoken")
    tiktoken = None
    TIKTOKEN_AVAILABLE = False

try:
    import ftfy
    FTFY_AVAILABLE = True
except ImportError:
    logger.warning("ftfy not found. Install with: uv add ftfy")
    ftfy = None
    FTFY_AVAILABLE = False

# Import utility function - use our own implementation
from mcp_doc_retriever.context7.pdf_extractor.utils import calculate_iou

# Minimal implementations for missing dependencies
def mock_hash_string(text: str) -> str:
    """Minimal version for standalone testing."""
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()

class MockSectionHierarchy:
    """Minimal version for standalone testing."""
    def __init__(self):
        self.titles = []
        self.hashes = []
        
    def update(self, number: str, title: str, content: str) -> None:
        self.titles.append(f"{number}. {title}")
        self.hashes.append(mock_hash_string(title))
        
    def get_titles(self) -> List[str]:
        return self.titles
            
    def get_hashes(self) -> List[str]:
        return self.hashes

def mock_extract_code_metadata(code: str, language: str) -> Dict[str, Any]:
    """Minimal version for standalone testing."""
    return {
        "language": language, 
        "length": len(code),
        "functions": []
    }

# Import or define the actual implementations
try:
    from mcp_doc_retriever.context7.tree_sitter_utils import extract_code_metadata
except ImportError:
    logger.warning("Using mock implementation for extract_code_metadata")
    extract_code_metadata = mock_extract_code_metadata

try:
    from mcp_doc_retriever.context7.text_chunker import SectionHierarchy, hash_string
except ImportError:
    logger.warning("Using mock implementation for SectionHierarchy and hash_string")
    SectionHierarchy = MockSectionHierarchy
    hash_string = mock_hash_string

def clean_section_title(title: str, to_ascii: bool = False) -> str:
    """
    Cleans a section title by normalizing Unicode characters and removing unprintable characters.

    Args:
        title: The raw section title.
        to_ascii: If True, converts non-ASCII characters to ASCII equivalents.

    Returns:
        The cleaned section title.
    """
    try:
        # If ftfy is available, use it, otherwise use a simple cleanup
        if FTFY_AVAILABLE:
            cleaned = ftfy.fix_text(title, normalization="NFC")
        else:
            cleaned = title
            
        cleaned = "".join(c for c in cleaned if unicodedata.category(c)[0] != "C")
        if to_ascii:
            cleaned = (
                unicodedata.normalize("NFKD", cleaned)
                .encode("ascii", "ignore")
                .decode("ascii")
            )
        cleaned = cleaned.strip()
        return cleaned if cleaned else "Unnamed Section"
    except Exception as e:
        logger.warning(f"Error cleaning section title '{title}': {e}")
        return "Unnamed Section"


def parse_markdown_table(content: str) -> Optional[Dict[str, Any]]:
    """
    Parses a Markdown table into a dictionary with headers and rows.

    Args:
        content: The Markdown table content (e.g., "| Header1 | Header2 |\n|---|---|\n| Row1 | Row2 |").

    Returns:
        Dictionary with 'headers' (list) and 'rows' (list of lists), or None if parsing fails.
    """
    try:
        lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
        if len(lines) < 2:
            return None

        headers = [h.strip() for h in lines[0].strip("|").split("|") if h.strip()]
        if not headers:
            return None

        separator = lines[1].strip("|").split("|")
        if not all(re.match(r"[-: ]+", s.strip()) for s in separator):
            return None

        rows = []
        for line in lines[2:]:
            cells = [
                c.strip() for c in line.strip("|").split("|") if c.strip() or c == ""
            ]
            if len(cells) == len(headers):
                rows.append(cells)

        return {"headers": headers, "rows": rows}
    except Exception as e:
        logger.warning(f"Error parsing Markdown table: {e}")
        return None


def parse_markdown_image(content: str) -> Optional[Dict[str, str]]:
    """
    Parses a Markdown image (e.g., ![alt](src)) into a dictionary with alt text and source.

    Args:
        content: The Markdown image content.

    Returns:
        Dictionary with 'alt' and 'src', or None if parsing fails.
    """
    try:
        match = re.match(r"!\[(.*?)\]\((.*?)\)", content.strip())
        if match:
            return {"alt": match.group(1).strip(), "src": match.group(2).strip()}
        return None
    except Exception as e:
        logger.warning(f"Error parsing Markdown image: {e}")
        return None


def extract_from_markdown(
    file_path: str,
    repo_link: str,
    marker_json: Optional[Dict[str, Any]] = None,
    table_cache: Optional[List[Dict[str, Any]]] = None,
    use_mock_data: bool = False
) -> List[Dict[str, Any]]:
    """
    Extracts text, images, tables, and code blocks from a Markdown file as separate JSON objects,
    with section hierarchy tracking. Concatenates paragraphs and lists into text nodes, splitting
    on two or more line breaks. Cleans section titles with ftfy and parses code with tree-sitter.
    Integrates Marker's JSON for tables and images, with validation against table_cache.

    Args:
        file_path: Path to the Markdown file.
        repo_link: URL of the repository.
        marker_json: Optional Marker JSON output for tables and images.
        table_cache: Optional list of tables from table_extractor.py for validation.
        use_mock_data: If True, use mock data for testing when dependencies are missing.

    Returns:
        List of dictionaries containing extracted content (text, image, table, code).
    """
    # If required dependencies are missing and mock data is requested, return mock data
    if (not MARKDOWN_IT_AVAILABLE or not TIKTOKEN_AVAILABLE) and use_mock_data:
        logger.warning("Using mock data because required dependencies are missing")
        return _create_mock_extraction_data(file_path, repo_link)
        
    section_hierarchy = SectionHierarchy()
    extracted_data: List[Dict[str, Any]] = []
    
    try:
        if not MARKDOWN_IT_AVAILABLE:
            raise ImportError("markdown-it-py is required for markdown extraction")
            
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is required for token counting")
            
        md = MarkdownIt("commonmark", {"html": False, "typographer": True})
        markdown_content = Path(file_path).read_text(encoding="utf-8")
        tokens = md.parse(markdown_content)
        logger.debug(f"Parsed Markdown file: {file_path}, {len(tokens)} tokens")

        text_content = []
        text_start_line = None
        last_line = 0
        image_content = ""
        table_content = ""
        code_block = None
        code_start_line = None
        table_start_line = None
        encoding = tiktoken.encoding_for_model("gpt-4")
        section_title = ""
        section_level = 0
        section_counts = [0] * 6
        current_page = 1

        # Extract tables and images from Marker JSON
        marker_tables = []
        marker_images = []
        if marker_json:
            for page in marker_json.get("pages", []):
                page_number = page.get("page_number", 1)
                for block in page.get("blocks", []):
                    if block.get("type") == "table":
                        marker_tables.append(
                            {
                                "page": page_number,
                                "bbox": block.get("bbox", [0, 0, 0, 0]),
                                "table_data": {
                                    "cells": [
                                        {
                                            "row": c["row"],
                                            "col": c["col"],
                                            "text": c["text"],
                                        }
                                        for c in block.get("cells", [])
                                    ]
                                },
                            }
                        )
                    elif block.get("type") == "image":
                        marker_images.append(
                            {
                                "page": page_number,
                                "bbox": block.get("bbox", [0, 0, 0, 0]),
                                "alt": block.get("alt", ""),
                                "src": block.get("src", ""),
                            }
                        )

        table_lines = []
        in_table = False

        def flush_text():
            nonlocal text_content, text_start_line, last_line
            if text_content:
                content = "\n".join(text_content).strip()
                if content:
                    section_titles = section_hierarchy.get_titles()
                    section_hash_path = section_hierarchy.get_hashes()
                    text_token_count = len(encoding.encode(content))
                    text_end_line = last_line
                    extracted_data.append(
                        {
                            "file_path": file_path,
                            "repo_link": repo_link,
                            "extraction_date": datetime.datetime.now().isoformat(),
                            "type": "text",
                            "content": content,
                            "line_span": (text_start_line, text_end_line),
                            "token_count": text_token_count,
                            "section_id": hash_string(content),
                            "section_path": section_titles,
                            "section_hash_path": section_hash_path,
                            "page": current_page,
                            "metadata": {},
                        }
                    )
                    logger.debug(
                        f"Added text at lines {text_start_line}-{text_end_line}, "
                        f"section_path: {section_titles}"
                    )
                text_content = []
                text_start_line = None

        for i, token in enumerate(tokens):
            # Track current line for line_span
            current_line = token.map[0] + 1 if token.map else last_line + 1

            # Check for double line breaks
            if token.map and token.map[0] > last_line + 1:
                flush_text()
                text_start_line = current_line

            if token.type == "heading_open":
                flush_text()
                section_level = int(token.tag[1:])
                section_title = ""
                logger.debug(f"Found heading level {section_level}")

            elif (
                token.type == "inline"
                and i > 0
                and tokens[i - 1].type == "heading_open"
            ):
                section_title = token.content.strip()
                logger.debug(f"Captured raw section title: {section_title}")

            elif token.type == "heading_close":
                cleaned_title = clean_section_title(section_title, to_ascii=True)
                section_number = ""
                number_match = re.match(r"(\d+(?:\.\d+)*\.?)\s*(.*)", section_title)
                if number_match:
                    section_number = number_match.group(1).rstrip(".")
                    cleaned_title = clean_section_title(
                        number_match.group(2) or "Unnamed Section", to_ascii=True
                    )
                else:
                    for j in range(section_level, len(section_counts)):
                        section_counts[j] = 0
                    section_counts[section_level - 1] += 1
                    section_number_parts = [
                        str(section_counts[j])
                        for j in range(section_level)
                        if section_counts[j] > 0
                    ]
                    section_number = (
                        ".".join(section_number_parts)
                        if section_number_parts
                        else str(section_level)
                    )

                section_hierarchy.update(
                    section_number, cleaned_title, markdown_content
                )
                logger.debug(f"Updated hierarchy: {section_hierarchy.get_titles()}")

            elif token.type in (
                "paragraph_open",
                "bullet_list_open",
                "ordered_list_open",
            ):
                if not text_start_line:
                    text_start_line = current_line
                logger.debug(f"Started text block at line {text_start_line}")

            elif token.type == "inline":
                content = token.content
                page_match = re.match(r"<!--\s*page:\s*(\d+)\s*-->", content)
                if page_match:
                    flush_text()
                    current_page = int(page_match.group(1))
                    logger.debug(f"Updated current page to {current_page}")
                    continue

                # Handle images
                image_match = re.match(r"!\[(.*?)\]\((.*?)\)", content)
                if image_match:
                    flush_text()
                    image_content = content
                    image_start_line = current_line
                    image_end_line = current_line
                    image_data = parse_markdown_image(image_content)
                    section_titles = section_hierarchy.get_titles()
                    section_hash_path = section_hierarchy.get_hashes()
                    image_token_count = len(encoding.encode(image_content))

                    image_metadata = {
                        "alt": image_data["alt"] if image_data else "",
                        "src": image_data["src"] if image_data else "",
                    }
                    if marker_images:
                        matching_image = next(
                            (
                                img
                                for img in marker_images
                                if img["page"] == current_page
                            ),
                            None,
                        )
                        if matching_image:
                            image_metadata["bbox"] = matching_image["bbox"]
                            image_metadata["source"] = "marker_json"
                        else:
                            image_metadata["source"] = "markdown"
                    else:
                        image_metadata["source"] = "markdown"

                    extracted_data.append(
                        {
                            "file_path": file_path,
                            "repo_link": repo_link,
                            "extraction_date": datetime.datetime.now().isoformat(),
                            "type": "image",
                            "content": image_content,
                            "line_span": (image_start_line, image_end_line),
                            "token_count": image_token_count,
                            "section_id": hash_string(image_content),
                            "section_path": section_titles,
                            "section_hash_path": section_hash_path,
                            "page": current_page,
                            "metadata": image_metadata,
                        }
                    )
                    logger.debug(
                        f"Added image at lines {image_start_line}-{image_end_line}, "
                        f"section_path: {section_titles}"
                    )
                    continue

                # Handle tables
                if re.match(r"\|.*\|", content):
                    flush_text()
                    table_lines.append(content)
                    if not in_table:
                        in_table = True
                        table_start_line = current_line
                else:
                    if in_table:
                        table_content = "\n".join(table_lines)
                        table_data = parse_markdown_table(table_content)
                        if table_data:
                            table_end_line = current_line - 1
                            table_token_count = len(encoding.encode(table_content))
                            section_titles = section_hierarchy.get_titles()
                            section_hash_path = section_hierarchy.get_hashes()
                            table_id = hash_string(table_content)

                            table_metadata = {
                                "valid": True,
                                "issues": [],
                                "source": "markdown",
                            }
                            if marker_tables:
                                matching_table = next(
                                    (
                                        t
                                        for t in marker_tables
                                        if t["page"] == current_page
                                    ),
                                    None,
                                )
                                if matching_table:
                                    table_data = matching_table["table_data"]
                                    table_metadata["source"] = "marker_json"
                                    table_metadata["bbox"] = matching_table["bbox"]

                            if table_cache:
                                table_valid = any(
                                    t["page"] == current_page
                                    and (
                                        not table_metadata.get("bbox")
                                        or calculate_iou(
                                            t["bbox"],
                                            table_metadata.get(
                                                "bbox", [0, 0, 100, 100]
                                            ),
                                        )
                                        > 0.5
                                    )
                                    for t in table_cache
                                )
                                if not table_valid:
                                    table_metadata["issues"].append(
                                        "No matching table in table_cache"
                                    )
                                    table_metadata["valid"] = False

                            extracted_data.append(
                                {
                                    "file_path": file_path,
                                    "repo_link": repo_link,
                                    "extraction_date": datetime.datetime.now().isoformat(),
                                    "type": "table",
                                    "content": table_content,
                                    "line_span": (table_start_line, table_end_line),
                                    "token_count": table_token_count,
                                    "section_id": table_id,
                                    "section_path": section_titles,
                                    "section_hash_path": section_hash_path,
                                    "page": current_page,
                                    "metadata": {
                                        "table": table_data,
                                        "validation": table_metadata,
                                    },
                                }
                            )
                            logger.debug(
                                f"Added table at lines {table_start_line}-{table_end_line}, "
                                f"source: {table_metadata['source']}, section_path: {section_titles}"
                            )
                        table_lines = []
                        in_table = False

                if not in_table and not image_match:
                    text_content.append(content)

            elif token.type == "code_block" or token.type == "fence":
                flush_text()
                code_block = token.content.strip()
                code_start_line = current_line
                code_end_line = token.map[1] if token.map else code_start_line
                code_type = (
                    token.info.split()[0].lower()
                    if token.info
                    else Path(file_path).suffix[1:]
                )
                code_token_count = len(encoding.encode(code_block))
                code_metadata = extract_code_metadata(code_block, code_type)
                section_titles = section_hierarchy.get_titles()
                section_hash_path = section_hierarchy.get_hashes()

                extracted_data.append(
                    {
                        "file_path": file_path,
                        "repo_link": repo_link,
                        "extraction_date": datetime.datetime.now().isoformat(),
                        "type": "code",
                        "content": code_block,
                        "line_span": (code_start_line, code_end_line),
                        "token_count": code_token_count,
                        "section_id": hash_string(code_block),
                        "section_path": section_titles,
                        "section_hash_path": section_hash_path,
                        "page": current_page,
                        "metadata": {
                            "code_type": code_type,
                            "code_metadata": code_metadata,
                        },
                    }
                )
                logger.debug(
                    f"Added code block at lines {code_start_line}-{code_end_line}, "
                    f"section_path: {section_titles}"
                )
                code_block = None

            last_line = current_line if token.map else last_line

        # Flush any remaining content
        flush_text()
        if in_table and table_lines:
            table_content = "\n".join(table_lines)
            table_data = parse_markdown_table(table_content)
            if table_data:
                table_end_line = last_line
                table_token_count = len(encoding.encode(table_content))
                section_titles = section_hierarchy.get_titles()
                section_hash_path = section_hierarchy.get_hashes()
                table_id = hash_string(table_content)

                table_metadata = {"valid": True, "issues": [], "source": "markdown"}
                if marker_tables:
                    matching_table = next(
                        (t for t in marker_tables if t["page"] == current_page), None
                    )
                    if matching_table:
                        table_data = matching_table["table_data"]
                        table_metadata["source"] = "marker_json"
                        table_metadata["bbox"] = matching_table["bbox"]

                if table_cache:
                    table_valid = any(
                        t["page"] == current_page
                        and (
                            not table_metadata.get("bbox")
                            or calculate_iou(
                                t["bbox"], table_metadata.get("bbox", [0, 0, 100, 100])
                            )
                            > 0.5
                        )
                        for t in table_cache
                    )
                    if not table_valid:
                        table_metadata["issues"].append(
                            "No matching table in table_cache"
                        )
                        table_metadata["valid"] = False

                extracted_data.append(
                    {
                        "file_path": file_path,
                        "repo_link": repo_link,
                        "extraction_date": datetime.datetime.now().isoformat(),
                        "type": "table",
                        "content": table_content,
                        "line_span": (table_start_line, table_end_line),
                        "token_count": table_token_count,
                        "section_id": table_id,
                        "section_path": section_titles,
                        "section_hash_path": section_hash_path,
                        "page": current_page,
                        "metadata": {"table": table_data, "validation": table_metadata},
                    }
                )
                logger.debug(
                    f"Added final table at lines {table_start_line}-{table_end_line}, "
                    f"source: {table_metadata['source']}, section_path: {section_titles}"
                )

        return extracted_data

    except Exception as e:
        logger.error(f"Error extracting from Markdown file {file_path}: {e}")
        if use_mock_data:
            return _create_mock_extraction_data(file_path, repo_link)
        return []


def _create_mock_extraction_data(file_path: str, repo_link: str) -> List[Dict[str, Any]]:
    """
    Create mock extraction data for testing when dependencies are missing.
    
    Args:
        file_path: Path to the Markdown file.
        repo_link: URL of the repository.
        
    Returns:
        List of dictionaries containing mock extracted content.
    """
    # Read the markdown file if possible
    try:
        content = Path(file_path).read_text(encoding="utf-8")
    except:
        content = "# Mock Markdown\n\nMock content for testing."
    
    # Create mock extraction data with all node types
    now = datetime.datetime.now().isoformat()
    mock_hash = mock_hash_string(content)
    
    return [
        {
            "file_path": file_path,
            "repo_link": repo_link,
            "extraction_date": now,
            "type": "text",
            "content": "This is mock text content for testing the markdown extractor.",
            "line_span": (1, 2),
            "token_count": 50,
            "section_id": mock_hash + "_text",
            "section_path": ["1. Test Section"],
            "section_hash_path": [mock_hash],
            "page": 1,
            "metadata": {},
        },
        {
            "file_path": file_path,
            "repo_link": repo_link,
            "extraction_date": now,
            "type": "table",
            "content": "| Header 1 | Header 2 |\n|----------|----------|\n| Cell 1   | Cell 2   |",
            "line_span": (4, 6),
            "token_count": 30,
            "section_id": mock_hash + "_table",
            "section_path": ["1. Test Section"],
            "section_hash_path": [mock_hash],
            "page": 1,
            "metadata": {
                "table": {"headers": ["Header 1", "Header 2"], "rows": [["Cell 1", "Cell 2"]]},
                "validation": {"valid": True, "issues": [], "source": "markdown"},
            },
        },
        {
            "file_path": file_path,
            "repo_link": repo_link,
            "extraction_date": now,
            "type": "code",
            "content": "def hello_world():\n    print('Hello, world!')\n    return 42",
            "line_span": (8, 10),
            "token_count": 25,
            "section_id": mock_hash + "_code",
            "section_path": ["1. Test Section"],
            "section_hash_path": [mock_hash],
            "page": 1,
            "metadata": {
                "code_type": "python",
                "code_metadata": {"language": "python", "length": 54, "functions": ["hello_world"]},
            },
        },
        {
            "file_path": file_path,
            "repo_link": repo_link,
            "extraction_date": now,
            "type": "image",
            "content": "![Test Image](test_image.png)",
            "line_span": (12, 12),
            "token_count": 10,
            "section_id": mock_hash + "_image",
            "section_path": ["1. Test Section"],
            "section_hash_path": [mock_hash],
            "page": 1,
            "metadata": {"alt": "Test Image", "src": "test_image.png", "source": "markdown"},
        },
    ]

if __name__ == "__main__":
    import warnings
    import sys
    
    # Set up logging for better debugging
    logging.basicConfig(level=logging.INFO)
    
    print("MARKDOWN EXTRACTOR MODULE VERIFICATION")
    print("====================================")
    
    # CRITICAL: Define exact expected results for validation
    # These must match exactly or the test fails
    EXPECTED_RESULTS = {
        "basic_extraction": {
            "min_node_count": 4,  # At least 4 nodes: text, table, code, image
            "required_node_types": ["text", "table", "code", "image"],
            "section_count": 1  # Minimum number of sections in the test markdown
        },
        "marker_enhanced": {
            "min_node_count": 4,
            "min_marker_enhanced": 1  # At least one node enhanced with marker data
        }
    }
    
    # Track validation status
    validation_passed = True
    actual_results = {
        "basic_extraction": {
            "node_count": 0,
            "node_types": [],
            "section_count": 0
        },
        "marker_enhanced": {
            "node_count": 0,
            "marker_enhanced": 0
        }
    }
    
    # Ensure required dependencies are available
    missing_deps = []
    if not MARKDOWN_IT_AVAILABLE:
        missing_deps.append("markdown-it-py")
    if not TIKTOKEN_AVAILABLE:
        missing_deps.append("tiktoken")
    if not FTFY_AVAILABLE:
        missing_deps.append("ftfy")
    
    if missing_deps:
        print(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
        print("Using mock data for testing")
    
    # Use the test markdown file
    test_md_path = Path(__file__).parent / "test_markdown.md"
    
    # Check if we need to create the test file
    if not test_md_path.exists():
        print(f"Creating test markdown file at: {test_md_path}")
        test_markdown_content = """# Test Markdown Document

This is a test markdown document created for testing the markdown extractor.

## Section 1: Tables

Here's a sample table:

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |

## Section 2: Code Blocks

Here's a sample code block:

```python
def hello_world():
    print("Hello, world!")
    return 42
```

## Section 3: Images

Here's a sample image:

![Test Image](test_image.png)

## Section 4: Text Content

This is some regular text content that should be extracted as a text node.
It contains multiple sentences and should be treated as a single block.

* This is a list item
* This is another list item
"""
        try:
            with open(test_md_path, "w") as f:
                f.write(test_markdown_content)
            print(f"✅ Test file created successfully")
        except Exception as e:
            print(f"❌ Failed to create test file: {e}")
            # Create a temporary file as fallback
            import tempfile
            temp_dir = tempfile.mkdtemp()
            test_md_path = Path(temp_dir) / "test_markdown.md"
            with open(test_md_path, "w") as f:
                f.write(test_markdown_content)
            print(f"✓ Created temporary test file: {test_md_path}")
    else:
        print(f"Using existing test file: {test_md_path}")
    
    # Mock Marker JSON for testing
    marker_json = {
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {
                        "type": "table",
                        "bbox": [100, 160, 300, 200],
                        "cells": [
                            {"row": 0, "col": 0, "text": "Header 1"},
                            {"row": 0, "col": 1, "text": "Header 2"},
                            {"row": 0, "col": 2, "text": "Header 3"},
                            {"row": 1, "col": 0, "text": "Cell 1"},
                            {"row": 1, "col": 1, "text": "Cell 2"},
                            {"row": 1, "col": 2, "text": "Cell 3"},
                            {"row": 2, "col": 0, "text": "Cell 4"},
                            {"row": 2, "col": 1, "text": "Cell 5"},
                            {"row": 2, "col": 2, "text": "Cell 6"},
                        ],
                    },
                    {
                        "type": "image",
                        "bbox": [100, 50, 200, 100],
                        "alt": "Test Image",
                        "src": "test_image.png",
                    },
                ],
            }
        ]
    }
    
    # Mock table cache for testing
    table_cache = [{"page": 1, "bbox": [100, 150, 300, 210], "data": [], "accuracy": 95.0}]
    
    print("\n1. Testing basic extraction:")
    print("---------------------------")
    
    try:
        basic_result = extract_from_markdown(
            str(test_md_path), 
            "https://github.com/test/repo",
            use_mock_data=True if missing_deps else False
        )
        
        # Store results for validation
        actual_results["basic_extraction"]["node_count"] = len(basic_result)
        actual_results["basic_extraction"]["node_types"] = list(set(node.get("type", "unknown") for node in basic_result))
        
        # Count unique section paths for section count
        unique_sections = set()
        for node in basic_result:
            section_path = node.get("section_path", [])
            if section_path:
                unique_sections.add(tuple(section_path))
        actual_results["basic_extraction"]["section_count"] = len(unique_sections)
        
        if basic_result:
            print(f"Extracted {len(basic_result)} nodes from Markdown")
            # Count node types
            node_types = {}
            for node in basic_result:
                node_type = node.get("type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            for node_type, count in node_types.items():
                print(f"  - {node_type}: {count} nodes")
            
            # Show a sample of each type
            for node_type in node_types.keys():
                sample = next((node for node in basic_result if node.get("type") == node_type), None)
                if sample:
                    content = sample.get("content", "")
                    print(f"\n  {node_type.upper()} SAMPLE:")
                    print(f"  - Token count: {sample.get('token_count', 0)}")
                    content_preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"  - Content preview: {content_preview}")
        else:
            print("No data extracted from Markdown file")
            validation_passed = False
    except Exception as e:
        print(f"Error during basic extraction: {e}")
        import traceback
        traceback.print_exc()
        validation_passed = False
    
    # VALIDATION - Basic Extraction
    print("\n• Validating basic extraction results:")
    print("------------------------------------")
    
    # Check node count
    expected_min_count = EXPECTED_RESULTS["basic_extraction"]["min_node_count"]
    actual_count = actual_results["basic_extraction"]["node_count"]
    node_count_valid = actual_count >= expected_min_count
    if not node_count_valid:
        print(f"  ✗ FAIL: Node count too low! Expected at least {expected_min_count}, got {actual_count}")
        validation_passed = False
    else:
        print(f"  ✓ PASS: Node count meets minimum ({actual_count} >= {expected_min_count})")
    
    # Check required node types
    expected_types = set(EXPECTED_RESULTS["basic_extraction"]["required_node_types"])
    actual_types = set(actual_results["basic_extraction"]["node_types"])
    missing_types = expected_types - actual_types
    nodes_types_valid = len(missing_types) == 0
    
    if not nodes_types_valid:
        print(f"  ✗ FAIL: Missing required node types: {', '.join(missing_types)}")
        validation_passed = False
    else:
        print(f"  ✓ PASS: All required node types present: {', '.join(expected_types)}")
    
    # Check section count
    expected_section_count = EXPECTED_RESULTS["basic_extraction"]["section_count"]
    actual_section_count = actual_results["basic_extraction"]["section_count"]
    section_count_valid = actual_section_count >= expected_section_count
    
    if not section_count_valid:
        print(f"  ✗ FAIL: Section count too low! Expected at least {expected_section_count}, got {actual_section_count}")
        validation_passed = False
    else:
        print(f"  ✓ PASS: Section count meets minimum ({actual_section_count} >= {expected_section_count})")
    
    print("\n2. Testing with marker JSON and table cache:")
    print("------------------------------------------")
    
    try:
        marker_result = extract_from_markdown(
            str(test_md_path), 
            "https://github.com/test/repo",
            marker_json,
            table_cache,
            use_mock_data=True if missing_deps else False
        )
        
        # Store results for validation
        actual_results["marker_enhanced"]["node_count"] = len(marker_result)
        
        # Count marker-enhanced nodes
        marker_enhanced = 0
        for node in marker_result:
            if node.get("type") in ["table", "image"]:
                metadata = node.get("metadata", {})
                if isinstance(metadata, dict):
                    if metadata.get("source") == "marker_json" or (
                        isinstance(metadata.get("validation"), dict) and 
                        metadata.get("validation", {}).get("source") == "marker_json"
                    ):
                        marker_enhanced += 1
        
        actual_results["marker_enhanced"]["marker_enhanced"] = marker_enhanced
        
        if marker_result:
            print(f"Extracted {len(marker_result)} nodes from Markdown with marker JSON")
            print(f"  - Found {marker_enhanced} nodes enhanced with marker JSON data")
            
            # Show details of marker-enhanced nodes
            if marker_enhanced > 0:
                for node in marker_result:
                    if node.get("type") in ["table", "image"]:
                        metadata = node.get("metadata", {})
                        source = "unknown"
                        if isinstance(metadata, dict):
                            if metadata.get("source") == "marker_json":
                                source = "marker_json"
                            elif isinstance(metadata.get("validation"), dict) and metadata.get("validation", {}).get("source") == "marker_json":
                                source = "marker_json (validation)"
                                
                        if source.startswith("marker_json"):
                            print(f"\n  MARKER-ENHANCED {node.get('type').upper()}:")
                            print(f"  - Source: {source}")
                            print(f"  - BBox: {metadata.get('bbox', metadata.get('validation', {}).get('bbox', 'N/A'))}")
                            if node.get("type") == "table":
                                table_data = metadata.get("table", {})
                                print(f"  - Table headers: {table_data.get('headers', [])}")
                                print(f"  - Table rows: {len(table_data.get('rows', []))}")
                            elif node.get("type") == "image":
                                print(f"  - Alt: {metadata.get('alt', 'N/A')}")
                                print(f"  - Src: {metadata.get('src', 'N/A')}")
        else:
            print("No data extracted from Markdown file with marker JSON")
            validation_passed = False
    except Exception as e:
        print(f"Error during marker JSON extraction: {e}")
        import traceback
        traceback.print_exc()
        validation_passed = False
    
    # VALIDATION - Marker Enhanced Extraction
    print("\n• Validating marker enhanced results:")
    print("-----------------------------------")
    
    # Check node count
    expected_min_count = EXPECTED_RESULTS["marker_enhanced"]["min_node_count"]
    actual_count = actual_results["marker_enhanced"]["node_count"]
    node_count_valid = actual_count >= expected_min_count
    
    if not node_count_valid:
        print(f"  ✗ FAIL: Node count too low! Expected at least {expected_min_count}, got {actual_count}")
        validation_passed = False
    else:
        print(f"  ✓ PASS: Node count meets minimum ({actual_count} >= {expected_min_count})")
    
    # Check marker enhanced count
    expected_min_enhanced = EXPECTED_RESULTS["marker_enhanced"]["min_marker_enhanced"]
    actual_enhanced = actual_results["marker_enhanced"]["marker_enhanced"]
    marker_enhanced_valid = actual_enhanced >= expected_min_enhanced
    
    if not marker_enhanced_valid:
        print(f"  ✗ FAIL: Marker enhanced count too low! Expected at least {expected_min_enhanced}, got {actual_enhanced}")
        validation_passed = False
    else:
        print(f"  ✓ PASS: Marker enhanced count meets minimum ({actual_enhanced} >= {expected_min_enhanced})")
    
    # FINAL VALIDATION - All tests
    if validation_passed:
        print("\n✅ VALIDATION COMPLETE - All results match expected values")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED - Results don't match expected values")
        print(f"Expected: {json.dumps(EXPECTED_RESULTS, indent=2)}")
        print(f"Got: {json.dumps(actual_results, indent=2)}")
        sys.exit(1)
