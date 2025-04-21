# JSON_SCHEMA_FORMAT
> Expected JSON output format for PDF extraction

## Core JSON Structure
All PDF extraction output must conform to this schema:

```json
{
  "metadata": {
    "filename": "example.pdf",
    "page_count": 10,
    "created_date": "2023-01-15",
    "last_modified": "2023-02-01",
    "extraction_date": "2025-04-21"
  },
  "sections": [
    {
      "id": "section-1",
      "title": "Introduction",
      "level": 1,
      "text": "Full extracted text content...",
      "page": 1,
      "metadata": {
        "font_size": 12,
        "is_bold": true,
        "section_type": "heading"
      }
    }
  ],
  "tables": [
    {
      "id": "table-1",
      "page": 3,
      "caption": "Table 1: Performance Metrics",
      "headers": ["Metric", "Value", "Unit"],
      "rows": [
        ["Throughput", "125", "MB/s"],
        ["Latency", "10", "ms"],
        ["Power", "0.5", "W"]
      ],
      "metadata": {
        "confidence": 0.95,
        "extraction_method": "visual"
      }
    }
  ],
  "figures": [
    {
      "id": "figure-1",
      "page": 4,
      "caption": "Figure 1: System Architecture",
      "description": "Diagram showing the system components and connections",
      "metadata": {
        "width": 500,
        "height": 300,
        "figure_type": "diagram"
      }
    }
  ],
  "references": [
    {
      "id": "ref-1",
      "text": "[1] Smith, J. et al. (2022). PDF Processing Techniques.",
      "page": 9
    }
  ]
}
```

## Schema Requirements

### Metadata Section
- **Required fields**: filename, page_count, extraction_date
- **Optional fields**: created_date, last_modified, author, title, subject

### Section Objects
- **Required fields**: id, text, page
- **Optional fields**: title, level, metadata
- Sections must preserve the hierarchical structure of the document
- Text must maintain original formatting, including paragraph breaks

### Table Objects
- **Required fields**: id, page, headers, rows
- **Optional fields**: caption, footnotes, metadata
- Tables must be fully structured with proper alignment of cells
- Each row must have the same number of cells as there are headers

### Figure Objects
- **Required fields**: id, page
- **Optional fields**: caption, description, metadata
- Figures should include descriptive text even if the image itself is not extracted

### Reference Objects
- **Required fields**: id, text
- **Optional fields**: page, url, doi
- References should maintain original citation format

## Validation Requirements
- All JSON output must be validated against this schema
- Missing required fields must trigger validation failures
- See VALIDATION_REQUIREMENTS.md for complete validation process

## Example Usage
When implementing a PDF extractor, ensure the output follows this schema:

```python
def extract_pdf(pdf_path):
    """Extract structured content from a PDF file."""
    # Initialize the output structure
    result = {
        "metadata": {},
        "sections": [],
        "tables": [],
        "figures": [],
        "references": []
    }
    
    # Extraction logic here...
    
    # Ensure all required fields are present
    validate_schema(result)
    
    return result
```

## Schema Evolution
- The schema may be extended but backward compatibility must be maintained
- New field additions are allowed, but existing fields must not change meaning
- Schema changes must be documented in this file with date of change
