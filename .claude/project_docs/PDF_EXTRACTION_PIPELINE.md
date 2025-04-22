# PDF EXTRACTION PIPELINE
> Complete workflow for PDF processing in the pdf_extractor project

## Pipeline Overview

The pdf_extractor implements a complete pipeline for extracting structured content from PDFs:

1. **PDF Loading**: Read PDF document into memory
2. **Content Extraction**: Extract content using a layered approach:
   - Marker-based extraction (primary)
   - Camelot table detection (fallback)
   - Qwen VL-2B vision language model (for complex visual elements)
3. **Conversion to Markdown**: Structure extracted content in markdown format
4. **JSON Conversion**: Convert markdown to ordered JSON objects
5. **Section Hierarchy Management**: Maintain document structure with section hierarchy class
6. **ArangoDB Storage**: Insert JSON objects into ArangoDB
7. **Query Interface**: Enable filtered queries of PDF content from ArangoDB

## Component Interactions

```
PDF Document
    ↓
PDF Loading
    ↓
┌───────────────────────┐
│ Content Extraction    │
│ ┌─────────────────┐   │
│ │ Marker          │   │
│ └─────────────────┘   │
│         ↓             │
│ ┌─────────────────┐   │
│ │ Camelot Tables  │   │
│ └─────────────────┘   │
│         ↓             │
│ ┌─────────────────┐   │
│ │ Qwen VL-2B      │   │
│ └─────────────────┘   │
└───────────────────────┘
    ↓
Markdown Generation
    ↓
Section Hierarchy Management
    ↓
Ordered JSON Conversion
    ↓
ArangoDB Storage
    ↓
Query Interface
```

## Key Components

1. **Markdown Extraction**:
   - Uses Marker for primary extraction
   - Preserves document hierarchy and structure
   - Handles text formatting and basic elements

2. **Table Detection**:
   - Camelot for text-based PDFs
   - Custom fallbacks for complex table structures
   - Specialized handling of merged cells and headers

3. **Visual Element Processing**:
   - Qwen VL-2B for visually complex elements
   - Optical Character Recognition (OCR) for scanned content
   - Image identification and description

4. **Section Hierarchy**:
   - Maintains parent-child relationships between sections
   - Preserves document structure in the JSON output
   - Enables hierarchical querying of content

5. **ArangoDB Integration**:
   - Storage of extracted JSON documents
   - Support for semantic, BM25, keyword, and hybrid queries
   - Query filtering capabilities

## Processing Characteristics

- **Computational Requirements**: Processing is computationally expensive
  - Expected runtime: Minutes per document
  - System requirements: 256GB RAM, 24GB GPU
  - No optimization for speed is necessary given hardware constraints

- **Error Handling Strategy**:
  - Log errors but continue processing
  - Skip elements that can't be processed
  - PDF extraction is inherently messy; robustness is prioritized over completeness

## Deployment Options

The pdf_extractor is deployable as:

1. **Docker Container**: Primary deployment method
   - FastAPI endpoints for agent interaction
   - CLI interface with identical functionality
   - Self-contained with all dependencies

2. **Interfaces**:
   - FastAPI endpoints for programmatic access
   - CLI for local or scripted interaction
   - Both interfaces provide identical functionality

## Integration Points

- **ArangoDB**: `/Users/robert/Documents/dev/workspace/experiments/mcp-doc-retriver/src/mcp_doc_retriever/arangodb`
  - Provides semantic, BM25, keyword, and hybrid query capabilities
  - Essential for document retrieval after extraction
