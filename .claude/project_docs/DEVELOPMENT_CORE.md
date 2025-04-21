# DEVELOPMENT_CORE
> Essential standards and workflows for the pdf_extractor project

## Project Context
- Working directory: `/home/graham/workspace/experiments/pdf_extractor`
- Project configuration: `pyproject.toml`
- Python version: 3.11.12
- Package manager: uv (not pip)

## Critical Development Standards

### Execution & Validation (HIGHEST PRIORITY)
- Use **uv run** to execute Python scripts (never use `python` directly)
- Every `if __name__ == "__main__":` block MUST include:
  1. Defined expected outputs as constants for PDF extraction results
  2. Explicit JSON schema validation with assertions
  3. Clear success/failure reporting for PDF processing

### Code Quality Essentials
- Prioritize runtime correctness for PDF extraction accuracy
- Prefer official PDF processing libraries (PyPDF, pdfminer, etc.)
- Keep modules under 500 lines for better maintainability
- Document module purpose, expected PDF inputs, and JSON output format
- Validate extraction results after every code change

### Workflow Discipline
- Execute standalone validation after editing PDF processing components
- Never move to next task until current PDF extraction features are fully validated
- Focus on addressing extraction accuracy issues before stylistic improvements
- Document techniques for handling edge cases in PDF parsing
