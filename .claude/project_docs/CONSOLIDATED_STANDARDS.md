# CONSOLIDATED_STANDARDS
> Single source of truth for pdf_extractor development standards

## Project Information
- **Project**: pdf_extractor
- **Purpose**: Convert PDFs to ordered JSON objects
- **Path**: `/home/graham/workspace/experiments/pdf_extractor`
- **Python Version**: 3.11.12
- **Package Manager**: uv (not pip)

## Access Information
- **SSH Connection**: `ssh -i ~/.ssh/id_ed25519_wsl2 graham@192.168.86.49`
- **Project Root**: `/home/graham/workspace/experiments/pdf_extractor/`
- **Configuration**: `/home/graham/workspace/experiments/pdf_extractor/pyproject.toml`
- **Main Task List**: `/home/graham/workspace/experiments/pdf_extractor/task.md`

## Development Standards Index
This document serves as an index to the complete development standards. Each document covers a specific aspect of development:

1. **VALIDATION_REQUIREMENTS.md** - Critical requirements for validating PDF extraction results
2. **CODE_DETAILS.md** - Comprehensive code standards for PDF extraction
3. **TASK_WORKFLOW.md** - Structured approach for task implementation and tracking
4. **FILE_OPERATIONS.md** - File operations and research capabilities
5. **PROJECT_ACCESS.md** - Connection details and environment configuration

## Key Standards (Cross-referenced)

### Execution Environment
- **ALWAYS** use `uv run` to execute Python scripts (never use `python` directly)
- For environment variables, use the `env` command: `env VAR_NAME="value" uv run command`
- Install dependencies with `llm install -e .` (not pip)

### Validation Requirements (Most Critical)
- **NEVER** rely on absence of errors as success criteria
- **ALWAYS** validate against known good expected outputs (stored as fixtures)
- **EVERY** module must include comprehensive validation in its `__main__` block
- See VALIDATION_REQUIREMENTS.md for complete validation specifications

### Development Workflow
- Complete each sub-task with thorough validation before proceeding
- Update the task list after completing each sub-task
- No task is complete until validation confirms expected outputs
- When confused, use perplexity to research with multiple search queries

### Code Quality
- Prioritize extraction accuracy over code style
- Keep modules under 500 lines
- Document all functions with purpose, inputs, and expected outputs
- Prefer official, maintained libraries for PDF processing

### Troubleshooting
- Use perplexity research with multiple search queries
- Review the codebase for similar implementations
- Test with simpler PDFs before complex ones
- Document solutions for future reference

## Document Changelog
- 2025-04-21: Initial creation of consolidated standards document
