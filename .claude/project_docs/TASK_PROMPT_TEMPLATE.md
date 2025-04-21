# Task Implementation Request

## Project Context
- **Project**: pdf_extractor
- **Purpose**: Convert PDFs to ordered JSON objects
- **Path**: `/home/graham/workspace/experiments/pdf_extractor`

## Task Status
- **Current Task List**: {task.md}
- **Completed Tasks**: {completed_tasks}
- **Current Task**: {current_task_id} - {task_description}

## Implementation Requirements
1. Follow all standards in DEVELOPMENT_CORE.md
2. **CRITICAL**: Implement validation according to VALIDATION_REQUIREMENTS.md
   - Validation must check actual results against expected results
   - "No errors" is NOT sufficient for validation
   - Create fixtures for proper validation if needed
3. Ensure JSON output conforms to JSON_SCHEMA_FORMAT.md specification
4. Track progress according to TASK_WORKFLOW.md
5. Test with sample PDFs from src/input directory

## Expected Outputs
- Function should handle PDF extraction errors gracefully
- Output must match expected JSON schema format
- **MUST** include comprehensive validation against known good results
- Document any new PDF parsing techniques

## Troubleshooting Guidance
If you encounter issues:
1. Generate multiple search queries for perplexity
2. Review similar PDF extractors in the codebase
3. Start with minimal implementation and iterate
4. Document the solution for future reference

## Validation Checklist (Required for ALL Tasks)
- [ ] Created/updated test fixtures with expected results
- [ ] Implemented field-by-field validation against fixtures
- [ ] Validation compares actual content, not just structure
- [ ] Error reporting shows specific field mismatches
- [ ] Added comprehensive validation in __main__ block
